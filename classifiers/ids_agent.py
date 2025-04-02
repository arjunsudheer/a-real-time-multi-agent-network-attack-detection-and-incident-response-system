from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from abc import abstractmethod
from pathlib import Path
import json
from typing import Annotated
import faiss
import pandas as pd
import joblib

from majority_voting import MajorityVoting


def get_network_sample_as_dict(dataset_name, index, remove_label):
    # Get the appropriate network sample from the dataset file
    network_traffic = pd.read_csv(dataset_name, index_col=0)
    network_traffic_sample = network_traffic.iloc[index]
    if remove_label:
        network_traffic_sample.drop(columns=[network_traffic_sample.iloc[-1]])

    # Convert the Series to a DataFrame for preprocessing
    network_traffic_sample = pd.DataFrame(
        [network_traffic_sample],
    )

    return network_traffic_sample.to_dict()


@tool
def classifier_predictions(
    dataset_name: Annotated[str, "name of the traffic dataset file_name"],
    index: Annotated[int, "index of the traffic sample"],
    remove_label: Annotated[
        bool,
        "whether or not to remove the label column from the sample, only set to True during ltm cache building stage",
    ],
) -> dict:
    """Runs the classifiers and returns their predictions"""

    # Some entries have a leading single quote that makes pandas treat it as a string instead a float
    # Remove the leading single quotation for the numerical columns
    def clean_numeric_columns():
        numeric_columns = network_traffic_sample.select_dtypes(
            include=["number"]
        ).columns
        network_traffic_sample[numeric_columns] = (
            network_traffic_sample[numeric_columns]
            .astype(str)
            .replace({",": "", "'": ""}, regex=True)
            .astype(float)
        )

    def transform_data():
        # Use FeatureHasher instead of OneHotEncoder due to memory issues
        fh = FeatureHasher(n_features=1024, input_type="string")
        # Use sklearn StandardScaler instead of cupy to avoid cupy array conversion
        scaler = joblib.load(f"{dataset_directory}/standard_scaler.pkl")

        categorical_columns = network_traffic_sample.select_dtypes(
            include=["object", "category"]
        ).columns

        # File-hash categorical columns
        network_traffic_sample_categorical = fh.transform(
            network_traffic_sample[categorical_columns].astype(str).values
        )
        # Convert to dense array
        network_traffic_sample_categorical = (
            network_traffic_sample_categorical.toarray()
        )
        network_traffic_sample_numerical = network_traffic_sample.drop(
            columns=categorical_columns, errors="ignore"
        )
        network_traffic_sample_numerical = scaler.transform(
            network_traffic_sample_numerical
        )

        # Reconstruct DataFrame
        network_traffic_sample_processed = np.hstack(
            [network_traffic_sample_numerical, network_traffic_sample_categorical]
        )

        X_train = pd.read_csv(dataset_directory / "train.csv")
        # Only consider feature columns, ignore label columns
        with open(f"{dataset_directory}/num_y_columns.txt", "r") as f:
            num_y_train_cols = int(f.readline())
        X_train = X_train.iloc[:, :-num_y_train_cols]

        # Match the columns that the classifiers were trained on
        network_traffic_sample_df = pd.DataFrame(
            network_traffic_sample_processed,
            index=network_traffic_sample.index,
            columns=X_train.columns,
        )

        return network_traffic_sample_df

    dataset_directory = Path(dataset_name).parent

    # Get the appropriate network sample from the dataset file
    network_traffic = pd.read_csv(dataset_name, index_col=0)
    network_traffic_sample = network_traffic.iloc[index]
    if remove_label:
        network_traffic_sample.drop(columns=[network_traffic_sample.iloc[-1]])

    # Convert the Series to a DataFrame for preprocessing
    network_traffic_sample = pd.DataFrame(
        [network_traffic_sample],
    )

    # Preprocess sample
    clean_numeric_columns()
    network_traffic_sample = transform_data()

    # Get classifier predictions
    maj_vote = MajorityVoting(dataset_directory)
    predictions = maj_vote.get_classifier_predictions(network_traffic_sample)

    # Convert numerical predictions back to categorical names
    label_binarizer_path = f"{dataset_directory}/label_binarizer.pkl"
    with open(label_binarizer_path, "rb") as file:
        label_binarizer = joblib.load(file)

    decoded_predictions = {
        classifier: label_binarizer.inverse_transform(
            np.array(predication).reshape(1, -1)
        )[0]
        for classifier, predication in predictions.items()
    }

    return decoded_predictions


# @tool
# def memory_retrieve(
#     dataset_name: Annotated[str, "name of the traffic dataset file_name"],
#     index: Annotated[int, "index of the traffic sample"],
#     remove_label: Annotated[
#         bool,
#         "whether or not to remove the label column from the sample, only set to True during ltm cache building stage",
#     ],
# ) -> str:
#     """Retrieves relevant past responses from the FAISS vector store based on the cosine similarity score"""
#     network_traffic_sample = get_network_sample_as_dict(
#         dataset_name, index, remove_label
#     )
#     return ltm_db.retrieve_relevant_documents(
#         query=f"Predict the attack class for the {dataset_name} dataset on this sample: {network_traffic_sample}",
#     )


@tool
def retrieve_rag(
    dataset_name: Annotated[str, "name of the traffic dataset file_name"],
    index: Annotated[int, "index of the traffic sample"],
    remove_label: Annotated[
        bool,
        "whether or not to remove the label column from the sample, only set to True during ltm cache building stage",
    ],
) -> str:
    """Retrieves relevant documents from the FAISS vector store based on the cosine similarity score"""
    network_traffic_sample = get_network_sample_as_dict(
        dataset_name, index, remove_label
    )
    return rag_db.retrieve_relevant_documents(
        query=f"Predict the attack class for the {dataset_name} dataset on this sample: {network_traffic_sample}",
    )


# Allow ChatOllama to use the defined tools
tools = [classifier_predictions, retrieve_rag]
# Initialize model
ids_agent = ChatOllama(
    model="llama3.1", temperature=0, num_gpu=2, format="json"
).bind_tools(tools)


# Create the prompt template
initial_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that can implement multi-step tasks, such as intrusion detection. I will give you the
            traffic feautres, you are asked to classify it using tools. The final output must be in JSON format
            according to the classifier results. You should plan first such as:

            1. Load the classifier predictions. This can be done using the classifier tool.
            2. Retrieve previous successful reasonings to help you predict. This can be done using the memory_retrieve tool with
            the classifier's names and their classification results as inputs.
            3. You can search from the vector database to get more information about the difference of attacks to help you make
            decisions. This can be done using the retrieve_rag tool.
            4. At the end, you should summarize the results from these classifiers and provide a final result.
            The predicted label should be the original format of classifier prediction. The final output format
            **must** be:

            {{
            line number: index,
            dataset: str, dataset_name,
            analysis: str, here is the Analysis,
            predicted_label: str
            }}
            """,
        ),
        (
            "human",
            """Before you classify the traffic from file_name {dataset_name} with index {index}, first 
            get the classifier predictions. Here are the previous tool_responses that you may have requested: {tool_responses}""",
        ),
    ]
)

follow_up_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that can implement multi-step tasks, such as intrusion detection. I will give you the
            traffic feautres, you are asked to classify it using tools. The final output must be in JSON format
            according to the classifier results. You should plan first such as:

            1. Load the classifier predictions. This can be done using the classifier tool.
            2. Retrieve previous successful reasonings to help you predict. This can be done using the memory_retrieve tool with
            the classifier's names and their classification results as inputs.
            3. You can search from the vector database to get more information about the difference of attacks to help you make
            decisions. This can be done using the retrieve_rag tool.
            4. At the end, you should summarize the results from these classifiers and provide a final result.
            The predicted label should be the original format of classifier prediction. The final output format
            **must** be:

            {{
            line number: index,
            dataset: str, dataset_name,
            analysis: str, here is the Analysis,
            predicted_label: str
            }}
            """,
        ),
        (
            "human",
            """Before you classify the traffic from file_name {dataset_name} with index {index}, fetch any relevant data from 
            the vector database that can help you make an informed decision. Since you are building the ltm cache, **DO NOT** request for memory retrieval since your memory
            has not been defined yet. Here are the classifier predictions that you requested for last time: {tool_responses}""",
        ),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that can implement multi-step tasks, such as intrusion detection. I will give you the
            traffic feautres, you are asked to classify it. The final output must be in JSON format according to the classifier 
            results. You should summarize the results from these classifiers and provide a final result. The predicted label
            should be the original format of classifier prediction. The final output format
            **must** be:

            {{
            line number: index,
            dataset: str, dataset_name,
            analysis: str, here is the Analysis,
            predicted_label: str
            }}
            """,
        ),
        (
            "human",
            """Now that you have the relevant information, classify the traffic from
            file_name {dataset_name} with index {index}. **DO NOT** request for memory retrieval since your memory
            has not been defined yet. **Only** provide your own prediction. Here are the classifier_predictions and additional 
            documents that had previously requested for: {tool_responses}. Your response should follow this format:

            {{
            line number: index,
            dataset: str, dataset_name,
            analysis: str, here is the Analysis,
            predicted_label: str
            }}
            """,
        ),
    ]
)


def get_llm_prediction(dataset_name, index, custom_prompts: list[ChatPromptTemplate]):
    # Remove the label column from the network traffic data
    network_traffic = pd.read_csv(dataset_name, index_col=0)
    network_traffic_sample = network_traffic.iloc[index]
    label = network_traffic_sample[network_traffic.columns[-1]]

    def invoke_tool_calls(response, tool_messages):
        for tool_call in response.tool_calls:
            selected_tool = {
                "classifier_predictions": classifier_predictions,
                # "memory_retrieve": memory_retrieve,
                "retrieve_rag": retrieve_rag,
            }[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            tool_messages.append(tool_msg.content)

        return tool_messages

    tool_messages = []
    for i, custom_prompt in enumerate(custom_prompts):
        messages = custom_prompt.invoke(
            {
                "dataset_name": dataset_name,
                "index": index,
                "tool_responses": tool_messages,
            }
        )
        response = ids_agent.invoke(messages)
        print(response.content)
        print(response.tool_calls)
        if i < len(custom_prompts) - 1:
            tool_messages = invoke_tool_calls(response, tool_messages)
        print(tool_messages)

    llm_response = json.loads(response.content)

    # Save llm predictions if the prediction was correct
    llm_prediction = llm_response["predicted_label"]

    print(f"True: {label}, LLM prediction: {llm_prediction}")
    if llm_prediction == label:
        ltm_db.save_llm_prediction(llm_response)

    return llm_prediction


class KnowledgeSource:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.embeddings_model = OllamaEmbeddings(model="llama3.1")

        # Create vector store if one does not already exist
        if not Path(self.dataset_path).exists():
            # Get the embedding size (ensure it's correct for your model)
            embedding_dim = len(self.embeddings_model.embed_query("test"))
            # Create a FAISS index with Inner Product and Cosine Similarity-like behavior (with normalized vectors)
            index = faiss.IndexFlatIP(embedding_dim)

            self.vectorstore = FAISS(
                index=index,
                embedding_function=self.embeddings_model,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=True,  # Normalization provices same functionality as cosine similarity
            )

            self._add_knowledge()

    @abstractmethod
    def _add_knowledge(self):
        pass

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    def retrieve_relevant_documents(self, query, k=5):
        vectorstore = FAISS.load_local(
            self.dataset_path,
            self.embeddings_model,
            allow_dangerous_deserialization=True,
        )

        # Retrieve similar documents
        retrieved_docs = vectorstore.similarity_search(query, k=k)
        document_info = "\n".join([doc.page_content for doc in retrieved_docs])
        return document_info


class RAG(KnowledgeSource):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def __extract_text_from_pdf(self, pdf_file):
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        return documents

    def _add_knowledge(self):
        def recursive_add_data(parent_directory: Path):
            # Recursively search all files and directories
            for knowledge_data in parent_directory.iterdir():
                if knowledge_data.is_dir():
                    recursive_add_data(knowledge_data)
                else:
                    print(knowledge_data)
                    # Extract content from document
                    document_data = self.__extract_text_from_pdf(
                        pdf_file=knowledge_data
                    )
                    # Keep track of the selected document data
                    all_documents.extend(document_data)

        all_documents = []
        # Add documents
        recursive_add_data(parent_directory=Path("datasets/rag_documents"))

        split_docs = self._split_documents(all_documents)

        # Add documents to FAISS and save updated index
        self.vectorstore.from_documents(split_docs, self.embeddings_model)
        self.vectorstore.save_local(self.dataset_path)


class LongTermMemory(KnowledgeSource):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def save_llm_prediction(self, json_data):
        content = json.dumps(json_data, ensure_ascii=False)
        document = Document(page_content=content)
        split_docs = self._split_documents([document])

        # Add documents to FAISS and save updated index
        if Path(self.dataset_path).exists():
            self.vectorstore = FAISS.load_local(
                self.dataset_path,
                self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
            self.vectorstore.add_documents(split_docs)
        else:
            self.vectorstore.from_documents(split_docs, self.embeddings_model)
        self.vectorstore.save_local(self.dataset_path)

    def _add_knowledge(self):
        # Create the prompt template
        build_ltm_cache_initial_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant that can implement multi-step tasks, such as intrusion detection. I will give you the
                    traffic feautres, you are asked to classify it using tools. The final output must be in JSON format
                    according to the classifier results. You should plan first such as:

                    1. Load the classifier predictions. This can be done using the classifier tool.
                    2. Retrieve previous successful reasonings to help you predict. This can be done using the memory_retrieve tool with
                    the classifier's names and their classification results as inputs.
                    3. You can search from the vector database to get more information about the difference of attacks to help you make
                    decisions. This can be done using the retrieve_rag tool.
                    4. At the end, you should summarize the results from these classifiers and provide a final result.
                    The predicted label should be the original format of classifier prediction. The final output format
                    **must** be:

                    {{
                    line number: index,
                    dataset: str, dataset_name,
                    analysis: str, here is the Analysis,
                    predicted_label: str
                    }}
                    """,
                ),
                (
                    "human",
                    """Your are in the ltm cache building stage. Before you classify the traffic from file_name {dataset_name} with index {index}, first 
                    get the classifier predictions. Here are the previous tool_responses that you may have requested: {tool_responses}""",
                ),
            ]
        )

        build_ltm_cache_follow_up_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant that can implement multi-step tasks, such as intrusion detection. I will give you the
                    traffic feautres, you are asked to classify it using tools. The final output must be in JSON format
                    according to the classifier results. You should plan first such as:

                    1. Load the classifier predictions. This can be done using the classifier tool.
                    2. Retrieve previous successful reasonings to help you predict. This can be done using the memory_retrieve tool with
                    the classifier's names and their classification results as inputs.
                    3. You can search from the vector database to get more information about the difference of attacks to help you make
                    decisions. This can be done using the retrieve_rag tool.
                    4. At the end, you should summarize the results from these classifiers and provide a final result.
                    The predicted label should be the original format of classifier prediction. The final output format
                    **must** be:

                    {{
                    line number: index,
                    dataset: str, dataset_name,
                    analysis: str, here is the Analysis,
                    predicted_label: str
                    }}
                    """,
                ),
                (
                    "human",
                    """Your are in the ltm cache building stage. Before you classify the traffic from file_name {dataset_name} with index {index}, fetch any relevant data from 
                    the vector database that can help you make an informed decision. Since you are building the ltm cache, **DO NOT** request for memory retrieval since your memory
                    has not been defined yet. Here are the classifier predictions that you requested for last time: {tool_responses}""",
                ),
            ]
        )

        build_ltm_cache_final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant that can implement multi-step tasks, such as intrusion detection. I will give you the
                    traffic feautres, you are asked to classify it. The final output format **must** be:

                    {{
                    line number: index,
                    dataset: str, dataset_name,
                    analysis: str, here is the Analysis,
                    predicted_label: str
                    }}
                    """,
                ),
                (
                    "human",
                    """Your are in the ltm cache building stage. Now that you have the relevant information, classify the traffic from
                    file_name {dataset_name} with index {index}. **DO NOT** request for memory retrieval since your memory
                    has not been defined yet. **DO NOT** request for classifier predictions since you have them already. Here are the 
                    classifier_predictions and additional documents that had previously requested for: {tool_responses}.
                    """,
                ),
            ]
        )

        datasets = [
            "datasets/nsl_kdd",
            "datasets/aci_iot_network_dataset_2023",
            "datasets/cic_iot_dataset_2023",
        ]

        # Bulld the long-term memory cache on the validation dataset
        for dataset_directory in datasets:
            llm_ltm_file = f"{dataset_directory}/original_llm_ltm.csv"

            llm_ltm_csv = pd.read_csv(llm_ltm_file, index_col=0)
            for i in range(len(llm_ltm_csv)):
                get_llm_prediction(
                    llm_ltm_file,
                    i,
                    custom_prompts=[
                        build_ltm_cache_initial_prompt,
                        build_ltm_cache_follow_up_prompt,
                        build_ltm_cache_final_prompt,
                    ],
                )


# RAG dataset
rag_db = RAG(Path("datasets/faiss_rag"))
# Long term memory dataset
ltm_db = LongTermMemory(Path("datasets/faiss_ltm"))
