from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss
from pathlib import Path


class KnowledgeSource:
    def __init__(
        self,
        dataset_path: str,
        embeddings_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"tokenizer_kwargs": {"clean_up_tokenization_spaces": True}},
        ),
    ) -> None:
        """
        __init__ initializes the dataset path and embeddings path.

        Args:
            dataset_path (str): The path to store the FAISS index in.
            embeddings_model (HuggingFaceEmbeddings, optional): The embeddings model to use when writing data to the FAISS index. Defaults to HuggingFaceEmbeddings( model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"tokenizer_kwargs": {"clean_up_tokenization_spaces": True}}, ).
        """
        self.dataset_path = dataset_path
        self.embeddings_model = embeddings_model

    def _split_documents(self, documents: Document) -> list[Document]:
        """
        _split_documents splits a large document for adding to the FAISS index.

        Args:
            documents (Document): The document to be split.

        Returns:
            list[Document]: The split documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    def retrieve_relevant_knowledge(self, query: str, k: int = 5) -> str:
        """
        retrieve_relevant_knowledge retrieves data that are similar to the query from the FAISS index.

        Elements are fetched based on the cosine-similarity score.

        Args:
            query (str): The query to use when retrieving data from the FAISS index.
            k (int, optional): The number of results to return. Defaults to 5.

        Returns:
            str: The similar data based on the cosine similarity score that matches the query.
        """
        if not Path(self.dataset_path).exists():
            return "Your FAISS index has not been saved yet. Please save documents before accessing them."

        vectorstore = FAISS.load_local(
            self.dataset_path,
            self.embeddings_model,
            allow_dangerous_deserialization=True,
        )

        # Retrieve similar documents
        retrieved_docs = vectorstore.similarity_search(query, k=k)
        document_info = "\n".join([doc.page_content for doc in retrieved_docs])
        return document_info

    def add_knowledge(self, data: str) -> None:
        """
        add_knowledge adds data to the FAISS index.

        Internally converts the text-based data to a Document format to add to the FAISS index.

        Args:
            data (str): The data to be added to the FAISS index.
        """
        document = Document(page_content=data)
        split_docs = self._split_documents([document])

        # Load the index if it exists
        if Path(self.dataset_path).exists():
            vectorstore = FAISS.load_local(
                self.dataset_path,
                self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
        # Create a new index if it does not exist
        else:
            # Get the embedding size for the embeddings_model
            embedding_dim = len(self.embeddings_model.embed_query("test"))
            # Create a FAISS index with Inner Product and Cosine Similarity-like behavior (with normalized vectors)
            index = faiss.IndexFlatIP(embedding_dim)

            vectorstore = FAISS(
                index=index,
                embedding_function=self.embeddings_model,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=True,  # Normalization provides same functionality as cosine similarity
            )

        # Add documents to FAISS and save updated index
        vectorstore.add_documents(split_docs)
        vectorstore.save_local(self.dataset_path)
