from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss
from pathlib import Path
import json


class KnowledgeSource:
    def __init__(self, dataset_path: str, embeddings_model) -> None:
        self.dataset_path = dataset_path
        self.embeddings_model = embeddings_model

        # Create vector store if one does not already exist
        if not Path(self.dataset_path).exists():
            # Get the embedding size for the embeddings_model
            embedding_dim = len(self.embeddings_model.embed_query("test"))
            # Create a FAISS index with Inner Product and Cosine Similarity-like behavior (with normalized vectors)
            index = faiss.IndexFlatIP(embedding_dim)

            self.vectorstore = FAISS(
                index=index,
                embedding_function=self.embeddings_model,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=True,  # Normalization provides same functionality as cosine similarity
            )

    def _split_documents(self, documents: Document) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    def retrieve_relevant_documents(self, query: str, k: int = 5) -> str | None:
        if not Path(self.dataset_path).exists():
            print(
                "Your FAISS index has not been saved yet. Please save documents before accessing them."
            )
            return

        vectorstore = FAISS.load_local(
            self.dataset_path,
            self.embeddings_model,
            allow_dangerous_deserialization=True,
        )

        # Retrieve similar documents
        retrieved_docs = vectorstore.similarity_search(query, k=k)
        document_info = "\n".join([doc.page_content for doc in retrieved_docs])
        return document_info

    def add_knowledge(self, data: str, is_json: bool = False) -> None:
        if is_json:
            data = json.dumps(data, ensure_ascii=False)

        document = Document(page_content=data)
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
