from typing import List, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class CVRetriever:
    def __init__(self, documents: List[Document]) -> None:
        self.documents: List[Document] = documents
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retriever: Optional[FAISS] = None

    def build_retriever(self) -> None:
        db = FAISS.from_documents(self.documents, self.embedding_model)
        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.retriever:
            raise ValueError("Retriever not built. Call build_retriever() first.")
        return self.retriever.get_relevant_documents(query)
