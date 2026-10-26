
import os
import time
import hashlib

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


class GovernanceRetriever:
    def __init__(
        self,
        index_name="governance-rag-index",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        k=3
    ):
        self.index_name = index_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.k = k

        self.api_key = os.getenv("PINECONE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY is missing from your .env file."
            )

        self.pc = Pinecone(api_key=self.api_key)

        # Generate embeddings locally
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_store = None

    def create_index(self):
        """Create the Pinecone index if it doesn't exist."""
        existing_indexes = self.pc.list_indexes().names()

        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}")

            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            while not self.pc.describe_index(
                self.index_name
            ).status["ready"]:
                time.sleep(1)

            print("Index created successfully.")
        else:
            print(f"Using existing index: {self.index_name}")

        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings
        )

    def upload_documents(self, chunks):
        """Generate embeddings and upload chunks in batches."""
        if not chunks:
            raise ValueError("No document chunks provided.")

        if self.vector_store is None:
            self.create_index()

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]
            ids = []

            for doc in batch:
                source = str(doc.metadata.get("source", ""))
                page = str(doc.metadata.get("page", ""))
                content = doc.page_content

                unique_text = f"{source}|{page}|{content}"
                ids.append(
                    hashlib.sha256(
                        unique_text.encode("utf-8")
                    ).hexdigest()
                )

            self.vector_store.add_documents(
                documents=batch,
                ids=ids
            )

            print(
                f"Uploaded {min(start + len(batch), len(chunks))}"
                f"/{len(chunks)} chunks"
            )

        print("All chunks uploaded successfully.")

    def get_retriever(self):
        """Return a retriever for similarity search."""
        if self.vector_store is None:
            self.create_index()

        return self.vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )

    def retrieve(self, question):
        """Retrieve relevant chunks for a question."""
        retriever = self.get_retriever()
        return retriever.invoke(question)


