import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class RAGPipeline:
    def __init__(
        self,
        index_name="governance-rag-index",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="qwen/qwen3.8-27b",
        k=3
    ):
        # Validate API keys
        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY missing from .env")

        if not groq_key:
            raise ValueError("GROQ_API_KEY missing from .env")

        # Connect to existing Pinecone index
        self.pc = Pinecone(api_key=pinecone_key)

        if index_name not in self.pc.list_indexes().names():
            raise ValueError(
                f"Index '{index_name}' not found. "
                "Check your Pinecone index name."
            )

        # Use the same embedding model used during indexing
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Connect to existing vectors (no uploading)
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(index_name),
            embedding=self.embeddings
        )

        # Retrieve the top-k relevant chunks
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        # Initialize Groq LLM
        self.llm = ChatGroq(
            model_name=llm_model,
            temperature=0,
            streaming=True
        )

        # Prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a Governance RAG Assistant.

Answer the question using only the provided context.

Instructions:
- Provide clear and accurate answers.
- Do not invent facts or requirements.
- If the answer is not present in the context, say:
  "I could not find the answer in the provided documents."
- Use bullet points where appropriate.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        # Build the RAG pipeline
        self.rag_pipeline = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    def ask(self, question):
        return self.rag_pipeline.invoke(question)


