from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.genaration import RAGPipeline

app = FastAPI(
    title="Governance RAG API",
    description="Ask questions about governance documents",
    version="1.0.0"
)

# Initialize RAG pipeline
rag = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Governance RAG API is running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        answer = rag.ask(request.question)
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))