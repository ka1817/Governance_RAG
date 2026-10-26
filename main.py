
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.genaration import RAGPipeline


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Governance RAG Assistant",
    description="Ask questions about governance documents using RAG.",
    version="1.0.0"
)


# --------------------------------------------------
# Frontend configuration
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")


# --------------------------------------------------
# Initialize RAG pipeline once
# --------------------------------------------------

rag = RAGPipeline()


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Question about the governance documents"
    )


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Governance RAG Assistant"
    }


# --------------------------------------------------
# RAG question-answering endpoint
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        answer = rag.ask(question)

        return {
            "question": question,
            "answer": answer
        }

    except Exception:
        # Avoid exposing API keys or internal details to clients.
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check the server logs."
        )
