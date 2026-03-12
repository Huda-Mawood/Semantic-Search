from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from controllers.rag_controller import run_rag

router = APIRouter(prefix="/api/v1", tags=["RAG"])


class RAGRequest(BaseModel):
    question: str


@router.post("/rag")
async def rag_query(request: RAGRequest):
    """
    Full Agentic RAG pipeline:
    Retrieve → Grade → Generate → Hallucination Check → Answer Check
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = run_rag(request.question)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))