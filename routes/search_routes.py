from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from controllers.search_controller import search

router = APIRouter(prefix="/api/v1", tags=["Search"])


class SearchRequest(BaseModel):
    query: str


@router.post("/search")
async def search_query(request: SearchRequest):
    """
    Search the vector database for the most relevant chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = search(request.query)

        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))