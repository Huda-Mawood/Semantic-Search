import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from controllers.ingest_controller import ingest_pdf
from models.embeddings_model import embed_documents
from models.vector_store import store_chunks

router = APIRouter(prefix="/api/v1", tags=["Ingest"])

TEMP_DIR = "./data/temp"
BATCH_SIZE = 90  # Cohere max is 96, we use 90 to be safe
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, chunk it,
    embed it with Cohere in batches, and store in ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_path = os.path.join(TEMP_DIR, file.filename)

    try:
        # Step 1: Save PDF temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Extract text + chunk
        chunks = ingest_pdf(temp_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        # Step 3: Embed chunks in batches + store in ChromaDB
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i: i + BATCH_SIZE]
            batch_texts = [chunk["text"] for chunk in batch_chunks]

            batch_embeddings = embed_documents(batch_texts)
            store_chunks(batch_chunks, batch_embeddings)

        return {
            "message": "PDF ingested successfully.",
            "filename": file.filename,
            "total_chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)