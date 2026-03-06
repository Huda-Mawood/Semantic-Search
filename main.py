from fastapi import FastAPI
from routes.ingest_routes import router as ingest_router
from routes.search_routes import router as search_router
from views.gradio_view import create_ui
import gradio as gr

app = FastAPI(
    title="Semantic Search API",
    description="Semantic search on Hands-On LLM book using Cohere + ChromaDB",
    version="1.0.0"
)

# Register routes
app.include_router(ingest_router)
app.include_router(search_router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


# Mount Gradio UI at /ui
gradio_app = create_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")