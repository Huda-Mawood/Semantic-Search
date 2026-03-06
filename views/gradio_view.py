import gradio as gr
from controllers.search_controller import search


def search_interface(query: str) -> str:
    """
    Gradio interface function for semantic search.
    """
    if not query.strip():
        return "Please enter a question."

    results = search(query)

    if not results:
        return "No results found."

    output = ""
    for i, result in enumerate(results, start=1):
        output += f"### Result {i}\n"
        output += f"**Score:** {result['score']}\n"
        output += f"**Page:** {result['page_number']}\n"
        output += f"**Text:**\n{result['text']}\n"
        output += "---\n"

    return output


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="LLM Book Search") as demo:
        gr.Markdown("# 📚 Hands-On LLM Book - Semantic Search")
        gr.Markdown("Ask any question about the book and get the most relevant answers.")

        with gr.Row():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are transformer models?",
                lines=2
            )

        search_btn = gr.Button("🔍 Search", variant="primary")
        output = gr.Markdown(label="Results")

        search_btn.click(
            fn=search_interface,
            inputs=query_input,
            outputs=output
        )

    return demo