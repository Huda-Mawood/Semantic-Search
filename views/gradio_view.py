import gradio as gr
from controllers.search_controller import search
from controllers.rag_controller import run_rag


def run_search(query: str, mode: str):
    if not query.strip():
        return []

    if mode == "Semantic Search":
        results = search(query)
        if not results:
            return []

        cards = []
        for result in results:
            cards.append({
                "type": "semantic",
                "page": result["page_number"],
                "text": result["text"],
                "score": result["score"]
            })
        return cards

    elif mode == "RAG":
        result = run_rag(query)
        cards = [{"type": "answer", "text": result["generation"]}]
        for source in result["sources"]:
            cards.append({
                "type": "rag_source",
                "page": source["page_number"],
                "text": source["text"]
            })
        return cards

    return []


def create_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LLM Book Search",
        css="""
        .card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            background: #fafafa;
        }
        """
    ) as demo:

        gr.Markdown("# Hands-On LLM Book - Search")
        gr.Markdown("Ask any question about the book.")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What are transformer models?",
                    lines=2
                )
            with gr.Column(scale=1):
                mode_selector = gr.Radio(
                    choices=["Semantic Search", "RAG"],
                    value="Semantic Search",
                    label="Mode"
                )

        search_btn = gr.Button("Search", variant="primary")

        @gr.render(inputs=[query_input, mode_selector], triggers=[search_btn.click])
        def show_results(query, mode):
            if not query.strip():
                gr.Markdown("Please enter a question.")
                return

            cards = run_search(query, mode)

            if not cards:
                gr.Markdown("No results found.")
                return

            sources_header_shown = False

            for card in cards:
                if card["type"] == "answer":
                    with gr.Group():
                        gr.Markdown("### Answer")
                        gr.Textbox(
                            value=card["text"],
                            label="",
                            lines=5,
                            interactive=False
                        )

                elif card["type"] == "semantic":
                    with gr.Group():
                        with gr.Row():
                            with gr.Column(scale=4):
                                gr.Textbox(
                                    value=card["text"],
                                    label=f"Page {card['page']}",
                                    lines=3,
                                    interactive=False
                                )
                            with gr.Column(scale=1, min_width=100):
                                gr.Textbox(
                                    value=str(card["score"]),
                                    label="Score",
                                    interactive=False
                                )

                elif card["type"] == "rag_source":
                    if not sources_header_shown:
                        gr.Markdown("### Referenced Passages from the Book")
                        sources_header_shown = True
                    with gr.Group():
                        gr.Textbox(
                            value=card["text"],
                            label=f"Page {card['page']}",
                            lines=3,
                            interactive=False
                        )

    return demo