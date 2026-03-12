from typing import List
from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph

from models.embeddings_model import embed_query
from models.vector_store import query_collection
from config import settings

# LLM
llm_json = ChatOllama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    format="json",
    temperature=0
)
llm_text = ChatOllama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0
)


# Graph State
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]


# Retrieval Grader
retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' as a JSON with a single key 'score'.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Retrieved document: {document}
User question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
retrieval_grader = retrieval_grader_prompt | llm_json | JsonOutputParser()


# RAG Generator
generation_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant for question-answering tasks about Large Language Models.
Use the following retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and accurate.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
Context: {context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = generation_prompt | llm_text | StrOutputParser()


# Hallucination Grader
hallucination_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an answer is grounded in a set of facts.
Give a binary score 'yes' or 'no' as a JSON with a single key 'score'.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Facts: {documents}
Answer: {generation}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = hallucination_grader_prompt | llm_json | JsonOutputParser()


# Answer Grader
answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an answer resolves a question.
Give a binary score 'yes' or 'no' as a JSON with a single key 'score'.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Answer: {generation}
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = answer_grader_prompt | llm_json | JsonOutputParser()


# Graph Nodes
def retrieve(state: GraphState) -> GraphState:
    """Retrieve chunks from ChromaDB using HuggingFace embeddings."""
    print("--- RETRIEVE ---")
    question = state["question"]

    query_embedding = embed_query(question)
    results = query_collection(query_embedding, top_k=settings.TOP_K_RESULTS)

    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
        )
    ]
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState) -> GraphState:
    """Filter out irrelevant documents."""
    print("--- GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        if score.get("score", "no").lower() == "yes":
            print("  Relevant")
            filtered_docs.append(doc)
        else:
            print("  Not relevant")

    return {"documents": filtered_docs, "question": question}


def generate(state: GraphState) -> GraphState:
    """Generate answer from relevant documents."""
    print("--- GENERATE ---")
    question = state["question"]
    documents = state["documents"]

    context = format_docs(documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# Conditional Edges
def decide_to_generate(state: GraphState) -> str:
    """Generate only if we have relevant documents."""
    print("--- DECIDE TO GENERATE ---")
    if not state["documents"]:
        print("No relevant documents found.")
        return "no_docs"
    return "generate"


def grade_generation(state: GraphState) -> str:
    """Check hallucination then check if answer resolves question."""
    print("--- GRADE GENERATION ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({
        "documents": format_docs(documents),
        "generation": generation
    })
    if score.get("score", "no").lower() != "yes":
        print("Hallucination detected - retrying")
        return "not supported"

    score = answer_grader.invoke({
        "question": question,
        "generation": generation
    })
    if score.get("score", "no").lower() == "yes":
        print("Generation is useful")
        return "useful"

    print("Generation does not address question")
    return "not useful"


# Build Graph
def build_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "no_docs": END,
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "useful": END,
            "not useful": END,
            "not supported": "generate",
        },
    )

    return workflow.compile()


rag_app = build_rag_graph()


def run_rag(question: str) -> dict:
    """
    Run the full Agentic RAG pipeline.

    Args:
        question: User question

    Returns:
        Dict with question, generation, and sources
    """
    result = None
    for output in rag_app.stream({"question": question}):
        result = output

    if not result:
        return {
            "question": question,
            "generation": "No relevant information found.",
            "sources": []
        }

    final = list(result.values())[-1]
    generation = final.get("generation", "No answer generated.")
    documents = final.get("documents", [])

    sources = [
        {
            "text": doc.page_content,
            "page_number": doc.metadata.get("page_number"),
            "source": doc.metadata.get("source"),
        }
        for doc in documents
    ]

    return {
        "question": question,
        "generation": generation,
        "sources": sources,
    }