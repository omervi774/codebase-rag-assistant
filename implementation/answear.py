from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model=MODEL, temperature=0.0)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the custom-closet-app codebase.
You are chatting with a user about the custom-closet-app codebase.
Use the given context when it is relevant.
If you don't know the answer, say so.

Context:
{context}
"""


def normalize_content(content) -> str:
    """
    Normalize Gradio / UI message content into plain text.
    Handles strings, lists, dicts, and None safely.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, (int, float, bool)):
        return str(content)

    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(normalize_content(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        # Most useful common fields first
        for key in ("text", "content", "value", "message"):
            if key in content:
                return normalize_content(content[key])
        return str(content)

    return str(content)


def fetch_context(question: str) -> list[Document]:
    question = normalize_content(question).strip()
    if not question:
        return []
    return retriever.invoke(question)


def history_to_langchain_messages(history):
    messages = []
    for msg in history:
        role = msg.get("role")
        content = normalize_content(msg.get("content", ""))

        if not content.strip():
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def combined_question(question: str, history) -> str:
    current_question = normalize_content(question).strip()

    prior_user_questions = []
    for msg in history:
        if msg.get("role") == "user":
            text = normalize_content(msg.get("content", "")).strip()
            if text:
                prior_user_questions.append(text)

    if prior_user_questions:
        return "\n".join(prior_user_questions + [current_question])

    return current_question


def answer_question(question: str, history=None) -> tuple[str, list[Document]]:
    if history is None:
        history = []

    question = normalize_content(question).strip()
    if not question:
        return "Please enter a question.", []

    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(history_to_langchain_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, docs