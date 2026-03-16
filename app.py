import gradio as gr
from dotenv import load_dotenv

from implementation.answear import answer_question

load_dotenv(override=True)

APP_TITLE = "Closet App Assistant"
APP_SUBTITLE = (
    "Ask questions about the closet app codebase. "
    "The assistant answers from the indexed project files, and the relevant retrieved docs are shown on the right."
)


def normalize_content(content) -> str:
    """
    Normalize message content coming from Gradio into plain text.
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
        for key in ("text", "content", "value", "message"):
            if key in content:
                return normalize_content(content[key])
        return str(content)

    return str(content)


def format_docs_text(docs):
    if not docs:
        return "Relevant retrieved documents will appear here."

    parts = []
    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        file_path = metadata.get("file_path", "Unknown path")
        source_type = metadata.get("source_type", "unknown")
        extension = metadata.get("extension", "")

        content = (doc.page_content or "").strip()
        preview = content[:1500]
        if len(content) > 1500:
            preview += "\n..."

        section = (
            f"[Document {i}]\n"
            f"file_path: {file_path}\n"
            f"source_type: {source_type}\n"
            f"extension: {extension}\n\n"
            f"{preview}"
        )
        parts.append(section)

    separator = "\n\n" + "=" * 80 + "\n\n"
    return separator.join(parts)


def put_message_in_chatbot(message, history):
    if history is None:
        history = []

    message = normalize_content(message).strip()
    if not message:
        return "", history

    return "", history + [{"role": "user", "content": message}]


def chat(history):
    if not history:
        return history, "Relevant retrieved documents will appear here."

    last_user_message = normalize_content(history[-1].get("content", "")).strip()
    prior_history = history[:-1]

    if not last_user_message:
        return history, "Relevant retrieved documents will appear here."

    answer, docs = answer_question(last_user_message, prior_history)

    history.append({"role": "assistant", "content": answer})
    docs_text = format_docs_text(docs)

    return history, docs_text


def clear_all():
    return [], "Relevant retrieved documents will appear here.", ""


def main():
    theme = gr.themes.Soft()

    with gr.Blocks(title=APP_TITLE, theme=theme) as ui:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_SUBTITLE}")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                )

                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the closet app...",
                    show_label=False,
                )

                clear_btn = gr.Button("Clear Chat")

            with gr.Column(scale=1):
                docs_box = gr.Textbox(
                    label="Relevant Docs",
                    value="Relevant retrieved documents will appear here.",
                    lines=30,
                    max_lines=30,
                    interactive=False,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, docs_box],
        )

        clear_btn.click(
            clear_all,
            inputs=[],
            outputs=[chatbot, docs_box, message],
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()