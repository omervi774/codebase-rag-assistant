import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv
project_folder = "custom-closet-app"
MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = Path(__file__).parent.parent / f"knowledge-base/{project_folder}"

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

ALLOWED_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx",
    ".css", ".scss",
    ".html", ".json",
    ".md", ".txt"
}

IGNORED_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "coverage",
    "__pycache__"
}


def fetch_documents():
    documents = []
    

    for file_path in KNOWLEDGE_BASE.rglob("*"):
        # Skip directories
        if not file_path.is_file():
            continue

        # Skip files inside ignored directories
        if any(part in IGNORED_DIRS for part in file_path.parts):
            continue

        # Keep only relevant file types
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"Skipping non-utf8 file: {file_path}")
            continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        document = {
            "content": content,
            "metadata": {
                "file_path": str(file_path.relative_to(KNOWLEDGE_BASE)),
                "file_name": file_path.name,
                "extension": file_path.suffix.lower(),
                "source_type": infer_source_type(file_path),
            }
        }

        documents.append(document)

    return documents


def infer_source_type(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext in {".js", ".jsx", ".ts", ".tsx"}:
        return "code"
    elif ext in {".md", ".txt"}:
        return "documentation"
    elif ext in {".css", ".scss", ".html"}:
        return "frontend_asset"
    elif ext == ".json":
        return "config"
    else:
        return "other"

def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.create_documents([doc["content"]], metadatas=[doc["metadata"]])
        all_chunks.extend(chunks)
    return all_chunks

def create_embeddings(chunks):
    # if Chroma exsists with db then delete it, otherwise create Chroma vector store and add chunks to it
    if os.path.exists(DB_NAME):     
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )
    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


    
if __name__ == "__main__":
    documents = fetch_documents() 
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")    
    vectorstore = create_embeddings(chunks)
    # print(f"Fetched {len(documents)} documents from {KNOWLEDGE_BASE}")
    # print(documents[0]["metadata"]["file_name"])
    # print(documents[0]["metadata"]["source_type"])
    # print(documents[10]["metadata"]["file_name"])
    # print(documents[10]["metadata"]["source_type"])
    # print(documents[10]["content"][:500])


