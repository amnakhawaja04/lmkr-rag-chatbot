# utils.py
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_lmkr_documents(path: Path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"DATA_DIR {path} does not exist. Create it and put your LMKR .txt files inside.")
    all_docs = []
    txt_files = list(path.rglob("*.txt"))
    print(f"[LMKR] Found {len(txt_files)} .txt files under {path}")
    for fp in txt_files:
        try:
            loader = TextLoader(str(fp), autodetect_encoding=True)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"[LMKR] WARNING: Skipping file {fp} due to error: {e}")
    print(f"[LMKR] Loaded {len(all_docs)} documents after skipping bad files.")
    if not all_docs:
        raise ValueError("No valid .txt documents could be loaded from {path}. Check that your files are readable text.")
    return all_docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=60,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[LMKR] Split into {len(chunks)} chunks")
    if not chunks:
        raise ValueError("No chunks created. Check that your .txt files have content.")
    return chunks


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def extract_answer(text: str) -> str:
    marker = "Helpful LMKR-specific answer:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    for prefix in ["Assistant:", "assistant:", "Answer:", "answer:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip()


def history_to_str(history_pairs: List[Tuple[str, str]]) -> str:
    return "\n".join([f"User: {u}\nAssistant: {a}" for (u, a) in history_pairs])


def is_smalltalk(question: str) -> bool:
    q = question.lower().strip()
    greetings = {
        "hi",
        "hello",
        "hey",
        "salam",
        "salaam",
        "asalam o alaikum",
        "assalam o alaikum",
        "good morning",
        "good evening",
    }
    if q in greetings:
        return True
    if any(p in q for p in ["how are you", "how's it going"]):
        return True
    if any(
        p in q
        for p in [
            "what can you do",
            "what can you help me with",
            "how can you help",
            "who are you",
            "what are you",
        ]
    ):
        return True
    return False


def smalltalk_answer(question: str) -> str:
    q = question.lower().strip()
    if q in {
        "hi",
        "hello",
        "hey",
        "salam",
        "salaam",
        "asalam o alaikum",
        "assalam o alaikum",
        "good morning",
        "good evening",
    }:
        return "Hi! 👋 I’m LMKR Assistant. How can I help you?\n\n"
    if "what can you" in q or "how can you help" in q:
        return (
            "I’m LMKR’s AI assistant. I can:\n"
            "- Explain about LMKR\n"
            "- Describe LMKR products and solutions (e.g., GeoGraphix, GVERSE)\n"
            "- Summarize services, projects, and industries from your LMKR knowledge base\n"
            "- Extract contact details and other info from LMKR documents\n\n"
            "Just ask any LMKR-related question to get started."
        )
    if "how are you" in q:
        return "I’m running perfectly! How can I help you with LMKR today 😊?"
    return (
        "Hi! I’m LMKR Assistant. I’m here to answer LMKR-related questions — "
        "company info, products, services, projects, and contact details.\n"
        "What would you like to know?"
    )
