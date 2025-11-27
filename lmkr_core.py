import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch 

from huggingface_hub import InferenceClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser



BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(r"C:\Users\afarooq\Desktop\lmkr-chatbot\lmkr_data")
INDEX_DIR = Path(r"C:\Users\afarooq\Desktop\lmkr-chatbot\faiss_lmkr")

INDEX_DIR.mkdir(parents=True, exist_ok=True)




# -------------------------------------------------------------------
# HUGGING FACE INFERENCE API CLIENT (LLM)
# -------------------------------------------------------------------

from huggingface_hub import InferenceClient

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

if HF_API_TOKEN is None:
    raise RuntimeError(
        "HF_API_TOKEN environment variable is not set.\n"
        "Create a read token at https://huggingface.co/settings/tokens "
        "and set it in your environment."
    )

# Let HF route to the right provider (Featherless, etc.)
hf_client = InferenceClient(
    model=HF_MODEL_ID,
    token=HF_API_TOKEN,
    timeout=60,
)



def hf_generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Call Hugging Face Inference chat-completion API with a single user message.

    `prompt` is already the full RAG prompt that includes context + history.
    We send it as the user content.
    """
    completion = hf_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Robust access (dataclass or dict-like)
    msg = completion.choices[0].message
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content", "")

    return content or ""

def _call_hf_from_promptvalue(prompt_value):
    """
    LangChain passes a PromptValue (e.g., StringPromptValue) here, not a raw str.
    Convert it to a string before sending to HF.
    """
    if hasattr(prompt_value, "to_string"):
        prompt_text = prompt_value.to_string()
    else:
        prompt_text = str(prompt_value)
    return hf_generate(prompt_text)




llm = RunnableLambda(_call_hf_from_promptvalue)



# -------------------------------------------------------------------
# DOCUMENT LOADING, CHUNKING, EMBEDDINGS, VECTOR STORE
# -------------------------------------------------------------------

def load_lmkr_documents(path: Path):
    """Load all .txt files recursively, skipping any broken ones."""
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"DATA_DIR {path} does not exist. "
            f"Create it and put your LMKR .txt files inside."
        )

    all_docs = []
    txt_files = list(path.rglob("*.txt"))
    print(f"[LMKR] Found {len(txt_files)} .txt files under {path}")

    for fp in txt_files:
        try:
            loader = TextLoader(str(fp), autodetect_encoding=True)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            # Just warn and skip this file instead of crashing everything
            print(f"[LMKR] WARNING: Skipping file {fp} due to error: {e}")

    print(f"[LMKR] Loaded {len(all_docs)} documents after skipping bad files.")

    if not all_docs:
        raise ValueError(
            f"No valid .txt documents could be loaded from {path}. "
            f"Check that your files are readable text."
        )

    return all_docs



def split_documents(docs):
    """Split docs into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[LMKR] Split into {len(chunks)} chunks")
    if not chunks:
        raise ValueError("No chunks created. Check that your .txt files have content.")
    return chunks



embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

index_file = INDEX_DIR / "index.faiss"

if not index_file.exists():
    print("[LMKR] No existing FAISS index found. Building a new one...")
    docs = load_lmkr_documents(DATA_DIR)
    chunks = split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(str(INDEX_DIR))
    print(f"[LMKR] Index saved to {INDEX_DIR}")
else:
    print("[LMKR] FAISS index found. Loading from disk...")
    vectordb = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

print("[LMKR] Vector store ready.")



# -------------------------------------------------------------------
# RAG PIPELINE + SMALL-TALK ROUTER
# -------------------------------------------------------------------

retriever = vectordb.as_retriever(search_kwargs={"k": 10})


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def extract_answer(text: str) -> str:
    """
    Post-process the model output to remove prompt echoes or extra headers.
    """
    marker = "Helpful LMKR-specific answer:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    for prefix in ["Assistant:", "assistant:", "Answer:", "answer:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip()


def history_to_str(history_pairs: List[Tuple[str, str]]) -> str:
    """
    Convert list of (user, assistant) tuples into a chat history text block.
    """
    return "\n".join(
        [f"User: {u}\nAssistant: {a}" for (u, a) in history_pairs]
    )


# ---- Small-talk routing (same behavior) ----
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


rag_template = """
You are LMKR Assistant, a knowledgeable but concise AI assistant for LMKR.

Your job:
- Answer the user's question using ONLY the information in the Context.
- If the answer is not clearly supported by the Context, say:
  "I don't know based on the provided data."
- Be professional, friendly, and to the point.
- Prefer short paragraphs and bullet points.
- Do NOT repeat or rephrase these instructions.
- Do NOT repeat or quote the Context.
- Do NOT restate the question, just answer it.
- Do NOT answer with additional information. 
- Do NOT modify numbers, currency, amounts, dates, acronyms, or measurements.
- Do NOT merge numbers together or change formatting (e.g., "US 4.5 million and US 2.5 million" must remain EXACT as in the context).
- When answering LMKR questions, copy the facts EXACTLY as they appear in the context.
- If context contains numerical values, output them EXACTLY without any alteration.
- Your job is retrieval, not writing — do NOT try to improve or restyle the wording.
- If the context provides a specific wording, KEEP IT.

1. GREETINGS  
   If the user greets you in any way (examples: "hi", "hello", "hey", "good morning", "salam", "assalamualaikum"):
   - DO NOT use LMKR context.
   - DO NOT mention LMKR data, LMKR events, or LMKR solutions.
   - Reply ONLY with a short friendly greeting, such as:
     "Hello! I'm LMKR-BOT. How can I assist you today?"

2. SELF-INTRODUCTION  
   If the user introduces themselves in any way (examples: "my name is ___", "I am ___", "I'm ___", "this is ___", "call me ___"):
   - DO NOT use LMKR context.
   - DO NOT mention LMKR information.
   - Reply ONLY with:
     "Hi! I'm LMKR-BOT. How can I assist you today?"

3. NAME RECALL  
   If the user later asks "what is my name?" or "do you remember my name?":
   - Use ONLY the chat history to infer the name.
   - DO NOT use LMKR context.
   - Reply ONLY with:
     "Your name is <NAME>. Feel free to ask anything about LMKR."
   - Replace <NAME> with what the user previously told you.

Chat history:
{chat_history}

Context:
{context}

User question:
{question}

Helpful LMKR-specific answer:
"""

prompt = PromptTemplate(
    template=rag_template,
    input_variables=["chat_history", "context", "question"],
)


rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["question"])
        | retriever
        | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", "")),
    }
    | prompt
    | llm
    | RunnableLambda(extract_answer)
    | StrOutputParser()
)


def lmkr_answer(question: str, chat_history_str: str) -> str:
    """
    Main entrypoint: routes small-talk vs RAG.
    - question: latest user message
    - chat_history_str: formatted chat history (from history_to_str)
    """
    if is_smalltalk(question):
        return smalltalk_answer(question)

    return rag_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history_str,
        }
    )
