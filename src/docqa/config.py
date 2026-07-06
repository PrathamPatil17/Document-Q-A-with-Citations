"""Central configuration: paths, model ids, thresholds, and the document registry."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Repo-root-relative paths so scripts work regardless of CWD.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "ai_policy_docs"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Chroma returns cosine DISTANCE (0 = identical). A chunk counts as relevant
# only when its distance to the query is <= this threshold. Tuned so that
# clearly off-topic questions (e.g. "tomorrow's weather") retrieve nothing.
DISTANCE_THRESHOLD = 0.75
TOP_K_ASK = 5
TOP_K_CONTRADICT = 4

# doc_id is the stable key used in the API; source_file matches data/documents/.
DOCUMENTS = [
    {"doc_id": "eu_ai_act", "source_file": "eu_ai_act.txt",
     "title": "EU AI Act (Regulation 2024/1689) — excerpt"},
    {"doc_id": "nist_ai_rmf", "source_file": "nist_ai_rmf.txt",
     "title": "NIST AI Risk Management Framework 1.0"},
    {"doc_id": "oecd_ai_principles", "source_file": "oecd_ai_principles.txt",
     "title": "OECD Recommendation on AI (OECD/LEGAL/0449)"},
    {"doc_id": "us_ai_bill_of_rights", "source_file": "us_ai_bill_of_rights.txt",
     "title": "Blueprint for an AI Bill of Rights (US, 2022)"},
    {"doc_id": "unesco_ai_ethics", "source_file": "unesco_ai_ethics.txt",
     "title": "UNESCO Recommendation on the Ethics of AI (2021)"},
    {"doc_id": "g7_hiroshima_principles", "source_file": "g7_hiroshima_principles.txt",
     "title": "G7 Hiroshima Process Guiding Principles for Advanced AI (2023)"},
]

DOC_BY_ID = {d["doc_id"]: d for d in DOCUMENTS}


def get_groq_key() -> str:
    """Return the Groq API key or raise a clear error if it is missing."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
