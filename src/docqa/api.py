"""FastAPI backend exposing the RAG pipeline."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .config import DOCUMENTS
from . import rag, llm

app = FastAPI(title="Document Q&A with Citations")


class AskRequest(BaseModel):
    question: str


class ContradictRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str
    topic: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    return DOCUMENTS


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return rag.ask(req.question)
    except llm.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/contradict")
def contradict(req: ContradictRequest):
    try:
        return rag.contradict(req.doc_id_1, req.doc_id_2, req.topic)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except llm.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
