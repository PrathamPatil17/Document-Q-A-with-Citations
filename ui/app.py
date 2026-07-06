"""Streamlit UI for the Document Q&A system. Calls the FastAPI backend.

Run:  streamlit run ui/app.py   (backend must be running on port 8000)
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("DOCQA_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A with Citations", layout="wide")
st.title("Document Q&A with Citations")


@st.cache_data(ttl=60)
def load_documents():
    return requests.get(f"{API_URL}/documents", timeout=10).json()


try:
    docs = load_documents()
except Exception:
    st.error(f"Cannot reach backend at {API_URL}. Start it with: "
             "uvicorn src.docqa.api:app --port 8000")
    st.stop()

ask_tab, contradict_tab = st.tabs(["Ask", "Contradict"])

with ask_tab:
    st.caption("Ask in any language — the answer comes back in the same language.")
    question = st.text_input("Your question")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving and answering..."):
            resp = requests.post(f"{API_URL}/ask", json={"question": question},
                                 timeout=60)
        if resp.status_code != 200:
            st.error(resp.json().get("detail", "Request failed"))
        else:
            data = resp.json()
            if not data["covered"]:
                st.warning(data["answer"])  # explicit not-covered message
            else:
                st.markdown(f"**Answer** _(detected language: {data['language']})_")
                st.write(data["answer"])
                st.markdown("**Citations**")
                for c in data["citations"]:
                    with st.expander(f"{c['source_file']} — {c['section']} "
                                     f"(chunk {c['chunk_index']})"):
                        st.write(c["snippet"])

with contradict_tab:
    st.caption("Check whether two documents conflict on a topic.")
    id_to_title = {d["doc_id"]: d["title"] for d in docs}
    ids = list(id_to_title)
    col1, col2 = st.columns(2)
    doc1 = col1.selectbox("Document 1", ids, format_func=lambda i: id_to_title[i])
    doc2 = col2.selectbox("Document 2", ids, index=min(1, len(ids) - 1),
                          format_func=lambda i: id_to_title[i])
    topic = st.text_input("Topic", placeholder="e.g. real-time biometric surveillance")
    if st.button("Compare", type="primary") and topic.strip():
        with st.spinner("Comparing documents..."):
            resp = requests.post(f"{API_URL}/contradict",
                                 json={"doc_id_1": doc1, "doc_id_2": doc2,
                                       "topic": topic}, timeout=60)
        if resp.status_code != 200:
            st.error(resp.json().get("detail", "Request failed"))
        else:
            data = resp.json()
            verdict = data["verdict"]
            color = {"conflict": "red", "agree": "green"}.get(verdict, "orange")
            st.markdown(f"### Verdict: :{color}[{verdict}]")
            st.write(data["reasoning"])
            e1, e2 = st.columns(2)
            with e1:
                st.markdown(f"**{id_to_title[doc1]} evidence**")
                for c in data["doc_1_evidence"]:
                    with st.expander(f"{c['section']} (chunk {c['chunk_index']})"):
                        st.write(c["snippet"])
            with e2:
                st.markdown(f"**{id_to_title[doc2]} evidence**")
                for c in data["doc_2_evidence"]:
                    with st.expander(f"{c['section']} (chunk {c['chunk_index']})"):
                        st.write(c["snippet"])
