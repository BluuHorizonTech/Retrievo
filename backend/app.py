from flask import Flask, request, jsonify
from rag_engine import RAGEngine
from PyPDF2 import PdfReader
from flask_cors import CORS
from flask import Response

import requests
import json

import os

app = Flask(__name__)
CORS(app)

OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
GEN_MODEL = "mistral"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

engine = RAGEngine()

# Chat memory stores conversation history.
# session based memory
chat_memory = {}

# This msg means 10 messages per session we will keep
MEMORY_WINDOW = 10

@app.route("/")
def home():
    return "RAG Server Running 🚀"


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename
    pages = []   # list of (page_number, text)

    try:
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(file)
            for page_no, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((page_no, text))
        else:
            text = file.read().decode("utf-8", errors="ignore")
            pages.append((None, text))
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 500

    if not pages:
        return jsonify({"error": "No extractable text in file"}), 400

    result = engine.add_document(pages, filename=filename)
    return jsonify({
        "message": "Document processed and stored in RAG",
        **result,
    })

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json or {}
    question = data.get("question")
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    chat_memory.setdefault(session_id, [])

    # retrieve returns dicts with text + page + filename
    hits = engine.retrieve(question, top_k=4)

    context = "\n\n".join(
        f"[{h['filename']} — page {h['page']}]\n{h['text']}"
        for h in hits
    )

    history = "".join(
        f"{m['role']}: {m['content']}\n"
        for m in chat_memory[session_id]
    )

    prompt = f"""You are a helpful assistant.
Use the context below to answer. When possible, cite the source
in the form (filename, page N).

Conversation so far:
{history}

Context from documents:
{context}

User question:
{question}

Answer clearly.
"""

    return stream_answer(prompt, question, session_id, hits)


# ------------------------------------------------------------------
def stream_answer(prompt, question, session_id, sources):
    try:
        response = requests.post(
            OLLAMA_GEN_URL,
            json={
                "model": GEN_MODEL,
                "prompt": prompt,
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Ollama call failed: {e}"}), 502

    def generate():
        full_answer = ""
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response", "")
                full_answer += chunk
                yield chunk
        finally:
            # Save memory even if client disconnects
            chat_memory[session_id].append({"role": "user", "content": question})
            chat_memory[session_id].append({"role": "assistant", "content": full_answer})

            if len(chat_memory[session_id]) > MEMORY_WINDOW:
                chat_memory[session_id] = chat_memory[session_id][-MEMORY_WINDOW:]

    resp = Response(generate(), content_type="text/plain")

    # Optional: expose sources as a header so the frontend can render citations.
    # Header values must be latin-1 safe, so keep it JSON with ensure_ascii=True.
    try:
        resp.headers["X-Sources"] = json.dumps(
            [
                {"filename": s["filename"], "page": s["page"], "chunk_id": s["chunk_id"]}
                for s in sources
            ],
            ensure_ascii=True,
        )
    except Exception:
        pass

    return resp


# ------------------------------------------------------------------
@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(engine.stats())


# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        app.run(debug=True, port=5000)
    finally:
        engine.shutdown()