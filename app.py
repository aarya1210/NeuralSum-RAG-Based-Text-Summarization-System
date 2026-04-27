"""
Flask web app — serves the summarization UI and inference endpoint.
Now extended with RAG (retrieval + generation).
"""

import os
import json
import time
import torch
import numpy as np
import faiss

from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────
MODEL_DIR   = "./model_output"
FALLBACK    = "t5-small"
MAX_INPUT   = 512

CHUNK_SIZE  = 300
TOP_K       = 3

# ─── LOAD T5 MODEL ────────────────────────────────────────────────────────
print("Loading summarization model...")
model_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else FALLBACK

tokenizer = T5Tokenizer.from_pretrained(model_path)
model     = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ─── LOAD EMBEDDING MODEL (RAG) ───────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─── METADATA ─────────────────────────────────────────────────────────────
meta = {}
meta_path = os.path.join(MODEL_DIR, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)

print(f"✅ Models ready | Device: {device}")

# ─────────────────────────────────────────────────────────────────────────
# 🔹 RAG HELPERS
# ─────────────────────────────────────────────────────────────────────────

def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

def build_index(chunks):
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(chunks, index, query):
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec).astype("float32")

    D, I = index.search(query_vec, TOP_K)
    return [chunks[i] for i in I[0]]

# ─────────────────────────────────────────────────────────────────────────
# 🔹 ROUTES
# ─────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", meta=meta, model_path=model_path)

# ─── ORIGINAL SUMMARIZATION ───────────────────────────────────────────────
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    length_preset = data.get("length", "medium")
    length_map    = {"short": 60, "medium": 120, "long": 180}
    max_len       = length_map.get(length_preset, 120)
    min_len       = max(20, max_len // 3)

    num_beams = int(data.get("beams", 4))
    no_repeat_ngram = int(data.get("no_repeat_ngram", 3))

    inp = "summarize: " + text

    tokens = tokenizer(
        inp,
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
    ).to(device)

    t0 = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            **tokens,
            max_new_tokens=max_len,
            min_new_tokens=min_len,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram,
            early_stopping=True,
            length_penalty=1.2,
            max_length=None
        )

    elapsed = round(time.time() - t0, 2)

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    input_words   = len(text.split())
    summary_words = len(summary.split())
    compression   = round((1 - summary_words / max(input_words, 1)) * 100, 1)

    return jsonify({
        "summary": summary,
        "stats": {
            "input_words": input_words,
            "summary_words": summary_words,
            "compression": compression,
            "time_sec": elapsed,
        },
    })

# ─── RAG SUMMARIZATION (FIXED) ────────────────────────────────────────────
@app.route("/summarize_rag", methods=["POST"])
def summarize_rag():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    length_preset = data.get("length", "medium")
    length_map    = {"short": 60, "medium": 120, "long": 180}
    max_len       = length_map.get(length_preset, 120)
    min_len       = max(20, max_len // 3)

    num_beams = int(data.get("beams", 4))

    t0 = time.time()

    # 1. Chunk text
    chunks = chunk_text(text)

    # 2. Build index
    index = build_index(chunks)

    # 3. Retrieve relevant chunks
    query = "summarize key important information"
    relevant_chunks = retrieve(chunks, index, query)

    # 4. Combine
    final_text = " ".join(relevant_chunks)

    # 5. Generate summary
    inp = "summarize: " + final_text

    tokens = tokenizer(
        inp,
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **tokens,
            max_new_tokens=max_len,
            min_new_tokens=min_len,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.2,
            max_length=None
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elapsed = round(time.time() - t0, 2)

    # ✅ FIXED: stats added (prevents UI crash)
    input_words   = len(text.split())
    summary_words = len(summary.split())
    compression   = round((1 - summary_words / max(input_words, 1)) * 100, 1)

    return jsonify({
        "summary": summary,
        "stats": {
            "input_words": input_words,
            "summary_words": summary_words,
            "compression": compression,
            "time_sec": elapsed,
            "rag_used": True,
            "chunks_retrieved": len(relevant_chunks),
        }
    })

# ─── HEALTH ───────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model": model_path
    })

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)