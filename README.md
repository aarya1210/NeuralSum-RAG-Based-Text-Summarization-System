# 🧠 NeuralSum – RAG-Based Text Summarization System

NeuralSum is an end-to-end AI-powered web application that generates concise summaries from long text inputs using a fine-tuned transformer model combined with a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features

* ✨ Abstractive text summarization using fine-tuned T5 model
* 🔍 Retrieval-Augmented Generation (RAG) for handling long documents
* ⚡ Real-time summarization via Flask REST API
* 📊 Displays summary statistics (compression ratio, latency, word count)
* 🌐 Simple and interactive web interface

---

## 🧠 How It Works

NeuralSum improves summarization by combining retrieval and generation:

1. **Input Text** → User provides long document
2. **Chunking** → Text split into smaller chunks
3. **Embedding** → Convert chunks into vector representations
4. **Retrieval (FAISS)** → Select most relevant chunks
5. **Generation (T5)** → Generate final summary

---

## 🏗️ Tech Stack

* **Languages:** Python, JavaScript
* **Frontend:** HTML, CSS
* **Backend:** Flask
* **AI/ML:** Transformers (T5), Sentence-Transformers
* **Vector DB:** FAISS
* **Libraries:** PyTorch, HuggingFace Transformers

---

## 📂 Project Structure

```bash
NeuralSum/
│
├── app.py                 # Flask backend (API + RAG pipeline)
├── finetune.py           # Model training script
├── model_output/         # Fine-tuned model + tokenizer
├── templates/
│   └── index.html        # Frontend UI
├── static/               # CSS / JS files
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/your-username/NeuralSum.git
cd NeuralSum

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🔄 API Endpoints

### ➤ Basic Summarization

```http
POST /summarize
```

### ➤ RAG-based Summarization

```http
POST /summarize_rag
```

---

## 📊 Example Output

```json
{
  "summary": "Generated summary text...",
  "stats": {
    "input_words": 500,
    "summary_words": 120,
    "compression": 76.0,
    "time_sec": 1.5
  }
}
```

---

## 🎯 Key Highlights

* Handles long documents efficiently using RAG
* Reduces irrelevant information in summaries
* Demonstrates real-world AI system design (LLM + Retrieval)
* Scalable architecture for NLP applications

---

## 🔮 Future Improvements

* 📄 PDF / document upload support
* ☁️ Cloud deployment (AWS / Render)
* 🔗 LangChain integration
* 📈 Model performance improvements

---

## 👨‍💻 Author

**Aarya Ingawale**
📧 [aaryaingawale12@gmail.com](mailto:aaryaingawale12@gmail.com)
🔗 https://github.com/aarya1210
🔗 https://linkedin.com/in/aarya-ingawale-8080bb220

---

## ⭐ If you found this useful, consider giving a star!
