
# Mental Health PDF Q&A System using RAG and Ollama

This project sets up a Retrieval-Augmented Generation (RAG) pipeline to interact with a mental health-related PDF. It extracts the contents, creates embeddings, stores them in a FAISS vector store, and enables question-answering using a local LLM (`deepseek-r1:1.5b`) with fallback support if the context is insufficient.

#Features

-  Converts a PDF to Markdown using `docling`
-  Splits the Markdown into structured chunks based on headers
-  Uses `Ollama` embeddings (`nomic-embed-text`) for vector representation
-  Stores and retrieves chunks using `FAISS`
-  RAG setup using `LangChain` with fallback to model-generated answers if context is insufficient
-  Outputs bullet-point answers

---

# Project Structure

```
mental_health_rag/
│
├── app.py                    # Main script with RAG chain
├── mental_health.pdf         # Source document
├── .env                      # Environment variables (optional)
└── README.md                 # Project documentation
```

---

# Installation

1. **Clone the repo:**

```bash
git clone https://github.com/yourname/mental_health_rag.git
cd mental_health_rag
```

2. **Set up a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run Ollama and pull models:**
```bash
ollama run nomic-embed-text
ollama run deepseek-r1:1.5b
```

---

# Required Libraries

Here’s a list of required libraries (add them in `requirements.txt`):

```
streamlit
python-dotenv
faiss-cpu
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-ollama
docling
```

---

# How It Works

1. **Document Loading & Conversion:**
   - Uses `DocumentConverter` from `docling` to convert PDF to Markdown.

2. **Markdown Splitting:**
   - Uses `MarkdownHeaderTextSplitter` to chunk content by headers (`#`, `##`, `###`).

3. **Embeddings & Vector Store:**
   - Generates embeddings using `nomic-embed-text` model via Ollama.
   - FAISS stores vectors for retrieval.

4. **Retriever & Chain:**
   - Retrieves top 3 relevant chunks using MMR.
   - If context is insufficient, the model generates a fallback answer.

5. **Model Answering:**
   - Uses `deepseek-r1:1.5b` for generating answers in bullet points.

---

# Example Usage

```python
question = "What is distress?"
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
```

---

# Output Sample

```
Question: What is distress?
Distress refers to a state of emotional suffering or mental pain.

It often arises in response to stressful, overwhelming, or traumatic events.

Can manifest as anxiety, sadness, fear, irritability, or a sense of helplessness.

May be short-term (acute) or long-lasting (chronic).


--------------------------------------------------


# Future Improvements
- Multi-file PDF support
- Answer highlighting from source
- Switch models dynamically (DeepSeek, LLaMA, etc.)

---

#Author: Yuvika Ajmera
