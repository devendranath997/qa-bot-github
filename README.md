# 📄 QA Bot — Ask Questions About Your PDF

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask natural-language questions about its content.

Built with **IBM WatsonX**, **LangChain**, **ChromaDB**, and **Gradio**.

---

## ✨ Features

- Upload any PDF and ask questions in plain English
- Powered by IBM Granite-3 (8B) LLM via WatsonX
- Semantic search over document chunks using IBM Slate embeddings + ChromaDB
- Clean Gradio web interface — no frontend code needed
- Secure credential management via environment variables

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader ──► RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
                        │
                        ▼
              WatsonxEmbeddings (ibm/slate-125m-english-rtrvr-v2)
                        │
                        ▼
                  ChromaDB (in-memory vector store)
                        │
              similarity search (top-3 chunks)
                        │
                        ▼
         WatsonxLLM (ibm/granite-3-8b-instruct)
                        │
                        ▼
                    Answer ✅
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/qa-bot.git
cd qa-bot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your WatsonX credentials

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
WATSONX_APIKEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> **Where to find these values:**
> - **API Key** → [IBM Cloud](https://cloud.ibm.com) → Manage → Access (IAM) → API keys → Create
> - **Project ID** → [watsonx.ai](https://dataplatform.cloud.ibm.com/wx/home) → Your project → Manage → General → Project ID

### 5. Run the app

```bash
python qa_bot.py
```

Open **http://localhost:7860** in your browser.

---

## 📁 Project Structure

```
qa-bot/
├── qa_bot.py          # Main application (RAG pipeline + Gradio UI)
├── requirements.txt   # Python dependencies
├── .env.example       # Credential template (safe to commit)
├── .env               # Your actual credentials (git-ignored — never commit!)
├── .gitignore         # Ensures secrets and temp files are excluded
└── README.md          # This file
```

---

## 🔐 Security Notes

- **Never commit `.env`** — it is listed in `.gitignore` and will be excluded automatically.
- Rotate your IBM Cloud API key immediately if it is ever accidentally exposed.
- The `.env.example` file is a safe template with no real credentials — it is fine to commit.

---

## 🛠️ VS Code — Push to GitHub

If you are deploying from VS Code for the first time:

```bash
# 1. Initialise git inside the project folder
git init

# 2. Stage all files
git add .

# 3. Commit
git commit -m "Initial commit: QA Bot with WatsonX RAG pipeline"

# 4. Create a new repo on github.com, then link it
git remote add origin https://github.com/<your-username>/qa-bot.git

# 5. Push
git branch -M main
git push -u origin main
```

Or use the VS Code **Source Control** panel (Ctrl+Shift+G) to do the same thing visually.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | RAG orchestration framework |
| `langchain-community` | PDF loader integration |
| `langchain-ibm` | WatsonX LLM & embedding connectors |
| `langchain-chroma` | ChromaDB vector store integration |
| `chromadb` | In-memory vector database |
| `pypdf` | PDF text extraction |
| `gradio` | Web UI |
| `python-dotenv` | `.env` file loading |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
