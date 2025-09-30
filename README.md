# Chatbot powered by Weaviate

Lightweight Streamlit app that:
- Converts uploaded documents (PDF/DOCX/TXT) into cleaned, semantic chunks.
- Stores chunks in a Weaviate collection (cloud) and vectors them.
- Provides a chat UI that uses Weaviate's hybrid generative search (OpenAI - Can be customized to your own provider) to answer questions from the documents.

Key components
- `streamlit_app.py` — main UI (upload → chunk → vectorize → chat)
- `utils/recursive_chunker.py` — document extraction and chunking
- `weaviatedb/` — connection, schema, ingestion, retrieval logic
- `UI/chatbot.py` — chat interface

Prerequisites
- Python 3.10+ (3.11 recommended)
- Weaviate Cloud instance and API key
- OpenAI API key (for generative responses)

Quick install
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set CLUSTER_URL, API_KEY, OPENAI_API_KEY
```

Run the app
```bash
streamlit run streamlit_app.py
```

Makefile targets (if present)
- `make venv` — create virtualenv
- `make install` — install requirements
- `make run` — run Streamlit inside venv
- `make test` — run tests (integration tests require env keys)

Environment variables
- `CLUSTER_URL` — Weaviate Cloud URL
- `API_KEY` — Weaviate Cloud API key
- `OPENAI_API_KEY` — OpenAI API key for generative responses

License
- See `LICENSE` in the repository root.
