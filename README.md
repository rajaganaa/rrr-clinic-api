# RRR Clinic API

FastAPI backend for RRR Clinic MedAssist — AI-powered medical assistant for Dr. Rajeswari's clinic, Tamil Nadu.

**Author:** Rajaganapathy M — M.Tech AI, SRM University | Patent: 202641043947

## Architecture — 7-step Antahkarana pipeline

```
Question + Image
      │
   Manas ──── Question routing + entity extraction
      │
   Chitta ─── ChromaDB RAG retrieval (drug PDFs)
      │
   Buddhi ─── Groq LLM reasoning (llama3-70b, Tamil + English)
      │
  Ahamkara ── Confidence scoring
      │
   Sakshi ─── Hallucination detection + correction
      │
  /api/reason response
```

## Stack

| Layer | Tool | Cost |
|-------|------|------|
| LLM | Groq — llama3-70b-8192 | Free tier |
| Vision | OpenAI GPT-4o (GitHub Models) | Free |
| Embeddings | all-MiniLM-L6-v2 | Free, CPU |
| Vector DB | ChromaDB (local persistent) | Free |
| Hosting | Azure Container Apps | Student credits |
| CI/CD | GitHub Actions → GHCR | Free |

## Local development

```bash
cp .env.example .env
# Fill in GROQ_API_KEY and GITHUB_TOKEN in .env

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Docker

```bash
docker build -t rrr-clinic-api .
docker run -p 8000:8000 --env-file .env rrr-clinic-api
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Groq API key — get free at console.groq.com |
| `GITHUB_TOKEN` | ✅ | GitHub PAT — for GPT-4o vision via GitHub Models |
| `PORT` | optional | Default 8000 |
| `CHROMA_PATH` | optional | Default ./data/chroma_db |
| `MEDASSIST_DATA_DIR` | optional | Default ./data/drug_guides |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/reason` | Main AI reasoning (question + optional image) |
| POST | `/api/vision` | Medicine image analysis only |
| POST | `/api/search` | Direct ChromaDB search |
| GET | `/api/sources` | List indexed drug PDFs |

## Adding drug PDFs

Drop any medicine PDF into `data/drug_guides/`. On next restart ChromaDB will index it automatically.

## Deployment

Push to `main` branch → GitHub Actions builds Docker image → pushes to GHCR → deploys to Azure Container Apps automatically.

See `.github/workflows/deploy.yml` for the full pipeline.
