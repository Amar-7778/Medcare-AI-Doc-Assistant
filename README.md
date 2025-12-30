# Medcare AI Doc Assistant

Medcare AI Doc Assistant is an open-source assistant designed to help healthcare professionals and administrators generate, summarize, and manage clinical documents such as patient notes, discharge summaries, referral letters, and patient education materials using large language models (LLMs) and document processing tools.

IMPORTANT: This project may process protected health information (PHI). It is provided as a research/demo tool and is NOT certified for clinical use. Follow your organization's policies and applicable laws (e.g., HIPAA) before using this software with real patient data.

## Key Features

- Generate clinical notes and summaries from structured or unstructured inputs.
- Convert medical documents into structured data and extract key entities.
- Support for configurable LLM backends (OpenAI, local LLMs, etc.).
- API-first architecture so it can be integrated into existing EHR/workflows.
- Docker and local development support for reproducible setups.

## Quick Start (recommended)

Prerequisites

- Git
- Docker & Docker Compose (recommended) OR Python 3.10+ and Node.js 16+
- An API key for your chosen LLM provider (e.g., OpenAI) if you plan to use hosted models

Clone the repo

```bash
git clone https://github.com/Amar-7778/Medcare-AI-Doc-Assistant.git
cd Medcare-AI-Doc-Assistant
```

Option A — Run with Docker (fastest)

1. Copy or create a `.env` file from `.env.example` and add your secrets.
2. Start services:

```bash
docker compose up --build
```

Option B — Run locally (backend + optional frontend)

Backend (example Python/Node - replace with actual stack in this repo):

```bash
# create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r backend/requirements.txt
cd backend
export FLASK_ENV=development
# set env vars e.g. OPENAI_API_KEY
python app.py
```

Frontend (if included):

```bash
cd frontend
npm install
npm run dev
```

## Configuration

Create a `.env` file in the project root (or use your deployment secret manager) with the following example variables:

```env
# LLM provider
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Server
PORT=8000

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost:5432/medcare

# Security
SECRET_KEY=replace-with-a-secure-random-value
```

Adjust variables to match your deployment and provider.

## API

This repository provides a REST API for document generation and summarization (adjust endpoints to match implementation).

Example: Generate a clinical note

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Patient presents with...", "type": "progress_note"}'
```

Response (example):

```json
{
  "note": "...generated clinical note...",
  "meta": {"model":"gpt-4o-mini","tokens": 512}
}
```

## Architecture (high level)

- API layer — exposes REST/GraphQL endpoints for document generation and retrieval.
- LLM adapter — abstracts calls to OpenAI or other LLM providers.
- Document processor — handles parsing, summarization, entity extraction.
- Storage — database or object store for documents and metadata.
- Frontend — lightweight UI for uploading documents and previewing output.

## Privacy, Security, and Compliance

- DO NOT use with real PHI in production unless you have appropriate safeguards and agreements with your LLM provider.
- If you need to process PHI, ensure you have a Business Associate Agreement (BAA) and follow encryption, access control, and audit practices.
- Consider on-premises or private LLM deployments for sensitive workloads.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests, and submit pull requests for changes. Include tests and update documentation as needed.

Suggested workflow:

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-change`
3. Make changes and add tests
4. Submit a pull request with a clear description of the change

## Roadmap / Ideas

- Improve clinical templates (SOAP, H&P, discharge summaries)
- Add structured data export (FHIR, HL7)
- Add role-based access control and audit logging
- Add e2e tests and CI/CD for model-dependent workflows

## License

Specify the repository license here (e.g., MIT). If not decided, add a LICENSE file before publishing.

## Disclaimer

This software is provided for research and demonstration purposes only. It is not medical advice and should not be used as a substitute for professional clinical judgment.

---

If you'd like, I can also:
- add a `.env.example` file,
- scaffold a minimal Docker Compose setup,
- or tailor the README to the actual tech stack used in this repository — tell me which languages/frameworks are in the repo and I'll update the README accordingly.
