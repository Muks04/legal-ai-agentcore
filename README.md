# Legal AI Bot — Indian Case Law Research + Draft Checker

An AI-powered legal research assistant for Indian litigating lawyers, deployed on **Amazon Bedrock AgentCore**. Provides instant case law search via FAISS vector similarity, draft defect checking for High Court filings, and devil's advocate analysis — accessible via WhatsApp and a web app.

## Live System

- **Web App:** https://main.d1nvmh7pabkyfy.amplifyapp.com (Research + Defect Checker tabs)
- **WhatsApp:** Integrated via multi-bot router
- **AgentCore Runtime:** `legalAiAgentRuntime-mteSKGE2Ym` (us-east-1)
- **FAISS Index:** 1,452 vectors (Indian judgments — Criminal, Arbitration, Civil, Revenue)

## Features

### 1. Case Law Research (FAISS Vector Search)
Query natural language legal questions → get relevant Indian court judgments with citations in ~4.5 seconds.

```
"What is the test for granting anticipatory bail in economic offences?"
→ Returns 5 relevant judgments with citations, ratio, and relevance scores
```

### 2. Filing Defect Checker
Upload a draft pleading → AI checks against 8 mandatory High Court registry requirements. Catches defects BEFORE filing, saving days of rejection.

**Defects checked:**
1. Margin alignment (4cm left, 2.5cm right, 3cm top)
2. Typed vs handwritten (T/C required)
3. Pagination order + T/C marking
4. Translation requirement (non-Hindi/English documents)
5. Index vs annexure cross-check
6. Typo errors in legal terms and section numbers
7. Court fee, verification clause, advocate details
8. Miscellaneous formatting issues

### 3. Devil's Advocate
Submit your legal argument → AI attacks it from opposing counsel's perspective.

```
"My argument is that Section 138 NI Act requires presentation within 6 months"
→ Returns: weaknesses, counter-precedents, procedural vulnerabilities, risk rating
```

## Architecture

```
                  ┌─────────────────────────┐
                  │     Web App (Amplify)    │
                  │  Research | Defect Check │
                  └───────────┬─────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
   API Gateway         API Gateway       WhatsApp Router
   (Research)          (Defect Check)    (Lambda)
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │  AgentCore Runtime       │
                  │  (Legal AI Agent)        │
                  └───────────┬─────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      FAISS Index      Bedrock Nova Pro   S3 (Drafts)
      (1,452 vectors)  (Generation)       (Saved docs)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Runtime | Amazon Bedrock AgentCore (us-east-1) |
| Model | Amazon Nova Pro v1.0 (reasoning + generation) |
| Embeddings | Amazon Titan Embed Text v2 (1024 dimensions) |
| Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| Storage | S3 (judgments, FAISS index, generated drafts) |
| Web App | AWS Amplify (React) |
| APIs | API Gateway + Lambda (defect checker, research) |
| Container | Python 3.12 + FastAPI + Uvicorn (ARM64) |
| CI/CD | AWS CodeBuild → ECR → AgentCore |

## Agent Tools (Bedrock Converse API)

The agent uses Bedrock's tool-use loop (ReAct pattern):

| Tool | Function |
|------|----------|
| `research_judgments` | FAISS vector search across 1,452 indexed judgments |
| `draft_document` | Generates legal documents in Indian court format |
| `devils_advocate` | Attacks arguments from opposing counsel perspective |

## Project Structure

```
legal-ai-bot/
├── agentcore/
│   ├── agent.py              # Main AgentCore agent (FAISS + tools)
│   ├── Dockerfile
│   ├── deploy_agentcore.py
│   └── requirements.txt
├── functions/
│   ├── defect_checker.py     # Filing defect checker (Lambda)
│   ├── legal_orchestrator.py # Orchestration logic
│   ├── legal_tools.py        # Tool implementations
│   ├── build_index.py        # FAISS index builder
│   └── browser/              # SCC Online browser (pending creds)
├── web-app/                  # Amplify frontend (React)
├── judgments/                # Raw judgment PDFs/text
├── Dockerfile
├── deploy.py
├── buildspec.yml
└── README.md
```

## APIs

| Endpoint | Function |
|----------|----------|
| `https://w8zn7cr8ta.execute-api.us-east-1.amazonaws.com/` | Defect Checker |
| `https://3t5punay3k.execute-api.us-east-1.amazonaws.com/` | Research API |

## Deployment

```bash
# Build and push container
python deploy.py --build

# Deploy to AgentCore
python deploy.py --deploy

# Rebuild FAISS index (after adding new judgments)
python functions/build_index.py
```

## Data Sources

- **Current:** 1,452 vectors from Indian court judgments (Criminal, Arbitration, Civil, Revenue)
- **Planned:** SCC Online integration (code complete, awaiting direct credentials)
- **Copyright safe:** Only operates on lawyer's own SCC/Manupatra credentials

## Pricing Model

| Tier | Features | Price |
|------|----------|-------|
| Basic | Case law search (5 queries/day) | Free trial |
| Pro | Unlimited search + defect checker + devil's advocate | ₹2,999/month |
| Enterprise | Pro + custom document templates + priority support | ₹9,999/month |

## Author

**Saurabh Mukherjee** — AWS Solutions Architect Professional | GenAI Professional | ML Engineer Associate

Co-founded with ADC Kaushal Sharma (practicing lawyer, 15+ years litigation experience).
