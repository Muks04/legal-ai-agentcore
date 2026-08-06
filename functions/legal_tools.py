"""
Legal AI Bot - Core Tools for Indian Litigating Lawyers
========================================================
Tool 1: research_judgments — RAG over Indian case law (Bedrock KB)
Tool 2: draft_document — AI-powered legal drafting with Indian court formatting
Tool 3: devils_advocate — Adversarial analysis of legal arguments
"""

import json
import os
import uuid
import time
import pickle
import tempfile
from datetime import datetime

import boto3
import numpy as np
import faiss

# AWS Clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")

# Environment
RESEARCH_TABLE = os.environ.get("RESEARCH_TABLE", "legal-research-prod")
DRAFTS_TABLE = os.environ.get("DRAFTS_TABLE", "legal-drafts-prod")
DRAFTS_BUCKET = os.environ.get("DRAFTS_BUCKET", "legal-ai-drafts-008714537357")
JUDGMENTS_BUCKET = os.environ.get("JUDGMENTS_BUCKET", "legal-ai-judgments-008714537357")
INDEX_KEY = "faiss-index/legal_index.faiss"
METADATA_KEY = "faiss-index/legal_metadata.pkl"
MODEL_ID = "amazon.nova-pro-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

# FAISS index cache (loaded once per Lambda cold start)
_faiss_index = None
_faiss_metadata = None


# ============================================================
# TOOL 1: Research Judgments (RAG on Indian Case Law)
# ============================================================

RESEARCH_SYSTEM_PROMPT = """You are a legal research assistant specializing in Indian law.
Your role is to analyze retrieved judgment excerpts and provide structured legal research.

Rules:
- Always cite the case name, court, year, and paragraph number where possible
- Use standard Indian legal citation format: Party v. Party, (Year) Volume Reporter Page
- Distinguish between ratio decidendi (binding) and obiter dicta (persuasive)
- Note if a judgment has been overruled or distinguished
- Reference specific sections of statutes (e.g., Section 34 of the Arbitration and Conciliation Act, 1996)
- If the retrieved excerpts don't fully answer the query, say so clearly
- Provide the legal principle extracted from each cited passage
- Organize findings by relevance to the query
"""


def _load_faiss_index():
    """Load FAISS index and metadata from S3 (cached per Lambda cold start)."""
    global _faiss_index, _faiss_metadata

    if _faiss_index is not None:
        return _faiss_index, _faiss_metadata

    # Download index from S3
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        s3.download_file(JUDGMENTS_BUCKET, INDEX_KEY, f.name)
        _faiss_index = faiss.read_index(f.name)
        os.unlink(f.name)

    # Download metadata from S3
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        s3.download_file(JUDGMENTS_BUCKET, METADATA_KEY, f.name)
        with open(f.name, "rb") as mf:
            _faiss_metadata = pickle.load(mf)
        os.unlink(f.name)

    return _faiss_index, _faiss_metadata


def _get_query_embedding(text: str) -> np.ndarray:
    """Get embedding for a query using Titan Embed V2."""
    response = bedrock_runtime.invoke_model(
        modelId=EMBEDDING_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text[:8000],
            "dimensions": 1024,
            "normalize": True,
        }),
    )
    result = json.loads(response["body"].read())
    return np.array([result["embedding"]], dtype="float32")


def research_judgments(query: str, lawyer_phone: str) -> dict:
    """
    Search Indian case law using FAISS vector search + Bedrock for synthesis.
    Strategy:
      1. First search local FAISS index (instant, offline)
      2. If results are insufficient, trigger AgentCore Browser → SCC Online (live)
      3. Cache SCC results in FAISS for next time
    """
    try:
        # Strategy 1: Search local FAISS index first
        index, metadata = _load_faiss_index()
        chunks = metadata["chunks"]
        meta_list = metadata["metadata"]

        # Get query embedding
        query_embedding = _get_query_embedding(query)

        # Search FAISS (top 5 most relevant chunks)
        k = min(5, index.ntotal)
        scores, indices = index.search(query_embedding, k)

        # Collect relevant passages
        retrieved_passages = []
        citations = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            chunk_text = chunks[idx]
            chunk_meta = meta_list[idx]
            score = float(scores[0][i])

            # Only include if relevance score is above threshold
            if score < 0.3:
                continue

            retrieved_passages.append(
                f"[Source: {chunk_meta.get('source', chunk_meta.get('s3_key', 'Unknown'))}, "
                f"Citation: {chunk_meta.get('citation', 'N/A')}]\n{chunk_text}"
            )
            citations.append({
                "source": chunk_meta.get("source", "Unknown"),
                "citation": chunk_meta.get("citation", ""),
                "relevance_score": round(score, 4),
                "content_snippet": chunk_text[:200],
            })

        # Strategy 2: If local results are weak, try SCC Online (AgentCore Browser)
        scc_results = []
        if len(retrieved_passages) < 2 or (scores[0][0] < 0.5 if len(scores[0]) > 0 else True):
            try:
                from browser.scc_browser_agent import search_scc_online, save_research
                print(f"[LEGAL-BOT] Local results insufficient (score={scores[0][0] if len(scores[0]) > 0 else 0:.3f}). Searching SCC Online...")
                scc_result = search_scc_online(query, max_results=5)
                if scc_result.get("status") == "success":
                    scc_results = scc_result.get("results", [])
                    # Add SCC results to retrieved passages
                    for r in scc_results[:3]:
                        passage_text = r.get("full_text", r.get("headnote", ""))
                        if passage_text:
                            retrieved_passages.append(
                                f"[Source: SCC Online — {r.get('case_name', 'Unknown')}, "
                                f"Citation: {r.get('citation', 'N/A')}, "
                                f"Court: {r.get('court', 'N/A')}]\n{passage_text[:2000]}"
                            )
                            citations.append({
                                "source": f"SCC Online: {r.get('case_name', '')}",
                                "citation": r.get("citation", ""),
                                "court": r.get("court", ""),
                                "relevance_score": 0.95,  # Live search = high relevance
                                "content_snippet": passage_text[:200],
                            })
                    save_research(query, scc_results, lawyer_phone)
            except ImportError:
                print("[LEGAL-BOT] AgentCore Browser module not available — using FAISS only")
            except Exception as e:
                print(f"[LEGAL-BOT] SCC Online search failed (non-fatal): {e}")

        if not retrieved_passages:
            return {
                "status": "success",
                "research": "इस विषय पर मेरे पास पर्याप्त जानकारी नहीं है। कृपया अधिक विशिष्ट प्रश्न पूछें या नया judgment upload करें।\n\nI couldn't find sufficient relevant judgments for this query. Please try a more specific question or upload relevant judgment PDFs.",
                "citations": [],
                "queryId": str(uuid.uuid4()),
            }

        # Collect relevant passages
        retrieved_passages = []
        citations = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            chunk_text = chunks[idx]
            chunk_meta = meta_list[idx]
            score = float(scores[0][i])

            retrieved_passages.append(
                f"[Source: {chunk_meta['source']}, Chunk {chunk_meta['chunk_index']+1}/{chunk_meta['total_chunks']}]\n{chunk_text}"
            )
            citations.append({
                "source": chunk_meta["source"],
                "relevance_score": round(score, 4),
                "content_snippet": chunk_text[:200],
            })

        # Synthesize with Bedrock Nova Pro
        context = "\n\n---\n\n".join(retrieved_passages)
        synthesis_prompt = f"""Based on the following retrieved judgment excerpts, answer this legal research query:

Query: {query}

Retrieved Excerpts:
{context}

Provide your research findings with proper citations and legal principles:"""

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": synthesis_prompt}]}
                ],
                "system": [{"text": RESEARCH_SYSTEM_PROMPT}],
                "inferenceConfig": {
                    "maxTokens": 2048,
                    "temperature": 0.2,
                    "topP": 0.9,
                },
            }),
        )

        result = json.loads(response["body"].read())
        result_text = result["output"]["message"]["content"][0]["text"]

        # Save research to DynamoDB
        query_id = str(uuid.uuid4())
        research_table = dynamodb.Table(RESEARCH_TABLE)
        research_table.put_item(
            Item={
                "queryId": query_id,
                "lawyerPhone": lawyer_phone,
                "query": query,
                "result": result_text[:5000],
                "citationCount": len(citations),
                "timestamp": datetime.utcnow().isoformat(),
                "ttl": int(time.time()) + (90 * 86400),
            }
        )

        return {
            "status": "success",
            "research": result_text,
            "citations": citations,
            "queryId": query_id,
        }

    except FileNotFoundError:
        return {
            "status": "error",
            "message": "FAISS index not found. Run build_index.py first to index your judgments.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Research failed: {str(e)}"}


# ============================================================
# TOOL 2: Draft Document (Legal Drafting Assistant)
# ============================================================

DRAFTING_TEMPLATES = {
    "reply": {
        "system": """You are a legal drafting assistant for Indian courts.
Draft a Reply/Written Statement following Indian court formatting:

Format:
- Title: IN THE [COURT NAME], [CITY]
- Case Number and Parties
- Paragraph-wise replies (numbered)
- Each paragraph should address specific contentions
- Use legal language appropriate for Indian courts
- Include prayer clause at the end
- Place and Date at bottom

Maintain the legal essence while ensuring clarity and proper structure.
Use standard legal phrases: "It is respectfully submitted that...", "The contents of paragraph X are denied...", etc.
""",
    },
    "petition": {
        "system": """You are a legal drafting assistant for Indian courts.
Draft a Petition following Indian court formatting:

Format:
- Title: IN THE [COURT NAME], [CITY]
- Case Type and Number
- Details of Petitioner and Respondent
- Facts of the case (numbered paragraphs)
- Grounds (lettered sub-paragraphs)
- Prayer clause
- Verification clause
- Advocate details

Use formal Indian legal language with proper citations to statutes and precedents.
""",
    },
    "application": {
        "system": """You are a legal drafting assistant for Indian courts.
Draft an Application/Interim Application following Indian court formatting:

Format:
- Title: IN THE [COURT NAME], [CITY]
- Application type (IA/MA/CMA)
- Under which Section/Order/Rule
- Facts necessitating the application
- Grounds
- Prayer
- Affidavit support (if applicable)

Be concise but thorough. Indian courts prefer brevity with substance.
""",
    },
    "vakalatnama": {
        "system": """You are a legal drafting assistant for Indian courts.
Generate a Vakalatnama (Power of Attorney for Advocate):

Include:
- Court name and jurisdiction
- Case details
- Client name and address
- Advocate name, enrollment number
- Powers conferred
- Client signature block
- Witness signatures
- Date and place
""",
    },
    "general": {
        "system": """You are a legal drafting assistant for Indian courts.
Draft the requested legal document following Indian court formatting conventions.
Maintain proper legal language, structure, and citation format.
Use numbered paragraphs, proper party descriptions, and formal legal phraseology.
Include all standard clauses appropriate for the document type.
""",
    },
}


def draft_document(
    instruction: str,
    doc_type: str = "general",
    case_context: str = "",
    lawyer_phone: str = "",
) -> dict:
    """
    Generate a legal document draft based on instructions.
    Supports: reply, petition, application, vakalatnama, general
    """
    try:
        template = DRAFTING_TEMPLATES.get(doc_type, DRAFTING_TEMPLATES["general"])

        # Build the prompt
        user_prompt = f"Instructions: {instruction}"
        if case_context:
            user_prompt += f"\n\nCase Context:\n{case_context}"

        # Call Bedrock
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": [{"text": user_prompt}]}
                    ],
                    "system": [{"text": template["system"]}],
                    "inferenceConfig": {
                        "maxTokens": 4096,
                        "temperature": 0.3,  # Lower temp for legal precision
                        "topP": 0.9,
                    },
                }
            ),
        )

        result = json.loads(response["body"].read())
        draft_text = result["output"]["message"]["content"][0]["text"]

        # Save draft
        draft_id = str(uuid.uuid4())
        drafts_table = dynamodb.Table(DRAFTS_TABLE)
        drafts_table.put_item(
            Item={
                "draftId": draft_id,
                "lawyerPhone": lawyer_phone,
                "docType": doc_type,
                "instruction": instruction[:500],
                "draftPreview": draft_text[:1000],
                "timestamp": datetime.utcnow().isoformat(),
                "ttl": int(time.time()) + (30 * 86400),  # 30 days TTL
            }
        )

        # Upload full draft to S3
        s3_key = f"drafts/{lawyer_phone}/{draft_id}.txt"
        s3.put_object(
            Bucket=DRAFTS_BUCKET,
            Key=s3_key,
            Body=draft_text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

        return {
            "status": "success",
            "draft": draft_text,
            "draftId": draft_id,
            "docType": doc_type,
            "s3Key": s3_key,
        }

    except Exception as e:
        return {"status": "error", "message": f"Drafting failed: {str(e)}"}


# ============================================================
# TOOL 3: Devil's Advocate (Adversarial Argument Analysis)
# ============================================================

DEVILS_ADVOCATE_SYSTEM = """You are an experienced opposing counsel in an Indian court.
Your role is to ATTACK the legal argument presented to you. Be thorough and ruthless but fair.

Your analysis must include:

1. WEAKNESSES IN THE ARGUMENT
   - Logical fallacies or gaps
   - Facts that could be disputed
   - Assumptions that may not hold

2. COUNTER-ARGUMENTS
   - What the opposing side will likely argue
   - Precedents that go AGAINST this position
   - Statutory provisions that could be used against this argument

3. DISTINGUISHING CASES
   - If precedents are cited, explain how they can be distinguished
   - Find factual differences that weaken reliance on cited cases

4. PROCEDURAL VULNERABILITIES
   - Limitation issues
   - Jurisdiction challenges
   - Maintainability objections
   - Non-joinder / mis-joinder

5. RISK ASSESSMENT
   - Rate the argument strength: Strong / Moderate / Weak
   - Identify the single biggest vulnerability
   - Suggest what would make the argument stronger

Use Indian legal framework — cite CPC, CrPC, Evidence Act, specific statutes as relevant.
Be specific with section numbers and case citations where possible.
"""


def devils_advocate(
    argument: str, legal_context: str = "", lawyer_phone: str = ""
) -> dict:
    """
    Analyze a legal argument from the opposing side's perspective.
    Finds weaknesses, counter-arguments, and procedural vulnerabilities.
    """
    try:
        user_prompt = f"Analyze this argument as opposing counsel:\n\n{argument}"
        if legal_context:
            user_prompt += (
                f"\n\nAdditional context (statute/case details):\n{legal_context}"
            )

        # Call Bedrock with adversarial system prompt
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": [{"text": user_prompt}]}
                    ],
                    "system": [{"text": DEVILS_ADVOCATE_SYSTEM}],
                    "inferenceConfig": {
                        "maxTokens": 4096,
                        "temperature": 0.5,  # Slightly higher for creative counterarguments
                        "topP": 0.95,
                    },
                }
            ),
        )

        result = json.loads(response["body"].read())
        analysis_text = result["output"]["message"]["content"][0]["text"]

        # Save to research table (as adversarial analysis)
        query_id = str(uuid.uuid4())
        research_table = dynamodb.Table(RESEARCH_TABLE)
        research_table.put_item(
            Item={
                "queryId": query_id,
                "lawyerPhone": lawyer_phone,
                "query": f"[DEVILS_ADVOCATE] {argument[:300]}",
                "result": analysis_text[:5000],
                "timestamp": datetime.utcnow().isoformat(),
                "ttl": int(time.time()) + (90 * 86400),
            }
        )

        return {
            "status": "success",
            "analysis": analysis_text,
            "queryId": query_id,
        }

    except Exception as e:
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}


# ============================================================
# Tool Router (called by orchestrator)
# ============================================================

TOOLS_SCHEMA = [
    {
        "name": "research_judgments",
        "description": "Search Indian case law and legal precedents. Use when the lawyer asks about case law, judgments, legal principles, or statutory interpretation.",
        "parameters": {
            "query": "The legal research question",
        },
    },
    {
        "name": "draft_document",
        "description": "Draft legal documents (reply, petition, application, vakalatnama). Use when the lawyer asks to draft, write, or prepare a legal document.",
        "parameters": {
            "instruction": "What to draft and any specific requirements",
            "doc_type": "reply | petition | application | vakalatnama | general",
            "case_context": "Optional case details for context",
        },
    },
    {
        "name": "devils_advocate",
        "description": "Analyze a legal argument from opposing counsel's perspective. Use when the lawyer wants to test their argument, find weaknesses, or prepare for counterarguments.",
        "parameters": {
            "argument": "The legal argument to analyze",
            "legal_context": "Optional statute/case context",
        },
    },
]


def execute_tool(tool_name: str, parameters: dict, lawyer_phone: str) -> dict:
    """Route tool execution based on name."""
    if tool_name == "research_judgments":
        return research_judgments(
            query=parameters.get("query", ""),
            lawyer_phone=lawyer_phone,
        )
    elif tool_name == "draft_document":
        return draft_document(
            instruction=parameters.get("instruction", ""),
            doc_type=parameters.get("doc_type", "general"),
            case_context=parameters.get("case_context", ""),
            lawyer_phone=lawyer_phone,
        )
    elif tool_name == "devils_advocate":
        return devils_advocate(
            argument=parameters.get("argument", ""),
            legal_context=parameters.get("legal_context", ""),
            lawyer_phone=lawyer_phone,
        )
    else:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
