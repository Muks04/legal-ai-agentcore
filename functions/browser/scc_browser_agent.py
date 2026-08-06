"""
Legal AI Bot — SCC Online Browser Agent (AgentCore Browser)
=============================================================
Uses Amazon Bedrock AgentCore Browser to:
1. Login to SCC Online (advocate's credentials)
2. Search case law by query (full-text, citation, section, topic)
3. Extract judgment text + metadata (court, bench, date, headnotes)
4. Cache extracted judgments in FAISS for future instant retrieval
5. Return structured research results to the Legal Bot orchestrator

Architecture:
  WhatsApp → Legal Bot → SCC Browser Agent → AgentCore Browser → SCC Online
                                           → FAISS Cache (S3)
                                           → Bedrock Nova Pro (synthesis)

Prerequisites:
  - AgentCore Browser Tool created (aws.browser.v1 or custom)
  - SCC Online credentials stored in Secrets Manager
  - IAM role with bedrock-agentcore:* permissions
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import boto3

# AWS Clients
bedrock_agentcore = boto3.client("bedrock-agentcore", region_name="us-east-1")
secrets_manager = boto3.client("secretsmanager", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

# Configuration
BROWSER_TOOL_ID = os.environ.get("BROWSER_TOOL_ID", "")  # Set after creating browser tool
SCC_SECRET_ARN = os.environ.get("SCC_SECRET_ARN", "")  # Secrets Manager ARN for SCC creds
JUDGMENTS_BUCKET = os.environ.get("JUDGMENTS_BUCKET", "legal-ai-judgments-008714537357")
RESEARCH_TABLE = os.environ.get("RESEARCH_TABLE", "legal-research-prod")
CACHE_INDEX_KEY = "faiss-index/legal_index.faiss"
CACHE_METADATA_KEY = "faiss-index/legal_metadata.pkl"


# ============================================================
# SCC Online Navigation Actions
# ============================================================

SCC_ACTIONS = {
    "login": {
        "url": "https://www.scconline.com/Members/Login",
        "steps": [
            {"action": "fill", "selector": "input[name='email'], input#txtUserName, input[type='email']", "value": "{email}"},
            {"action": "fill", "selector": "input[name='password'], input#txtPassword, input[type='password']", "value": "{password}"},
            {"action": "click", "selector": "button[type='submit'], input#btnLogin, .login-btn"},
            {"action": "wait", "duration": 3},
        ],
    },
    "search": {
        "url": "https://www.scconline.com/Members/SearchResult.aspx",
        "steps": [
            {"action": "fill", "selector": "input#txtSearch, input.search-input, input[name='query']", "value": "{query}"},
            {"action": "click", "selector": "button#btnSearch, input.search-btn, button[type='submit']"},
            {"action": "wait", "duration": 5},
        ],
    },
    "extract_results": {
        "selector": "div.search-result, div.case-item, tr.result-row",
        "fields": {
            "case_name": "a.case-title, .case-name, td:first-child a",
            "citation": ".citation, .cite-text, td:nth-child(2)",
            "court": ".court-name, .court, td:nth-child(3)",
            "date": ".date, .judgment-date, td:nth-child(4)",
            "headnote": ".headnote, .summary, .snippet",
            "link": "a.case-title[href], .case-name a[href]",
        },
    },
    "extract_judgment": {
        "selector": "div.judgment-text, div#divJudgment, div.content-body, article",
        "metadata_selectors": {
            "bench": ".bench-info, .judges, .coram",
            "decided_on": ".decided-date, .date-of-judgment",
            "parties": ".parties, .case-title h1",
            "citations_in_judgment": "a.citation-link, .cited-cases a",
        },
    },
}


# ============================================================
# AgentCore Browser Session Management
# ============================================================

class SCCBrowserSession:
    """Manages an AgentCore Browser session for SCC Online."""

    def __init__(self):
        self.session_id = None
        self.websocket_url = None
        self.credentials = None

    def get_scc_credentials(self) -> dict:
        """Retrieve SCC Online credentials from Secrets Manager."""
        if self.credentials:
            return self.credentials

        try:
            if SCC_SECRET_ARN:
                response = secrets_manager.get_secret_value(SecretId=SCC_SECRET_ARN)
                self.credentials = json.loads(response["SecretString"])
            else:
                # Fallback to environment variables (for testing)
                self.credentials = {
                    "email": os.environ.get("SCC_EMAIL", ""),
                    "password": os.environ.get("SCC_PASSWORD", ""),
                }
            return self.credentials
        except Exception as e:
            raise Exception(f"Failed to get SCC credentials: {e}")

    def start_session(self, timeout_minutes: int = 10) -> str:
        """Start an AgentCore Browser session."""
        try:
            response = bedrock_agentcore.start_browser_session(
                browserToolId=BROWSER_TOOL_ID,
                sessionTimeoutMinutes=timeout_minutes,
            )
            self.session_id = response["sessionId"]
            self.websocket_url = response["automationEndpoint"]
            print(f"[SCC-BROWSER] Session started: {self.session_id}")
            return self.session_id
        except Exception as e:
            raise Exception(f"Failed to start browser session: {e}")

    def navigate(self, url: str):
        """Navigate to a URL."""
        self._send_action({
            "type": "navigate",
            "url": url,
        })
        time.sleep(2)  # Wait for page load

    def fill(self, selector: str, value: str):
        """Fill a form field."""
        self._send_action({
            "type": "fill",
            "selector": selector,
            "value": value,
        })

    def click(self, selector: str):
        """Click an element."""
        self._send_action({
            "type": "click",
            "selector": selector,
        })

    def extract_text(self, selector: str) -> str:
        """Extract text content from element(s)."""
        response = self._send_action({
            "type": "extract",
            "selector": selector,
        })
        return response.get("text", "")

    def screenshot(self) -> str:
        """Take a screenshot (returns base64)."""
        response = self._send_action({
            "type": "screenshot",
        })
        return response.get("image", "")

    def stop_session(self):
        """Stop the browser session."""
        if self.session_id:
            try:
                bedrock_agentcore.stop_browser_session(
                    browserToolId=BROWSER_TOOL_ID,
                    sessionId=self.session_id,
                )
                print(f"[SCC-BROWSER] Session stopped: {self.session_id}")
            except Exception:
                pass
            self.session_id = None

    def _send_action(self, action: dict) -> dict:
        """Send an action to the browser session via AgentCore API."""
        try:
            response = bedrock_agentcore.send_browser_action(
                browserToolId=BROWSER_TOOL_ID,
                sessionId=self.session_id,
                action=json.dumps(action),
            )
            return json.loads(response.get("result", "{}"))
        except Exception as e:
            print(f"[SCC-BROWSER] Action failed: {action.get('type')} — {e}")
            return {}


# ============================================================
# SCC Online Research Functions
# ============================================================

def search_scc_online(query: str, max_results: int = 5) -> dict:
    """
    Search SCC Online for judgments matching the query.
    Uses AgentCore Browser to navigate, search, and extract results.

    Returns:
        {
            "status": "success",
            "results": [
                {
                    "case_name": "...",
                    "citation": "...",
                    "court": "Supreme Court",
                    "date": "2023-05-15",
                    "headnote": "...",
                    "full_text": "..." (if extracted)
                }
            ],
            "source": "scc_online_live"
        }
    """
    session = SCCBrowserSession()

    try:
        # 1. Start browser session
        session.start_session(timeout_minutes=10)

        # 2. Login to SCC Online
        creds = session.get_scc_credentials()
        if not creds.get("email"):
            return {
                "status": "error",
                "message": "SCC Online credentials not configured. Please set SCC_SECRET_ARN or SCC_EMAIL/SCC_PASSWORD.",
            }

        print(f"[SCC-BROWSER] Logging in to SCC Online...")
        session.navigate("https://www.scconline.com/Members/Login")
        time.sleep(2)
        session.fill("input[type='email'], input#txtUserName", creds["email"])
        session.fill("input[type='password'], input#txtPassword", creds["password"])
        session.click("button[type='submit'], input#btnLogin")
        time.sleep(3)

        # 3. Search
        print(f"[SCC-BROWSER] Searching: {query[:80]}...")
        session.navigate("https://www.scconline.com/Members/BrowseModule/Search")
        time.sleep(2)
        session.fill("input.search-input, input#txtSearch, input[name='q']", query)
        session.click("button.search-btn, button#btnSearch, button[type='submit']")
        time.sleep(5)  # Wait for search results

        # 4. Extract search results
        results_text = session.extract_text("div.search-results, div#results, table.results-table")
        screenshot = session.screenshot()

        # 5. Parse results (use Bedrock to structure the raw text)
        parsed_results = parse_search_results(results_text, query, max_results)

        # 6. Extract full text for top results
        for i, result in enumerate(parsed_results[:3]):  # Top 3 only
            if result.get("link"):
                try:
                    print(f"[SCC-BROWSER] Extracting judgment {i+1}: {result.get('case_name', '')[:50]}...")
                    session.navigate(result["link"])
                    time.sleep(3)
                    full_text = session.extract_text(
                        "div.judgment-text, div#divJudgment, div.content-body, article, .judgment-content"
                    )
                    result["full_text"] = full_text[:10000]  # Cap at 10K chars

                    # Extract metadata
                    bench = session.extract_text(".bench-info, .judges, .coram")
                    if bench:
                        result["bench"] = bench

                except Exception as e:
                    print(f"[SCC-BROWSER] Failed to extract judgment {i+1}: {e}")

        # 7. Cache extracted judgments in FAISS
        cache_in_faiss(parsed_results)

        return {
            "status": "success",
            "results": parsed_results,
            "total_found": len(parsed_results),
            "source": "scc_online_live",
            "query": query,
            "searched_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"SCC Online search failed: {str(e)}",
            "fallback": "Using offline FAISS index instead.",
        }
    finally:
        session.stop_session()


def parse_search_results(raw_text: str, query: str, max_results: int = 5) -> list:
    """
    Use Bedrock Nova Pro to parse raw extracted text into structured results.
    This handles the variability of SCC Online's HTML structure.
    """
    if not raw_text or len(raw_text) < 50:
        return []

    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt = f"""You are parsing search results from SCC Online (Indian legal database).

Raw extracted text from the search results page:
---
{raw_text[:5000]}
---

Original search query: "{query}"

Extract up to {max_results} case results. For each, provide:
- case_name: Full party names (e.g., "Mahnoor Fatima v. State of Maharashtra")
- citation: SCC citation (e.g., "(2024) 5 SCC 123" or "2024 SCC OnLine SC 2490")
- court: Court name (Supreme Court / Delhi High Court / etc.)
- date: Date of judgment (YYYY-MM-DD if possible)
- headnote: Brief summary/headnote (1-2 sentences)
- link: URL if visible (or null)

Return as JSON array. If you can't parse the text, return empty array [].
"""

    try:
        response = bedrock_runtime.invoke_model(
            modelId="amazon.nova-pro-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": "You are a legal data parser. Return valid JSON only. No explanation."}],
                "inferenceConfig": {"maxTokens": 2048, "temperature": 0.1},
            }),
        )
        result = json.loads(response["body"].read())
        response_text = result["output"]["message"]["content"][0]["text"]

        # Parse JSON from response
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]

        return json.loads(clean)

    except Exception as e:
        print(f"[SCC-BROWSER] Parse failed: {e}")
        return []


def cache_in_faiss(results: list):
    """
    Add newly extracted judgments to the FAISS index.
    Downloads current index from S3, adds new vectors, re-uploads.
    """
    import tempfile
    import pickle

    # Only cache results that have full_text
    texts_to_cache = []
    metadata_to_cache = []

    for result in results:
        if result.get("full_text") and len(result["full_text"]) > 200:
            # Chunk the full text
            text = result["full_text"]
            chunk_size = 1000
            overlap = 200
            start = 0
            chunk_idx = 0

            while start < len(text):
                chunk = text[start:start + chunk_size].strip()
                if chunk:
                    texts_to_cache.append(chunk)
                    metadata_to_cache.append({
                        "source": f"SCC Online: {result.get('case_name', 'Unknown')}",
                        "citation": result.get("citation", ""),
                        "court": result.get("court", ""),
                        "date": result.get("date", ""),
                        "chunk_index": chunk_idx,
                        "cached_at": datetime.utcnow().isoformat(),
                    })
                    chunk_idx += 1
                start += chunk_size - overlap

    if not texts_to_cache:
        return

    print(f"[SCC-BROWSER] Caching {len(texts_to_cache)} new chunks in FAISS...")

    try:
        import numpy as np
        import faiss

        bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Generate embeddings for new chunks
        embeddings = []
        for text in texts_to_cache:
            response = bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "inputText": text[:8000],
                    "dimensions": 1024,
                    "normalize": True,
                }),
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])

        new_embeddings = np.array(embeddings, dtype="float32")

        # Download existing index from S3
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            s3.download_file(JUDGMENTS_BUCKET, CACHE_INDEX_KEY, f.name)
            index = faiss.read_index(f.name)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            s3.download_file(JUDGMENTS_BUCKET, CACHE_METADATA_KEY, f.name)
            with open(f.name, "rb") as mf:
                existing_metadata = pickle.load(mf)
            os.unlink(f.name)

        # Add new vectors
        index.add(new_embeddings)
        existing_metadata["chunks"].extend(texts_to_cache)
        existing_metadata["metadata"].extend(metadata_to_cache)

        # Upload updated index
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            faiss.write_index(index, f.name)
            s3.upload_file(f.name, JUDGMENTS_BUCKET, CACHE_INDEX_KEY)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(existing_metadata, f)
            f.flush()
            s3.upload_file(f.name, JUDGMENTS_BUCKET, CACHE_METADATA_KEY)
            os.unlink(f.name)

        print(f"[SCC-BROWSER] FAISS updated: +{len(texts_to_cache)} chunks (total: {index.ntotal})")

    except ImportError:
        print("[SCC-BROWSER] FAISS not available in this environment — skipping cache")
    except Exception as e:
        print(f"[SCC-BROWSER] Cache update failed (non-fatal): {e}")


# ============================================================
# Save Research to DynamoDB
# ============================================================

def save_research(query: str, results: list, lawyer_phone: str):
    """Save the research session to DynamoDB for history tracking."""
    try:
        table = dynamodb.Table(RESEARCH_TABLE)
        table.put_item(Item={
            "queryId": str(uuid.uuid4()),
            "lawyerPhone": lawyer_phone,
            "query": query,
            "source": "scc_online_live",
            "resultCount": len(results),
            "topCases": json.dumps([{
                "case_name": r.get("case_name", ""),
                "citation": r.get("citation", ""),
            } for r in results[:5]]),
            "timestamp": datetime.utcnow().isoformat(),
            "ttl": int(time.time()) + (180 * 86400),  # 6 months
        })
    except Exception:
        pass
