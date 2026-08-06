"""
Legal AI Bot — AgentCore Runtime Agent
Migrated from Lambda monolith to managed container with MCP tools.
POST /invocations → Agent logic | GET /ping → Health check
"""
import json, os, time, uuid, pickle, tempfile
from datetime import datetime
import boto3, numpy as np, faiss
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-pro-v1:0")
JUDGMENTS_BUCKET = os.environ.get("JUDGMENTS_BUCKET", "legal-ai-judgments-008714537357")
DRAFTS_BUCKET = os.environ.get("DRAFTS_BUCKET", "legal-ai-drafts-008714537357")
INDEX_KEY = "faiss-index/legal_index.faiss"
METADATA_KEY = "faiss-index/legal_metadata.pkl"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
_faiss_index = None
_faiss_metadata = None
app = FastAPI(title="Legal AI Agent", version="2.0")

SYSTEM_PROMPT = """You are an expert Legal AI Assistant for Indian litigating lawyers.
Tools: research_judgments, draft_document, devils_advocate.
Rules: Always cite cases, use Indian citation format, distinguish ratio from obiter,
reference statute sections, keep WhatsApp replies under 1500 chars, never fabricate citations."""

TOOLS = [
    {"toolSpec": {"name": "research_judgments",
        "description": "Search Indian case law via FAISS vector search and live SCC Online",
        "inputSchema": {"json": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Legal research question"}
        }, "required": ["query"]}}}},
    {"toolSpec": {"name": "draft_document",
        "description": "Draft legal documents in Indian court format",
        "inputSchema": {"json": {"type": "object", "properties": {
            "instruction": {"type": "string", "description": "What to draft and requirements"},
            "doc_type": {"type": "string", "description": "reply, petition, application, vakalatnama, or general"},
            "case_context": {"type": "string", "description": "Optional case details for context"}
        }, "required": ["instruction"]}}}},
    {"toolSpec": {"name": "devils_advocate",
        "description": "Attack a legal argument from opposing counsel perspective to find weaknesses",
        "inputSchema": {"json": {"type": "object", "properties": {
            "argument": {"type": "string", "description": "The legal argument to analyze"},
            "legal_context": {"type": "string", "description": "Optional statute or case context"}
        }, "required": ["argument"]}}}},
]

def load_faiss_index():
    global _faiss_index, _faiss_metadata
    if _faiss_index is not None:
        return _faiss_index, _faiss_metadata
    try:
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            s3.download_file(JUDGMENTS_BUCKET, INDEX_KEY, f.name)
            _faiss_index = faiss.read_index(f.name); os.unlink(f.name)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            s3.download_file(JUDGMENTS_BUCKET, METADATA_KEY, f.name)
            with open(f.name, "rb") as mf: _faiss_metadata = pickle.load(mf)
            os.unlink(f.name)
        print(f"[AGENT] FAISS loaded: {_faiss_index.ntotal} vectors")
    except Exception as e:
        print(f"[AGENT] FAISS load failed: {e}")
        _faiss_index = faiss.IndexFlatIP(1024)
        _faiss_metadata = {"chunks": [], "metadata": []}
    return _faiss_index, _faiss_metadata

def get_embedding(text: str) -> np.ndarray:
    r = bedrock_runtime.invoke_model(modelId=EMBEDDING_MODEL, contentType="application/json",
        accept="application/json", body=json.dumps({"inputText": text[:8000], "dimensions": 1024, "normalize": True}))
    return np.array([json.loads(r["body"].read())["embedding"]], dtype="float32")

def tool_research(query):
    idx, meta = load_faiss_index()
    qe = get_embedding(query)
    k = min(5, idx.ntotal)
    scores, indices = idx.search(qe, k)
    passages = []
    for i, ix in enumerate(indices[0]):
        if ix == -1 or scores[0][i] < 0.3: continue
        passages.append({"source": meta["metadata"][ix].get("source",""), "score": round(float(scores[0][i]),3), "text": meta["chunks"][ix][:500]})
    return json.dumps({"found": len(passages), "passages": passages})

def tool_draft(instruction, doc_type="general", case_context=""):
    r = bedrock_runtime.invoke_model(modelId=MODEL_ID, contentType="application/json", accept="application/json",
        body=json.dumps({"messages": [{"role": "user", "content": [{"text": f"Draft ({doc_type}): {instruction}\nContext: {case_context}"}]}],
            "system": [{"text": "Draft legal document in Indian court format. Numbered paragraphs, prayer clause, proper legal language."}],
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0.3}}))
    draft = json.loads(r["body"].read())["output"]["message"]["content"][0]["text"]
    s3.put_object(Bucket=DRAFTS_BUCKET, Key=f"drafts/{uuid.uuid4()}.txt", Body=draft.encode())
    return json.dumps({"draft": draft})

def tool_devils_advocate(argument, legal_context=""):
    r = bedrock_runtime.invoke_model(modelId=MODEL_ID, contentType="application/json", accept="application/json",
        body=json.dumps({"messages": [{"role": "user", "content": [{"text": f"Attack this argument:\n{argument}\nContext: {legal_context}"}]}],
            "system": [{"text": "You are opposing counsel. Attack with: weaknesses, counter-precedents, procedural vulnerabilities, risk rating. Indian legal framework."}],
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0.5}}))
    return json.dumps({"analysis": json.loads(r["body"].read())["output"]["message"]["content"][0]["text"]})

TOOL_HANDLERS = {
    "research_judgments": lambda p: tool_research(p["query"]),
    "draft_document": lambda p: tool_draft(p["instruction"], p.get("doc_type","general"), p.get("case_context","")),
    "devils_advocate": lambda p: tool_devils_advocate(p["argument"], p.get("legal_context","")),
}

def run_agent(user_message: str, history: list = None) -> str:
    """Run agent with Bedrock Converse API + tool use loop."""
    messages = (history or []) + [{"role": "user", "content": [{"text": user_message}]}]
    for _ in range(3):
        response = bedrock_runtime.converse(
            modelId=MODEL_ID, messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            toolConfig={"tools": TOOLS},
            inferenceConfig={"maxTokens": 4096, "temperature": 0.3},
        )
        output = response["output"]["message"]
        messages.append(output)
        if response.get("stopReason") == "tool_use":
            tool_results = []
            for block in output["content"]:
                if block.get("toolUse"):
                    tc = block["toolUse"]
                    print(f"[AGENT] Tool: {tc['name']}")
                    try:
                        result = TOOL_HANDLERS[tc["name"]](tc["input"])
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                    tool_results.append({"toolResult": {"toolUseId": tc["toolUseId"], "content": [{"text": result}]}})
            messages.append({"role": "user", "content": tool_results})
        else:
            return "".join(b.get("text", "") for b in output["content"])
    return "Unable to complete. Try a more specific query."

@app.get("/ping")
async def ping():
    return JSONResponse(content={"status": "Healthy"}, status_code=200)

@app.post("/invocations")
async def invocations(request: Request):
    try:
        body = await request.json()
        # AgentCore sends {"prompt": "..."} per HTTP protocol contract
        msg = body.get("prompt", body.get("message", body.get("input", "")))
        session_id = body.get("session_id", body.get("sessionId", str(uuid.uuid4())))
        if not msg:
            return JSONResponse(content={"response": "No message received", "status": "error"}, status_code=400)
        print(f"[AGENT] Session: {session_id} | Msg: {msg[:80]}")
        response_text = run_agent(msg, body.get("conversationHistory", []))
        return JSONResponse(content={"response": response_text, "status": "success"})
    except Exception as e:
        print(f"[AGENT] Error: {e}")
        return JSONResponse(content={"response": f"Error: {str(e)}", "status": "error"}, status_code=500)

@app.on_event("startup")
async def startup():
    print("[AGENT] Legal AI Agent v2 (AgentCore) starting...")
    load_faiss_index()
    print("[AGENT] Ready.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
