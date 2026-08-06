"""
Legal AI Bot - WhatsApp Orchestrator
=====================================
Handles Twilio WhatsApp webhooks, detects legal intent,
routes to appropriate tools, and manages conversation flow.
"""

import json
import os
import time
import urllib.parse
import urllib.error
from datetime import datetime

import boto3

from legal_tools import TOOLS_SCHEMA, execute_tool

# AWS Clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

# Environment
CHAT_HISTORY_TABLE = os.environ.get("CHAT_HISTORY_TABLE", "legal-chat-history-prod")
MODEL_ID = "amazon.nova-pro-v1:0"
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

# ============================================================
# Intent Detection & Orchestration
# ============================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Legal AI Assistant for Indian litigating lawyers.
You help with three core tasks:
1. RESEARCH — Finding relevant judgments, case law, statutes, and legal principles
2. DRAFTING — Creating legal documents (petitions, replies, applications, vakalatnama)
3. DEVIL'S ADVOCATE — Testing legal arguments by arguing the opposing side

Analyze the lawyer's message and decide which tool to use.

Rules:
- If the message is about finding cases, judgments, or legal principles → use research_judgments
- If the message asks to draft, write, prepare, or generate a document → use draft_document
- If the message asks to test an argument, find weaknesses, or "what will the other side argue" → use devils_advocate
- If the message is general conversation or unclear → respond directly without a tool
- Always be respectful (address as "Sir/Ma'am" or use professional language)
- Keep responses concise for WhatsApp (under 1500 chars unless drafting a full document)

Respond in JSON format:
{
    "use_tool": true/false,
    "tool_name": "research_judgments" | "draft_document" | "devils_advocate" | null,
    "parameters": { ... tool-specific parameters ... },
    "direct_response": "response if no tool needed"
}
"""


def get_chat_history(phone: str, limit: int = 5) -> list:
    """Fetch recent chat history for context."""
    try:
        table = dynamodb.Table(CHAT_HISTORY_TABLE)
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("phone").eq(phone),
            ScanIndexForward=False,  # Most recent first
            Limit=limit,
        )
        messages = response.get("Items", [])
        messages.reverse()  # Chronological order
        return messages
    except Exception:
        return []


def save_chat_message(phone: str, role: str, content: str):
    """Save a message to chat history."""
    try:
        table = dynamodb.Table(CHAT_HISTORY_TABLE)
        table.put_item(
            Item={
                "phone": phone,
                "timestamp": datetime.utcnow().isoformat(),
                "role": role,
                "content": content[:3000],  # Truncate for DynamoDB
                "ttl": int(time.time()) + (30 * 86400),  # 30 days
            }
        )
    except Exception:
        pass  # Non-critical — don't fail the response


def detect_intent_and_route(message: str, phone: str) -> dict:
    """
    Use Bedrock to detect the lawyer's intent and route to the correct tool.
    Returns the tool result or a direct response.
    """
    # Get recent context
    history = get_chat_history(phone)
    history_text = ""
    if history:
        history_text = "\n\nRecent conversation:\n"
        for msg in history[-3:]:  # Last 3 messages for context
            history_text += f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}\n"

    # Call Bedrock for intent detection
    user_prompt = f"Lawyer's message: {message}{history_text}"

    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": [{"text": user_prompt}]}
                    ],
                    "system": [{"text": ORCHESTRATOR_SYSTEM_PROMPT}],
                    "inferenceConfig": {
                        "maxTokens": 1024,
                        "temperature": 0.1,  # Low temp for reliable routing
                        "topP": 0.9,
                    },
                }
            ),
        )

        result = json.loads(response["body"].read())
        response_text = result["output"]["message"]["content"][0]["text"]

        # Parse the JSON response
        # Handle potential markdown code blocks in response
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1]  # Remove first line
            clean_text = clean_text.rsplit("```", 1)[0]  # Remove last ```

        intent = json.loads(clean_text)

        if intent.get("use_tool") and intent.get("tool_name"):
            # Execute the tool
            tool_result = execute_tool(
                tool_name=intent["tool_name"],
                parameters=intent.get("parameters", {}),
                lawyer_phone=phone,
            )
            return {
                "tool_used": intent["tool_name"],
                "result": tool_result,
            }
        else:
            return {
                "tool_used": None,
                "result": {
                    "status": "success",
                    "response": intent.get(
                        "direct_response", "I'm here to help with legal research, drafting, and argument analysis. How can I assist you?"
                    ),
                },
            }

    except json.JSONDecodeError:
        # If Bedrock response isn't valid JSON, treat as direct response
        return {
            "tool_used": None,
            "result": {
                "status": "success",
                "response": response_text if "response_text" in dir() else "I can help you with legal research, document drafting, and testing your arguments. What would you like to do?",
            },
        }
    except Exception as e:
        return {
            "tool_used": None,
            "result": {
                "status": "error",
                "message": f"I apologize, I encountered an issue: {str(e)}. Please try again.",
            },
        }


def format_response_for_whatsapp(orchestration_result: dict) -> str:
    """Format the tool output for WhatsApp (concise, readable)."""
    tool_used = orchestration_result.get("tool_used")
    result = orchestration_result.get("result", {})

    if result.get("status") == "error":
        return f"⚠️ {result.get('message', 'Something went wrong. Please try again.')}"

    if tool_used is None:
        return result.get("response", "How can I help you today?")

    if tool_used == "research_judgments":
        research = result.get("research", "No results found.")
        citations = result.get("citations", [])
        response = f"📚 *Legal Research*\n\n{research}"
        if citations:
            response += "\n\n📎 *Sources:*"
            for i, cite in enumerate(citations[:3], 1):
                source = cite.get("source", "").split("/")[-1]
                response += f"\n{i}. {source}"
        # Truncate for WhatsApp
        if len(response) > 4000:
            response = response[:3900] + "\n\n... [Full research saved. Ask for details on any point]"
        return response

    elif tool_used == "draft_document":
        draft = result.get("draft", "Could not generate draft.")
        doc_type = result.get("docType", "document")
        response = f"📝 *Draft ({doc_type.title()})*\n\n{draft}"
        if len(response) > 4000:
            response = response[:3900] + "\n\n... [Full draft saved. I can send remaining sections.]"
        return response

    elif tool_used == "devils_advocate":
        analysis = result.get("analysis", "Could not analyze.")
        response = f"⚔️ *Devil's Advocate Analysis*\n\n{analysis}"
        if len(response) > 4000:
            response = response[:3900] + "\n\n... [Full analysis saved. Ask about specific points.]"
        return response

    return "✅ Done. Anything else I can help with?"


# ============================================================
# Lambda Handler (Twilio WhatsApp Webhook)
# ============================================================


def send_whatsapp_reply(to: str, body: str):
    """Send WhatsApp reply via Twilio API."""
    import urllib.request
    import base64

    # Twilio WhatsApp has 1600 char limit — truncate if needed
    if len(body) > 1500:
        body = body[:1450] + "\n\n... [Message truncated. Ask for more details.]"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"

    # Manual encoding to avoid urllib.parse.urlencode mangling the + sign
    encoded_to = urllib.parse.quote(to, safe='')
    encoded_from = urllib.parse.quote(
        f"whatsapp:{os.environ.get('TWILIO_WHATSAPP_NUMBER', '+14155238886')}",
        safe=''
    )
    encoded_body = urllib.parse.quote(body, safe='')
    data = f"To={encoded_to}&From={encoded_from}&Body={encoded_body}".encode()

    credentials = base64.b64encode(
        f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode()
    ).decode()

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    try:
        urllib.request.urlopen(req)
        print(f"WhatsApp reply sent successfully to {to}")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"Failed to send WhatsApp reply: {e} | Response: {error_body[:500]}")
    except Exception as e:
        print(f"Failed to send WhatsApp reply: {e}")


def lambda_handler(event, context):
    """
    Main Lambda handler for Twilio WhatsApp webhook.
    Parses incoming message, orchestrates response, sends reply.
    """
    try:
        # Parse Twilio webhook (URL-encoded form data)
        if "body" in event:
            body = event["body"]
            if event.get("isBase64Encoded"):
                import base64
                body = base64.b64decode(body).decode("utf-8")
            params = dict(urllib.parse.parse_qsl(body))
        else:
            params = event  # Direct invocation for testing

        # Extract message details
        from_number = params.get("From", "").replace("whatsapp:", "")
        message_body = params.get("Body", "").strip()
        sender_name = params.get("ProfileName", "Counsellor")

        if not message_body:
            return {
                "statusCode": 200,
                "body": '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                "headers": {"Content-Type": "application/xml"},
            }

        print(f"[LEGAL-BOT] From: {from_number} | Name: {sender_name} | Message: {message_body[:100]}")

        # Save incoming message
        save_chat_message(from_number, "user", message_body)

        # Route and execute
        orchestration_result = detect_intent_and_route(message_body, from_number)

        # Format for WhatsApp
        reply_text = format_response_for_whatsapp(orchestration_result)

        # Save bot response
        save_chat_message(from_number, "assistant", reply_text)

        # Send reply via Twilio
        send_whatsapp_reply(f"whatsapp:{from_number}", reply_text)

        # Return TwiML empty response (we send asynchronously)
        return {
            "statusCode": 200,
            "body": '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            "headers": {"Content-Type": "application/xml"},
        }

    except Exception as e:
        print(f"[LEGAL-BOT] ERROR: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"},
        }
