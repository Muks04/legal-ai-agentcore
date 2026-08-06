"""
Legal AI Bot — Setup AgentCore Browser Tool
=============================================
One-time setup script to create the AgentCore Browser Tool
and store SCC Online credentials in Secrets Manager.

Usage:
    python setup_browser_tool.py --create-browser
    python setup_browser_tool.py --store-credentials --email "x@y.com" --password "pass"
    python setup_browser_tool.py --test
"""

import argparse
import json
import sys

import boto3

REGION = "us-east-1"
ACCOUNT_ID = "008714537357"

bedrock_agentcore = boto3.client("bedrock-agentcore", region_name=REGION)
secrets_manager = boto3.client("secretsmanager", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)


def create_browser_execution_role():
    """Create IAM role for AgentCore Browser."""
    role_name = "legal-ai-browser-execution-role"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": ACCOUNT_ID}
            },
        }],
    }

    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Execution role for Legal AI AgentCore Browser",
        )
        role_arn = response["Role"]["Arn"]

        # Attach policies
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="BrowserToolPolicy",
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:PutObject",
                            "s3:GetObject",
                        ],
                        "Resource": f"arn:aws:s3:::legal-ai-judgments-{ACCOUNT_ID}/*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                        "Resource": "*",
                    },
                ],
            }),
        )

        print(f"✅ Created role: {role_arn}")
        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{role_name}"
        print(f"✓ Role already exists: {role_arn}")
        return role_arn


def create_browser_tool():
    """Create the AgentCore Browser Tool."""
    print("\n[1/3] Creating execution role...")
    role_arn = create_browser_execution_role()

    print("\n[2/3] Creating AgentCore Browser Tool...")
    try:
        response = bedrock_agentcore.create_browser_tool(
            name="legal-ai-scc-browser",
            description="Browser tool for searching SCC Online Indian legal database",
            networkMode="PUBLIC",  # SCC Online is public internet
            executionRoleArn=role_arn,
            sessionRecording={
                "enabled": True,
                "s3Bucket": f"legal-ai-judgments-{ACCOUNT_ID}",
                "s3Prefix": "browser-recordings/",
            },
        )

        browser_tool_id = response["browserToolId"]
        print(f"\n✅ Browser Tool created!")
        print(f"   ID: {browser_tool_id}")
        print(f"   Name: legal-ai-scc-browser")
        print(f"   Network: PUBLIC (internet access)")
        print(f"   Recording: S3 (for debugging)")
        print(f"\n   ⚠️  Set this as BROWSER_TOOL_ID in Lambda environment:")
        print(f"   BROWSER_TOOL_ID={browser_tool_id}")
        return browser_tool_id

    except Exception as e:
        if "already exists" in str(e).lower():
            print("✓ Browser tool may already exist. List existing tools to get ID.")
        else:
            print(f"❌ Failed: {e}")
            print("\n   If AgentCore Browser is not available in your region,")
            print("   check: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-tool.html")
        return None


def store_scc_credentials(email: str, password: str):
    """Store SCC Online credentials in Secrets Manager."""
    secret_name = "legal-ai-bot/scc-online-credentials"

    try:
        response = secrets_manager.create_secret(
            Name=secret_name,
            Description="SCC Online login credentials for Legal AI Bot",
            SecretString=json.dumps({
                "email": email,
                "password": password,
                "platform": "scconline.com",
            }),
        )
        print(f"✅ Credentials stored!")
        print(f"   Secret ARN: {response['ARN']}")
        print(f"   Set this as SCC_SECRET_ARN in Lambda environment:")
        print(f"   SCC_SECRET_ARN={response['ARN']}")

    except secrets_manager.exceptions.ResourceExistsException:
        # Update existing
        secrets_manager.update_secret(
            SecretId=secret_name,
            SecretString=json.dumps({
                "email": email,
                "password": password,
                "platform": "scconline.com",
            }),
        )
        print(f"✅ Credentials updated in existing secret: {secret_name}")


def test_browser():
    """Quick test — start a session, navigate to SCC, take screenshot."""
    print("\n[TEST] Starting browser session...")

    # List browser tools
    try:
        response = bedrock_agentcore.list_browser_tools()
        tools = response.get("browserTools", [])
        if not tools:
            print("❌ No browser tools found. Run --create-browser first.")
            return

        tool_id = tools[0]["browserToolId"]
        print(f"   Using tool: {tool_id}")

        # Start session
        session = bedrock_agentcore.start_browser_session(
            browserToolId=tool_id,
            sessionTimeoutMinutes=5,
        )
        session_id = session["sessionId"]
        print(f"   Session: {session_id}")
        print(f"   Live View: {session.get('liveViewEndpoint', 'N/A')}")

        # Navigate
        bedrock_agentcore.send_browser_action(
            browserToolId=tool_id,
            sessionId=session_id,
            action=json.dumps({"type": "navigate", "url": "https://www.scconline.com"}),
        )
        print("   ✅ Navigated to scconline.com")

        # Screenshot
        result = bedrock_agentcore.send_browser_action(
            browserToolId=tool_id,
            sessionId=session_id,
            action=json.dumps({"type": "screenshot"}),
        )
        print("   ✅ Screenshot captured")

        # Stop
        bedrock_agentcore.stop_browser_session(
            browserToolId=tool_id,
            sessionId=session_id,
        )
        print("   ✅ Session stopped")
        print("\n   Browser test PASSED!")

    except Exception as e:
        print(f"❌ Test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Setup AgentCore Browser for Legal AI Bot")
    parser.add_argument("--create-browser", action="store_true", help="Create the Browser Tool")
    parser.add_argument("--store-credentials", action="store_true", help="Store SCC Online creds")
    parser.add_argument("--email", type=str, help="SCC Online email")
    parser.add_argument("--password", type=str, help="SCC Online password")
    parser.add_argument("--test", action="store_true", help="Test browser session")
    args = parser.parse_args()

    print("=" * 60)
    print("  LEGAL AI BOT — AgentCore Browser Setup")
    print("=" * 60)

    if args.create_browser:
        create_browser_tool()
    elif args.store_credentials:
        if not args.email or not args.password:
            print("❌ Provide --email and --password")
            sys.exit(1)
        store_scc_credentials(args.email, args.password)
    elif args.test:
        test_browser()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
