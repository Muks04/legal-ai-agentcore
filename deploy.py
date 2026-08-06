"""
Legal AI Bot - Deployment Script
=================================
Deploys the Legal AI Bot to AWS:
1. Creates ECR repository (if not exists)
2. Builds and pushes Docker image
3. Deploys CloudFormation stack (DynamoDB + S3 + Lambda)
4. Updates Lambda with new image
5. Creates Bedrock Knowledge Base (if not exists)

Usage:
    python deploy.py                    # Full deploy
    python deploy.py --stack-only       # Only deploy CloudFormation
    python deploy.py --image-only       # Only build and push image
    python deploy.py --create-kb        # Create Bedrock Knowledge Base
"""

import argparse
import json
import os
import subprocess
import sys
import time

import boto3

# Configuration
REGION = "us-east-1"
ACCOUNT_ID = "008714537357"
STACK_NAME = "legal-ai-bot-prod"
ECR_REPO = "legal-ai-bot"
FUNCTION_NAME = "legal-ai-bot-prod"
KB_NAME = "indian-legal-judgments"
JUDGMENTS_BUCKET = f"legal-ai-judgments-{ACCOUNT_ID}"

# AWS Clients
ecr = boto3.client("ecr", region_name=REGION)
cfn = boto3.client("cloudformation", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)
bedrock_agent_client = boto3.client("bedrock-agent", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)


def run_cmd(cmd: str, check: bool = True) -> str:
    """Run shell command and return output."""
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ✗ Error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def ensure_ecr_repo():
    """Create ECR repository if it doesn't exist."""
    print("\n[1/5] Ensuring ECR repository exists...")
    try:
        ecr.describe_repositories(repositoryNames=[ECR_REPO])
        print(f"  ✓ Repository '{ECR_REPO}' already exists")
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(
            repositoryName=ECR_REPO,
            imageScanningConfiguration={"scanOnPush": True},
        )
        print(f"  ✓ Created repository '{ECR_REPO}'")


def build_and_push_image():
    """Build Docker image and push to ECR."""
    print("\n[2/5] Building and pushing Docker image...")

    repo_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}"

    # Login to ECR
    login_cmd = f"aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
    run_cmd(login_cmd)

    # Build
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_cmd(f"docker build -t {ECR_REPO}:latest {script_dir}")

    # Tag and push
    run_cmd(f"docker tag {ECR_REPO}:latest {repo_uri}:latest")
    run_cmd(f"docker push {repo_uri}:latest")

    print(f"  ✓ Image pushed: {repo_uri}:latest")
    return f"{repo_uri}:latest"


def deploy_stack():
    """Deploy CloudFormation stack."""
    print("\n[3/5] Deploying CloudFormation stack...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "infra", "legal-ai-stack.yaml")

    with open(template_path, "r") as f:
        template_body = f.read()

    try:
        # Check if stack exists
        cfn.describe_stacks(StackName=STACK_NAME)
        # Update
        print("  Updating existing stack...")
        try:
            cfn.update_stack(
                StackName=STACK_NAME,
                TemplateBody=template_body,
                Parameters=[
                    {"ParameterKey": "Environment", "ParameterValue": "prod"},
                    {"ParameterKey": "AccountId", "ParameterValue": ACCOUNT_ID},
                ],
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            waiter = cfn.get_waiter("stack_update_complete")
            waiter.wait(StackName=STACK_NAME)
        except cfn.exceptions.ClientError as e:
            if "No updates" in str(e):
                print("  ✓ Stack already up to date")
                return
            raise
    except cfn.exceptions.ClientError:
        # Create new stack
        print("  Creating new stack...")
        cfn.create_stack(
            StackName=STACK_NAME,
            TemplateBody=template_body,
            Parameters=[
                {"ParameterKey": "Environment", "ParameterValue": "prod"},
                {"ParameterKey": "AccountId", "ParameterValue": ACCOUNT_ID},
            ],
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
        waiter = cfn.get_waiter("stack_create_complete")
        waiter.wait(StackName=STACK_NAME)

    print("  ✓ Stack deployed successfully")


def update_lambda(image_uri: str):
    """Update Lambda function with new image."""
    print("\n[4/5] Updating Lambda function...")
    try:
        lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME, ImageUri=image_uri
        )
        # Wait for update
        waiter = lambda_client.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=FUNCTION_NAME)
        print(f"  ✓ Lambda updated with image: {image_uri}")
    except lambda_client.exceptions.ResourceNotFoundException:
        print("  ⚠ Lambda function not found — it will be created by CloudFormation")
        print("    Run deploy again after stack creation completes.")


def create_bedrock_kb():
    """Create Bedrock Knowledge Base for Indian judgments."""
    print("\n[5/5] Setting up Bedrock Knowledge Base...")

    # Check if KB already exists
    try:
        kbs = bedrock_agent_client.list_knowledge_bases()
        for kb in kbs.get("knowledgeBaseSummaries", []):
            if kb["name"] == KB_NAME:
                print(f"  ✓ Knowledge Base '{KB_NAME}' already exists (ID: {kb['knowledgeBaseId']})")
                return kb["knowledgeBaseId"]
    except Exception:
        pass

    # Create KB role
    kb_role_name = "legal-ai-bedrock-kb-role"
    try:
        role = iam.get_role(RoleName=kb_role_name)
        role_arn = role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role = iam.create_role(
            RoleName=kb_role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for Legal AI Bedrock Knowledge Base",
        )
        role_arn = role["Role"]["Arn"]

        # Attach S3 access policy
        iam.put_role_policy(
            RoleName=kb_role_name,
            PolicyName="S3Access",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": [
                                f"arn:aws:s3:::{JUDGMENTS_BUCKET}",
                                f"arn:aws:s3:::{JUDGMENTS_BUCKET}/*",
                            ],
                        }
                    ],
                }
            ),
        )

        # Attach Bedrock model access
        iam.put_role_policy(
            RoleName=kb_role_name,
            PolicyName="BedrockAccess",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["bedrock:InvokeModel"],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )
        time.sleep(10)  # Wait for role propagation

    # Create Knowledge Base
    try:
        kb_response = bedrock_agent_client.create_knowledge_base(
            name=KB_NAME,
            description="Indian court judgments, statutes, and legal precedents for litigation research",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed-text-v2:0",
                },
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": "PLACEHOLDER",  # Will need OpenSearch Serverless collection
                    "vectorIndexName": "legal-judgments-index",
                    "fieldMapping": {
                        "vectorField": "embedding",
                        "textField": "text",
                        "metadataField": "metadata",
                    },
                },
            },
        )
        kb_id = kb_response["knowledgeBase"]["knowledgeBaseId"]
        print(f"  ✓ Knowledge Base created: {kb_id}")
        print(f"\n  ⚠ IMPORTANT: Update BEDROCK_KB_ID in Lambda environment to: {kb_id}")
        print(f"  ⚠ You also need to set up an OpenSearch Serverless collection for vector storage.")
        print(f"  ⚠ Upload judgment PDFs to: s3://{JUDGMENTS_BUCKET}/")
        return kb_id
    except Exception as e:
        print(f"  ⚠ KB creation needs manual setup: {e}")
        print(f"\n  Alternative: Use Bedrock Console to create KB with:")
        print(f"    - Name: {KB_NAME}")
        print(f"    - S3 source: s3://{JUDGMENTS_BUCKET}/")
        print(f"    - Embedding: Amazon Titan Embed Text v2")
        print(f"    - Vector store: OpenSearch Serverless (auto-create)")
        return None


def main():
    parser = argparse.ArgumentParser(description="Deploy Legal AI Bot")
    parser.add_argument("--stack-only", action="store_true", help="Only deploy CloudFormation")
    parser.add_argument("--image-only", action="store_true", help="Only build/push image")
    parser.add_argument("--create-kb", action="store_true", help="Create Bedrock Knowledge Base")
    args = parser.parse_args()

    print("=" * 60)
    print("  LEGAL AI BOT - Deployment")
    print("  Region: us-east-1 | Account: 008714537357")
    print("=" * 60)

    if args.create_kb:
        create_bedrock_kb()
        return

    if args.stack_only:
        deploy_stack()
        return

    if args.image_only:
        ensure_ecr_repo()
        build_and_push_image()
        return

    # Full deploy
    ensure_ecr_repo()
    image_uri = build_and_push_image()
    deploy_stack()
    update_lambda(image_uri)

    print("\n" + "=" * 60)
    print("  ✓ DEPLOYMENT COMPLETE")
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Upload judgment PDFs to S3:")
    print(f"     aws s3 cp ./judgments/ s3://{JUDGMENTS_BUCKET}/ --recursive")
    print("  2. Create Bedrock KB:")
    print("     python deploy.py --create-kb")
    print("  3. Configure Twilio webhook to point to Lambda Function URL")
    print("  4. Set env vars: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN")
    print()


if __name__ == "__main__":
    main()
