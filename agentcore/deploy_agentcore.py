"""
Legal AI Bot — Deploy to AgentCore Runtime
=============================================
Deploys the Legal AI Agent as a managed container on AgentCore.

Steps:
1. Build ARM64 Docker image
2. Push to ECR (legal-ai-agent repo)
3. Create AgentCore Runtime
4. Create AgentCore Endpoint
5. Update WhatsApp webhook gateway to point to AgentCore

Usage:
    python deploy_agentcore.py --build
    python deploy_agentcore.py --deploy
    python deploy_agentcore.py --full
"""

import argparse
import json
import os
import subprocess
import sys
import time

import boto3

REGION = "us-east-1"
ACCOUNT_ID = "008714537357"
ECR_REPO = "legal-ai-agent"
RUNTIME_NAME = "legal-ai-agent-runtime"
ENDPOINT_NAME = "legal-ai-agent-endpoint"

# AWS Clients
ecr = boto3.client("ecr", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)
bedrock_agentcore = boto3.client("bedrock-agentcore", region_name=REGION)


def run_cmd(cmd: str) -> str:
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ {result.stderr[:200]}")
    return result.stdout.strip()


def create_ecr_repo():
    """Create ECR repository for the agent image."""
    print("\n[1/5] Creating ECR repository...")
    try:
        ecr.create_repository(
            repositoryName=ECR_REPO,
            imageScanningConfiguration={"scanOnPush": True},
        )
        print(f"  ✓ Created: {ECR_REPO}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"  ✓ Already exists: {ECR_REPO}")


def build_and_push():
    """Build ARM64 image and push to ECR."""
    print("\n[2/5] Building and pushing Docker image (ARM64)...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}"

    # ECR login
    run_cmd(f"aws ecr get-login-password --region {REGION} --profile SaurabhArupMukherjee | "
            f"docker login --username AWS --password-stdin {ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com")

    # Build for ARM64 (AgentCore uses Graviton)
    run_cmd(f"docker buildx build --platform linux/arm64 -t {ECR_REPO}:latest {script_dir} --load")
    run_cmd(f"docker tag {ECR_REPO}:latest {repo_uri}:latest")
    run_cmd(f"docker push {repo_uri}:latest")
    print(f"  ✓ Pushed: {repo_uri}:latest")
    return f"{repo_uri}:latest"


def create_runtime_role():
    """Create IAM role for AgentCore Runtime."""
    role_name = "legal-ai-agentcore-runtime-role"
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT_ID}},
        }],
    }
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
        )
        arn = response["Role"]["Arn"]
    except iam.exceptions.EntityAlreadyExistsException:
        arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{role_name}"

    # Attach policies
    iam.put_role_policy(RoleName=role_name, PolicyName="AgentCorePolicy",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:Converse"], "Resource": "*"},
                {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": [
                    f"arn:aws:s3:::{JUDGMENTS_BUCKET}/*",
                    f"arn:aws:s3:::legal-ai-drafts-{ACCOUNT_ID}/*",
                ]},
                {"Effect": "Allow", "Action": ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"],
                 "Resource": f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/legal-*"},
                {"Effect": "Allow", "Action": ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
                 "Resource": f"arn:aws:ecr:{REGION}:{ACCOUNT_ID}:repository/{ECR_REPO}"},
                {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            ],
        }))
    print(f"  ✓ Role: {arn}")
    return arn


def create_agentcore_runtime(image_uri: str, role_arn: str):
    """Create AgentCore Runtime."""
    print("\n[3/5] Creating AgentCore Runtime...")
    try:
        response = bedrock_agentcore.create_runtime(
            name=RUNTIME_NAME,
            description="Legal AI Agent — Indian case law research, drafting, and analysis",
            containerConfig={
                "containerUri": image_uri,
                "protocol": "HTTP",
            },
            roleArn=role_arn,
        )
        runtime_id = response["runtimeId"]
        print(f"  ✓ Runtime created: {runtime_id}")
        return runtime_id
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  ✓ Runtime already exists")
            return RUNTIME_NAME
        raise


def create_agentcore_endpoint(runtime_id: str):
    """Create AgentCore Endpoint."""
    print("\n[4/5] Creating AgentCore Endpoint...")
    try:
        response = bedrock_agentcore.create_endpoint(
            name=ENDPOINT_NAME,
            runtimeId=runtime_id,
        )
        endpoint_url = response.get("endpointUrl", "")
        print(f"  ✓ Endpoint: {endpoint_url}")
        return endpoint_url
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  ✓ Endpoint already exists")
            return ""
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  LEGAL AI AGENT — AgentCore Deployment")
    print("=" * 60)

    if args.full or args.build:
        create_ecr_repo()
        image_uri = build_and_push()

    if args.full or args.deploy:
        image_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}:latest"
        role_arn = create_runtime_role()
        time.sleep(10)
        runtime_id = create_agentcore_runtime(image_uri, role_arn)
        time.sleep(5)
        endpoint_url = create_agentcore_endpoint(runtime_id)

        print(f"\n{'=' * 60}")
        print(f"  ✓ DEPLOYMENT COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Runtime: {runtime_id}")
        print(f"  Endpoint: {endpoint_url}")
        print(f"\n  Next: Update WhatsApp gateway Lambda to invoke this endpoint")


if __name__ == "__main__":
    main()
