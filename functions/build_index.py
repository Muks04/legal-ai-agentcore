"""
Legal AI Bot - Build FAISS Index from Judgment PDFs
====================================================
Reads PDFs from S3, chunks them, embeds with Titan Embed V2,
and saves FAISS index back to S3.

Run this locally or as a one-time Lambda whenever new judgments are uploaded.

Usage:
    python build_index.py
    
Cost: ~$0.02 for 5 PDFs (Titan Embed pricing: $0.00002 per 1K tokens)
"""

import io
import json
import os
import pickle
import tempfile

import boto3
import numpy as np

# pip install PyPDF2 faiss-cpu numpy
import PyPDF2
import faiss

# Configuration
REGION = "us-east-1"
PROFILE = os.environ.get("AWS_PROFILE", "SaurabhArupMukherjee")
JUDGMENTS_BUCKET = "legal-ai-judgments-008714537357"
INDEX_KEY = "faiss-index/legal_index.faiss"
METADATA_KEY = "faiss-index/legal_metadata.pkl"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

# AWS Clients
session = boto3.Session(profile_name=PROFILE, region_name=REGION)
s3 = session.client("s3")
bedrock = session.client("bedrock-runtime")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def get_embedding(text: str) -> list:
    """Get embedding from Bedrock Titan Embed V2."""
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text[:8000],  # Titan Embed V2 max input
            "dimensions": 1024,
            "normalize": True,
        }),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def list_pdfs_in_s3() -> list:
    """List all PDFs in the judgments bucket."""
    response = s3.list_objects_v2(
        Bucket=JUDGMENTS_BUCKET,
        Prefix="judgments/",
    )
    pdfs = []
    for obj in response.get("Contents", []):
        if obj["Key"].lower().endswith(".pdf"):
            pdfs.append(obj["Key"])
    return pdfs


def build_index():
    """Main function: Build FAISS index from all judgment PDFs in S3."""
    print("=" * 60)
    print("  LEGAL AI BOT - Building FAISS Index")
    print("=" * 60)

    # 1. List PDFs
    pdf_keys = list_pdfs_in_s3()
    print(f"\n[1/4] Found {len(pdf_keys)} PDFs in S3:")
    for key in pdf_keys:
        print(f"  - {key.split('/')[-1]}")

    # 2. Extract text and chunk
    all_chunks = []
    all_metadata = []

    print(f"\n[2/4] Extracting text and chunking...")
    for pdf_key in pdf_keys:
        filename = pdf_key.split("/")[-1]
        print(f"  Processing: {filename}")

        # Download PDF
        response = s3.get_object(Bucket=JUDGMENTS_BUCKET, Key=pdf_key)
        pdf_bytes = response["Body"].read()

        # Extract text
        text = extract_text_from_pdf(pdf_bytes)
        print(f"    Extracted {len(text)} characters")

        # Chunk
        chunks = chunk_text(text)
        print(f"    Created {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": filename,
                "s3_key": pdf_key,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    print(f"\n  Total chunks: {len(all_chunks)}")

    # 3. Generate embeddings
    print(f"\n[3/4] Generating embeddings (Titan Embed V2)...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        if i % 10 == 0:
            print(f"  Embedding chunk {i+1}/{len(all_chunks)}...")
        emb = get_embedding(chunk)
        embeddings.append(emb)

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype="float32")
    print(f"  Embeddings shape: {embeddings_array.shape}")

    # 4. Build FAISS index
    print(f"\n[4/4] Building FAISS index...")
    dimension = embeddings_array.shape[1]  # 1024 for Titan Embed V2
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity since normalized)
    index.add(embeddings_array)
    print(f"  Index size: {index.ntotal} vectors")

    # Save to temp files then upload to S3
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        faiss.write_index(index, f.name)
        faiss_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(
            {"chunks": all_chunks, "metadata": all_metadata},
            f,
        )
        metadata_path = f.name

    # Upload to S3
    print(f"\n  Uploading index to S3...")
    s3.upload_file(faiss_path, JUDGMENTS_BUCKET, INDEX_KEY)
    s3.upload_file(metadata_path, JUDGMENTS_BUCKET, METADATA_KEY)

    # Cleanup
    os.unlink(faiss_path)
    os.unlink(metadata_path)

    print(f"\n{'=' * 60}")
    print(f"  ✓ INDEX BUILT SUCCESSFULLY")
    print(f"    Vectors: {index.ntotal}")
    print(f"    Dimension: {dimension}")
    print(f"    S3 Index: s3://{JUDGMENTS_BUCKET}/{INDEX_KEY}")
    print(f"    S3 Metadata: s3://{JUDGMENTS_BUCKET}/{METADATA_KEY}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_index()
