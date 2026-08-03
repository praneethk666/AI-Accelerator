#!/usr/bin/env python3
"""
scripts/test_jina_reranker_live.py
A test script to verify whether the hosted Jina Reranker API is working and to demonstrate
how to enforce configured limits (reranker_max_pairs, reranker_max_tokens_per_pair)
from config/global.yaml.
"""

import os
import sys
import yaml
from dotenv import load_dotenv

# Add project root to path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.models import JinaRerankerAPIClient

def load_project_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "global.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def test_jina_reranker():
    # Load .env file
    load_dotenv()
    
    config = load_project_config()
    
    # 1. Retrieve configurations
    embed_cfg = config.get("embeddings") or {}
    guardrails_cfg = config.get("guardrails") or {}
    deg_cfg = guardrails_cfg.get("degradation") or {}
    
    # Reranker settings
    # If the configured model is local, default to jina's standard hosted model
    model_name = embed_cfg.get("reranker_model", "jina-reranker-v2-base-multilingual")
    if "bge-reranker" in model_name or "local" in embed_cfg.get("reranker_provider", "local"):
        model_name = "jina-reranker-v2-base-multilingual"

    api_key = os.environ.get("JINA_API_KEY") or embed_cfg.get("reranker_api_key")
    
    # Guardrail Limits
    max_pairs = int(deg_cfg.get("reranker_max_pairs", 50))
    max_tokens_per_pair = int(deg_cfg.get("reranker_max_tokens_per_pair", 512))
    
    print("=" * 60)
    print("JINA RERANKER CONFIGURATION & LIMITS")
    print(f"Model:      {model_name}")
    print(f"API Key:    {'*' * 8 if api_key else 'NOT SET'}")
    print(f"Limits:     reranker_max_pairs = {max_pairs}")
    print(f"            reranker_max_tokens_per_pair = {max_tokens_per_pair} tokens")
    print("=" * 60)
    
    if not api_key:
        print("\n[ERROR] JINA_API_KEY is not set. Please add it to your .env file:")
        print("JINA_API_KEY=jua_...")
        return False

    # 2. Prepare mock search hits to rerank
    query = "What is the torque specification for the cylinder head bolts?"
    
    # Create 6 test document chunks (some highly relevant, some irrelevant)
    raw_documents = [
        "The cylinder head bolt torque specification is 85 N-m (62.7 lb-ft), tightened in three stages in the sequence shown.",
        "Ensure all engine gasket surfaces are clean and free of oil before head bolt re-assembly.",
        "Perform cylinder head bolt lubrication on threads prior to torqueing.",
        "The vehicle cabin air filter must be replaced every 15,000 miles or 12 months under normal driving conditions.",
        "Refer to the electrical diagram for pinout connections of the engine control module (ECM) connectors.",
        "Make sure to check the tire pressure monthly. Recommended pressure is 32 psi for both front and rear tires.",
    ]
    
    # Apply limit: reranker_max_pairs
    print(f"\n[Limit Check] Limiting candidate count from {len(raw_documents)} to max_pairs={max_pairs}...")
    limited_documents = raw_documents[:max_pairs]
    
    # Apply limit: reranker_max_tokens_per_pair (truncate document text to stay within tokens limit)
    # Since 1 token ≈ 4 characters, we can approximate characters limit: max_tokens * 4
    char_limit = max_tokens_per_pair * 4
    processed_documents = []
    for doc in limited_documents:
        if len(doc) > char_limit:
            print(f"  Truncating document (exceeded {max_tokens_per_pair} tokens / {char_limit} chars)...")
            doc = doc[:char_limit] + "..."
        processed_documents.append(doc)
        
    pairs = [(query, doc) for doc in processed_documents]
    
    # 3. Instantiate client and predict
    print(f"\nSending {len(pairs)} pairs to Jina Reranker API...")
    try:
        client = JinaRerankerAPIClient(model_name=model_name, api_key=api_key)
        scores = client.predict(pairs)
        
        # Display Results
        print("\nReranker Results:")
        results = sorted(zip(scores, raw_documents), key=lambda x: x[0], reverse=True)
        for i, (score, doc) in enumerate(results, 1):
            print(f"  {i}. [Score: {score:.4f}] - {doc[:80]}...")
            
        print("\n[SUCCESS] Jina Reranker API is working properly!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Jina Reranker failed: {e}")
        return False

if __name__ == "__main__":
    success = test_jina_reranker()
    sys.exit(0 if success else 1)
