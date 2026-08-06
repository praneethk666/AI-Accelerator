import os
import psycopg
import json
import yaml

from dotenv import load_dotenv

load_dotenv(override=True)

with open("config/global.yaml") as f:
    config = yaml.safe_load(f)

costs = config.get("models_cost", {})

url = os.getenv("POSTGRES_URL")
if not url:
    url = f"host={os.getenv('POSTGRES_HOST', 'localhost')} port={os.getenv('POSTGRES_PORT', '5432')} dbname={os.getenv('POSTGRES_DB', 'accelerator')} user={os.getenv('POSTGRES_USER', 'postgres')} password={os.getenv('POSTGRES_PASSWORD', 'postgres')}"

with psycopg.connect(url, autocommit=True, prepare_threshold=None) as conn:
    rows = conn.execute("SELECT document_id, token_usage, indexed_tokens FROM documents WHERE token_usage IS NOT NULL").fetchall()
    updated = 0
    for row in rows:
        doc_id, token_usage, indexed_tokens = row
        if not token_usage:
            continue
        
        # force recompute, clear old by_model
        token_usage.pop("by_model", None)
        token_usage.pop("total_cost_usd", None)
        
        by_kind = token_usage.get("by_kind", {})
        model_costs = {}
        total_cost = 0.0
        
        llm_model = config.get("llm_answer_model", "gpt-4o-mini")
        vision_model = config.get("vision_ocr_model", "meta/llama-3.2-11b-vision-instruct")
        embed_model = config.get("embeddings", {}).get("model", "text-embedding-3-small")
        
        mapping = {
            "categorization": config.get("categorization_model", llm_model),
            "chunk_llm": config.get("categorization_model", llm_model),
            "enrichment": config.get("enrichment_model", llm_model),
            "vision": vision_model,
            "embedding": embed_model
        }
        
        for kind, usage in by_kind.items():
            model = mapping.get(kind, llm_model)
            if model not in model_costs:
                model_costs[model] = {"input_tokens": 0, "output_tokens": 0, "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            
            it = usage.get("input_tokens", 0)
            ot = usage.get("output_tokens", 0)
            
            model_costs[model]["input_tokens"] += it
            model_costs[model]["output_tokens"] += ot
            
            if model in costs:
                c = costs[model]
                it_cost = (it / 1_000_000.0) * c.get("input", 0.0)
                ot_cost = (ot / 1_000_000.0) * c.get("output", 0.0)
                cost = it_cost + ot_cost
                model_costs[model]["input_cost"] += it_cost
                model_costs[model]["output_cost"] += ot_cost
                model_costs[model]["total_cost"] += cost
                total_cost += cost
        
        # Backfill embedding cost if it was missing and we have indexed_tokens
        if indexed_tokens and embed_model not in [m for m in model_costs.keys() if "embedding" in " ".join(by_kind.keys())]:
            if embed_model not in model_costs:
                model_costs[embed_model] = {"input_tokens": 0, "output_tokens": 0, "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            
            it = indexed_tokens
            model_costs[embed_model]["input_tokens"] += it
            if embed_model in costs:
                c = costs[embed_model]
                it_cost = (it / 1_000_000.0) * c.get("input", 0.0)
                model_costs[embed_model]["input_cost"] += it_cost
                model_costs[embed_model]["total_cost"] += it_cost
                total_cost += it_cost
                
        token_usage["by_model"] = model_costs
        token_usage["total_cost_usd"] = total_cost
        
        conn.execute("UPDATE documents SET token_usage = %s WHERE document_id = %s", (json.dumps(token_usage), doc_id))
        updated += 1
        print(f"Updated {doc_id} -> ${total_cost:.4f}")
        
    print(f"Migration complete. Updated {updated} documents.")
