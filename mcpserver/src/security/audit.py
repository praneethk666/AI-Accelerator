import json
import hashlib
import sqlite3
from typing import Dict, Any, Optional

class AuditLogger:
    def __init__(self, db_path="audit.db"):
        self.db_path = db_path
        self._last_hash = "0" * 64
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entryHash TEXT UNIQUE,
                    previousEntryHash TEXT,
                    agent TEXT,
                    role TEXT,
                    capability TEXT,
                    server TEXT,
                    tool TEXT,
                    policyDecision TEXT,
                    executionStatus TEXT,
                    durationMs REAL,
                    resultMetadata TEXT,
                    isRetry BOOLEAN,
                    isFallback BOOLEAN
                )
            ''')
            
            # Fetch last hash for chaining
            cursor.execute('SELECT entryHash FROM audit_records ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            if row:
                self._last_hash = row[0]
            conn.commit()

    def log(self, 
            agent: str, 
            role: str,
            capability: str,
            server: str,
            tool: str,
            policy_decision: str,
            execution_status: str,
            duration_ms: float,
            result_metadata: Dict[str, Any],
            is_retry: bool = False,
            is_fallback: bool = False) -> Dict[str, Any]:
            
        record = {
            "agent": agent,
            "role": role,
            "capability": capability,
            "server": server,
            "tool": tool,
            "policyDecision": policy_decision,
            "executionStatus": execution_status,
            "durationMs": duration_ms,
            "resultMetadata": result_metadata,
            "isRetry": is_retry,
            "isFallback": is_fallback,
            "previousEntryHash": self._last_hash
        }
        
        # Calculate new hash
        record_str = json.dumps(record, sort_keys=True)
        new_hash = hashlib.sha256(record_str.encode("utf-8")).hexdigest()
        
        record["entryHash"] = new_hash
        self._last_hash = new_hash
        
        # Persist to local database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_records (
                    entryHash, previousEntryHash, agent, role, capability, 
                    server, tool, policyDecision, executionStatus, 
                    durationMs, resultMetadata, isRetry, isFallback
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                new_hash,
                record["previousEntryHash"],
                agent,
                role,
                capability,
                server,
                tool,
                policy_decision,
                execution_status,
                duration_ms,
                json.dumps(result_metadata),
                is_retry,
                is_fallback
            ))
            conn.commit()
        
        return record
