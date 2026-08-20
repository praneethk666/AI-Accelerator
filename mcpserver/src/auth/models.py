from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class AuthContext:
    subject: str
    issuer: str
    audience: str
    claims: Dict[str, Any]
    token: str
    
    @property
    def client_id(self) -> Optional[str]:
        return self.claims.get("client_id") or self.subject
