import os
from datetime import datetime, timedelta, timezone
import jwt
from dotenv import load_dotenv
import sys

# Load environment variables (pulls JWT_SECRET from .env)
load_dotenv()
jwt_secret = os.getenv("JWT_SECRET", "super-secret-key-123")
issuer = "local-mcp-issuer"
audience = "local-mcp-audience"

def create_jwt(subject: str) -> str:
    # A valid token requires 'sub', 'exp' (expiration), 'iss' (issuer) and 'aud' (audience)
    payload = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "exp": datetime.now(timezone.utc) + timedelta(days=30),  # Valid for 30 days
        "iat": datetime.now(timezone.utc)
    }
    
    # Sign it using HS256 and your JWT_SECRET
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return token

if __name__ == "__main__":
    if len(sys.argv) > 1:
        identities = sys.argv[1:]
    else:
        identities = ["vishal_engineer", "agent_alpha"]
        
    print("====================================")
    print(f"Signing keys with JWT_SECRET: {jwt_secret}")
    print("====================================\n")
    
    for ident in identities:
        t = create_jwt(ident)
        print(f"Token for [ {ident} ]:\n{t}\n")
    
    print("IMPORTANT: To use JWT tokens in your application:")
    print("1. Update config.yaml -> auth.jwt.enabled: true")
    print("2. In ui/index.html, replace the dummy token ('vishal-test-token') with this generated eyJhbG... token!")
    print("3. Restart your servers so they can validate real JWTs.")
