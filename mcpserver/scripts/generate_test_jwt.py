import jwt
import os
import argparse
from datetime import datetime, timedelta, timezone

def generate_token(subject, secret, issuer, audience, hours_valid=24):
    payload = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "exp": datetime.now(timezone.utc) + timedelta(hours=hours_valid)
    }
    # Create the JWT
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test JWT for Custom MCP Server")
    parser.add_argument("--sub", type=str, default="vishal_engineer", help="Subject (agentId) for the token")
    parser.add_argument("--secret", type=str, default="super-secret-key-123", help="JWT Secret key (should match .env JWT_SECRET)")
    parser.add_argument("--issuer", type=str, default="local-mcp-issuer", help="JWT Issuer")
    parser.add_argument("--audience", type=str, default="local-mcp-audience", help="JWT Audience")
    
    args = parser.parse_args()
    
    token = generate_token(args.sub, args.secret, args.issuer, args.audience)
    print("\n" + "="*50)
    print(f"Generated JWT for agent '{args.sub}'")
    print("="*50 + "\n")
    print(token)
    print("\n" + "="*50)
    print("To test the server, run:")
    print(f'curl -X POST http://localhost:8100/mcp \\')
    print(f'     -H "Authorization: Bearer {token}" \\')
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d "{{\\"jsonrpc\\": \\"2.0\\", \\"id\\": 1, \\"method\\": \\"ping\\"}}"')
    print("\n" + "="*50)
