"""
auth_gmail.py — Standalone fallback script for first-time Gmail authentication.

Use this ONLY if the browser-based OAuth flow via the server
(http://localhost:8100/auth/gmail/start) doesn't work.

Usage:
  python scripts/auth_gmail.py

What it does:
  1. Reads credentials from credentials/google_credentials.json
  2. Opens a browser for you to consent
  3. Saves the token to credentials/gmail_token.json

After running this, start the server normally — it will pick up the saved token.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google_auth_oauthlib.flow import InstalledAppFlow

CREDENTIALS_PATH = Path("credentials/google_credentials.json")
TOKEN_PATH = Path("credentials/gmail_token.json")
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def main():
    if not CREDENTIALS_PATH.exists():
        print(f"ERROR: credentials file not found at {CREDENTIALS_PATH}")
        print("Download it from Google Cloud Console → APIs & Services → Credentials")
        sys.exit(1)

    print(f"Starting OAuth flow using {CREDENTIALS_PATH}...")
    print("A browser window will open. Log in and grant Gmail send permission.")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CREDENTIALS_PATH),
        scopes=SCOPES,
    )
    # Uses local server on port 0 (random) for the callback
    creds = flow.run_local_server(port=0)

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())

    print(f"\n✅ Authentication successful!")
    print(f"   Token saved to: {TOKEN_PATH}")
    print(f"\nYou can now start the server:")
    print(f"   python -m src.server")


if __name__ == "__main__":
    main()
