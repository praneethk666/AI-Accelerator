"""
auth.py — Gmail OAuth 2.0 Web App flow + token persistence + auto-refresh.

Two HTTP endpoints (mounted in server.py):
  GET /auth/gmail/start    → redirects browser to Google consent page
  GET /auth/gmail/callback → receives ?code= from Google, exchanges for token, saves to disk

On subsequent server starts, load_credentials() reads the saved token and
auto-refreshes it silently if expired (as long as refresh_token is still valid).

If no token exists, all mail_send calls return an error telling the agent to
visit /auth/gmail/start first.
"""

import logging
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from starlette.requests import Request as StarletteRequest
from starlette.responses import HTMLResponse, RedirectResponse

logger = logging.getLogger(__name__)

# Module-level credentials cache — avoids re-reading token.json on every tool call
_credentials: Optional[Credentials] = None


# ── OAuth flow helpers ────────────────────────────────────────────────────────

def _build_flow(cfg) -> Flow:
    """Build an OAuth2 Flow from the credentials.json file."""
    return Flow.from_client_secrets_file(
        cfg.credentials_path,
        scopes=cfg.scopes,
        redirect_uri=cfg.oauth_redirect_uri,
    )


# ── HTTP route handlers (mounted in server.py) ────────────────────────────────

async def gmail_oauth_start(request: StarletteRequest) -> RedirectResponse:
    """
    Step 1 of OAuth flow.
    Visit http://localhost:8100/auth/gmail/start in your browser.
    Redirects to Google's consent page.
    """
    from src.config import load_config
    cfg = load_config().channels["mail"]

    if not Path(cfg.credentials_path).exists():
        return HTMLResponse(
            "<h2>❌ credentials file not found</h2>"
            f"<p>Expected at: <code>{cfg.credentials_path}</code></p>"
            "<p>Download your OAuth credentials from Google Cloud Console "
            "and place the file there.</p>",
            status_code=500,
        )

    flow = _build_flow(cfg)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",  # Always request refresh_token
    )
    logger.info("Redirecting to Google OAuth consent page")
    return RedirectResponse(url=auth_url)


async def gmail_oauth_callback(request: StarletteRequest) -> HTMLResponse:
    """
    Step 2 of OAuth flow.
    Google redirects here with ?code=... after user consents.
    Exchanges the code for access + refresh tokens and saves them to disk.
    """
    global _credentials
    from src.config import load_config
    cfg = load_config().channels["mail"]

    error = request.query_params.get("error")
    if error:
        logger.error(f"OAuth error from Google: {error}")
        return HTMLResponse(
            f"<h2>❌ OAuth Error</h2><p>Google returned: <code>{error}</code></p>",
            status_code=400,
        )

    code = request.query_params.get("code")
    if not code:
        return HTMLResponse(
            "<h2>❌ No authorization code received</h2>"
            "<p>Google did not send a code. Try visiting /auth/gmail/start again.</p>",
            status_code=400,
        )

    try:
        flow = _build_flow(cfg)
        flow.fetch_token(code=code)
        creds = flow.credentials

        # Persist token to disk
        token_path = Path(cfg.token_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

        _credentials = creds
        logger.info(f"Gmail authenticated successfully. Token saved to {cfg.token_path}")

        return HTMLResponse(
            "<h1>✅ Gmail Authenticated!</h1>"
            "<p>The notifications-mcp server can now send emails.</p>"
            "<p>You can close this browser tab.</p>"
            "<hr><small>Token saved to <code>"
            + cfg.token_path
            + "</code></small>"
        )

    except Exception as e:
        logger.error(f"Failed to exchange OAuth code for token: {e}")
        return HTMLResponse(
            f"<h2>❌ Token Exchange Failed</h2><p><code>{e}</code></p>",
            status_code=500,
        )


# ── Credential loader (called by client.py on every tool call) ────────────────

async def get_credentials(cfg) -> Credentials:
    """
    Return valid Gmail credentials.

    Priority:
      1. In-memory cache (_credentials) if still valid
      2. Load from token_path file, auto-refresh if expired
      3. Raise RuntimeError if no token exists (user must visit /auth/gmail/start)
    """
    global _credentials

    # Return cached credentials if still valid
    if _credentials and _credentials.valid:
        return _credentials

    token_path = Path(cfg.token_path)
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), cfg.scopes)

        # Auto-refresh if expired and we have a refresh token
        if creds.expired and creds.refresh_token:
            logger.info("Gmail token expired — refreshing silently...")
            creds.refresh(Request())
            # Persist the refreshed token
            with open(token_path, "w") as f:
                f.write(creds.to_json())
            logger.info("Gmail token refreshed and saved.")

        _credentials = creds
        return creds

    # No token at all — user must authenticate first
    raise RuntimeError(
        "Gmail not authenticated. "
        "Visit http://localhost:8100/auth/gmail/start to complete setup."
    )
