"""Gmail send tool with allowlist enforcement."""

import asyncio
import base64
import json
import os
import importlib
import re
from email.message import EmailMessage
from typing import Optional, Any

from langchain.tools import tool

from config import SECRETS_DIR
from contacts import resolve_contact, load_contacts


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
EMAIL_RE = r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"


def _credentials_path() -> str:
    return str(SECRETS_DIR / "credentials.json")


def _token_path() -> str:
    return str(SECRETS_DIR / "token.json")


def _save_token(creds: Any) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_token_path(), "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def _load_credentials(interactive: bool = False) -> Any:
    credentials_path = _credentials_path()
    token_path = _token_path()

    try:
        google_auth_transport = importlib.import_module("google.auth.transport.requests")
        google_oauth2 = importlib.import_module("google.oauth2.credentials")
        google_oauthlib = importlib.import_module("google_auth_oauthlib.flow")
    except Exception as exc:
        raise RuntimeError(
            "Missing Gmail dependencies. Install with: "
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    Request = getattr(google_auth_transport, "Request")
    Credentials = getattr(google_oauth2, "Credentials")
    InstalledAppFlow = getattr(google_oauthlib, "InstalledAppFlow")

    if not os.path.exists(credentials_path):
        raise RuntimeError(
            "Missing Gmail OAuth credentials. Place credentials.json in secrets/."
        )

    creds: Optional[Any] = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)

    if not creds or not creds.valid:
        if not interactive:
            raise RuntimeError(
                "Missing or invalid token.json. Run: python app/gmail_oauth.py"
            )
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        _save_token(creds)

    return creds


def _build_service():
    try:
        google_api = importlib.import_module("googleapiclient.discovery")
    except Exception as exc:
        raise RuntimeError(
            "Missing Gmail dependencies. Install with: "
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc
    build = getattr(google_api, "build")
    creds = _load_credentials(interactive=False)
    return build("gmail", "v1", credentials=creds)


def _allowed_emails() -> set:
    return {str(c.get("email", "")).lower() for c in load_contacts() if c.get("email")}


def send_email_message(to_email: str, subject: str, body: str, enforce_allowlist: bool = True) -> str:
    if not to_email:
        raise ValueError("to_email is required")
    if not re.match(EMAIL_RE, to_email.strip()):
        raise ValueError("Invalid recipient email format")
    if enforce_allowlist:
        allowed = _allowed_emails()
        if to_email.lower() not in allowed:
            raise ValueError("Recipient email is not in the allowlist")

    service = _build_service()
    footer = "\n\n---\nThis is an AI automated email sent by hierarchical_qa_bot."
    message = EmailMessage()
    message["To"] = to_email
    message["From"] = "me"
    message["Subject"] = subject
    message.set_content(f"{body}{footer}")

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return sent.get("id", "unknown")


def send_email_to_contact(to_name: str, subject: str, body: str) -> str:
    match, matches = resolve_contact(to_name)
    if matches:
        names = ", ".join(str(m.get("name", "")) for m in matches)
        raise ValueError(f"Multiple matches for name '{to_name}': {names}")
    if not match:
        raise ValueError(f"No allowlisted contact found for '{to_name}'")

    email = str(match.get("email", "")).strip()
    if not email:
        raise ValueError(f"No email address set for '{to_name}'")

    message_id = send_email_message(email, subject, body)
    return message_id


def send_email_to_address(to_email: str, subject: str, body: str) -> str:
    """Send to an explicit user-provided email after confirmation."""
    return send_email_message(to_email, subject, body, enforce_allowlist=False)


async def send_email_to_contact_async(to_name: str, subject: str, body: str) -> str:
    """Async wrapper for email sending to avoid blocking event-loop driven callers."""
    return await asyncio.to_thread(send_email_to_contact, to_name, subject, body)


@tool
def send_email(to_name: str, subject: str, body: str) -> str:
    """Send an email to an allowlisted contact name. Confirmation handled by caller."""
    message_id = send_email_to_contact(to_name, subject, body)
    return json.dumps({"status": "sent", "message_id": message_id})
