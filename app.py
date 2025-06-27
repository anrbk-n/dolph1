"""
app.py — Dolphin OCR-VQA + Google OAuth2 (with userinfo fallback) + E-mail confirm
=================================================================================
Endpoints
---------
POST /markdown                 – PDF / image → Markdown
GET  /login  → /auth           – Google OAuth2 sign-in
POST /send-confirmation-email  – send confirmation mail
GET  /confirm/{token}          – confirm user e-mail
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import gc
import os
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from uuid import uuid4

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.integrations.starlette_client import OAuth
import anyio
import torch
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import PlainTextResponse, JSONResponse
from omegaconf import OmegaConf

from all_utils.utils import setup_output_dirs
from chat import DOLPHIN
from processor_api import process_document, generate_markdown

# ───────────────────────────────────── GPU helper ───────────────────────────
def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# ───────────────────────────────────── Model init ───────────────────────────
CFG_PATH = os.getenv("DOLPHIN_CONFIG", "config/Dolphin.yaml")
CONFIG   = OmegaConf.load(CFG_PATH)
ROOT_OUT = Path.cwd() / "result"
setup_output_dirs(ROOT_OUT)

def get_model() -> DOLPHIN:                           # lazy singleton
    if not hasattr(get_model, "_model") or get_model._model is None:   # type: ignore[attr-defined]
        get_model._model = DOLPHIN(CONFIG)                            # type: ignore[attr-defined]
    return get_model._model                                           # type: ignore[return-value]

# ───────────────────────────────────── FastAPI app ──────────────────────────
app = FastAPI(title="Dolphin API", version="0.3.0", docs_url="/docs")

# cookie-based session for OAuth state/nonce
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY"),
)

# ─────────────────────────────── Markdown endpoint ──────────────────────────
@app.post("/markdown", response_class=PlainTextResponse, summary="PDF/image → MD")
async def markdown(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    _, res = await anyio.to_thread.run_sync(
        process_document,
        tmp_path,
        get_model(),
        str(ROOT_OUT),
        CONFIG.model.get("max_batch_size", 4),
    )
    md = generate_markdown(res)

    try:
        os.remove(tmp_path)
    finally:
        clear_cuda()

    return PlainTextResponse(md, media_type="text/markdown")

# ─────────────────────────────── Google OAuth2 flow ─────────────────────────
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

@app.get("/login", summary="Start Google OAuth2 sign-in")
async def login(request: Request):
    redirect_uri = request.url_for("google_auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth", name="google_auth_callback")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)

    if "id_token" in token:                                  # ← сheck
        userinfo = await oauth.google.parse_id_token(token, nonce=None)

    else:                                                    # ← fallback
        async with AsyncOAuth2Client(token=token) as c:
            resp = await c.get(oauth.google.server_metadata["userinfo_endpoint"])
            if resp.status_code != 200:
                raise HTTPException(400, "Failed to fetch userinfo")
            userinfo = resp.json()

    return {
        "id":    userinfo.get("sub") or userinfo.get("id"),
        "name":  userinfo.get("name"),
        "email": userinfo.get("email"),
        "message": (
            "Google sign-in successful"
            if "id_token" in token
            else "Google sign-in successful (via userinfo fallback)"
        ),
    }
# ───────────────────────────── Mail confirmation flow ───────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
CONFIRM_BASE_URL = os.getenv("CONFIRM_BASE_URL", "http://localhost:8000/confirm")

def _send_mail(receiver: str, link: str) -> None:
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = EMAIL_FROM, receiver, "E-mail confirmation"
    msg.attach(MIMEText(
        f"Hello!\n\nPlease confirm your account:\n{link}\n\nIf you didn't request this, ignore.",
        "plain",
    ))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.sendmail(EMAIL_FROM, receiver, msg.as_string())
    except Exception as exc:
        raise HTTPException(500, f"SMTP error: {exc}") from exc

@app.post("/send-confirmation-email", summary="Send confirmation e-mail")
async def send_confirmation_email(
    background_tasks: BackgroundTasks,
    user_email: str = Form(..., description="Recipient e-mail"),
):
    token = uuid4().hex
    link  = f"{CONFIRM_BASE_URL}/{token}"
    # TODO: store token↔e-mail in DB / cache
    background_tasks.add_task(_send_mail, user_email, link)
    return {"message": "Confirmation e-mail sent", "debug_link": link}

@app.get("/confirm/{token}", summary="Confirm user e-mail")
async def confirm_email(token: str):
    # TODO: validate token and activate user
    return {"message": "E-mail confirmed", "token": token}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
