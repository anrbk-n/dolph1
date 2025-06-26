from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from authlib.integrations.starlette_client import OAuth
from pydantic import BaseModel
import os

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="templates")

# ─────────────────────────────────────────────────────────────────────────────
# OAuth2 — Google
# ─────────────────────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
    raise RuntimeError(
        "Не заданы переменные окружения GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET"
    )


oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic
# ─────────────────────────────────────────────────────────────────────────────
class User(BaseModel):
    id: str
    name: str | None = None
    email: str


# ─────────────────────────────────────────────────────────────────────────────
# /login — log with google
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/login", summary="Войти через Google")
async def login(request: Request):
    redirect_uri = request.url_for("google_auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


# ─────────────────────────────────────────────────────────────────────────────
# /auth — callback  Google
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/auth", name="google_auth_callback", summary="Callback от Google")
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        raw_user = await oauth.google.parse_id_token(request, token)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OAuth error: {exc}") from exc

    user = User(
        id=raw_user["sub"],
        name=raw_user.get("name"),
        email=raw_user["email"],
    )

    # TODO: for the bd in the future

    return templates.TemplateResponse("welcome.html", {"request": request, "user": user})
