"""
Accounts: passwordless sign-in by email link, and collections saved per user.

  POST /auth/request            {email} → emails a single-use link (15 min)
  GET  /auth/verify?token=…     confirmation page (mail scanners may open links;
  POST /auth/verify             only this POST consumes the token)
  GET  /auth/session            {authenticated, email}
  POST /auth/logout
  GET  /api/collections         the user's collections (+ ids deleted elsewhere)
  PUT  /api/collections/{id}    create / update, last write wins on updated_at
  DELETE /api/collections/{id}  leaves a tombstone so other devices drop it too
  POST /flag                    {faiss_id} report an irrelevant image
"""
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from typing import List, Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from app.backend import db, mailer

ENV = os.environ.get("ENV", "local")
DEV_SECRET = "dev-secret-change-me"
SECRET_KEY = os.environ.get("AUTH_SECRET", "" if ENV == "production" else DEV_SECRET)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "" if ENV == "production" else "http://localhost:5173").rstrip("/")
COOKIE_NAME = "glane_session"
SESSION_DAYS = 30
TOKEN_MINUTES = 15
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_COLLECTIONS = 300

router = APIRouter()


def enabled() -> bool:
    """Accounts need a real secret and a public URL for the links in production."""
    if not SECRET_KEY or not PUBLIC_BASE_URL:
        return False
    return not (ENV == "production" and SECRET_KEY == DEV_SECRET)


def _require_enabled():
    if not enabled() or not mailer.can_send():
        raise HTTPException(503, "Sign-in is not available yet")


def _now() -> int:
    return int(time.time())


def _hash(token: str) -> str:
    return hmac.new(SECRET_KEY.encode(), token.encode(), hashlib.sha256).hexdigest()


# ── Rate limiting (single instance: in-memory is enough) ─────────────────────
_hits = defaultdict(deque)


def _allow(key: str, limit: int, window: int) -> bool:
    q = _hits[key]
    now = time.monotonic()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[-1].strip() if forwarded else (request.client.host if request.client else "?")


# ── Sessions: signed cookie "user_id:expiry:signature" ───────────────────────
def _sign(payload: str) -> str:
    return hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _session_value(user_id: str) -> str:
    payload = f"{user_id}:{_now() + SESSION_DAYS * 86400}"
    return f"{payload}:{_sign(payload)}"


def current_user_id(request: Request) -> Optional[str]:
    if not enabled():
        return None
    raw = request.cookies.get(COOKIE_NAME, "")
    try:
        payload, sig = raw.rsplit(":", 1)
        user_id, expiry = payload.split(":", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(payload)) or _now() > int(expiry):
        return None
    return user_id


def _require_user(request: Request) -> str:
    uid = current_user_id(request)
    if not uid:
        raise HTTPException(401, "Not signed in")
    return uid


# ── Sign-in ──────────────────────────────────────────────────────────────────
class SignInRequest(BaseModel):
    email: str = Field(max_length=254)


@router.post("/auth/request")
def request_link(req: SignInRequest, request: Request):
    _require_enabled()
    email = req.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Please enter a valid email address")
    if not (_allow(f"email:{email}", 3, 600) and _allow(f"ip:{_client_ip(request)}", 20, 3600)):
        raise HTTPException(429, "Too many attempts. Please try again in a few minutes.")

    token = secrets.token_urlsafe(32)
    with db.transaction() as d:
        d.execute("DELETE FROM login_tokens WHERE expires_at < ?", (_now() - 86400,))
        d.execute("INSERT INTO login_tokens (token_hash, email, expires_at, used_at) VALUES (?, ?, ?, NULL)",
                  (_hash(token), email, _now() + TOKEN_MINUTES * 60))
    try:
        mailer.send_login_email(email, f"{PUBLIC_BASE_URL}/auth/verify?token={token}", TOKEN_MINUTES, _lang(request))
    except Exception as e:
        print(f"[auth] email to {email} failed: {e}", flush=True)
        raise HTTPException(502, "The email could not be sent. Please try again.")
    body = {"status": "sent"}
    if ENV != "production" and not mailer.RESEND_API_KEY:
        body["dev_outbox"] = "/auth/dev-outbox"
    return body


def _lang(request: Request) -> str:
    return "fr" if request.headers.get("accept-language", "").lower().startswith("fr") else "en"


PAGE_TEXT = {
    "en": {"expired_title": "Link expired", "expired": "This link has expired.",
           "expired_body": "Sign-in links work once and for {minutes} minutes. Ask for a new one from the Glane menu.",
           "back": "BACK TO GLANE", "title": "Sign in", "heading": "Sign in to Glane",
           "as": "You are signing in as", "button": "SIGN IN"},
    "fr": {"expired_title": "Lien expiré", "expired": "Ce lien a expiré.",
           "expired_body": "Les liens de connexion fonctionnent une fois, pendant {minutes} minutes. Demandez-en un nouveau depuis le menu de Glane.",
           "back": "RETOUR À GLANE", "title": "Connexion", "heading": "Se connecter à Glane",
           "as": "Vous vous connectez en tant que", "button": "SE CONNECTER"},
}


def _page(title: str, content: str, lang: str = "en") -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
<title>{html.escape(title)} · Glane</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%23000'/%3E%3Ctext x='16' y='23' font-family='Arial' font-weight='700' font-size='20' fill='%23fff' text-anchor='middle'%3EG%3C/text%3E%3C/svg%3E">
<style>
@font-face {{ font-family: Junicode; src: url('/fonts/JunicodeVF-Roman.woff2') format('woff2'); font-weight: 300 700; }}
@font-face {{ font-family: Junicode; src: url('/fonts/JunicodeVF-Italic.woff2') format('woff2'); font-weight: 300 700; font-style: italic; }}
body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #E3E3E3; font-family: Junicode, Georgia, serif; color: #000; }}
main {{ background: #fff; border-radius: 8px; padding: 2.2rem; width: min(420px, calc(100vw - 2rem)); box-sizing: border-box; animation: in .5s cubic-bezier(.16,1,.3,1) both; }}
@keyframes in {{ from {{ opacity: 0; transform: translateY(12px); }} }}
.brand {{ font-size: 1.3rem; letter-spacing: .16em; font-weight: 600; }}
h1 {{ font-weight: 400; font-style: italic; font-size: 1.7rem; margin: 1.6rem 0 .6rem; }}
p {{ color: #555; line-height: 1.5; margin: 0 0 1.4rem; }}
button, a.btn {{ font: inherit; font-variant-caps: all-small-caps; letter-spacing: .07em; font-weight: 600; font-size: 1.05rem;
  background: #000; color: #fff; border: 0; border-radius: 999px; padding: .65rem 1.3rem; cursor: pointer; text-decoration: none; display: inline-block; }}
</style></head><body><main><div class="brand">GLANE</div>{content}</main></body></html>""")


def _pending_email(token: str) -> Optional[str]:
    with db.transaction() as d:
        row = d.one("SELECT email FROM login_tokens WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?",
                    (_hash(token), _now()))
    return row["email"] if row else None


@router.get("/auth/verify", response_class=HTMLResponse)
def verify_page(request: Request, token: str = ""):
    _require_enabled()
    lang = _lang(request)
    m = PAGE_TEXT[lang]
    email = _pending_email(token) if token else None
    if not email:
        return _page(m["expired_title"], f"<h1>{m['expired']}</h1><p>{m['expired_body'].format(minutes=TOKEN_MINUTES)}</p>"
                     f'<a class="btn" href="/">{m["back"]}</a>', lang)
    return _page(m["title"], f"<h1>{m['heading']}</h1><p>{m['as']} <b>{html.escape(email)}</b>.</p>"
                 f'<form method="post" action="/auth/verify"><input type="hidden" name="token" value="{html.escape(token)}">'
                 f'<button type="submit">{m["button"]}</button></form>', lang)


@router.post("/auth/verify")
def verify(token: str = Form(...)):
    _require_enabled()
    now = _now()
    with db.transaction() as d:
        claimed = d.execute(
            "UPDATE login_tokens SET used_at = ? WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?",
            (now, _hash(token), now)).rowcount
        if claimed != 1:
            return RedirectResponse("/?login_error=expired", status_code=303)
        email = d.one("SELECT email FROM login_tokens WHERE token_hash = ?", (_hash(token),))["email"]
        user = d.one("SELECT id FROM users WHERE email = ?", (email,))
        if user:
            user_id = user["id"]
            d.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))
        else:
            user_id = uuid.uuid4().hex
            d.execute("INSERT INTO users (id, email, created_at, last_login_at) VALUES (?, ?, ?, ?)",
                      (user_id, email, now, now))
    resp = RedirectResponse("/?signed_in=1", status_code=303)
    resp.set_cookie(COOKIE_NAME, _session_value(user_id), max_age=SESSION_DAYS * 86400, httponly=True,
                    samesite="lax", secure=PUBLIC_BASE_URL.startswith("https://"), path="/")
    return resp


@router.get("/auth/session")
def session(request: Request):
    uid = current_user_id(request)
    if not uid:
        return {"authenticated": False, "available": enabled() and mailer.can_send()}
    with db.transaction() as d:
        user = d.one("SELECT email FROM users WHERE id = ?", (uid,))
    if not user:
        return {"authenticated": False, "available": True}
    return {"authenticated": True, "email": user["email"]}


@router.post("/auth/logout")
def logout():
    resp = JSONResponse({"status": "ok"})
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


if ENV != "production":
    @router.get("/auth/dev-outbox", response_class=HTMLResponse)
    def dev_outbox():
        rows = "".join(
            f'<p><b>{html.escape(m["to"])}</b> · {time.strftime("%H:%M:%S", time.localtime(m["at"]))}<br>'
            f'<a href="{html.escape(m["link"])}">Open the sign-in link</a></p>' for m in mailer.DEV_OUTBOX)
        return _page("Dev outbox", "<h1>Dev outbox</h1><p>Emails are not sent in local development; "
                     "the latest sign-in links are listed here.</p>" + (rows or "<p><i>Empty.</i></p>"))


# ── Collections ──────────────────────────────────────────────────────────────
class ItemIn(BaseModel):
    id: Optional[int] = None
    src: str = Field(max_length=2048)                 # grid thumbnail
    hd: str = Field(default="", max_length=2048)      # full-page image
    orig: str = Field(default="", max_length=2048)    # museum file, fallback
    url: str = Field(default="", max_length=2048)     # museum page
    title: str = Field(default="", max_length=500)
    author: str = Field(default="", max_length=300)
    source: str = Field(default="", max_length=40)
    members: Optional[List[int]] = Field(default=None, max_length=100)

    @field_validator("src", "hd", "orig", "url")
    @classmethod
    def _http_only(cls, v):
        if v and not v.startswith(("https://", "http://")):
            raise ValueError("must be an http(s) URL")
        return v


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    created_at: int
    updated_at: int
    cover_key: Optional[str] = Field(default=None, max_length=2100)
    items: List[ItemIn] = Field(default_factory=list, max_length=2000)


def _out(row: dict) -> dict:
    return {"id": row["id"], "name": row["name"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], "cover_key": row["cover_key"], "items": json.loads(row["items"])}


def _check_id(cid: str):
    if not COLLECTION_ID_RE.match(cid):
        raise HTTPException(400, "Invalid collection id")


@router.get("/api/collections")
def list_collections(request: Request):
    uid = _require_user(request)
    with db.transaction() as d:
        rows = d.all("SELECT * FROM collections WHERE user_id = ? ORDER BY created_at", (uid,))
    return {"collections": [_out(r) for r in rows if not r["deleted"]],
            "deleted": [r["id"] for r in rows if r["deleted"]]}


@router.put("/api/collections/{cid}")
def save_collection(cid: str, body: CollectionIn, request: Request):
    uid = _require_user(request)
    _check_id(cid)
    items = json.dumps([i.model_dump() for i in body.items], ensure_ascii=False)
    with db.transaction() as d:
        row = d.one("SELECT * FROM collections WHERE user_id = ? AND id = ?", (uid, cid))
        if row and row["updated_at"] > body.updated_at:
            return {"status": "stale", "deleted": bool(row["deleted"]), "collection": _out(row)}
        if not row:
            count = d.one("SELECT COUNT(*) AS n FROM collections WHERE user_id = ? AND deleted = 0", (uid,))["n"]
            if count >= MAX_COLLECTIONS:
                raise HTTPException(409, "Too many collections")
        d.execute(
            """INSERT INTO collections (user_id, id, name, created_at, updated_at, cover_key, items, deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)
               ON CONFLICT (user_id, id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at,
                   cover_key = excluded.cover_key, items = excluded.items, deleted = 0""",
            (uid, cid, body.name.strip(), body.created_at, body.updated_at, body.cover_key, items))
    return {"status": "saved"}


@router.delete("/api/collections/{cid}")
def delete_collection(cid: str, request: Request, updated_at: int = 0):
    uid = _require_user(request)
    _check_id(cid)
    stamp = max(updated_at, int(time.time() * 1000))
    with db.transaction() as d:
        d.execute(
            """INSERT INTO collections (user_id, id, name, created_at, updated_at, cover_key, items, deleted)
               VALUES (?, ?, '', 0, ?, NULL, '[]', 1)
               ON CONFLICT (user_id, id) DO UPDATE SET deleted = 1, items = '[]', updated_at = excluded.updated_at""",
            (uid, cid, stamp))
    return {"status": "deleted"}


# ── Irrelevant-image reports ("WTF flag") ────────────────────────────────────
class FlagIn(BaseModel):
    faiss_id: int = Field(ge=0)


@router.post("/flag")
def flag(body: FlagIn, request: Request):
    if not _allow(f"flag:{_client_ip(request)}", 60, 3600):
        raise HTTPException(429, "Too many reports")
    with db.transaction() as d:
        d.execute("""INSERT INTO flags (faiss_id, reports, last_at) VALUES (?, 1, ?)
                     ON CONFLICT (faiss_id) DO UPDATE SET reports = flags.reports + 1, last_at = excluded.last_at""",
                  (body.faiss_id, _now()))
    return {"status": "ok"}
