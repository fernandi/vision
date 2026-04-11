"""
Magic-link authentication backend.

Flow:
  POST /auth/request  { email }  → generates token, logs the link (no real email sent yet)
  GET  /auth/verify?token=xxx    → validates, sets session cookie
  GET  /auth/session             → returns { authenticated, email } from cookie
  POST /auth/logout              → clears cookie

Storage: SQLite (data/auth.db) — auto-created on first use.
Board persistence (for logged-in users):
  POST /boards                   → create board { name, images:[{src,...}] }
  GET  /boards                   → list boards for current user
  DELETE /boards/{id}            → delete board
"""
import time, hashlib, hmac, os, sqlite3, json, secrets
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH          = os.path.join("data", "auth.db")
TOKEN_TTL_MIN    = 15                            # magic-link valid for 15 minutes
SESSION_TTL_DAYS = 7
SECRET_KEY       = os.environ.get("AUTH_SECRET", "dev-secret-change-me")
COOKIE_NAME      = "vs_session"

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Database ────────────────────────────────────────────────────────────────────
def _get_db():
    os.makedirs("data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    UNIQUE NOT NULL,
            created_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS magic_tokens (
            token_hash TEXT    PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            expires_at TEXT    NOT NULL,
            used       INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS boards (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            name       TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS board_images (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            board_id   INTEGER NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
            src        TEXT    NOT NULL,
            title      TEXT,
            author     TEXT,
            source     TEXT,
            pinned_at  TEXT    NOT NULL
        );
    """)
    con.commit()
    return con

@contextmanager
def db():
    con = _get_db()
    try:
        yield con
    finally:
        con.close()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _hash_token(raw: str) -> str:
    return hmac.new(SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def _make_session_token(user_id: int) -> str:
    payload = f"{user_id}:{int(time.time()) + SESSION_TTL_DAYS * 86400}"
    sig     = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"

def _parse_session(token: str) -> Optional[int]:
    """Return user_id if session token is valid and not expired, else None."""
    try:
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            return None
        payload, sig = parts
        expected = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        uid_str, exp_str = payload.split(":", 1)
        if int(time.time()) > int(exp_str):
            return None
        return int(uid_str)
    except Exception:
        return None

def get_current_user(request: Request) -> Optional[int]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return _parse_session(token)

# ── Endpoints ─────────────────────────────────────────────────────────────────
class MagicLinkRequest(BaseModel):
    email: str

@router.post("/request")
def request_magic_link(req: MagicLinkRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email")

    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MIN)).isoformat()

    with db() as con:
        con.execute(
            "INSERT OR IGNORE INTO users (email, created_at) VALUES (?, ?)",
            (email, _now_utc())
        )
        user_id = con.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()["id"]
        con.execute(
            "INSERT INTO magic_tokens (token_hash, user_id, expires_at, used) VALUES (?, ?, ?, 0)",
            (token_hash, user_id, expires_at)
        )
        con.commit()

    # ── In dev mode: log the link instead of sending an email ────────────────
    verify_url = f"http://localhost:8000/auth/verify?token={raw_token}"
    print(f"\n🔗 MAGIC LINK for {email}:\n   {verify_url}\n   (expires in {TOKEN_TTL_MIN} min)\n")

    # TODO: swap for Resend call when API key is available:
    # import httpx
    # httpx.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_KEY}"},
    #   json={"from": "noreply@yourdomain.com", "to": email,
    #         "subject": "Your sign-in link", "text": f"Click here: {verify_url}"})

    return {"status": "sent", "dev_link": verify_url}

@router.get("/verify")
def verify_magic_link(token: str, response: Response):
    token_hash = _hash_token(token)
    with db() as con:
        row = con.execute(
            "SELECT * FROM magic_tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        if not row:
            raise HTTPException(401, "Invalid or expired link")
        if row["used"]:
            raise HTTPException(401, "Link already used")
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            raise HTTPException(401, "Link expired")
        con.execute("UPDATE magic_tokens SET used = 1 WHERE token_hash = ?", (token_hash,))
        con.commit()
        user_id = row["user_id"]

    session_token = _make_session_token(user_id)
    resp = JSONResponse({"status": "ok", "user_id": user_id})
    resp.set_cookie(
        key=COOKIE_NAME, value=session_token,
        max_age=SESSION_TTL_DAYS * 86400,
        httponly=True, samesite="lax"
    )
    return resp

@router.get("/session")
def session_status(request: Request):
    uid = get_current_user(request)
    if uid is None:
        return {"authenticated": False}
    with db() as con:
        user = con.execute("SELECT email FROM users WHERE id = ?", (uid,)).fetchone()
    return {"authenticated": True, "user_id": uid, "email": user["email"] if user else ""}

@router.post("/logout")
def logout(response: Response):
    resp = JSONResponse({"status": "ok"})
    resp.delete_cookie(COOKIE_NAME)
    return resp

# ── Board endpoints ────────────────────────────────────────────────────────────
class BoardCreate(BaseModel):
    name: str
    images: list = []   # [{src, title, author, source}]

@router.post("/boards")
def create_board(req: BoardCreate, request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(401, "Not authenticated")
    with db() as con:
        cur = con.execute(
            "INSERT INTO boards (user_id, name, created_at) VALUES (?, ?, ?)",
            (uid, req.name, _now_utc())
        )
        board_id = cur.lastrowid
        now = _now_utc()
        for img in req.images:
            con.execute(
                "INSERT INTO board_images (board_id, src, title, author, source, pinned_at) VALUES (?,?,?,?,?,?)",
                (board_id, img.get("src",""), img.get("title",""), img.get("author",""), img.get("source",""), now)
            )
        con.commit()
    return {"id": board_id, "name": req.name}

@router.get("/boards")
def list_boards(request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(401, "Not authenticated")
    with db() as con:
        boards = con.execute(
            "SELECT id, name, created_at FROM boards WHERE user_id = ? ORDER BY created_at DESC",
            (uid,)
        ).fetchall()
        result = []
        for b in boards:
            imgs = con.execute(
                "SELECT src, title, author, source FROM board_images WHERE board_id = ?",
                (b["id"],)
            ).fetchall()
            result.append({
                "id": b["id"], "name": b["name"], "created_at": b["created_at"],
                "images": [dict(r) for r in imgs]
            })
    return {"boards": result}

@router.delete("/boards/{board_id}")
def delete_board(board_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(401, "Not authenticated")
    with db() as con:
        row = con.execute("SELECT user_id FROM boards WHERE id = ?", (board_id,)).fetchone()
        if not row or row["user_id"] != uid:
            raise HTTPException(404, "Board not found")
        con.execute("DELETE FROM boards WHERE id = ?", (board_id,))
        con.commit()
    return {"status": "deleted"}
