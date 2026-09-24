"""
Sign-in emails. Sent through Resend when RESEND_API_KEY is set; otherwise, outside
production, kept in a small in-memory outbox (see /auth/dev-outbox) and printed.
"""
import html
import os
import time
from collections import deque

import httpx

ENV = os.environ.get("ENV", "local")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "Glane <login@heretique.fr>")

DEV_OUTBOX = deque(maxlen=20)


class EmailNotConfigured(RuntimeError):
    pass


def can_send() -> bool:
    return bool(RESEND_API_KEY) or ENV != "production"


def _render(link: str, minutes: int):
    subject = "Your Glane sign-in link"
    text = (
        "Hello,\n\n"
        f"Use this link to sign in to Glane. It works once and expires in {minutes} minutes:\n\n"
        f"{link}\n\n"
        "If you did not ask for it, you can ignore this email.\n"
    )
    safe = html.escape(link, quote=True)
    body = f"""<!doctype html><html><body style="margin:0;padding:32px 16px;background:#E3E3E3;font-family:Georgia,'Times New Roman',serif;color:#000">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
<table role="presentation" width="100%" style="max-width:480px;background:#fff;border-radius:6px" cellpadding="0" cellspacing="0"><tr><td style="padding:32px">
<div style="font-size:20px;letter-spacing:4px">GLANE</div>
<p style="font-size:22px;font-style:italic;margin:28px 0 12px">Your sign-in link</p>
<p style="font-size:15px;line-height:1.5;color:#444;margin:0 0 24px">Click the button below to sign in. It works once and expires in {minutes} minutes.</p>
<a href="{safe}" style="display:inline-block;background:#000;color:#fff;text-decoration:none;font-size:14px;letter-spacing:1px;padding:12px 22px;border-radius:999px">SIGN IN TO GLANE</a>
<p style="font-size:13px;line-height:1.5;color:#737373;margin:28px 0 0">If you did not ask for this link, you can ignore this email.</p>
</td></tr></table></td></tr></table></body></html>"""
    return subject, text, body


def send_login_email(to: str, link: str, minutes: int) -> None:
    subject, text, body = _render(link, minutes)
    if not RESEND_API_KEY:
        if ENV == "production":
            raise EmailNotConfigured("RESEND_API_KEY is not set")
        DEV_OUTBOX.appendleft({"to": to, "subject": subject, "link": link, "at": time.time()})
        print(f"[mail/dev] sign-in link for {to}: {link}", flush=True)
        return
    resp = httpx.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
        json={"from": EMAIL_FROM, "to": [to], "subject": subject, "text": text, "html": body},
        timeout=10,
    )
    resp.raise_for_status()
