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


MAIL_TEXT = {
    "en": {
        "subject": "Your Glane sign-in link",
        "hello": "Hello,",
        "intro": "Use this link to sign in to Glane. It works once and expires in {minutes} minutes:",
        "ignore": "If you did not ask for it, you can ignore this email.",
        "title": "Your sign-in link",
        "lead": "Click the button below to sign in. It works once and expires in {minutes} minutes.",
        "button": "Sign in to Glane",
        "footer": "If you did not ask for this link, you can ignore this email.",
    },
    "fr": {
        "subject": "Votre lien de connexion à Glane",
        "hello": "Bonjour,",
        "intro": "Utilisez ce lien pour vous connecter à Glane. Il fonctionne une fois et expire dans {minutes} minutes :",
        "ignore": "Si vous n’avez rien demandé, vous pouvez ignorer cet email.",
        "title": "Votre lien de connexion",
        "lead": "Cliquez sur le bouton ci-dessous pour vous connecter. Il fonctionne une fois et expire dans {minutes} minutes.",
        "button": "Se connecter à Glane",
        "footer": "Si vous n’avez pas demandé ce lien, vous pouvez ignorer cet email.",
    },
}


def _render(link: str, minutes: int, lang: str = "en"):
    m = {k: v.format(minutes=minutes) for k, v in MAIL_TEXT.get(lang, MAIL_TEXT["en"]).items()}
    text = f"{m['hello']}\n\n{m['intro']}\n\n{link}\n\n{m['ignore']}\n"
    safe = html.escape(link, quote=True)
    # The site's look: "Glane" in Flor de Ruina Semilla (mail apps that refuse web
    # fonts fall back to Georgia), text in Junicode, a black button with square
    # corners, and at the foot of the card a band of stretched pixels (an image:
    # CSS backgrounds are not reliable in mail). Fonts and band come from the site.
    base = html.escape(link.split("/auth/")[0], quote=True)
    serif = "Junicode,Georgia,'Times New Roman',serif"
    title = "'Flor Semilla',Georgia,'Times New Roman',serif"
    body = f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@font-face {{ font-family: 'Flor Semilla'; src: url('{base}/fonts/FlorDeRuina-Semilla.woff2') format('woff2'); }}
@font-face {{ font-family: Junicode; src: url('{base}/fonts/JunicodeVF-Roman.woff2') format('woff2'); font-weight: 300 700; }}
@font-face {{ font-family: Junicode; src: url('{base}/fonts/JunicodeVF-Italic.woff2') format('woff2'); font-weight: 300 700; font-style: italic; }}
</style></head>
<body style="margin:0;padding:32px 16px;background:#E3E3E3;font-family:{serif};color:#000">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
<table role="presentation" width="100%" style="max-width:480px;background:#fff;border-radius:2px;border-collapse:separate;overflow:hidden" cellpadding="0" cellspacing="0">
<tr><td style="padding:32px 32px 28px">
<div style="font-family:{title};font-size:40px;line-height:1;color:#000">Glane<span style="font-size:20px;vertical-align:top">*</span></div>
<p style="font-family:{serif};font-size:23px;font-style:italic;margin:30px 0 10px">{m['title']}</p>
<p style="font-family:{serif};font-size:16px;line-height:1.5;color:#444;margin:0 0 26px">{m['lead']}</p>
<a href="{safe}" style="display:inline-block;background:#000;color:#fff;text-decoration:none;font-family:{title};font-size:16px;padding:13px 24px;border-radius:2px">{m['button']}</a>
<p style="font-family:{serif};font-size:13px;line-height:1.5;font-style:italic;color:#737373;margin:30px 0 0">{m['footer']}</p>
</td></tr>
<tr><td style="padding:0;line-height:0;font-size:0"><img src="{base}/mail-band.png" width="480" height="6" alt="" style="display:block;width:100%;height:6px;border:0"></td></tr>
</table></td></tr></table></body></html>"""
    return m["subject"], text, body


def send_login_email(to: str, link: str, minutes: int, lang: str = "en") -> None:
    subject, text, body = _render(link, minutes, lang)
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
