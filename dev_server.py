"""
Local development server for the Glane front end.

    python dev_server.py        → http://localhost:5173

- serves app/frontend;
- search and images go to a remote backend (production by default), so the
  model and index (~3 GB of RAM) never load on this machine;
- accounts (/auth/*, /api/*) run here, on SQLite (data/dev-accounts.db); emails
  are not sent but listed at http://localhost:5173/auth/dev-outbox.

The remote backend may predate v0.2. Unless --native is given, v0.2 search
features are emulated on top of it: `reference_ids` / `negative_ids` become
uploaded images (fetched here, where museum CDNs don't enforce CORS), and
POST /collection-zip is built here with the backend's own code.
With --native, everything (accounts included) goes to --api untouched.

--mirror PATH serves images produced by scripts/mirror_images.py (dir: output)
and points search results at them, as IMAGE_BASE_URL does on the server.
"""
import argparse
import base64
import json
import mimetypes
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "app", "frontend")
SEARCH_PREFIXES = ("/search", "/cluster-members", "/collection-zip", "/health")
ACCOUNT_PREFIXES = ("/auth/", "/api/", "/flag")
PASS_HEADERS = ("content-type", "location", "set-cookie", "cache-control", "x-missing")
UA = {"User-Agent": "Mozilla/5.0 (Glane dev server)"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None   # hand 30x responses (and their cookies) back to the browser


_opener = urllib.request.build_opener(_NoRedirect)


def _post_json(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _ids_to_base64(api, ids, mirror_dir=None):
    if not ids:
        return []
    items = _post_json(f"{api}/cluster-members", {"faiss_ids": ids})["results"]
    out = []
    for it in items:
        local = mirror_dir and os.path.join(mirror_dir, "h", f"{int(it['faiss_id'])}.webp")
        if local and os.path.isfile(local):
            with open(local, "rb") as f:
                out.append(base64.b64encode(f.read()).decode())
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(it["image_url"], headers=UA), timeout=60) as r:
                out.append(base64.b64encode(r.read()).decode())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ! image {it.get('faiss_id')} not fetchable ({e}), skipped")
    return out


def start_accounts(port, front_port):
    """Run the real accounts router (no ML dependencies) in a background thread."""
    os.environ.setdefault("ENV", "local")
    os.environ.setdefault("PUBLIC_BASE_URL", f"http://localhost:{front_port}")
    os.environ.setdefault("SQLITE_PATH", os.path.join(ROOT, "data", "dev-accounts.db"))
    import uvicorn
    from fastapi import FastAPI
    from app.backend import accounts, db
    db.init()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(accounts.router)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()


class Handler(SimpleHTTPRequestHandler):
    api = ""
    accounts_api = ""
    native = False
    mirror_dir = None
    base = ""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=FRONTEND, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if self.path.startswith(("/search", "/cluster", "/auth", "/api", "/collection-zip")):
            super().log_message(fmt, *args)

    def _upstream(self):
        if self.path.startswith(ACCOUNT_PREFIXES):
            return self.accounts_api
        if self.path.startswith(SEARCH_PREFIXES):
            return self.api
        return None

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _with_mirror(self, items):
        for it in items:
            if self.mirror_dir and "thumb_url" not in it and it.get("faiss_id") is not None:
                fid = int(it["faiss_id"])
                it["original_url"] = it.get("image_url", "")
                it["thumb_url"] = f"{self.base}/mirror/t/{fid}.webp"
                it["image_url"] = f"{self.base}/mirror/h/{fid}.webp"
        return items

    def _serve_mirror(self):
        rel = urllib.parse.unquote(self.path.split("?", 1)[0][len("/mirror/"):])
        path = os.path.realpath(os.path.join(self.mirror_dir, *rel.split("/")))
        if not path.startswith(os.path.realpath(self.mirror_dir) + os.sep) or not os.path.isfile(path):
            return self._reply(404, b"", [("Content-Type", "text/plain")])
        with open(path, "rb") as f:
            self._reply(200, f.read(), [("Content-Type", "image/webp")])

    def do_GET(self):
        if self.mirror_dir and self.path.startswith("/mirror/"):
            return self._serve_mirror()
        upstream = self._upstream()
        if upstream:
            return self._forward("GET", None, upstream)
        return super().do_GET()

    def do_POST(self):
        body = self._body()
        if not self.native and self.path == "/collection-zip":
            return self._collection_zip(json.loads(body or b"{}"))
        if not self.native and self.path == "/search":
            payload = json.loads(body or b"{}")
            ref_ids = payload.pop("reference_ids", None)
            neg_ids = payload.pop("negative_ids", None)
            if ref_ids:
                payload["reference_images"] = (payload.get("reference_images") or []) + _ids_to_base64(self.api, ref_ids, self.mirror_dir)
            if neg_ids:
                payload["negative_images"] = (payload.get("negative_images") or []) + _ids_to_base64(self.api, neg_ids, self.mirror_dir)
            body = json.dumps(payload).encode()
        return self._forward("POST", body, self._upstream() or self.api)

    def do_PUT(self):
        return self._forward("PUT", self._body(), self._upstream() or self.api)

    def do_DELETE(self):
        return self._forward("DELETE", self._body() or None, self._upstream() or self.api)

    def _collection_zip(self, payload):
        from app.backend import zip_export
        ids = payload.get("faiss_ids", [])[:zip_export.MAX_ITEMS]
        items = self._with_mirror(_post_json(f"{self.api}/cluster-members", {"faiss_ids": ids})["results"]) if ids else []
        data, missing = zip_export.build_zip(items)
        name = urllib.parse.quote(f"{zip_export.safe_name(payload.get('name', ''))}.zip")
        self._reply(200, data, [("Content-Type", "application/zip"),
                                ("Content-Disposition", f"attachment; filename*=UTF-8''{name}"),
                                ("X-Missing", str(missing))])

    def _reply(self, status, data, headers):
        try:
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            pass  # the browser dropped a superseded request

    def _forward(self, method, body, upstream):
        headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "cookie", "accept")}
        req = urllib.request.Request(upstream + self.path, data=body, method=method, headers={**headers, **UA})
        try:
            resp = _opener.open(req, timeout=180)
        except urllib.error.HTTPError as e:
            resp = e   # includes 30x, kept un-followed by _NoRedirect
        except urllib.error.URLError as e:
            return self._reply(502, json.dumps({"detail": str(e)}).encode(), [("Content-Type", "application/json")])
        with resp:
            data = resp.read()
            passed = [(k, v) for k, v in resp.headers.items() if k.lower() in PASS_HEADERS]
            status = resp.code
        if self.mirror_dir and not self.native and status == 200 and self.path in ("/search", "/cluster-members"):
            payload = json.loads(data)
            self._with_mirror(payload.get("results", []))
            data = json.dumps(payload).encode()
        self._reply(status, data, passed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="https://glane.heretique.fr", help="backend for search and images")
    p.add_argument("--port", type=int, default=5173)
    p.add_argument("--accounts-port", type=int, default=5174)
    p.add_argument("--mirror", help="folder written by scripts/mirror_images.py --out dir:…")
    p.add_argument("--native", action="store_true",
                   help="--api already runs v0.2: forward everything to it, accounts included")
    args = p.parse_args()
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("font/woff2", ".woff2")
    Handler.api = args.api.rstrip("/")
    Handler.native = args.native
    Handler.mirror_dir = os.path.abspath(args.mirror) if args.mirror else None
    Handler.base = f"http://localhost:{args.port}"
    if args.native:
        Handler.accounts_api = Handler.api
    else:
        start_accounts(args.accounts_port, args.port)
        Handler.accounts_api = f"http://127.0.0.1:{args.accounts_port}"
    print(f"Glane dev: http://localhost:{args.port}   (search: {Handler.api}, accounts: {Handler.accounts_api})", flush=True)
    if not args.native:
        print(f"Sign-in emails (dev): http://localhost:{args.port}/auth/dev-outbox", flush=True)
    if Handler.mirror_dir:
        print(f"Images from the local mirror: {Handler.mirror_dir}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
