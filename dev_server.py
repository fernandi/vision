"""
Front-end dev server: serves app/frontend locally and forwards API calls to a
running Glane backend (production by default), so the UI can be worked on
without loading the model and index on this machine (~3 GB of RAM).

    python dev_server.py                      # → http://localhost:5173
    python dev_server.py --api http://localhost:8000 --native

The deployed backend may predate v0.2 endpoints. Unless --native is given,
they are emulated here on top of the older API:
- `reference_ids` / `negative_ids` are turned into uploaded images, fetched
  server-side where museum CDNs don't enforce CORS;
- POST /collection-zip is built here with the same code as the backend.
"""
import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from app.backend import zip_export

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "frontend")
API_PREFIXES = ("/search", "/cluster-members", "/collection-zip", "/health", "/auth/", "/flag")
UA = {"User-Agent": "Mozilla/5.0 (Glane dev server)"}


def _post_json(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _ids_to_base64(api, ids):
    if not ids:
        return []
    items = _post_json(f"{api}/cluster-members", {"faiss_ids": ids})["results"]
    out = []
    for it in items:
        try:
            with urllib.request.urlopen(urllib.request.Request(it["image_url"], headers=UA), timeout=60) as r:
                out.append(base64.b64encode(r.read()).decode())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ! image {it.get('faiss_id')} not fetchable ({e}), skipped")
    return out


class Handler(SimpleHTTPRequestHandler):
    api = ""
    native = False

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=FRONTEND, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith(("/search", "/cluster", "/auth")):
            return
        super().log_message(fmt, *args)

    def _is_api(self):
        return self.path.startswith(API_PREFIXES)

    def do_GET(self):
        if self._is_api():
            return self._forward("GET", None)
        return super().do_GET()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/collection-zip" and not self.native:
            return self._collection_zip(json.loads(body or b"{}"))
        if self.path == "/search" and not self.native:
            payload = json.loads(body or b"{}")
            ref_ids = payload.pop("reference_ids", None)
            neg_ids = payload.pop("negative_ids", None)
            if ref_ids:
                payload["reference_images"] = (payload.get("reference_images") or []) + _ids_to_base64(self.api, ref_ids)
            if neg_ids:
                payload["negative_images"] = (payload.get("negative_images") or []) + _ids_to_base64(self.api, neg_ids)
            body = json.dumps(payload).encode()
        return self._forward("POST", body)

    def _collection_zip(self, payload):
        ids = payload.get("faiss_ids", [])[:zip_export.MAX_ITEMS]
        items = _post_json(f"{self.api}/cluster-members", {"faiss_ids": ids})["results"] if ids else []
        data, missing = zip_export.build_zip(items)
        name = urllib.parse.quote(f"{zip_export.safe_name(payload.get('name', ''))}.zip")
        self._reply(200, data, "application/zip", {
            "Content-Disposition": f"attachment; filename*=UTF-8''{name}",
            "X-Missing": str(missing),
        })

    def _reply(self, status, data, ctype, extra=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            pass  # the browser dropped a superseded request

    def _forward(self, method, body):
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() in ("content-type", "cookie", "accept")}
        req = urllib.request.Request(self.api + self.path, data=body, method=method,
                                     headers={**headers, **UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                status, data, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            status, data, ctype = e.code, e.read(), e.headers.get("Content-Type", "text/plain")
        except urllib.error.URLError as e:
            status, data, ctype = 502, json.dumps({"detail": str(e)}).encode(), "application/json"
        self._reply(status, data, ctype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="https://glane.heretique.fr", help="backend to forward API calls to")
    p.add_argument("--port", type=int, default=5173)
    p.add_argument("--native", action="store_true",
                   help="backend already runs v0.2: forward everything untouched")
    args = p.parse_args()
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("font/woff2", ".woff2")
    Handler.api = args.api.rstrip("/")
    Handler.native = args.native
    print(f"Glane dev: http://localhost:{args.port}   (API: {Handler.api})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
