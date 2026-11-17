"""Read-only localhost viewer; payloads are inserted with textContent, never innerHTML."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from .report import dashboard_data


def serve(run, port):
    page = files("openrubrics_lab").joinpath("web/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                payload, content_type = page, "text/html; charset=utf-8"
            elif self.path == "/api/results":
                payload = json.dumps(dashboard_data(run), ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"OpenRubrics experiment viewer: http://127.0.0.1:{port} (saved actual outputs)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
