"""A minimal local HTTP server for serving a static directory.

Used to give the fixture pages a real http://127.0.0.1 origin instead of file://, so the
guardrail's domain-based allowlist has something realistic to check against, and so a discovery or
replay run against the fixture behaves the same way it would against a real deployed target
(file:// URLs have no hostname at all, which the allowlist has no meaningful way to check).
"""

import functools
import http.server
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def serve_directory(directory: str | Path, host: str = "127.0.0.1") -> Iterator[str]:
    """Serve `directory` over HTTP on an OS-assigned free port for the duration of the context.
    Yields the base URL (e.g. "http://127.0.0.1:54321"). Runs in a background thread; shuts down
    cleanly on exit."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    # Port 0 asks the OS for any free port, avoiding collisions between concurrent test runs.
    server = http.server.ThreadingHTTPServer((host, 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
