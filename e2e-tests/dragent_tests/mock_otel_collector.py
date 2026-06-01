# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process mock OTLP/HTTP collector for e2e tests.

The OTel SDK HTTP exporter and NAT's ``OTLPSpanAdapterExporter`` both POST
spans via ``requests.Session()``; an in-process ``responses``/``respx`` patch
in the test process would not intercept the dragent *subprocess*. A real
listener on ``127.0.0.1:<random>`` does.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from dataclasses import field
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer


@dataclass(frozen=True)
class CapturedRequest:
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class _State:
    requests: list[CapturedRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class MockOtelCollector:
    """Listens on 127.0.0.1:<random>; records every POST /otel/v1/traces."""

    def __init__(self) -> None:
        self._state = _State()
        # Bind to port 0 so the kernel picks a free port; avoids cross-test
        # port collisions when the suite is run with pytest-xdist.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._build_handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mock-otel-collector",
            daemon=True,
        )

    @property
    def requests(self) -> list[CapturedRequest]:
        with self._state.lock:
            return list(self._state.requests)

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def wait_for_requests(self, n: int = 1, timeout: float = 10.0) -> None:
        """Block until at least *n* requests have been captured, or raise."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state.lock:
                if len(self._state.requests) >= n:
                    return
            time.sleep(0.05)
        raise AssertionError(
            f"Timed out waiting for {n} request(s); captured {len(self.requests)}."
        )

    def __enter__(self) -> MockOtelCollector:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        state = self._state

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server API
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                with state.lock:
                    state.requests.append(
                        CapturedRequest(
                            path=self.path,
                            headers={k: v for k, v in self.headers.items()},
                            body=body,
                        )
                    )
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802 — http.server API
                # The moderation middleware (datarobot_dome) probes
                # ``<DATAROBOT_ENDPOINT>/account/info/`` at builder time to
                # validate the API token. Return an empty 200 so the workflow
                # can start; we don't otherwise assert on GETs.
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                # Quiet — the test asserts on captured state, not stderr noise.
                return

        return _Handler
