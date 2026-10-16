"""Tests for the Ollama client, run against a fake local server (no Ollama needed)."""

import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_client import OllamaClient, OllamaError, normalize_host  # noqa: E402


class FakeOllama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "qwen3:8b"}, {"name": "gemma3:4b"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        self.server.last_request = request
        model = request["model"]

        if model == "missing":
            return self._json(404, {"error": "model 'missing' not found"})

        if model == "crashes":
            lines = [{"message": {"content": "Start"}}, {"error": "boom"}]
        elif model == "reasoner":
            lines = [
                {"message": {"thinking": "Hmm, "}},
                {"message": {"thinking": "let's see."}},
                {"message": {"content": "Answer"}, "done": True, "eval_count": 10, "eval_duration": 5 * 10**9},
            ]
        elif model == "slow":
            lines = [{"message": {"content": "word%d " % i}} for i in range(200)]
        else:
            lines = [
                {"message": {"content": "Hel"}},
                {"message": {"content": "lo"}},
                {"message": {"content": ""}, "done": True, "eval_count": 20, "eval_duration": 10**9},
            ]

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for line in lines:
                data = (json.dumps(line) + "\n").encode()
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()
                if model == "slow":
                    time.sleep(0.02)
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            self.server.client_left = True


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.server.client_left = False
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = OllamaClient("127.0.0.1:%d" % cls.server.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def chat(self, model, **kwargs):
        return list(self.client.chat_stream(model, [{"role": "user", "content": "hi"}], **kwargs))

    def test_normalize_host(self):
        self.assertEqual(normalize_host(None), "http://localhost:11434")
        self.assertEqual(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(normalize_host("0.0.0.0"), "http://localhost:11434")
        self.assertEqual(normalize_host("http://server"), "http://server:11434")
        self.assertEqual(normalize_host("https://example.org/"), "https://example.org:443")

    def test_list_models_sorted(self):
        self.assertEqual(self.client.list_models(), ["gemma3:4b", "qwen3:8b"])

    def test_streaming_content_and_stats(self):
        events = self.chat("qwen3:8b")
        text = "".join(c for kind, c in events if kind == "content")
        self.assertEqual(text, "Hello")
        stats = [c for kind, c in events if kind == "stats"][0]
        self.assertEqual(stats["tokens"], 20)
        self.assertAlmostEqual(stats["tokens_per_second"], 20.0)

    def test_temperature_and_messages_are_sent(self):
        self.chat("qwen3:8b", temperature=0.3)
        request = self.server.last_request
        self.assertEqual(request["options"], {"temperature": 0.3})
        self.assertTrue(request["stream"])
        self.assertEqual(request["messages"][0]["content"], "hi")

    def test_thinking_is_separated(self):
        events = self.chat("reasoner")
        self.assertEqual("".join(c for k, c in events if k == "thinking"), "Hmm, let's see.")
        self.assertEqual("".join(c for k, c in events if k == "content"), "Answer")

    def test_http_error_uses_ollama_message(self):
        with self.assertRaisesRegex(OllamaError, "not found"):
            self.chat("missing")

    def test_error_in_stream(self):
        with self.assertRaisesRegex(OllamaError, "boom"):
            self.chat("crashes")

    def test_cancel_stops_generation(self):
        cancel = threading.Event()
        received = []
        for kind, text in self.client.chat_stream(
            "slow", [{"role": "user", "content": "x"}], cancel=cancel
        ):
            received.append(text)
            if len(received) == 3:
                cancel.set()
        self.assertLess(len(received), 10)

    def test_unreachable_server(self):
        client = OllamaClient("127.0.0.1:1")  # closed port
        with self.assertRaisesRegex(OllamaError, "Could not reach Ollama"):
            client.list_models()
        with self.assertRaises(OllamaError):
            list(client.chat_stream("x", []))


if __name__ == "__main__":
    unittest.main()
