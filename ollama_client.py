"""A small client for the Ollama HTTP API.

Uses only the Python standard library: nothing to install.
API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(Exception):
    """A user-readable error (Ollama unreachable, model missing, ...)."""


def normalize_host(host=None):
    """Turn a value such as OLLAMA_HOST into a usable base URL.

    >>> normalize_host("127.0.0.1:11434")
    'http://127.0.0.1:11434'
    >>> normalize_host("0.0.0.0")
    'http://localhost:11434'
    """
    host = (host or "").strip() or DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    parts = urllib.parse.urlsplit(host)
    hostname = parts.hostname or "localhost"
    if hostname == "0.0.0.0":  # a listen address, not an address to connect to
        hostname = "localhost"
    if ":" in hostname:  # IPv6
        hostname = "[" + hostname + "]"
    port = parts.port
    if port is None:
        port = 443 if parts.scheme == "https" else 11434
    return "{}://{}:{}".format(parts.scheme, hostname, port)


class OllamaClient:
    def __init__(self, host=None, timeout=600):
        self.host = normalize_host(host or os.environ.get("OLLAMA_HOST"))
        self.timeout = timeout
        # No proxy: Ollama usually runs on the local machine.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ------------------------------------------------------------------ HTTP
    def _open(self, path, payload=None, timeout=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.host + path, data=data, headers=headers)
        try:
            return self._opener.open(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:  # before URLError: it is a subclass of it
            raise OllamaError(self._http_error_message(exc)) from None
        except (urllib.error.URLError, OSError):
            raise OllamaError(
                "Could not reach Ollama at {}. Start the Ollama app "
                "(or run `ollama serve`).".format(self.host)
            ) from None

    @staticmethod
    def _http_error_message(exc):
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            message = body.get("error")
        except (ValueError, AttributeError):
            message = None
        return message or "HTTP error {} returned by Ollama.".format(exc.code)

    # ------------------------------------------------------------------- API
    def list_models(self):
        """Return the sorted names of the installed models."""
        with self._open("/api/tags", timeout=5) as response:
            try:
                data = json.loads(response.read().decode("utf-8"))
            except ValueError:
                raise OllamaError("Unexpected response from Ollama (invalid JSON).") from None
        return sorted(m["name"] for m in data.get("models", []) if "name" in m)

    def chat_stream(self, model, messages, temperature=None, cancel=None):
        """Send the conversation and yield the reply as it is generated.

        Yields ``(kind, value)`` tuples:
          - ``("content", "text")``   a chunk of the reply;
          - ``("thinking", "text")``  a chunk of "thinking" (reasoning models);
          - ``("stats", {...})``      final statistics (tokens, speed).

        ``cancel`` is a ``threading.Event``: as soon as it is set, the
        connection is closed, which tells Ollama to stop generating.
        """
        payload = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        response = self._open("/api/chat", payload)
        try:
            for raw_line in response:
                if cancel is not None and cancel.is_set():
                    return
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise OllamaError(str(chunk["error"]))
                message = chunk.get("message") or {}
                if message.get("thinking"):
                    yield ("thinking", message["thinking"])
                if message.get("content"):
                    yield ("content", message["content"])
                if chunk.get("done"):
                    yield ("stats", self._stats(chunk))
                    return
        except OllamaError:
            raise
        except Exception as exc:  # connection dropped mid-stream, etc.
            if cancel is not None and cancel.is_set():
                return
            raise OllamaError("Connection interrupted: {}".format(exc)) from None
        finally:
            try:
                response.close()
            except Exception:
                pass

    @staticmethod
    def _stats(chunk):
        tokens = chunk.get("eval_count") or 0
        nanoseconds = chunk.get("eval_duration") or 0
        speed = tokens / (nanoseconds / 1e9) if tokens and nanoseconds else None
        return {"tokens": tokens, "tokens_per_second": speed}
