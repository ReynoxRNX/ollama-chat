"""Petit client pour l'API HTTP d'Ollama.

Utilise uniquement la bibliothèque standard de Python : rien à installer.
Documentation de l'API : https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(Exception):
    """Erreur lisible par l'utilisateur (Ollama injoignable, modèle absent...)."""


def normalize_host(host=None):
    """Transforme une valeur comme OLLAMA_HOST en URL de base utilisable.

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
    if hostname == "0.0.0.0":  # adresse d'écoute, pas une adresse de connexion
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
        # Pas de proxy : Ollama tourne le plus souvent sur la machine locale.
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
        except urllib.error.HTTPError as exc:  # avant URLError : c'en est une sous-classe
            raise OllamaError(self._http_error_message(exc)) from None
        except (urllib.error.URLError, OSError):
            raise OllamaError(
                "Impossible de joindre Ollama sur {}. Lancez l'application Ollama "
                "(ou « ollama serve »).".format(self.host)
            ) from None

    @staticmethod
    def _http_error_message(exc):
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            message = body.get("error")
        except (ValueError, AttributeError):
            message = None
        return message or "Erreur HTTP {} renvoyée par Ollama.".format(exc.code)

    # ------------------------------------------------------------------- API
    def list_models(self):
        """Retourne la liste triée des noms de modèles installés."""
        with self._open("/api/tags", timeout=5) as response:
            try:
                data = json.loads(response.read().decode("utf-8"))
            except ValueError:
                raise OllamaError("Réponse inattendue d'Ollama (JSON invalide).") from None
        return sorted(m["name"] for m in data.get("models", []) if "name" in m)

    def chat_stream(self, model, messages, temperature=None, cancel=None):
        """Envoie la conversation et produit la réponse au fil de l'eau.

        Génère des tuples ``(type, contenu)`` :
          - ``("content", "texte")``   morceau de la réponse ;
          - ``("thinking", "texte")``  morceau de « réflexion » (modèles de raisonnement) ;
          - ``("stats", {...})``       statistiques finales (tokens, vitesse).

        ``cancel`` est un ``threading.Event`` : dès qu'il est activé, la
        connexion est fermée, ce qui demande à Ollama d'arrêter de générer.
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
        except Exception as exc:  # connexion coupée en cours de route, etc.
            if cancel is not None and cancel.is_set():
                return
            raise OllamaError("Connexion interrompue : {}".format(exc)) from None
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
