#!/usr/bin/env python3
"""Chat local avec Ollama : interface Tkinter, aucune dépendance externe.

Lancement :  python ollama_chat.py
"""

import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from ollama_client import OllamaClient, OllamaError

APP_TITLE = "Chat Ollama"
POLL_MS = 30
MODELS_EVENT = "models"  # identifiant des événements « liste des modèles »

# Couleurs fixes (fond clair) pour rester lisible quel que soit le thème du système.
BG = "#ffffff"
FG = "#1f2328"
USER_COLOR = "#0b5cad"
AI_COLOR = "#1a7f37"
MUTED = "#6e7781"
ERROR = "#cf222e"


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.client = OllamaClient()

        self.history = []  # messages {"role", "content"} envoyés à Ollama
        self.events = queue.Queue()  # worker -> interface
        self.active_stream = None  # identifiant de la génération en cours
        self.stream_counter = 0
        self.cancel_event = None

        self.reply_parts = []  # morceaux de la réponse en cours
        self.phase = None  # None, "thinking" ou "content"
        self.last_stats = None
        self.pending_text = ""  # dernier message envoyé (pour le remettre en cas d'échec)

        root.title(APP_TITLE)
        root.geometry("920x720")
        root.minsize(580, 440)
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.refresh_models()
        self.root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------ interface
    def _build_ui(self):
        base = tkfont.nametofont("TkTextFont")
        self.font = base.copy()
        self.font.configure(size=max(base.cget("size"), 11))
        self.font_bold = self.font.copy()
        self.font_bold.configure(weight="bold")
        self.font_italic = self.font.copy()
        self.font_italic.configure(slant="italic")
        self.font_small = self.font.copy()
        self.font_small.configure(size=self.font.cget("size") - 2)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        # Ligne 1 : modèle, température, actions
        bar = ttk.Frame(outer)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(bar, text="Modèle").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(bar, textvariable=self.model_var, state="readonly", width=30)
        self.model_box.pack(side="left", padx=(6, 4))
        ttk.Button(bar, text="Actualiser", command=self.refresh_models).pack(side="left")

        ttk.Label(bar, text="Température").pack(side="left", padx=(16, 4))
        self.temp_var = tk.StringVar(value="0.7")
        self.temp_box = ttk.Spinbox(
            bar, from_=0.0, to=2.0, increment=0.1, width=5, format="%.1f", textvariable=self.temp_var
        )
        self.temp_box.pack(side="left")

        ttk.Button(bar, text="Enregistrer…", command=self.save_conversation).pack(side="right")
        ttk.Button(bar, text="Nouvelle conversation", command=self.new_conversation).pack(
            side="right", padx=(0, 6)
        )

        # Ligne 2 : consigne système
        sys_row = ttk.Frame(outer)
        sys_row.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        sys_row.columnconfigure(1, weight=1)
        ttk.Label(sys_row, text="Consigne système").grid(row=0, column=0, padx=(0, 6))
        self.system_var = tk.StringVar()
        ttk.Entry(sys_row, textvariable=self.system_var).grid(row=0, column=1, sticky="ew")

        # Zone de conversation
        chat_frame = ttk.Frame(outer)
        chat_frame.grid(row=2, column=0, sticky="nsew")
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        self.chat = tk.Text(
            chat_frame, wrap="word", state="disabled", font=self.font, bg=BG, fg=FG,
            padx=12, pady=10, relief="flat", highlightthickness=1,
            highlightbackground="#d0d7de", highlightcolor="#d0d7de", spacing3=2,
        )
        scroll = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=scroll.set)
        self.chat.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        self.chat.tag_configure("role_user", foreground=USER_COLOR, font=self.font_bold, spacing1=6)
        self.chat.tag_configure("role_ai", foreground=AI_COLOR, font=self.font_bold, spacing1=6)
        self.chat.tag_configure("thinking", foreground=MUTED, font=self.font_italic)
        self.chat.tag_configure("meta", foreground=MUTED, font=self.font_small)
        self.chat.tag_configure("error", foreground=ERROR)

        # Zone de saisie
        entry_row = ttk.Frame(outer)
        entry_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        entry_row.columnconfigure(0, weight=1)
        self.input = tk.Text(
            entry_row, height=3, wrap="word", font=self.font, bg=BG, fg=FG, insertbackground=FG,
            padx=8, pady=6, relief="flat", highlightthickness=1,
            highlightbackground="#d0d7de", highlightcolor=USER_COLOR,
        )
        self.input.grid(row=0, column=0, sticky="ew")
        self.send_button = ttk.Button(entry_row, text="Envoyer", command=self.send, width=10)
        self.send_button.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.input.bind("<Return>", self._on_return)
        self.root.bind("<Escape>", lambda _e: self.stop())
        self.input.focus_set()

        # Barre d'état
        self.status_var = tk.StringVar(value="Connexion à Ollama…")
        ttk.Label(outer, textvariable=self.status_var, foreground=MUTED).grid(
            row=4, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Label(
            outer, text="Entrée : envoyer · Maj+Entrée : nouvelle ligne · Échap : arrêter",
            foreground=MUTED,
        ).grid(row=5, column=0, sticky="ew")

    def _on_return(self, event):
        if event.state & 0x0001:  # Maj enfoncée : saut de ligne normal
            return None
        self.send()
        return "break"

    # ---------------------------------------------------------- zone de chat
    def _append(self, text, *tags):
        at_bottom = self.chat.yview()[1] >= 0.99
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tags)
        self.chat.configure(state="disabled")
        if at_bottom:
            self.chat.see("end")

    def _clear_chat(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")

    def _set_busy(self, busy):
        if busy:
            self.send_button.configure(text="Arrêter", command=self.stop)
            self.model_box.configure(state="disabled")
        else:
            self.send_button.configure(text="Envoyer", command=self.send)
            self.model_box.configure(state="readonly")

    # --------------------------------------------------------------- modèles
    def refresh_models(self):
        self.status_var.set("Recherche des modèles installés…")

        def work():
            try:
                self.events.put((MODELS_EVENT, "ok", self.client.list_models()))
            except OllamaError as exc:
                self.events.put((MODELS_EVENT, "error", str(exc)))
            except Exception as exc:
                self.events.put((MODELS_EVENT, "error", "Erreur inattendue : {}".format(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _on_models(self, kind, payload):
        if kind == "error":
            self.status_var.set(payload + "  Puis cliquez sur « Actualiser ».")
            return
        models = payload
        self.model_box.configure(values=models)
        if not models:
            self.model_var.set("")
            self.status_var.set(
                "Aucun modèle installé. Dans un terminal, tapez par exemple : ollama pull qwen3:8b"
            )
            return
        if self.model_var.get() not in models:
            self.model_var.set(models[0])
        self.status_var.set("Connecté à {} · {} modèle(s) installé(s)".format(self.client.host, len(models)))

    # ----------------------------------------------------------------- envoi
    def send(self):
        if self.active_stream is not None:
            return
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            return
        model = self.model_var.get().strip()
        if not model:
            self.status_var.set("Choisissez d'abord un modèle (cliquez sur « Actualiser » si la liste est vide).")
            return

        self.input.delete("1.0", "end")
        self.pending_text = text
        self.history.append({"role": "user", "content": text})
        self._append("Vous\n", "role_user")
        self._append(text + "\n\n")
        self._append(model + "\n", "role_ai")

        messages = list(self.history)
        system_prompt = self.system_var.get().strip()
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        try:
            temperature = float(self.temp_var.get().replace(",", "."))
        except ValueError:
            temperature = None

        self.reply_parts = []
        self.phase = None
        self.last_stats = None
        self.stream_counter += 1
        self.active_stream = self.stream_counter
        self.cancel_event = threading.Event()
        self._set_busy(True)
        self.status_var.set("Génération en cours…")

        threading.Thread(
            target=self._worker,
            args=(self.active_stream, model, messages, temperature, self.cancel_event),
            daemon=True,
        ).start()

    def _worker(self, stream_id, model, messages, temperature, cancel):
        try:
            for kind, payload in self.client.chat_stream(model, messages, temperature, cancel):
                self.events.put((stream_id, kind, payload))
            self.events.put((stream_id, "end", None))
        except OllamaError as exc:
            self.events.put((stream_id, "error", str(exc)))
        except Exception as exc:
            self.events.put((stream_id, "error", "Erreur inattendue : {}".format(exc)))

    def stop(self):
        if self.active_stream is None:
            return
        self.cancel_event.set()
        self._finish(stopped=True)

    # ------------------------------------------------- événements du worker
    def _poll(self):
        try:
            while True:
                stream_id, kind, payload = self.events.get_nowait()
                if stream_id == MODELS_EVENT:
                    self._on_models(kind, payload)
                elif stream_id == self.active_stream:
                    self._handle(kind, payload)
                # sinon : événement d'une génération déjà arrêtée, on l'ignore
        except queue.Empty:
            pass
        finally:
            self.root.after(POLL_MS, self._poll)

    def _handle(self, kind, payload):
        if kind == "thinking":
            if self.phase != "thinking":
                self._append("[réflexion] ", "thinking")
                self.phase = "thinking"
            self._append(payload, "thinking")
        elif kind == "content":
            if self.phase == "thinking":
                self._append("\n\n")
            self.phase = "content"
            self.reply_parts.append(payload)
            self._append(payload)
        elif kind == "stats":
            self.last_stats = payload
        elif kind == "end":
            self._finish()
        elif kind == "error":
            self._finish(error=payload)

    def _finish(self, error=None, stopped=False):
        reply = "".join(self.reply_parts)
        self.active_stream = None
        self._set_busy(False)

        if reply:
            self.history.append({"role": "assistant", "content": reply})
        elif self.history and self.history[-1]["role"] == "user":
            # Aucune réponse : on retire la question de l'historique et on la
            # remet dans la zone de saisie pour pouvoir la renvoyer.
            self.history.pop()
            if not self.input.get("1.0", "end-1c").strip():
                self.input.insert("1.0", self.pending_text)

        self._append("\n")
        if error:
            self._append(error + "\n", "error")
            self.status_var.set("Erreur.")
        elif stopped:
            self._append("[arrêté]\n", "meta")
            self.status_var.set("Génération arrêtée.")
        else:
            stats = self.last_stats or {}
            speed = stats.get("tokens_per_second")
            if speed:
                self._append(
                    "{} tokens · {:.1f} tokens/s\n".format(stats.get("tokens", 0), speed), "meta"
                )
            self.status_var.set("Prêt.")
        self._append("\n")
        self.input.focus_set()

    # ------------------------------------------------------ actions du menu
    def new_conversation(self):
        if self.history and not messagebox.askyesno(
            APP_TITLE, "Effacer la conversation en cours ?", parent=self.root
        ):
            return
        self.stop()
        self.history = []
        self._clear_chat()
        self.status_var.set("Nouvelle conversation.")
        self.input.focus_set()

    def save_conversation(self):
        if not self.history:
            self.status_var.set("Rien à enregistrer pour l'instant.")
            return
        default = "conversation-{}.md".format(datetime.now().strftime("%Y%m%d-%H%M"))
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Enregistrer la conversation", initialfile=default,
            defaultextension=".md", filetypes=[("Markdown", "*.md"), ("Texte", "*.txt"), ("Tous", "*.*")],
        )
        if not path:
            return
        lines = ["# Conversation avec {}".format(self.model_var.get() or "Ollama"), ""]
        system_prompt = self.system_var.get().strip()
        if system_prompt:
            lines += ["**Consigne système :** {}".format(system_prompt), ""]
        for message in self.history:
            who = "Vous" if message["role"] == "user" else "Modèle"
            lines += ["**{}**".format(who), "", message["content"], ""]
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
        except OSError as exc:
            messagebox.showerror(APP_TITLE, "Impossible d'enregistrer : {}".format(exc), parent=self.root)
            return
        self.status_var.set("Conversation enregistrée : {}".format(path))

    def _on_close(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
        self.root.destroy()


def main():
    if sys.platform == "win32":
        try:  # texte net sur les écrans haute résolution
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    if sys.platform.startswith("linux"):
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
