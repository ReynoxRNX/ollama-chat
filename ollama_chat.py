#!/usr/bin/env python3
"""Local chat with Ollama: a Tkinter interface with no external dependencies.

Run it with:  python ollama_chat.py
"""

import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from ollama_client import OllamaClient, OllamaError

APP_TITLE = "Ollama Chat"
POLL_MS = 30
MODELS_EVENT = "models"  # identifier of "model list" events

# Fixed colors (light background) so the text stays readable with any system theme.
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

        self.history = []  # {"role", "content"} messages sent to Ollama
        self.events = queue.Queue()  # worker thread -> interface
        self.active_stream = None  # id of the generation in progress
        self.stream_counter = 0
        self.cancel_event = None

        self.reply_parts = []  # chunks of the reply being generated
        self.phase = None  # None, "thinking" or "content"
        self.last_stats = None
        self.pending_text = ""  # last message sent (restored if the request fails)

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

        # Row 1: model, temperature, actions
        bar = ttk.Frame(outer)
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(bar, text="Model").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(bar, textvariable=self.model_var, state="readonly", width=30)
        self.model_box.pack(side="left", padx=(6, 4))
        ttk.Button(bar, text="Refresh", command=self.refresh_models).pack(side="left")

        ttk.Label(bar, text="Temperature").pack(side="left", padx=(16, 4))
        self.temp_var = tk.StringVar(value="0.7")
        self.temp_box = ttk.Spinbox(
            bar, from_=0.0, to=2.0, increment=0.1, width=5, format="%.1f", textvariable=self.temp_var
        )
        self.temp_box.pack(side="left")

        ttk.Button(bar, text="Save…", command=self.save_conversation).pack(side="right")
        ttk.Button(bar, text="New conversation", command=self.new_conversation).pack(
            side="right", padx=(0, 6)
        )

        # Row 2: system prompt
        sys_row = ttk.Frame(outer)
        sys_row.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        sys_row.columnconfigure(1, weight=1)
        ttk.Label(sys_row, text="System prompt").grid(row=0, column=0, padx=(0, 6))
        self.system_var = tk.StringVar()
        ttk.Entry(sys_row, textvariable=self.system_var).grid(row=0, column=1, sticky="ew")

        # Conversation area
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

        # Input area
        entry_row = ttk.Frame(outer)
        entry_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        entry_row.columnconfigure(0, weight=1)
        self.input = tk.Text(
            entry_row, height=3, wrap="word", font=self.font, bg=BG, fg=FG, insertbackground=FG,
            padx=8, pady=6, relief="flat", highlightthickness=1,
            highlightbackground="#d0d7de", highlightcolor=USER_COLOR,
        )
        self.input.grid(row=0, column=0, sticky="ew")
        self.send_button = ttk.Button(entry_row, text="Send", command=self.send, width=10)
        self.send_button.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.input.bind("<Return>", self._on_return)
        self.root.bind("<Escape>", lambda _e: self.stop())
        self.input.focus_set()

        # Status bar
        self.status_var = tk.StringVar(value="Connecting to Ollama…")
        ttk.Label(outer, textvariable=self.status_var, foreground=MUTED).grid(
            row=4, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Label(
            outer, text="Enter: send · Shift+Enter: new line · Esc: stop", foreground=MUTED,
        ).grid(row=5, column=0, sticky="ew")

    def _on_return(self, event):
        if event.state & 0x0001:  # Shift held: insert a normal newline
            return None
        self.send()
        return "break"

    # ------------------------------------------------------------ chat area
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
            self.send_button.configure(text="Stop", command=self.stop)
            self.model_box.configure(state="disabled")
        else:
            self.send_button.configure(text="Send", command=self.send)
            self.model_box.configure(state="readonly")

    # --------------------------------------------------------------- models
    def refresh_models(self):
        self.status_var.set("Looking for installed models…")

        def work():
            try:
                self.events.put((MODELS_EVENT, "ok", self.client.list_models()))
            except OllamaError as exc:
                self.events.put((MODELS_EVENT, "error", str(exc)))
            except Exception as exc:
                self.events.put((MODELS_EVENT, "error", "Unexpected error: {}".format(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _on_models(self, kind, payload):
        if kind == "error":
            self.status_var.set(payload + "  Then click \"Refresh\".")
            return
        models = payload
        self.model_box.configure(values=models)
        if not models:
            self.model_var.set("")
            self.status_var.set(
                "No model installed. In a terminal, run for example: ollama pull qwen3:8b"
            )
            return
        if self.model_var.get() not in models:
            self.model_var.set(models[0])
        self.status_var.set("Connected to {} · {} model(s) installed".format(self.client.host, len(models)))

    # ----------------------------------------------------------------- send
    def send(self):
        if self.active_stream is not None:
            return
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            return
        model = self.model_var.get().strip()
        if not model:
            self.status_var.set("Pick a model first (click \"Refresh\" if the list is empty).")
            return

        self.input.delete("1.0", "end")
        self.pending_text = text
        self.history.append({"role": "user", "content": text})
        self._append("You\n", "role_user")
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
        self.status_var.set("Generating…")

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
            self.events.put((stream_id, "error", "Unexpected error: {}".format(exc)))

    def stop(self):
        if self.active_stream is None:
            return
        self.cancel_event.set()
        self._finish(stopped=True)

    # ------------------------------------------------------ worker events
    def _poll(self):
        try:
            while True:
                stream_id, kind, payload = self.events.get_nowait()
                if stream_id == MODELS_EVENT:
                    self._on_models(kind, payload)
                elif stream_id == self.active_stream:
                    self._handle(kind, payload)
                # otherwise: event from a generation that was already stopped, ignore it
        except queue.Empty:
            pass
        finally:
            self.root.after(POLL_MS, self._poll)

    def _handle(self, kind, payload):
        if kind == "thinking":
            if self.phase != "thinking":
                self._append("[thinking] ", "thinking")
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
            # No reply: remove the question from the history and put it back
            # in the input box so it can be sent again.
            self.history.pop()
            if not self.input.get("1.0", "end-1c").strip():
                self.input.insert("1.0", self.pending_text)

        self._append("\n")
        if error:
            self._append(error + "\n", "error")
            self.status_var.set("Error.")
        elif stopped:
            self._append("[stopped]\n", "meta")
            self.status_var.set("Generation stopped.")
        else:
            stats = self.last_stats or {}
            speed = stats.get("tokens_per_second")
            if speed:
                self._append(
                    "{} tokens · {:.1f} tokens/s\n".format(stats.get("tokens", 0), speed), "meta"
                )
            self.status_var.set("Ready.")
        self._append("\n")
        self.input.focus_set()

    # ------------------------------------------------------------- actions
    def new_conversation(self):
        if self.history and not messagebox.askyesno(
            APP_TITLE, "Clear the current conversation?", parent=self.root
        ):
            return
        self.stop()
        self.history = []
        self._clear_chat()
        self.status_var.set("New conversation.")
        self.input.focus_set()

    def save_conversation(self):
        if not self.history:
            self.status_var.set("Nothing to save yet.")
            return
        default = "conversation-{}.md".format(datetime.now().strftime("%Y%m%d-%H%M"))
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save conversation", initialfile=default,
            defaultextension=".md", filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        lines = ["# Conversation with {}".format(self.model_var.get() or "Ollama"), ""]
        system_prompt = self.system_var.get().strip()
        if system_prompt:
            lines += ["**System prompt:** {}".format(system_prompt), ""]
        for message in self.history:
            who = "You" if message["role"] == "user" else "Model"
            lines += ["**{}**".format(who), "", message["content"], ""]
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
        except OSError as exc:
            messagebox.showerror(APP_TITLE, "Could not save the file: {}".format(exc), parent=self.root)
            return
        self.status_var.set("Conversation saved: {}".format(path))

    def _on_close(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
        self.root.destroy()


def main():
    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens
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
