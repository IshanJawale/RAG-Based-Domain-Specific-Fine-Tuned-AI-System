"""
chat_store.py — tiny JSON-file persistence for per-domain chat history.

Layout: app/chat_history/<domain>.json -> {"chats": [ {id, title, created, updated, messages}, ... ]}
"""

import json
import time
import uuid
from pathlib import Path

HISTORY_DIR = Path(__file__).resolve().parent / "chat_history"


def _path(domain: str) -> Path:
    return HISTORY_DIR / f"{domain}.json"


def load_chats(domain: str) -> list[dict]:
    try:
        data = json.loads(_path(domain).read_text(encoding="utf-8"))
        return data.get("chats", [])
    except Exception:
        return []


def save_chats(domain: str, chats: list[dict]) -> None:
    HISTORY_DIR.mkdir(exist_ok=True)
    tmp = _path(domain).with_suffix(".tmp")
    tmp.write_text(json.dumps({"chats": chats}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_path(domain))


def new_chat() -> dict:
    now = time.time()
    return {"id": uuid.uuid4().hex[:12], "title": "New chat", "created": now, "updated": now, "messages": []}


def make_title(question: str, limit: int = 48) -> str:
    q = " ".join(question.split())
    return q if len(q) <= limit else q[: limit - 1].rstrip() + "…"


def group_label(ts: float) -> str:
    age_days = (time.time() - ts) / 86400
    if age_days < 1:
        return "Today"
    if age_days < 2:
        return "Yesterday"
    if age_days < 8:
        return "Previous 7 days"
    return "Older"
