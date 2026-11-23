"""Gửi tin Telegram. Biến môi trường: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID."""

from __future__ import annotations

import os

import requests

TOKEN_ENV, CHAT_ENV = "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"
MAX_LEN = 4000


def send(text: str, chat_id: str | None = None, token: str | None = None) -> None:
    token = token or os.environ.get(TOKEN_ENV, "").strip()
    chat_id = chat_id or os.environ.get(CHAT_ENV, "").strip()
    if not token or not chat_id:
        raise RuntimeError(f"Thiếu {TOKEN_ENV} hoặc {CHAT_ENV}")
    for i in range(0, len(text), MAX_LEN):
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[i : i + MAX_LEN], "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=20,
        )  # fmt: skip
        if resp.status_code != 200:
            raise RuntimeError(f"Telegram lỗi {resp.status_code}: {resp.text[:200]}")
