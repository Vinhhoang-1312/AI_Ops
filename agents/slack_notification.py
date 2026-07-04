# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Slack notification helpers for escalation workflows."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib import error, request


DEFAULT_SLACK_API_URL = "https://slack.com/api/chat.postMessage"
DEFAULT_TIMEOUT_SECONDS = 15


class SlackNotificationError(RuntimeError):
    """Raised when a Slack notification cannot be delivered."""


def slack_notifications_enabled() -> bool:
    """Return whether Slack notifications should be attempted."""

    raw = (os.getenv("IT_SUPPORT_SLACK_NOTIFICATIONS") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def send_slack_message(
    message: str,
    *,
    channel: Optional[str] = None,
    bot_token: Optional[str] = None,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    """Send a plain-text Slack message to a channel via `chat.postMessage`."""

    resolved_token = (bot_token or os.getenv("SLACK_BOT_TOKEN") or "").strip()
    resolved_channel = (
        channel
        or os.getenv("SLACK_ESCALATION_CHANNEL")
        or os.getenv("SLACK_DEFAULT_CHANNEL")
        or ""
    ).strip()
    resolved_timeout = timeout or _timeout_seconds()

    if not resolved_token:
        raise SlackNotificationError("SLACK_BOT_TOKEN is not set")
    if not resolved_channel:
        raise SlackNotificationError("No Slack channel was provided")
    if not message.strip():
        raise SlackNotificationError("Slack message is empty")

    payload = {
        "channel": resolved_channel,
        "text": message,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {resolved_token}",
    }
    req = request.Request(
        (os.getenv("SLACK_API_URL") or DEFAULT_SLACK_API_URL).strip(),
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=resolved_timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SlackNotificationError(f"Slack API request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise SlackNotificationError(f"Slack API request failed: {exc.reason}") from exc

    try:
        payload = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError as exc:
        raise SlackNotificationError("Slack API returned a non-JSON response") from exc

    if payload.get("ok") is not True:
        raise SlackNotificationError(f"Slack API returned an error: {payload.get('error', 'unknown_error')}")
    return payload


def _timeout_seconds() -> int:
    raw = (os.getenv("SLACK_TIMEOUT_SECONDS") or str(DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
