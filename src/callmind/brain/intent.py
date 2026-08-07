from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("callmind.brain.intent")

INTENT_LABELS = ("faq", "booking", "account_status", "escalation", "smalltalk")

INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a phone support agent. "
    "Read the latest caller utterance (and a few turns of context) and output STRICT JSON only.\n"
    "Schema: {\"intent\": one of "
    "[\"faq\",\"booking\",\"account_status\",\"escalation\",\"smalltalk\"], "
    "\"confidence\": number 0..1}.\n"
    "Use:\n"
    " - faq: questions about products, hours, policies, prices, locations.\n"
    " - booking: requests to schedule, reschedule, or cancel an appointment.\n"
    " - account_status: account balance, order status, delivery tracking.\n"
    " - escalation: explicit human request, frustration, abuse, or out-of-scope.\n"
    " - smalltalk: greetings, thanks, chit-chat, anything else.\n"
    "Confidence = how sure you are. Use 0.3 or lower if unsure."
)


@dataclass(frozen=True)
class Intent:
    label: str
    confidence: float


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class IntentChain:
    def __init__(self, llm) -> None:
        self.llm = llm

    async def classify(self, user_text: str, recent_turns: list[tuple[str, str]] | None = None) -> Intent:
        messages: list[dict[str, str]] = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
        for role, content in (recent_turns or [])[-6:]:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        chunks: list[str] = []
        async for delta in self.llm.stream_chat(messages, max_tokens=80, temperature=0.0):
            chunks.append(delta)
        raw = "".join(chunks).strip()
        obj = _extract_json(raw)
        if not obj:
            log.warning("intent parse failed: %s", raw[:200])
            return Intent(label="smalltalk", confidence=0.9)
        label = str(obj.get("intent", "smalltalk")).strip().lower()
        if label not in INTENT_LABELS:
            label = "smalltalk"
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return Intent(label=label, confidence=max(0.0, min(1.0, conf)))