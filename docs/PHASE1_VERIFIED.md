# Phase 1 — Live Verified

Date: 2026-08-07
Telnyx number: +1 437 900 8438
Result: end-to-end voice conversation with barge-in working.

## Bugs found and fixed during live test

| # | Component | Symptom | Fix |
|---|---|---|---|
| 1 | `stt/engine.py` | faster-whisper crashed with `'' is not a valid language code` | Coerce empty string to `None` (autodetect) |
| 2 | `.env` `CALLMIND_TTS_MODEL` | TTS rejected `speech-2.8-flash` (model doesn't exist on this account) | Use `speech-2.8-turbo` (probed against the account; works) |
| 3 | `tts/minimax.py` parser | TTS HTTP response was SSE `data: {json}\n` but parser called `json.loads` on the whole line | Strip `data:` SSE prefix before parsing |

## Verified config (this account)

| Setting | Value | Notes |
|---|---|---|
| LLM model | `MiniMax-Text-01` | `abab6.5-chat` and friends NOT supported on this endpoint |
| TTS model | `speech-2.8-turbo` | also confirmed: `speech-2.8-turbo`, `speech-2.6-{hd,turbo}`, `speech-2.5-{hd,turbo}-preview`, `speech-02-hd`, `speech-01-hd` |
| TTS voice_id | `female-shaonv` | placeholder; replace with one from MiniMax voice list |
| Telnyx account | Programmable Voice, app `CallMind` | webhook = `https://magazine-setup-robust.ngrok-free.dev/telnyx/webhook` |

## Telnyx streaming setup (verified)

- Inbound call → webhook → gateway POSTs `answer` with:
  - `stream_url = wss://.../ws/call`
  - `stream_track = inbound_track`
  - `stream_bidirectional_mode = rtp`
  - `stream_bidirectional_codec = PCMU`
- Telnyx opens WebSocket → sends `start`, `media` (base64 PCMU), `stop`
- Gateway sends back `media` (base64 PCMU) and `clear` for barge-in

## Latency (rough, single machine)

- STT (3s audio, GPU small/int8_float16): ~340 ms steady
- TTS streaming first chunk: ~300-500 ms (MiniMax)
- Total perceived turn latency: under 1 s (call felt natural)