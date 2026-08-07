from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CALLMIND_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000

    telephony_provider: str = "telnyx"
    telnyx_api_key: str = ""
    telnyx_api_base: str = "https://api.telnyx.com"
    telnyx_webhook_secret: str = ""
    public_ws_url: str = ""

    stt_model_size: str = "small"
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    stt_language: str | None = None

    vad_energy_dbfs: float = -45.0
    vad_start_frames: int = 3
    vad_end_frames: int = 25
    vad_min_speech_ms: int = 200
    vad_preroll_frames: int = 15

    llm_api_key: str = ""
    llm_base_url: str = "https://api.minimax.io/v1"
    llm_endpoint: str = "/text/chatcompletion_v2"
    llm_model: str = "MiniMax-Text-01"
    llm_max_tokens: int = 300
    llm_temperature: float = 0.7
    memory_window: int = 8

    embedding_model: str = "embo-01"
    embedding_endpoint: str = "/v1/embeddings"
    embedding_type: str = "db"
    embedding_dim: int = 1536
    embedding_batch_size: int = 32

    tts_api_key: str = ""
    tts_base_url: str = "https://api.minimax.io/v1"
    tts_endpoint: str = "/t2a_v2"
    tts_model: str = "speech-2.8-turbo"
    tts_voice_id: str = ""
    tts_sample_rate: int = 24000
    tts_speed: float = 1.0
    tts_volume: float = 1.0
    tts_pitch: int = 0

    greeting: str = "Hi, thanks for calling. How can I help you today?"

    memory_db_path: str = "callmind.db"
    kb_dir: str = "kb"
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.2

    business_id: str = "default"

    escalation_confidence_threshold: float = 0.55


@lru_cache
def get_settings() -> Settings:
    return Settings()
