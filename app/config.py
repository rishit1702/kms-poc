from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    LLM_PROVIDER: Literal["gemini", "anthropic", "ollama", "openrouter"] = "openrouter"

    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    GEMINI_MODEL: str = "gemini-2.0-flash"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"

    CHROMA_PATH: str = "./data/chroma"
    UPLOAD_PATH: str = "./uploads"

    class Config:
        env_file = ".env"


settings = Settings()
