from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    elevenlabs_api_key: str = ""
    anthropic_api_key: str = ""
    api_key: str = "dev-key"
    host: str = "0.0.0.0"
    port: int = 8000
    base_url: str = "http://localhost:8000"
    assets_dir: str = "assets"
    # LLM provider for creative director: "gemini" or "claude"
    creative_llm: str = "gemini"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
