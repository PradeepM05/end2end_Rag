import os
from dotenv import load_dotenv
from pydantic import Field
from typing import Optional

# For Pydantic v2, we need to use BaseSettings from pydantic-settings
try:
    # Try to import from pydantic-settings (for v2)
    from pydantic_settings import BaseSettings
except ImportError:
    # Fall back to pydantic directly (for v1)
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # OpenAI settings
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1000
    
    # Retrieval settings
    USE_COMPRESSION: bool = True
    RETRIEVAL_TOP_K: int = 5
    
    # Application settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = "config/production.env"
        env_file_encoding = "utf-8"

def get_settings():
    # First load from .env file
    load_dotenv()
    
    # Then create settings object (will override with env vars)
    return Settings()