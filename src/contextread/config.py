from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CACHE_ROOT = Path.home() / ".cache" / "contextread"


class Settings(BaseSettings):
    """Process settings, read from the environment and from a .env file in the working directory
    (`contextread init` writes one to fill in). The environment wins over the file."""

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")

    database_url: str = Field(alias="APP_PG_DSN")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_budget_usd: float = Field(default=10.0, alias="LLM_BUDGET_USD")
    llm_default_model: str = Field(default="gpt-4o-mini", alias="LLM_DEFAULT_MODEL")
    # a call cut off after this long is retried (twice): the gateway can hold a stalled call for 5 minutes
    llm_timeout_seconds: float = Field(default=90.0, alias="LLM_TIMEOUT_SECONDS")
    # the chat agent's model (indexing calls use llm_default_model): on the legal set gpt-4o-mini answered
    # 47 of the 56 questions Luna answered on the same documents, and took more steps to do it
    chat_model: str = Field(default="gpt-6-luna", alias="CHAT_MODEL")
    llm_cache_enabled: bool = Field(default=True, alias="LLM_CACHE")
    llm_cache_dir: Path = Field(default=CACHE_ROOT / "llm", alias="LLM_CACHE_DIR")
    deepinfra_api_key: str | None = Field(default=None, alias="deep_infra")
    deepinfra_base_url: str = Field(default="https://api.deepinfra.com/v1/openai", alias="DEEPINFRA_BASE_URL")
    deepinfra_budget_usd: float = Field(default=5.0, alias="DEEPINFRA_BUDGET_USD")
    # pages without a usable text layer are read by a vision model ("off" uses Tesseract only)
    ocr_vlm_model: str = Field(default="gpt-6-luna", alias="OCR_VLM_MODEL")
    ocr_vlm_provider: str = Field(default="openai", alias="OCR_VLM_PROVIDER")
    ocr_vlm_concurrency: int = Field(default=8, alias="OCR_VLM_CONCURRENCY")
    # flash's expand step: a model splits sections over 5 pages that the layout left whole (headings set
    # like body text, as in many contracts and reports); "off" keeps the layout-only tree
    tree_expand_model: str = Field(default="gpt-6-luna", alias="TREE_EXPAND_MODEL")
    # plan limits per user; 0 = no limit (the default). The reference free plan is 1,000 pages, 100 chats
    page_limit: int = Field(default=0, alias="PAGE_LIMIT")
    chat_limit: int = Field(default=0, alias="CHAT_LIMIT")

    internal_api_token: str | None = Field(default=None, alias="INTERNAL_API_TOKEN")
    web_url: str = Field(default="http://127.0.0.1:3000", alias="WEB_URL")
    public_api_url: str = Field(default="http://127.0.0.1:8000", alias="PUBLIC_API_URL")
    smtp_url: str | None = Field(default=None, alias="SMTP_URL")  # smtp(s)://user:pass@host:port
    mail_from: str = Field(default="ContextRead <no-reply@contextread.local>", alias="MAIL_FROM")
    dev_mail_dir: Path = Field(default=CACHE_ROOT / "dev-mail", alias="DEV_MAIL_DIR")

    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_access_key_id: str | None = Field(default=None, alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = Field(default=None, alias="S3_SECRET_ACCESS_KEY")
    s3_bucket: str = Field(default="contextread-documents", alias="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://") :]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
