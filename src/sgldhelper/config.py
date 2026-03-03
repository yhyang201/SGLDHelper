"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # GitHub
    github_token: str = ""
    github_repo: str = "sgl-project/sglang"

    # Slack
    slack_bot_token: str
    slack_app_token: str
    slack_pr_channel: str
    slack_ci_channel: str
    slack_features_channel: str

    # Polling intervals (seconds)
    pr_poll_interval: int = 60
    ci_poll_interval: int = 120
    feature_poll_interval: int = 3600

    # Auto-rerun
    max_auto_reruns: int = 2
    auto_rerun_enabled: bool = True

    # Roadmap issues (comma-separated)
    roadmap_issues: str = "14199"

    # Database
    db_path: str = "data/sgldhelper.db"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # AI / Kimi K2.5
    ai_enabled: bool = False
    moonshot_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k2.5"

    # AI rate limits
    ai_rate_limit_rpm: int = 10
    ai_user_cooldown_seconds: int = 60
    ai_user_cooldown_max: int = 5

    # Stall detection
    stall_days_threshold: int = 3
    review_nudge_days: int = 2
    stall_check_interval: int = 43200  # 12 hours

    # Diffusion file path prefixes — only multimodal_gen, NOT diffusion-llm
    diffusion_paths: list[str] = Field(
        default=[
            "python/sglang/srt/models/multimodal_gen/",
            "test/srt/models/multimodal_gen/",
        ]
    )

    # CI job name patterns for diffusion tests
    diffusion_ci_jobs: list[str] = Field(
        default=[
            "multimodal-gen-test-1-gpu",
            "multimodal-gen-test-2-gpu",
        ]
    )

    @property
    def github_configured(self) -> bool:
        return bool(self.github_token)

    @property
    def github_owner(self) -> str:
        return self.github_repo.split("/")[0]

    @property
    def github_repo_name(self) -> str:
        return self.github_repo.split("/")[1]

    @property
    def roadmap_issue_numbers(self) -> list[int]:
        return [int(n.strip()) for n in self.roadmap_issues.split(",") if n.strip()]
