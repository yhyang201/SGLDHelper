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

    # Polling intervals (seconds)
    pr_poll_interval: int = 60

    # Database
    db_path: str = "data/sgldhelper.db"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # AI / Kimi K2.5
    moonshot_api_key: str
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k2.5"

    # AI rate limits
    ai_rate_limit_rpm: int = 10
    ai_user_cooldown_seconds: int = 60
    ai_user_cooldown_max: int = 5

    # Cold start: max PRs to fetch on first run (when classification cache is empty)
    cold_start_max_prs: int = 500

    # Diffusion PR summary interval (seconds), default 2 hours
    diffusion_summary_interval: int = 7200

    # CI monitoring
    ci_poll_interval: int = 300
    ci_max_retries: int = 3
    ci_high_priority_max_retries: int = 10
    ci_high_priority_label: str = "high-priority"
    ci_high_priority_ping_user: str = "mickqian"
    ci_owner_rerun_max_retries: int = 5
    ci_approve_auto_ci_users: list[str] = Field(
        default=["mickqian", "bbuf"],
    )
    ci_approve_auto_ci_max_retries: int = 2
    ci_nvidia_workflow_id: int = 115218617
    ci_amd_workflow_id: int = 119055250

    # Auto merge
    auto_merge_enabled: bool = True
    auto_merge_delay_seconds: int = 300
    auto_merge_cancel_keywords: list[str] = Field(
        default=[
            "不要merge", "取消merge", "别merge", "停止merge",
            "cancel merge", "stop merge", "don't merge", "abort merge",
            "取消", "cancel",
        ]
    )

    # Tracked PR summary interval (seconds), default 12 hours
    tracked_pr_summary_interval: int = 43200

    # PR health check interval (seconds), default 2 hours
    pr_health_check_interval: int = 7200

    # Code quality report poll interval (seconds), default 1 hour
    # The report runs at most once per day; this is how often we check if it's time
    code_quality_poll_interval: int = 3600

    # Diffusion file path prefixes — only multimodal_gen, NOT diffusion-llm
    diffusion_paths: list[str] = Field(
        default=[
            "python/sglang/srt/models/multimodal_gen/",
            "python/sglang/multimodal_gen/",
            "test/srt/models/multimodal_gen/",
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
