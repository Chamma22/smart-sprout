"""Load local environment variables from .env files for development.

In Codespaces, environment variables come from Codespaces secrets and this
module has no effect.
"""
from pathlib import Path

from dotenv import load_dotenv


def load_local_env():
    """Load env vars from .env/sprout.env if the file exists, without overriding existing values."""
    env_path = Path(__file__).parent.parent / ".env" / "sprout.env"
    if env_path.exists():
        load_dotenv(env_path)
