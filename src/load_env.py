"""
Utility module for loading local environment variables from .env files.

This is used only for local development. In Codespaces, environment variables
are provided through Codespaces secrets, so this module has no effect there.
"""
from dotenv import load_dotenv
from pathlib import Path

def load_local_env():
    """
    Load environment variables from the local .env/sprout.env file.

    This function loads the .env file only if it exists. It does not
    override existing environment variables (e.g., Codespaces secrets).
    """
    env_path = Path(".env/sprout.env")
    if env_path.exists():
        load_dotenv(env_path)