"""
config.py - Configuration Management Module

This module handles loading and validating environment variables from .env file.
It provides a centralized configuration class for database and API credentials.

Usage:
    from config import Config

    config = Config()
    if config.validate():
        print(f"Database: {config.pg_database}")
"""

import os
import logging
from dotenv import load_dotenv


class Config:
    """
    Configuration manager that loads environment variables from .env file.

    This class is responsible for:
    - Loading environment variables from .env file
    - Providing access to database credentials
    - Providing access to API keys
    - Validating that all required settings are present

    Attributes:
        pg_host: PostgreSQL host address
        pg_port: PostgreSQL port number
        pg_database: Database name
        pg_user: Database username
        pg_password: Database password
        gemini_api_key: Google Gemini API key for embeddings
    """

    def __init__(self, env_path: str = None):
        """
        Initialize configuration by loading .env file.

        Args:
            env_path: Optional path to .env file. If not provided,
                     looks for .env in the same directory as this module.
        """
        # Determine .env file path
        if env_path is None:
            env_path = os.path.join(os.path.dirname(__file__), '.env')

        # Load environment variables from .env file
        load_dotenv(env_path)

        # PostgreSQL configuration
        self.pg_host = os.getenv('PG_HOST', 'localhost')
        self.pg_port = os.getenv('PG_PORT', '5433')
        self.pg_database = os.getenv('PG_DATABASE', 'jeen_db')
        self.pg_user = os.getenv('PG_USER', 'admin')
        self.pg_password = os.getenv('PG_PASSWORD', '')

        # Gemini API configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')

        # Logger for this module
        self.logger = logging.getLogger(__name__)

    def validate(self) -> bool:
        """
        Validate that all required configuration values are present.

        Checks that all required environment variables have been set.
        Logs an error message listing any missing variables.

        Returns:
            True if all required values are set, False otherwise
        """
        required = [
            ('PG_HOST', self.pg_host),
            ('PG_PORT', self.pg_port),
            ('PG_DATABASE', self.pg_database),
            ('PG_USER', self.pg_user),
            ('PG_PASSWORD', self.pg_password),
            ('GEMINI_API_KEY', self.gemini_api_key)
        ]

        missing = [name for name, value in required if not value]

        if missing:
            self.logger.error(f"Missing required configuration: {', '.join(missing)}")
            return False

        self.logger.debug("Configuration validated successfully")
        return True

    def get_db_connection_params(self) -> dict:
        """
        Get database connection parameters as a dictionary.

        Returns:
            Dictionary with host, port, database, user, password keys
        """
        return {
            'host': self.pg_host,
            'port': self.pg_port,
            'database': self.pg_database,
            'user': self.pg_user,
            'password': self.pg_password
        }

    def __repr__(self) -> str:
        """String representation of config (hides sensitive data)."""
        return (
            f"Config(pg_host='{self.pg_host}', pg_port='{self.pg_port}', "
            f"pg_database='{self.pg_database}', pg_user='{self.pg_user}', "
            f"pg_password='****', gemini_api_key='****')"
        )
