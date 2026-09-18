"""
Tests for configuration
"""

import config


class TestConfig:
    """Test configuration values"""

    def test_version_info(self):
        """Test version information exists and has correct format"""
        # Check version format (e.g., "2.4.1")
        assert isinstance(config.__version__, str)
        assert len(config.__version__.split(".")) == 3  # Major.Minor.Patch

        # Check author
        assert config.__author__ == "Fahad Gilani"

        # Check updated date exists (don't assert on specific format/value)
        assert isinstance(config.__updated__, str)

    def test_model_config(self):
        """Test model configuration"""
        # DEFAULT_MODEL is set in conftest.py for tests. Read the module
        # attribute (not a name captured at import time) so this reflects
        # whatever the autouse ``_runtime_env`` fixture's importlib.reload
        # last set, regardless of a local .env's DEFAULT_MODEL (issue #138).
        assert config.DEFAULT_MODEL == "gemini-2.5-flash"

    def test_temperature_defaults(self):
        """Test temperature constants"""
        assert config.TEMPERATURE_ANALYTICAL == 1.0
        assert config.TEMPERATURE_BALANCED == 1.0
        assert config.TEMPERATURE_CREATIVE == 1.0
