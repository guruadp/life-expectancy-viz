"""
Example configuration file for the lifestyle analysis project.
Copy this file to config.py and modify as needed.
"""

# Application Configuration
DEBUG = False
HOST = "localhost"
PORT = 8000
APP_TITLE = "How Lifestyle Choices Steal Your Years"
APP_DESCRIPTION = "Interactive Analysis of Lifestyle Impact on Life Expectancy"

# Database Configuration
DATA_PATH = "data"
SAMPLE_SIZE = 1000
RANDOM_STATE = 42

# Visualization Configuration
OUTPUT_PATH = "exports"
DPI = 300
FIGURE_SIZE = (16, 12)

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/app.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

