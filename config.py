"""
Configuration management for the lifestyle analysis project.
Handles environment variables, settings, and configuration validation.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    data_path: str = "data"
    sample_size: int = 1000
    random_state: int = 42

@dataclass
class VisualizationConfig:
    """Visualization configuration settings."""
    output_path: str = "exports"
    dpi: int = 300
    figure_size: tuple = (16, 12)
    color_scheme: Dict[str, str] = None
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'accent': '#F18F01',
                'success': '#C73E1D',
                'text': '#2C3E50',
                'background': '#F8F9FA'
            }

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    host: str = "localhost"
    port: int = 8000
    title: str = "How Lifestyle Choices Steal Your Years"
    description: str = "Interactive Analysis of Lifestyle Impact on Life Expectancy"
    
    # Sub-configurations
    database: DatabaseConfig = None
    visualization: VisualizationConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

class ConfigManager:
    """Manages application configuration with environment variable support."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from environment variables and config file."""
        # Load from environment variables
        config_data = {
            'debug': self._get_bool_env('DEBUG', False),
            'host': os.getenv('HOST', 'localhost'),
            'port': int(os.getenv('PORT', '8000')),
            'title': os.getenv('APP_TITLE', 'How Lifestyle Choices Steal Your Years'),
            'description': os.getenv('APP_DESCRIPTION', 'Interactive Analysis of Lifestyle Impact on Life Expectancy'),
        }
        
        # Database configuration
        database_config = DatabaseConfig(
            data_path=os.getenv('DATA_PATH', 'data'),
            sample_size=int(os.getenv('SAMPLE_SIZE', '1000')),
            random_state=int(os.getenv('RANDOM_STATE', '42'))
        )
        
        # Visualization configuration
        visualization_config = VisualizationConfig(
            output_path=os.getenv('OUTPUT_PATH', 'exports'),
            dpi=int(os.getenv('DPI', '300')),
            figure_size=eval(os.getenv('FIGURE_SIZE', '(16, 12)')),
        )
        
        # Logging configuration
        logging_config = LoggingConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            file_path=os.getenv('LOG_FILE', 'logs/app.log'),
            max_file_size=int(os.getenv('LOG_MAX_SIZE', str(10 * 1024 * 1024))),
            backup_count=int(os.getenv('LOG_BACKUP_COUNT', '5'))
        )
        
        self._config = AppConfig(
            debug=config_data['debug'],
            host=config_data['host'],
            port=config_data['port'],
            title=config_data['title'],
            description=config_data['description'],
            database=database_config,
            visualization=visualization_config,
            logging=logging_config
        )
    
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        return self._config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self._config.database
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self._config.visualization
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self._config.logging
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        try:
            # Validate paths exist or can be created
            Path(self._config.database.data_path).mkdir(parents=True, exist_ok=True)
            Path(self._config.visualization.output_path).mkdir(parents=True, exist_ok=True)
            
            if self._config.logging.file_path:
                Path(self._config.logging.file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Validate numeric values
            assert self._config.port > 0 and self._config.port < 65536, "Port must be between 1 and 65535"
            assert self._config.database.sample_size > 0, "Sample size must be positive"
            assert self._config.visualization.dpi > 0, "DPI must be positive"
            
            return True
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
config_manager = ConfigManager()
config = config_manager.config

