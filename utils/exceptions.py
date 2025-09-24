"""
Custom exceptions for the lifestyle analysis project.
"""

class LifestyleAnalysisError(Exception):
    """Base exception for all lifestyle analysis errors."""
    pass

class DataValidationError(LifestyleAnalysisError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(LifestyleAnalysisError):
    """Raised when configuration is invalid."""
    pass

class VisualizationError(LifestyleAnalysisError):
    """Raised when visualization generation fails."""
    pass

class DataGenerationError(LifestyleAnalysisError):
    """Raised when data generation fails."""
    pass

class FileOperationError(LifestyleAnalysisError):
    """Raised when file operations fail."""
    pass

class ValidationError(LifestyleAnalysisError):
    """Raised when input validation fails."""
    pass

