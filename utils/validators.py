"""
Validation utilities for the lifestyle analysis project.
Provides data validation and input validation functionality.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from .exceptions import DataValidationError, ValidationError
from .logger import get_logger

logger = get_logger(__name__)

class DataValidator:
    """Validates data integrity and structure."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that a DataFrame has the required columns and data types.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, raises DataValidationError if not
            
        Raises:
            DataValidationError: If validation fails
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame is None or empty")
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for null values in critical columns
        critical_columns = ['age_group', 'lifestyle_impact', 'adjusted_life_expectancy']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                raise DataValidationError(f"Column '{col}' contains null values")
        
        # Check data types
        numeric_columns = ['lifestyle_impact', 'adjusted_life_expectancy', 'years_lost', 'years_gained', 'risk_score']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise DataValidationError(f"Column '{col}' must be numeric")
        
        logger.info(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
        return True
    
    @staticmethod
    def validate_lifestyle_data(df: pd.DataFrame) -> bool:
        """
        Validate lifestyle data specific requirements.
        
        Args:
            df: Lifestyle data DataFrame
            
        Returns:
            True if valid
            
        Raises:
            DataValidationError: If validation fails
        """
        required_columns = [
            'age_group', 'smoking', 'exercise', 'diet', 'alcohol', 
            'sleep', 'stress', 'social_connections', 'lifestyle_impact',
            'adjusted_life_expectancy', 'years_lost', 'years_gained', 'risk_score'
        ]
        
        DataValidator.validate_dataframe(df, required_columns)
        
        # Validate age groups
        valid_age_groups = ['20-24', '25-29', '30-34', '35-39', '40-44', 
                           '45-49', '50-54', '55-59', '60-64', '65-69']
        invalid_age_groups = set(df['age_group'].unique()) - set(valid_age_groups)
        if invalid_age_groups:
            raise DataValidationError(f"Invalid age groups found: {invalid_age_groups}")
        
        # Validate lifestyle factor values
        factor_values = {
            'smoking': ['never', 'former', 'current_light', 'current_heavy'],
            'exercise': ['sedentary', 'light', 'moderate', 'vigorous'],
            'diet': ['poor', 'average', 'good', 'excellent'],
            'alcohol': ['none', 'light', 'moderate', 'heavy'],
            'sleep': ['poor', 'average', 'good', 'excellent'],
            'stress': ['high', 'moderate', 'low', 'minimal'],
            'social_connections': ['isolated', 'limited', 'moderate', 'strong']
        }
        
        for factor, valid_values in factor_values.items():
            if factor in df.columns:
                invalid_values = set(df[factor].unique()) - set(valid_values)
                if invalid_values:
                    raise DataValidationError(f"Invalid {factor} values: {invalid_values}")
        
        # Validate numeric ranges
        if 'risk_score' in df.columns:
            if not ((df['risk_score'] >= 0) & (df['risk_score'] <= 100)).all():
                raise DataValidationError("Risk score must be between 0 and 100")
        
        if 'adjusted_life_expectancy' in df.columns:
            if not (df['adjusted_life_expectancy'] >= 0).all():
                raise DataValidationError("Adjusted life expectancy must be non-negative")
        
        logger.info("Lifestyle data validation passed")
        return True

class InputValidator:
    """Validates user inputs and parameters."""
    
    @staticmethod
    def validate_age_group(age_group: str) -> bool:
        """
        Validate age group input.
        
        Args:
            age_group: Age group string
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        valid_age_groups = ['20-24', '25-29', '30-34', '35-39', '40-44', 
                           '45-49', '50-54', '55-59', '60-64', '65-69']
        
        if age_group not in valid_age_groups:
            raise ValidationError(f"Invalid age group: {age_group}. Must be one of {valid_age_groups}")
        
        return True
    
    @staticmethod
    def validate_lifestyle_choice(factor: str, value: str) -> bool:
        """
        Validate a lifestyle choice value.
        
        Args:
            factor: Lifestyle factor name
            value: Choice value
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        valid_choices = {
            'smoking': ['never', 'former', 'current_light', 'current_heavy'],
            'exercise': ['sedentary', 'light', 'moderate', 'vigorous'],
            'diet': ['poor', 'average', 'good', 'excellent'],
            'alcohol': ['none', 'light', 'moderate', 'heavy'],
            'sleep': ['poor', 'average', 'good', 'excellent'],
            'stress': ['high', 'moderate', 'low', 'minimal'],
            'social_connections': ['isolated', 'limited', 'moderate', 'strong']
        }
        
        if factor not in valid_choices:
            raise ValidationError(f"Unknown lifestyle factor: {factor}")
        
        if value not in valid_choices[factor]:
            raise ValidationError(f"Invalid {factor} value: {value}. Must be one of {valid_choices[factor]}")
        
        return True
    
    @staticmethod
    def validate_user_choices(choices: Dict[str, str]) -> bool:
        """
        Validate a complete set of user lifestyle choices.
        
        Args:
            choices: Dictionary of lifestyle choices
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        required_factors = ['age_group', 'smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']
        
        # Check all required factors are present
        missing_factors = set(required_factors) - set(choices.keys())
        if missing_factors:
            raise ValidationError(f"Missing required lifestyle factors: {missing_factors}")
        
        # Validate each choice
        for factor, value in choices.items():
            if factor == 'age_group':
                InputValidator.validate_age_group(value)
            else:
                InputValidator.validate_lifestyle_choice(factor, value)
        
        logger.info("User choices validation passed")
        return True
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float, name: str) -> bool:
        """
        Validate that a numeric value is within a specified range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name of the parameter for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric")
        
        if not (min_val <= value <= max_val):
            raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
        
        return True

