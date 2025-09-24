"""
Tests for the validators module.
"""

import pytest
import pandas as pd
import numpy as np
from utils.validators import DataValidator, InputValidator
from utils.exceptions import DataValidationError, ValidationError


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_validate_dataframe_success(self):
        """Test successful DataFrame validation."""
        df = pd.DataFrame({
            'age_group': ['20-24', '25-29'],
            'lifestyle_impact': [1.0, -2.0],
            'adjusted_life_expectancy': [50.0, 45.0]
        })
        
        required_columns = ['age_group', 'lifestyle_impact', 'adjusted_life_expectancy']
        result = DataValidator.validate_dataframe(df, required_columns)
        assert result is True
    
    def test_validate_dataframe_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="DataFrame is None or empty"):
            DataValidator.validate_dataframe(df, ['test'])
    
    def test_validate_dataframe_none_dataframe(self):
        """Test validation with None DataFrame."""
        with pytest.raises(DataValidationError, match="DataFrame is None or empty"):
            DataValidator.validate_dataframe(None, ['test'])
    
    def test_validate_dataframe_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({'age_group': ['20-24']})
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            DataValidator.validate_dataframe(df, ['age_group', 'missing_col'])
    
    def test_validate_dataframe_null_values(self):
        """Test validation with null values in critical columns."""
        df = pd.DataFrame({
            'age_group': ['20-24', None],
            'lifestyle_impact': [1.0, -2.0],
            'adjusted_life_expectancy': [50.0, 45.0]
        })
        
        with pytest.raises(DataValidationError, match="contains null values"):
            DataValidator.validate_dataframe(df, ['age_group', 'lifestyle_impact', 'adjusted_life_expectancy'])
    
    def test_validate_dataframe_invalid_numeric_type(self):
        """Test validation with invalid numeric data types."""
        df = pd.DataFrame({
            'age_group': ['20-24', '25-29'],
            'lifestyle_impact': ['not_numeric', 'also_not_numeric'],
            'adjusted_life_expectancy': [50.0, 45.0]
        })
        
        with pytest.raises(DataValidationError, match="must be numeric"):
            DataValidator.validate_dataframe(df, ['age_group', 'lifestyle_impact', 'adjusted_life_expectancy'])
    
    def test_validate_lifestyle_data_success(self):
        """Test successful lifestyle data validation."""
        df = pd.DataFrame({
            'age_group': ['20-24', '25-29'],
            'smoking': ['never', 'former'],
            'exercise': ['moderate', 'vigorous'],
            'diet': ['good', 'excellent'],
            'alcohol': ['light', 'moderate'],
            'sleep': ['good', 'excellent'],
            'stress': ['low', 'minimal'],
            'social_connections': ['moderate', 'strong'],
            'lifestyle_impact': [1.0, -2.0],
            'adjusted_life_expectancy': [50.0, 45.0],
            'years_lost': [0.0, 2.0],
            'years_gained': [1.0, 0.0],
            'risk_score': [30.0, 70.0]
        })
        
        result = DataValidator.validate_lifestyle_data(df)
        assert result is True
    
    def test_validate_lifestyle_data_invalid_age_groups(self):
        """Test validation with invalid age groups."""
        df = pd.DataFrame({
            'age_group': ['invalid_age'],
            'smoking': ['never'],
            'exercise': ['moderate'],
            'diet': ['good'],
            'alcohol': ['light'],
            'sleep': ['good'],
            'stress': ['low'],
            'social_connections': ['moderate'],
            'lifestyle_impact': [1.0],
            'adjusted_life_expectancy': [50.0],
            'years_lost': [0.0],
            'years_gained': [1.0],
            'risk_score': [30.0]
        })
        
        with pytest.raises(DataValidationError, match="Invalid age groups found"):
            DataValidator.validate_lifestyle_data(df)
    
    def test_validate_lifestyle_data_invalid_factor_values(self):
        """Test validation with invalid lifestyle factor values."""
        df = pd.DataFrame({
            'age_group': ['20-24'],
            'smoking': ['invalid_smoking'],
            'exercise': ['moderate'],
            'diet': ['good'],
            'alcohol': ['light'],
            'sleep': ['good'],
            'stress': ['low'],
            'social_connections': ['moderate'],
            'lifestyle_impact': [1.0],
            'adjusted_life_expectancy': [50.0],
            'years_lost': [0.0],
            'years_gained': [1.0],
            'risk_score': [30.0]
        })
        
        with pytest.raises(DataValidationError, match="Invalid smoking values"):
            DataValidator.validate_lifestyle_data(df)
    
    def test_validate_lifestyle_data_invalid_risk_score(self):
        """Test validation with invalid risk score range."""
        df = pd.DataFrame({
            'age_group': ['20-24'],
            'smoking': ['never'],
            'exercise': ['moderate'],
            'diet': ['good'],
            'alcohol': ['light'],
            'sleep': ['good'],
            'stress': ['low'],
            'social_connections': ['moderate'],
            'lifestyle_impact': [1.0],
            'adjusted_life_expectancy': [50.0],
            'years_lost': [0.0],
            'years_gained': [1.0],
            'risk_score': [150.0]  # Invalid: > 100
        })
        
        with pytest.raises(DataValidationError, match="Risk score must be between 0 and 100"):
            DataValidator.validate_lifestyle_data(df)
    
    def test_validate_lifestyle_data_negative_life_expectancy(self):
        """Test validation with negative adjusted life expectancy."""
        df = pd.DataFrame({
            'age_group': ['20-24'],
            'smoking': ['never'],
            'exercise': ['moderate'],
            'diet': ['good'],
            'alcohol': ['light'],
            'sleep': ['good'],
            'stress': ['low'],
            'social_connections': ['moderate'],
            'lifestyle_impact': [1.0],
            'adjusted_life_expectancy': [-10.0],  # Invalid: negative
            'years_lost': [0.0],
            'years_gained': [1.0],
            'risk_score': [30.0]
        })
        
        with pytest.raises(DataValidationError, match="Adjusted life expectancy must be non-negative"):
            DataValidator.validate_lifestyle_data(df)


class TestInputValidator:
    """Test cases for InputValidator class."""
    
    def test_validate_age_group_success(self):
        """Test successful age group validation."""
        valid_age_groups = ['20-24', '25-29', '30-34', '35-39', '40-44', 
                           '45-49', '50-54', '55-59', '60-64', '65-69']
        
        for age_group in valid_age_groups:
            result = InputValidator.validate_age_group(age_group)
            assert result is True
    
    def test_validate_age_group_invalid(self):
        """Test age group validation with invalid input."""
        with pytest.raises(ValidationError, match="Invalid age group"):
            InputValidator.validate_age_group("invalid_age")
    
    def test_validate_lifestyle_choice_success(self):
        """Test successful lifestyle choice validation."""
        valid_choices = {
            'smoking': ['never', 'former', 'current_light', 'current_heavy'],
            'exercise': ['sedentary', 'light', 'moderate', 'vigorous'],
            'diet': ['poor', 'average', 'good', 'excellent'],
            'alcohol': ['none', 'light', 'moderate', 'heavy'],
            'sleep': ['poor', 'average', 'good', 'excellent'],
            'stress': ['high', 'moderate', 'low', 'minimal'],
            'social_connections': ['isolated', 'limited', 'moderate', 'strong']
        }
        
        for factor, values in valid_choices.items():
            for value in values:
                result = InputValidator.validate_lifestyle_choice(factor, value)
                assert result is True
    
    def test_validate_lifestyle_choice_invalid_factor(self):
        """Test lifestyle choice validation with invalid factor."""
        with pytest.raises(ValidationError, match="Unknown lifestyle factor"):
            InputValidator.validate_lifestyle_choice("invalid_factor", "some_value")
    
    def test_validate_lifestyle_choice_invalid_value(self):
        """Test lifestyle choice validation with invalid value."""
        with pytest.raises(ValidationError, match="Invalid smoking value"):
            InputValidator.validate_lifestyle_choice("smoking", "invalid_value")
    
    def test_validate_user_choices_success(self):
        """Test successful user choices validation."""
        choices = {
            'age_group': '20-24',
            'smoking': 'never',
            'exercise': 'moderate',
            'diet': 'good',
            'alcohol': 'light',
            'sleep': 'good',
            'stress': 'low',
            'social_connections': 'moderate'
        }
        
        result = InputValidator.validate_user_choices(choices)
        assert result is True
    
    def test_validate_user_choices_missing_factors(self):
        """Test user choices validation with missing factors."""
        choices = {
            'age_group': '20-24',
            'smoking': 'never'
            # Missing other required factors
        }
        
        with pytest.raises(ValidationError, match="Missing required lifestyle factors"):
            InputValidator.validate_user_choices(choices)
    
    def test_validate_user_choices_invalid_values(self):
        """Test user choices validation with invalid values."""
        choices = {
            'age_group': 'invalid_age',
            'smoking': 'never',
            'exercise': 'moderate',
            'diet': 'good',
            'alcohol': 'light',
            'sleep': 'good',
            'stress': 'low',
            'social_connections': 'moderate'
        }
        
        with pytest.raises(ValidationError, match="Invalid age group"):
            InputValidator.validate_user_choices(choices)
    
    def test_validate_numeric_range_success(self):
        """Test successful numeric range validation."""
        result = InputValidator.validate_numeric_range(50, 0, 100, "test_value")
        assert result is True
    
    def test_validate_numeric_range_invalid_type(self):
        """Test numeric range validation with invalid type."""
        with pytest.raises(ValidationError, match="must be numeric"):
            InputValidator.validate_numeric_range("not_numeric", 0, 100, "test_value")
    
    def test_validate_numeric_range_out_of_range(self):
        """Test numeric range validation with out-of-range value."""
        with pytest.raises(ValidationError, match="must be between"):
            InputValidator.validate_numeric_range(150, 0, 100, "test_value")

