"""
Tests for the data generator module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data_generator import generate_lifestyle_data, generate_summary_statistics, save_data
from utils import DataGenerationError


class TestDataGenerator:
    """Test cases for data generation functions."""
    
    def test_generate_lifestyle_data_returns_dataframe(self):
        """Test that generate_lifestyle_data returns a DataFrame."""
        df = generate_lifestyle_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_generate_lifestyle_data_has_required_columns(self):
        """Test that generated data has all required columns."""
        df = generate_lifestyle_data()
        required_columns = [
            'age_group', 'smoking', 'exercise', 'diet', 'alcohol', 
            'sleep', 'stress', 'social_connections', 'lifestyle_impact',
            'adjusted_life_expectancy', 'years_lost', 'years_gained', 'risk_score'
        ]
        for col in required_columns:
            assert col in df.columns
    
    def test_generate_lifestyle_data_has_valid_age_groups(self):
        """Test that age groups are valid."""
        df = generate_lifestyle_data()
        valid_age_groups = ['20-24', '25-29', '30-34', '35-39', '40-44', 
                           '45-49', '50-54', '55-59', '60-64', '65-69']
        assert all(age in valid_age_groups for age in df['age_group'].unique())
    
    def test_generate_lifestyle_data_has_valid_lifestyle_factors(self):
        """Test that lifestyle factors have valid values."""
        df = generate_lifestyle_data()
        
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
            assert all(value in valid_values for value in df[factor].unique())
    
    def test_generate_lifestyle_data_has_valid_numeric_ranges(self):
        """Test that numeric values are within valid ranges."""
        df = generate_lifestyle_data()
        
        # Risk score should be between 0 and 100
        assert (df['risk_score'] >= 0).all()
        assert (df['risk_score'] <= 100).all()
        
        # Adjusted life expectancy should be non-negative
        assert (df['adjusted_life_expectancy'] >= 0).all()
        
        # Years lost and gained should be non-negative
        assert (df['years_lost'] >= 0).all()
        assert (df['years_gained'] >= 0).all()
    
    def test_generate_summary_statistics_returns_dict(self):
        """Test that generate_summary_statistics returns a dictionary."""
        df = generate_lifestyle_data()
        summary = generate_summary_statistics(df)
        assert isinstance(summary, dict)
    
    def test_generate_summary_statistics_has_required_keys(self):
        """Test that summary statistics has required keys."""
        df = generate_lifestyle_data()
        summary = generate_summary_statistics(df)
        
        required_keys = ['factor_impacts', 'age_analysis', 'total_combinations', 
                        'max_years_lost', 'max_years_gained', 'generation_timestamp']
        for key in required_keys:
            assert key in summary
    
    def test_generate_summary_statistics_factor_impacts(self):
        """Test that factor impacts are calculated correctly."""
        df = generate_lifestyle_data()
        summary = generate_summary_statistics(df)
        
        factors = ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']
        for factor in factors:
            assert factor in summary['factor_impacts']
            factor_data = summary['factor_impacts'][factor]
            assert 'worst' in factor_data
            assert 'best' in factor_data
            assert 'range' in factor_data
            assert factor_data['range'] == factor_data['best'] - factor_data['worst']
    
    @patch('data_generator.Path')
    @patch('data_generator.pd.DataFrame.to_csv')
    @patch('data_generator.json.dump')
    def test_save_data_success(self, mock_json_dump, mock_to_csv, mock_path):
        """Test successful data saving."""
        # Mock the path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir.return_value = None
        
        # Mock the data generation
        with patch('data_generator.generate_lifestyle_data') as mock_generate:
            with patch('data_generator.generate_summary_statistics') as mock_summary:
                mock_df = pd.DataFrame({'test': [1, 2, 3]})
                mock_summary_data = {'test': 'summary'}
                
                mock_generate.return_value = mock_df
                mock_summary.return_value = mock_summary_data
                
                df, summary = save_data()
                
                assert df.equals(mock_df)
                assert summary == mock_summary_data
                assert mock_to_csv.called
                assert mock_json_dump.called
    
    def test_generate_lifestyle_data_handles_errors(self):
        """Test that generate_lifestyle_data handles errors properly."""
        with patch('data_generator.pd.DataFrame') as mock_df:
            mock_df.side_effect = Exception("Test error")
            
            with pytest.raises(DataGenerationError):
                generate_lifestyle_data()
    
    def test_generate_summary_statistics_handles_errors(self):
        """Test that generate_summary_statistics handles errors properly."""
        with patch('data_generator.pd.Timestamp') as mock_timestamp:
            mock_timestamp.now.side_effect = Exception("Test error")
            
            df = generate_lifestyle_data()
            with pytest.raises(DataGenerationError):
                generate_summary_statistics(df)

