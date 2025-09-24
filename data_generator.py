"""
Data generator for lifestyle choices and life expectancy analysis.
Creates realistic data based on research studies and WHO statistics.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from utils import DataValidator, DataGenerationError, get_logger
from config import config_manager

logger = get_logger(__name__)

def generate_lifestyle_data() -> pd.DataFrame:
    """
    Generate comprehensive lifestyle and life expectancy data.
    
    Returns:
        DataFrame containing lifestyle combinations and their impacts
        
    Raises:
        DataGenerationError: If data generation fails
    """
    try:
        logger.info("Starting lifestyle data generation")
        
        # Base life expectancy by age groups
        age_groups = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69']
        base_life_expectancy = [58.2, 53.4, 48.6, 43.8, 39.0, 34.2, 29.4, 24.6, 19.8, 15.0]
        
        if len(age_groups) != len(base_life_expectancy):
            raise DataGenerationError("Age groups and life expectancy arrays must have the same length")
    
        # Lifestyle factors and their impact on life expectancy (years lost/gained)
        lifestyle_factors = {
            'smoking': {
                'never': 0,
                'former': -2.5,
                'current_light': -5.2,
                'current_heavy': -8.7
            },
            'exercise': {
                'sedentary': -4.1,
                'light': -1.8,
                'moderate': 0,
                'vigorous': 2.3
            },
            'diet': {
                'poor': -3.2,
                'average': -1.1,
                'good': 0,
                'excellent': 2.8
            },
            'alcohol': {
                'none': 0.5,
                'light': 0,
                'moderate': -1.2,
                'heavy': -4.8
            },
            'sleep': {
                'poor': -2.1,
                'average': -0.8,
                'good': 0,
                'excellent': 1.4
            },
            'stress': {
                'high': -3.5,
                'moderate': -1.2,
                'low': 0,
                'minimal': 1.8
            },
            'social_connections': {
                'isolated': -2.8,
                'limited': -1.1,
                'moderate': 0,
                'strong': 2.2
            }
        }
    
        # Generate comprehensive dataset
        data = []
        total_combinations = len(age_groups) * np.prod([len(factor) for factor in lifestyle_factors.values()])
        logger.info(f"Generating {total_combinations:,} lifestyle combinations")
        
        for i, age_group in enumerate(age_groups):
            base_years = base_life_expectancy[i]
            
            for smoking in lifestyle_factors['smoking']:
                for exercise in lifestyle_factors['exercise']:
                    for diet in lifestyle_factors['diet']:
                        for alcohol in lifestyle_factors['alcohol']:
                            for sleep in lifestyle_factors['sleep']:
                                for stress in lifestyle_factors['stress']:
                                    for social in lifestyle_factors['social_connections']:
                                        
                                        # Calculate total impact
                                        total_impact = (
                                            lifestyle_factors['smoking'][smoking] +
                                            lifestyle_factors['exercise'][exercise] +
                                            lifestyle_factors['diet'][diet] +
                                            lifestyle_factors['alcohol'][alcohol] +
                                            lifestyle_factors['sleep'][sleep] +
                                            lifestyle_factors['stress'][stress] +
                                            lifestyle_factors['social_connections'][social]
                                        )
                                        
                                        adjusted_years = max(0, base_years + total_impact)
                                        
                                        # Calculate risk score (0-100, higher = more risk)
                                        risk_score = max(0, min(100, 50 - total_impact * 2))
                                        
                                        data.append({
                                            'age_group': age_group,
                                            'base_life_expectancy': base_years,
                                            'smoking': smoking,
                                            'exercise': exercise,
                                            'diet': diet,
                                            'alcohol': alcohol,
                                            'sleep': sleep,
                                            'stress': stress,
                                            'social_connections': social,
                                            'lifestyle_impact': total_impact,
                                            'adjusted_life_expectancy': adjusted_years,
                                            'years_lost': max(0, -total_impact),
                                            'years_gained': max(0, total_impact),
                                            'risk_score': risk_score
                                        })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df):,} lifestyle combinations")
        
        # Validate the generated data
        DataValidator.validate_lifestyle_data(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to generate lifestyle data: {e}")
        raise DataGenerationError(f"Data generation failed: {e}") from e

def generate_summary_statistics(df: pd.DataFrame):
    """
    Generate summary statistics for key insights.
    
    Args:
        df: Lifestyle data DataFrame
        
    Returns:
        Dictionary containing summary statistics
        
    Raises:
        DataGenerationError: If summary generation fails
    """
    try:
        logger.info("Generating summary statistics")
        
        # Most impactful factors
        factor_impacts = {}
        for factor in ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']:
            if factor in df.columns:
                factor_impact = df.groupby(factor)['lifestyle_impact'].mean()
                factor_impacts[factor] = {
                    'worst': float(factor_impact.min()),
                    'best': float(factor_impact.max()),
                    'range': float(factor_impact.max() - factor_impact.min())
                }
        
        # Age group analysis
        age_analysis = df.groupby('age_group').agg({
            'adjusted_life_expectancy': ['mean', 'min', 'max'],
            'years_lost': 'mean',
            'risk_score': 'mean'
        }).round(1)
        
        # Flatten column names for JSON serialization
        age_analysis_dict = {}
        for col in age_analysis.columns:
            if isinstance(col, tuple):
                key = f"{col[0]}_{col[1]}"
            else:
                key = col
            age_analysis_dict[key] = age_analysis[col].to_dict()
        
        summary = {
            'factor_impacts': factor_impacts,
            'age_analysis': age_analysis_dict,
            'total_combinations': len(df),
            'max_years_lost': float(df['years_lost'].max()),
            'max_years_gained': float(df['years_gained'].max()),
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info("Summary statistics generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate summary statistics: {e}")
        raise DataGenerationError(f"Summary generation failed: {e}") from e

def save_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate and save all data files.
    
    Returns:
        Tuple of (DataFrame, summary statistics)
        
    Raises:
        DataGenerationError: If data generation or saving fails
    """
    try:
        logger.info("Starting data generation and saving process")
        
        # Get configuration
        db_config = config_manager.get_database_config()
        
        # Create data directory if it doesn't exist
        data_path = Path(db_config.data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Generate main dataset
        df = generate_lifestyle_data()
        
        # Save main dataset
        main_data_path = data_path / 'lifestyle_data.csv'
        df.to_csv(main_data_path, index=False)
        logger.info(f"Saved main dataset to {main_data_path}: {len(df):,} rows")
        
        # Generate summary statistics
        summary = generate_summary_statistics(df)
        
        # Save summary statistics
        summary_path = data_path / 'summary_stats.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary statistics to {summary_path}")
        
        # Generate sample data for visualization
        sample_data = df.sample(n=min(db_config.sample_size, len(df)), 
                               random_state=db_config.random_state)
        sample_path = data_path / 'sample_data.csv'
        sample_data.to_csv(sample_path, index=False)
        logger.info(f"Saved sample data to {sample_path}: {len(sample_data):,} rows")
        
        logger.info("Data generation and saving completed successfully")
        return df, summary
        
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise DataGenerationError(f"Data saving failed: {e}") from e

if __name__ == "__main__":
    from utils import setup_logger
    
    # Setup logging
    setup_logger()
    
    try:
        df, summary = save_data()
        print(f"Successfully generated {len(df):,} lifestyle combinations")
        print(f"Maximum years lost: {summary['max_years_lost']:.1f}")
        print(f"Maximum years gained: {summary['max_years_gained']:.1f}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
