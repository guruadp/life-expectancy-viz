"""
Main script to run the complete lifestyle choices visualization project.
Generates data, creates visualizations, and exports all formats.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from data_generator import save_data, DataGenerationError
from visualizer import LifestyleVisualizer, VisualizationError
from utils import setup_logger, get_logger
from config import config_manager

# Setup logging
logger = get_logger(__name__)

def main() -> None:
    """Main execution function."""
    
    try:
        logger.info("Starting lifestyle analysis project")
        
        print("=" * 60)
        print("HOW LIFESTYLE CHOICES STEAL YOUR YEARS")
        print("Data Visualization Portfolio Project")
        print("=" * 60)
        
        # Validate configuration
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            print("✗ Configuration validation failed. Please check your settings.")
            return
        
        # Create necessary directories
        create_directories()
        
        # Step 1: Generate data
        print("\n1. Generating comprehensive lifestyle and life expectancy data...")
        try:
            df, summary = save_data()
            print(f"   ✓ Generated {len(df):,} lifestyle combinations")
            print(f"   ✓ Maximum years lost: {summary['max_years_lost']:.1f}")
            print(f"   ✓ Maximum years gained: {summary['max_years_gained']:.1f}")
            logger.info(f"Data generation completed: {len(df):,} combinations")
        except DataGenerationError as e:
            print(f"   ✗ Error generating data: {e}")
            logger.error(f"Data generation failed: {e}")
            return
        except Exception as e:
            print(f"   ✗ Unexpected error generating data: {e}")
            logger.error(f"Unexpected error in data generation: {e}")
            return
        
        # Step 2: Create visualizations
        print("\n2. Creating visualizations...")
        try:
            visualizer = LifestyleVisualizer()
            visualizer.generate_all_visualizations()
            print("   ✓ Interactive HTML visualization created")
            print("   ✓ Social media PNG created")
            print("   ✓ Presentation PNG created")
            logger.info("All visualizations created successfully")
        except VisualizationError as e:
            print(f"   ✗ Error creating visualizations: {e}")
            logger.error(f"Visualization creation failed: {e}")
            return
        except Exception as e:
            print(f"   ✗ Unexpected error creating visualizations: {e}")
            logger.error(f"Unexpected error in visualization creation: {e}")
            return
        
        # Step 3: Generate additional analysis
        print("\n3. Generating additional analysis...")
        try:
            create_analysis_notebook()
            print("   ✓ Jupyter notebook for analysis created")
            logger.info("Analysis notebook created successfully")
        except Exception as e:
            print(f"   ✗ Error creating analysis notebook: {e}")
            logger.warning(f"Analysis notebook creation failed: {e}")
        
        # Display completion message
        display_completion_message()
        logger.info("Project completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)

def create_directories() -> None:
    """Create necessary directories."""
    try:
        # Get configuration
        db_config = config_manager.get_database_config()
        viz_config = config_manager.get_visualization_config()
        
        # Create directories
        Path(db_config.data_path).mkdir(parents=True, exist_ok=True)
        Path(viz_config.output_path).mkdir(parents=True, exist_ok=True)
        Path('notebooks').mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(parents=True, exist_ok=True)
        
        logger.info("Directories created successfully")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        raise

def display_completion_message() -> None:
    """Display project completion message."""
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("📊 Data Files:")
    print("   - data/lifestyle_data.csv (full dataset)")
    print("   - data/sample_data.csv (sample for quick analysis)")
    print("   - data/summary_stats.json (key statistics)")
    
    print("\n📈 Visualizations:")
    print("   - exports/interactive_visualization.html (interactive)")
    print("   - exports/social_media.png (clean for social media)")
    print("   - exports/presentation.png (annotated for presentations)")
    
    print("\n📓 Analysis:")
    print("   - notebooks/lifestyle_analysis.ipynb (exploratory analysis)")
    
    print("\n🚀 To run the interactive visualization:")
    print("   python -m http.server 8000")
    print("   Then open: http://localhost:8000/exports/interactive_visualization.html")
    
    print("\n💡 Key Insights:")
    print("   • Smoking has the largest negative impact on life expectancy")
    print("   • Exercise and diet choices can add 2-3 years to your life")
    print("   • Social connections significantly affect longevity")
    print("   • Lifestyle impact increases with age")

def create_analysis_notebook():
    """Create a Jupyter notebook for exploratory analysis."""
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lifestyle Choices and Life Expectancy Analysis\\n",
    "\\n",
    "This notebook provides an in-depth analysis of how lifestyle choices impact life expectancy.\\n",
    "\\n",
    "## Key Questions:\\n",
    "- Which lifestyle factors have the greatest impact on life expectancy?\\n",
    "- How do different combinations of lifestyle choices affect longevity?\\n",
    "- What are the most effective strategies for extending life expectancy?\\n",
    "- How does the impact of lifestyle choices vary by age group?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import plotly.express as px\\n",
    "import plotly.graph_objects as go\\n",
    "from plotly.subplots import make_subplots\\n",
    "\\n",
    "# Set style\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "df = pd.read_csv('../data/lifestyle_data.csv')\\n",
    "print(f'Dataset shape: {df.shape}')\\n",
    "print(f'Columns: {list(df.columns)}')\\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic statistics\\n",
    "print('Lifestyle Impact Statistics:')\\n",
    "print(df['lifestyle_impact'].describe())\\n",
    "print('\\nYears Lost Statistics:')\\n",
    "print(df['years_lost'].describe())\\n",
    "print('\\nYears Gained Statistics:')\\n",
    "print(df['years_gained'].describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Factor impact analysis\\n",
    "factors = ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']\\n",
    "\\n",
    "fig, axes = plt.subplots(2, 4, figsize=(20, 10))\\n",
    "axes = axes.ravel()\\n",
    "\\n",
    "for i, factor in enumerate(factors):\\n",
    "    factor_impact = df.groupby(factor)['lifestyle_impact'].mean().sort_values()\\n",
    "    bars = axes[i].bar(range(len(factor_impact)), factor_impact.values, \\n",
    "                      color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(factor_impact))))\\n",
    "    axes[i].set_title(f'{factor.replace(\"_\", \" \").title()} Impact')\\n",
    "    axes[i].set_xticks(range(len(factor_impact)))\\n",
    "    axes[i].set_xticklabels(factor_impact.index, rotation=45)\\n",
    "    axes[i].set_ylabel('Years Lost/Gained')\\n",
    "    \\n",
    "    # Add value labels\\n",
    "    for bar, value in zip(bars, factor_impact.values):\\n",
    "        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,\\n",
    "                    f'{value:.1f}', ha='center', va='bottom')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Interactive correlation heatmap\\n",
    "correlation_data = df[['lifestyle_impact', 'years_lost', 'years_gained', 'risk_score']].corr()\\n",
    "\\n",
    "fig = px.imshow(correlation_data, \\n",
    "                text_auto=True, \\n",
    "                aspect='auto',\\n",
    "                title='Correlation Matrix of Key Metrics')\\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Age group analysis\\n",
    "age_analysis = df.groupby('age_group').agg({\\n",
    "    'adjusted_life_expectancy': ['mean', 'std'],\\n",
    "    'lifestyle_impact': ['mean', 'std'],\\n",
    "    'risk_score': ['mean', 'std']\\n",
    "}).round(2)\\n",
    "\\n",
    "print('Age Group Analysis:')\\n",
    "print(age_analysis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Best and worst lifestyle combinations\\n",
    "print('Top 10 Best Lifestyle Combinations (Most Years Gained):')\\n",
    "best_combinations = df.nlargest(10, 'years_gained')[\\n",
    "    ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections', 'years_gained']\\n",
    "]\\n",
    "print(best_combinations)\\n",
    "\\n",
    "print('\\nTop 10 Worst Lifestyle Combinations (Most Years Lost):')\\n",
    "worst_combinations = df.nlargest(10, 'years_lost')[\\n",
    "    ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections', 'years_lost']\\n",
    "]\\n",
    "print(worst_combinations)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Interactive 3D scatter plot\\n",
    "fig = px.scatter_3d(df.sample(1000), \\n",
    "                    x='lifestyle_impact', \\n",
    "                    y='adjusted_life_expectancy', \\n",
    "                    z='risk_score',\\n",
    "                    color='risk_score',\\n",
    "                    hover_data=['smoking', 'exercise', 'diet'],\\n",
    "                    title='3D View: Lifestyle Impact vs Life Expectancy vs Risk Score')\\n",
    "fig.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    with open('notebooks/lifestyle_analysis.ipynb', 'w') as f:
        f.write(notebook_content)

if __name__ == "__main__":
    # Setup logging
    setup_logger()
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Project interrupted by user")
        print("\n✗ Project interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"✗ Fatal error: {e}")
        sys.exit(1)
