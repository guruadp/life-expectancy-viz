"""
Interactive and static visualizations for lifestyle choices and life expectancy.
Creates social media PNGs, presentation PNGs, and interactive HTML.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from utils import VisualizationError, get_logger, LoggerMixin
from config import config_manager

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = get_logger(__name__)

class LifestyleVisualizer(LoggerMixin):
    """Visualization class for lifestyle analysis data."""
    
    def __init__(self, data_path: str = 'data/lifestyle_data.csv'):
        """
        Initialize with data.
        
        Args:
            data_path: Path to the lifestyle data CSV file
            
        Raises:
            VisualizationError: If data loading fails
        """
        try:
            self.logger.info(f"Loading data from {data_path}")
            
            if not Path(data_path).exists():
                raise VisualizationError(f"Data file not found: {data_path}")
            
            self.df = pd.read_csv(data_path)
            
            # Validate data
            from utils import DataValidator
            DataValidator.validate_lifestyle_data(self.df)
            
            # Get colors from configuration
            viz_config = config_manager.get_visualization_config()
            self.colors = viz_config.color_scheme
            
            self.logger.info(f"Successfully loaded {len(self.df):,} records")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize visualizer: {e}")
            raise VisualizationError(f"Visualizer initialization failed: {e}") from e
        
    def create_interactive_visualization(self) -> go.Figure:
        """
        Create interactive HTML visualization with animations.
        
        Returns:
            Plotly figure object
            
        Raises:
            VisualizationError: If visualization creation fails
        """
        try:
            self.logger.info("Creating interactive visualization")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Life Expectancy by Lifestyle Factors', 
                              'Years Lost/Gained by Factor',
                              'Risk Score Distribution',
                              'Age Group Impact'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "box"}]]
            )
            
            # 1. Scatter plot: Life expectancy vs lifestyle impact
            scatter_data = self.df.groupby(['smoking', 'exercise', 'diet']).agg({
                'adjusted_life_expectancy': 'mean',
                'lifestyle_impact': 'mean',
                'risk_score': 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=scatter_data['lifestyle_impact'],
                    y=scatter_data['adjusted_life_expectancy'],
                    mode='markers',
                    marker=dict(
                        size=scatter_data['risk_score']/2,
                        color=scatter_data['risk_score'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Risk Score")
                    ),
                    text=scatter_data.apply(lambda x: f"Smoking: {x['smoking']}<br>Exercise: {x['exercise']}<br>Diet: {x['diet']}", axis=1),
                    hovertemplate='<b>%{text}</b><br>Life Expectancy: %{y:.1f} years<br>Lifestyle Impact: %{x:.1f} years<extra></extra>',
                    name='Lifestyle Combinations'
                ),
                row=1, col=1
            )
        
        # 2. Bar chart: Factor impacts
        factor_impacts = []
        factor_names = []
        
        for factor in ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']:
            factor_data = self.df.groupby(factor)['lifestyle_impact'].mean()
            worst_impact = factor_data.min()
            best_impact = factor_data.max()
            factor_impacts.extend([worst_impact, best_impact])
            factor_names.extend([f"{factor.replace('_', ' ').title()}<br>(Worst)", 
                               f"{factor.replace('_', ' ').title()}<br>(Best)"])
        
        fig.add_trace(
            go.Bar(
                x=factor_names,
                y=factor_impacts,
                marker_color=['#C73E1D' if x < 0 else '#2E86AB' for x in factor_impacts],
                name='Factor Impact',
                text=[f"{x:.1f}" for x in factor_impacts],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Histogram: Risk score distribution
        fig.add_trace(
            go.Histogram(
                x=self.df['risk_score'],
                nbinsx=20,
                marker_color=self.colors['primary'],
                name='Risk Score Distribution'
            ),
            row=2, col=1
        )
        
        # 4. Box plot: Age group impact
        age_data = []
        age_labels = []
        for age in self.df['age_group'].unique():
            age_subset = self.df[self.df['age_group'] == age]['lifestyle_impact']
            age_data.append(age_subset)
            age_labels.append(age)
        
        fig.add_trace(
            go.Box(
                y=age_data[0],
                name=age_labels[0],
                marker_color=self.colors['secondary']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'How Lifestyle Choices Steal Your Years',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.colors['text']}
            },
            showlegend=False,
            height=800,
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Lifestyle Impact (Years)", row=1, col=1)
        fig.update_yaxes(title_text="Adjusted Life Expectancy (Years)", row=1, col=1)
        fig.update_xaxes(title_text="Lifestyle Factors", row=1, col=2)
        fig.update_yaxes(title_text="Years Lost/Gained", row=1, col=2)
        fig.update_xaxes(title_text="Risk Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Age Groups", row=2, col=2)
        fig.update_yaxes(title_text="Lifestyle Impact (Years)", row=2, col=2)
        
        # Add annotations
        fig.add_annotation(
            x=0.5, y=0.95,
            xref="paper", yref="paper",
            text="Interactive visualization showing how lifestyle choices impact life expectancy",
            showarrow=False,
            font=dict(size=14, color=self.colors['text']),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=self.colors['primary'],
            borderwidth=1
        )
        
            self.logger.info("Interactive visualization created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive visualization: {e}")
            raise VisualizationError(f"Interactive visualization creation failed: {e}") from e
    
    def create_social_media_png(self) -> None:
        """
        Create clean PNG for social media sharing.
        
        Raises:
            VisualizationError: If PNG creation fails
        """
        try:
            self.logger.info("Creating social media PNG")
            
            # Get configuration
            viz_config = config_manager.get_visualization_config()
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=viz_config.figure_size)
            fig.suptitle('How Lifestyle Choices Steal Your Years', fontsize=24, fontweight='bold', y=0.95)
        
        # 1. Main scatter plot
        scatter_data = self.df.groupby(['smoking', 'exercise', 'diet']).agg({
            'adjusted_life_expectancy': 'mean',
            'lifestyle_impact': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        
        scatter = ax1.scatter(scatter_data['lifestyle_impact'], 
                             scatter_data['adjusted_life_expectancy'],
                             c=scatter_data['risk_score'], 
                             s=scatter_data['risk_score']*2,
                             cmap='RdYlGn_r', 
                             alpha=0.7,
                             edgecolors='white',
                             linewidth=0.5)
        
        ax1.set_xlabel('Lifestyle Impact (Years)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Adjusted Life Expectancy (Years)', fontsize=12, fontweight='bold')
        ax1.set_title('Life Expectancy vs Lifestyle Impact', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Risk Score', fontsize=10, fontweight='bold')
        
        # 2. Factor impact bar chart
        factors = ['Smoking', 'Exercise', 'Diet', 'Alcohol', 'Sleep', 'Stress', 'Social']
        worst_impacts = []
        best_impacts = []
        
        for factor in ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']:
            factor_data = self.df.groupby(factor)['lifestyle_impact'].mean()
            worst_impacts.append(factor_data.min())
            best_impacts.append(factor_data.max())
        
        x = np.arange(len(factors))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, worst_impacts, width, label='Worst Choice', color='#C73E1D', alpha=0.8)
        bars2 = ax2.bar(x + width/2, best_impacts, width, label='Best Choice', color='#2E86AB', alpha=0.8)
        
        ax2.set_xlabel('Lifestyle Factors', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Years Lost/Gained', fontsize=12, fontweight='bold')
        ax2.set_title('Impact of Lifestyle Choices', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(factors, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Risk score distribution
        ax3.hist(self.df['risk_score'], bins=20, color=self.colors['primary'], alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Age group comparison
        age_impact = self.df.groupby('age_group')['lifestyle_impact'].mean()
        bars = ax4.bar(age_impact.index, age_impact.values, color=self.colors['secondary'], alpha=0.8)
        ax4.set_xlabel('Age Group', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Lifestyle Impact (Years)', fontsize=12, fontweight='bold')
        ax4.set_title('Lifestyle Impact by Age Group', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
            plt.tight_layout()
            
            # Save the figure
            output_path = Path(viz_config.output_path) / 'social_media.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"Social media PNG saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create social media PNG: {e}")
            raise VisualizationError(f"Social media PNG creation failed: {e}") from e
        
    def create_presentation_png(self):
        """Create annotated PNG for presentations."""
        
        # Create figure with more space for annotations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('How Lifestyle Choices Steal Your Years: A Data-Driven Analysis', 
                    fontsize=28, fontweight='bold', y=0.95, color=self.colors['text'])
        
        # 1. Main scatter plot with annotations
        scatter_data = self.df.groupby(['smoking', 'exercise', 'diet']).agg({
            'adjusted_life_expectancy': 'mean',
            'lifestyle_impact': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        
        scatter = ax1.scatter(scatter_data['lifestyle_impact'], 
                             scatter_data['adjusted_life_expectancy'],
                             c=scatter_data['risk_score'], 
                             s=scatter_data['risk_score']*3,
                             cmap='RdYlGn_r', 
                             alpha=0.7,
                             edgecolors='white',
                             linewidth=1)
        
        ax1.set_xlabel('Lifestyle Impact (Years)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Adjusted Life Expectancy (Years)', fontsize=14, fontweight='bold')
        ax1.set_title('Life Expectancy vs Lifestyle Impact\n(Bubble size = Risk Score)', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add key insights as annotations
        ax1.annotate('High Risk:\nPoor lifestyle choices\ncan reduce life expectancy\nby up to 15+ years', 
                    xy=(-10, 25), xytext=(-15, 35),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.1))
        
        ax1.annotate('Low Risk:\nHealthy lifestyle choices\ncan extend life expectancy\nby 5+ years', 
                    xy=(5, 45), xytext=(10, 35),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.1))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Risk Score (0-100)', fontsize=12, fontweight='bold')
        
        # 2. Factor impact with detailed annotations
        factors = ['Smoking', 'Exercise', 'Diet', 'Alcohol', 'Sleep', 'Stress', 'Social']
        worst_impacts = []
        best_impacts = []
        
        for factor in ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']:
            factor_data = self.df.groupby(factor)['lifestyle_impact'].mean()
            worst_impacts.append(factor_data.min())
            best_impacts.append(factor_data.max())
        
        x = np.arange(len(factors))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, worst_impacts, width, label='Worst Choice', color='#C73E1D', alpha=0.8)
        bars2 = ax2.bar(x + width/2, best_impacts, width, label='Best Choice', color='#2E86AB', alpha=0.8)
        
        ax2.set_xlabel('Lifestyle Factors', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Years Lost/Gained', fontsize=14, fontweight='bold')
        ax2.set_title('Impact of Lifestyle Choices\n(Red = Years Lost, Blue = Years Gained)', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(factors, rotation=45)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and insights
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax2.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.2,
                    f'{height1:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.2,
                    f'{height2:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add factor-specific insights
            if i == 0:  # Smoking
                ax2.annotate('Smoking has the\nlargest negative impact', 
                           xy=(bar1.get_x() + bar1.get_width()/2, height1), 
                           xytext=(bar1.get_x() + bar1.get_width()/2, height1 - 3),
                           ha='center', va='top', fontsize=10, color='red', fontweight='bold')
        
        # 3. Risk score distribution with statistics
        n, bins, patches = ax3.hist(self.df['risk_score'], bins=20, color=self.colors['primary'], 
                                   alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Risk Score', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax3.set_title('Risk Score Distribution\n(Higher scores = Higher risk)', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_risk = self.df['risk_score'].mean()
        median_risk = self.df['risk_score'].median()
        ax3.axvline(mean_risk, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_risk:.1f}')
        ax3.axvline(median_risk, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_risk:.1f}')
        ax3.legend(fontsize=12)
        
        # 4. Age group comparison with trend line
        age_impact = self.df.groupby('age_group')['lifestyle_impact'].mean()
        bars = ax4.bar(age_impact.index, age_impact.values, color=self.colors['secondary'], alpha=0.8)
        ax4.set_xlabel('Age Group', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Lifestyle Impact (Years)', fontsize=14, fontweight='bold')
        ax4.set_title('Lifestyle Impact by Age Group\n(Impact increases with age)', fontsize=16, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add trend line
        x_trend = np.arange(len(age_impact))
        z = np.polyfit(x_trend, age_impact.values, 1)
        p = np.poly1d(z)
        ax4.plot(age_impact.index, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax4.legend(fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add overall insights box
        fig.text(0.5, 0.02, 
                'Key Insights: • Smoking has the largest negative impact on life expectancy • Exercise and diet choices can add 2-3 years • Social connections significantly affect longevity • Lifestyle impact increases with age',
                ha='center', va='bottom', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig('exports/presentation.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def generate_all_visualizations(self) -> None:
        """
        Generate all visualization types.
        
        Raises:
            VisualizationError: If any visualization generation fails
        """
        try:
            self.logger.info("Starting visualization generation process")
            
            # Get configuration
            viz_config = config_manager.get_visualization_config()
            
            # Create exports directory
            output_path = Path(viz_config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create interactive HTML visualization
            self.logger.info("Creating interactive HTML visualization...")
            interactive_fig = self.create_interactive_visualization()
            html_path = output_path / 'interactive_visualization.html'
            interactive_fig.write_html(html_path)
            self.logger.info(f"Interactive HTML saved to {html_path}")
            
            # Create social media PNG
            self.logger.info("Creating social media PNG...")
            self.create_social_media_png()
            
            # Create presentation PNG
            self.logger.info("Creating presentation PNG...")
            self.create_presentation_png()
            
            self.logger.info("All visualizations created successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")
            raise VisualizationError(f"Visualization generation failed: {e}") from e

if __name__ == "__main__":
    # Generate data first
    from data_generator import save_data
    save_data()
    
    # Create visualizations
    visualizer = LifestyleVisualizer()
    visualizer.generate_all_visualizations()
