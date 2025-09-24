"""
Interactive Dashboard for Lifestyle Choices and Life Expectancy Analysis.
Allows users to input their lifestyle choices and see personalized results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from data_generator import generate_lifestyle_data

# Page configuration
st.set_page_config(
    page_title="How Lifestyle Choices Steal Your Years",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .warning-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the lifestyle data."""
    try:
        df = pd.read_csv('data/lifestyle_data.csv')
        return df
    except FileNotFoundError:
        # Generate data if not found
        st.info("Generating data... This may take a moment.")
        df = generate_lifestyle_data()
        df.to_csv('data/lifestyle_data.csv', index=False)
        return df

def calculate_personal_impact(user_choices, df):
    """Calculate the impact of user's lifestyle choices."""
    
    # Find matching combinations
    mask = (
        (df['smoking'] == user_choices['smoking']) &
        (df['exercise'] == user_choices['exercise']) &
        (df['diet'] == user_choices['diet']) &
        (df['alcohol'] == user_choices['alcohol']) &
        (df['sleep'] == user_choices['sleep']) &
        (df['stress'] == user_choices['stress']) &
        (df['social_connections'] == user_choices['social_connections'])
    )
    
    matching_data = df[mask]
    
    if len(matching_data) == 0:
        return None
    
    # Calculate average impact for the user's age group
    age_group_data = matching_data[matching_data['age_group'] == user_choices['age_group']]
    
    if len(age_group_data) == 0:
        # Use closest age group
        age_groups = df['age_group'].unique()
        user_age = int(user_choices['age_group'].split('-')[0])
        closest_age = min(age_groups, key=lambda x: abs(int(x.split('-')[0]) - user_age))
        age_group_data = matching_data[matching_data['age_group'] == closest_age]
    
    if len(age_group_data) == 0:
        return None
    
    return {
        'age_group': user_choices['age_group'],
        'lifestyle_impact': age_group_data['lifestyle_impact'].mean(),
        'adjusted_life_expectancy': age_group_data['adjusted_life_expectancy'].mean(),
        'years_lost': age_group_data['years_lost'].mean(),
        'years_gained': age_group_data['years_gained'].mean(),
        'risk_score': age_group_data['risk_score'].mean(),
        'base_life_expectancy': age_group_data['base_life_expectancy'].mean()
    }

def create_personal_impact_chart(personal_data, df):
    """Create a personalized impact visualization."""
    
    if personal_data is None:
        return None
    
    # Create comparison with all data
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Your Life Expectancy Impact', 'Risk Score Comparison', 
                       'Years Lost/Gained Breakdown', 'Age Group Comparison'),
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "box"}]]
    )
    
    # 1. Life expectancy impact gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = personal_data['adjusted_life_expectancy'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Your Life Expectancy<br>(Years)"},
        delta = {'reference': personal_data['base_life_expectancy']},
        gauge = {
            'axis': {'range': [None, 80]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 50], 'color': "yellow"},
                {'range': [50, 80], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': personal_data['base_life_expectancy']
            }
        }
    ), row=1, col=1)
    
    # 2. Risk score comparison
    risk_comparison = df.groupby('risk_score').size().reset_index(name='count')
    fig.add_trace(go.Bar(
        x=risk_comparison['risk_score'],
        y=risk_comparison['count'],
        name='Population Distribution',
        marker_color='lightblue',
        opacity=0.7
    ), row=1, col=2)
    
    # Add user's risk score as a vertical line (using add_shape instead)
    fig.add_shape(
        type="line",
        x0=personal_data['risk_score'],
        x1=personal_data['risk_score'],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
        row=1, col=2
    )
    
    # Add annotation for user's risk score
    fig.add_annotation(
        x=personal_data['risk_score'],
        y=0.9,
        yref="paper",
        text=f"Your Risk: {personal_data['risk_score']:.1f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        row=1, col=2
    )
    
    # 3. Years lost/gained breakdown
    years_data = {
        'Years Lost': max(0, personal_data['years_lost']),
        'Years Gained': max(0, personal_data['years_gained']),
        'Net Impact': personal_data['lifestyle_impact']
    }
    
    colors = ['red' if v < 0 else 'green' if v > 0 else 'gray' for v in years_data.values()]
    
    fig.add_trace(go.Bar(
        x=list(years_data.keys()),
        y=list(years_data.values()),
        marker_color=colors,
        name='Years Impact',
        text=[f"{v:.1f}" for v in years_data.values()],
        textposition='auto'
    ), row=2, col=1)
    
    # 4. Age group comparison
    age_comparison = df.groupby('age_group')['lifestyle_impact'].mean().reset_index()
    fig.add_trace(go.Bar(
        x=age_comparison['age_group'],
        y=age_comparison['lifestyle_impact'],
        name='Age Group Average Impact',
        marker_color='lightgreen',
        opacity=0.7
    ), row=2, col=2)
    
    # Add user's data point
    fig.add_scatter(
        x=[personal_data['age_group']],
        y=[personal_data['lifestyle_impact']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Your Impact',
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Your Personalized Lifestyle Impact Analysis",
        title_x=0.5
    )
    
    return fig

def create_improvement_suggestions(personal_data, df):
    """Create suggestions for improving lifestyle choices."""
    
    if personal_data is None:
        return []
    
    suggestions = []
    
    # Analyze each factor
    factors = ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']
    factor_impacts = {}
    
    for factor in factors:
        factor_data = df.groupby(factor)['lifestyle_impact'].mean()
        factor_impacts[factor] = {
            'worst': factor_data.min(),
            'best': factor_data.max(),
            'range': factor_data.max() - factor_data.min()
        }
    
    # Generate suggestions based on impact potential
    if personal_data['years_lost'] > 5:
        suggestions.append({
            'priority': 'High',
            'category': 'Smoking',
            'message': 'Consider quitting smoking - this has the largest impact on life expectancy',
            'potential_gain': factor_impacts['smoking']['range']
        })
    
    if personal_data['risk_score'] > 70:
        suggestions.append({
            'priority': 'High',
            'category': 'Exercise',
            'message': 'Increase physical activity - even light exercise can add years to your life',
            'potential_gain': factor_impacts['exercise']['range']
        })
    
    if personal_data['years_lost'] > 2:
        suggestions.append({
            'priority': 'Medium',
            'category': 'Diet',
            'message': 'Improve your diet quality - focus on whole foods and reduce processed foods',
            'potential_gain': factor_impacts['diet']['range']
        })
    
    if personal_data['risk_score'] > 60:
        suggestions.append({
            'priority': 'Medium',
            'category': 'Sleep',
            'message': 'Prioritize good sleep hygiene - aim for 7-9 hours of quality sleep',
            'potential_gain': factor_impacts['sleep']['range']
        })
    
    if personal_data['years_lost'] > 1:
        suggestions.append({
            'priority': 'Low',
            'category': 'Social Connections',
            'message': 'Strengthen social relationships - strong social ties are linked to longevity',
            'potential_gain': factor_impacts['social_connections']['range']
        })
    
    return suggestions

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 How Lifestyle Choices Steal Your Years</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Interactive Analysis of Lifestyle Impact on Life Expectancy</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar for user input
    st.sidebar.markdown("## 🎯 Your Lifestyle Profile")
    st.sidebar.markdown("Tell us about your lifestyle choices to see their impact on your life expectancy.")
    
    # User input form
    with st.sidebar.form("lifestyle_form"):
        st.markdown("### Personal Information")
        age_group = st.selectbox(
            "Age Group",
            options=['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69'],
            index=2
        )
        
        st.markdown("### Lifestyle Choices")
        smoking = st.selectbox(
            "Smoking Status",
            options=['never', 'former', 'current_light', 'current_heavy'],
            index=0
        )
        
        exercise = st.selectbox(
            "Exercise Level",
            options=['sedentary', 'light', 'moderate', 'vigorous'],
            index=1
        )
        
        diet = st.selectbox(
            "Diet Quality",
            options=['poor', 'average', 'good', 'excellent'],
            index=2
        )
        
        alcohol = st.selectbox(
            "Alcohol Consumption",
            options=['none', 'light', 'moderate', 'heavy'],
            index=1
        )
        
        sleep = st.selectbox(
            "Sleep Quality",
            options=['poor', 'average', 'good', 'excellent'],
            index=2
        )
        
        stress = st.selectbox(
            "Stress Level",
            options=['high', 'moderate', 'low', 'minimal'],
            index=1
        )
        
        social_connections = st.selectbox(
            "Social Connections",
            options=['isolated', 'limited', 'moderate', 'strong'],
            index=2
        )
        
        submitted = st.form_submit_button("🔍 Analyze My Lifestyle Impact", use_container_width=True)
    
    if submitted:
        # Calculate personal impact
        user_choices = {
            'age_group': age_group,
            'smoking': smoking,
            'exercise': exercise,
            'diet': diet,
            'alcohol': alcohol,
            'sleep': sleep,
            'stress': stress,
            'social_connections': social_connections
        }
        
        personal_data = calculate_personal_impact(user_choices, df)
        
        if personal_data is None:
            st.error("No matching data found for your lifestyle combination. Please try different choices.")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write("**Your choices:**", user_choices)
                
                # Check what data exists for each factor
                for factor in ['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections']:
                    unique_values = df[factor].unique()
                    st.write(f"**{factor}:** Available options: {list(unique_values)}")
                
                # Check age groups
                age_groups = df['age_group'].unique()
                st.write(f"**Age groups:** Available options: {list(age_groups)}")
        else:
            # Display results
            st.markdown('<div class="sub-header">📈 Your Personalized Results</div>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Life Expectancy",
                    value=f"{personal_data['adjusted_life_expectancy']:.1f} years",
                    delta=f"{personal_data['lifestyle_impact']:+.1f} years"
                )
            
            with col2:
                st.metric(
                    label="Risk Score",
                    value=f"{personal_data['risk_score']:.1f}/100",
                    delta="Lower is better"
                )
            
            with col3:
                if personal_data['years_lost'] > 0:
                    st.metric(
                        label="Years Lost",
                        value=f"{personal_data['years_lost']:.1f} years",
                        delta="Due to lifestyle choices"
                    )
                else:
                    st.metric(
                        label="Years Gained",
                        value=f"{personal_data['years_gained']:.1f} years",
                        delta="Due to healthy choices"
                    )
            
            with col4:
                st.metric(
                    label="Base Life Expectancy",
                    value=f"{personal_data['base_life_expectancy']:.1f} years",
                    delta="For your age group"
                )
            
            # Impact assessment
            if personal_data['lifestyle_impact'] < -5:
                st.markdown('<div class="warning-box"><h4>⚠️ High Risk</h4><p>Your lifestyle choices are significantly reducing your life expectancy. Consider making changes to improve your health and longevity.</p></div>', unsafe_allow_html=True)
            elif personal_data['lifestyle_impact'] < -2:
                st.markdown('<div class="warning-box"><h4>⚠️ Moderate Risk</h4><p>Your lifestyle choices are moderately impacting your life expectancy. Small changes could make a big difference.</p></div>', unsafe_allow_html=True)
            elif personal_data['lifestyle_impact'] > 2:
                st.markdown('<div class="success-box"><h4>✅ Excellent Choices</h4><p>Your lifestyle choices are adding years to your life! Keep up the great work.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h4>📊 Average Impact</h4><p>Your lifestyle choices have a neutral to slightly positive impact on your life expectancy.</p></div>', unsafe_allow_html=True)
            
            # Personalized visualization
            st.markdown('<div class="sub-header">📊 Your Personalized Analysis</div>', unsafe_allow_html=True)
            try:
                personal_chart = create_personal_impact_chart(personal_data, df)
                if personal_chart:
                    st.plotly_chart(personal_chart, use_container_width=True)
                else:
                    st.warning("Unable to create personalized chart. Please try different lifestyle choices.")
            except Exception as e:
                st.error(f"Error creating personalized chart: {str(e)}")
                st.info("This might be due to insufficient data for your specific lifestyle combination. Try adjusting your choices.")
            
            # Improvement suggestions
            suggestions = create_improvement_suggestions(personal_data, df)
            if suggestions:
                st.markdown('<div class="sub-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
                
                for suggestion in suggestions:
                    priority_color = {
                        'High': '🔴',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }
                    
                    with st.expander(f"{priority_color[suggestion['priority']]} {suggestion['priority']} Priority: {suggestion['category']}"):
                        st.write(suggestion['message'])
                        st.write(f"**Potential Impact:** {suggestion['potential_gain']:.1f} years")
    
    # General insights section
    st.markdown('<div class="sub-header">🔍 General Insights</div>', unsafe_allow_html=True)
    
    # Key statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Overview")
        st.write(f"**Total Combinations Analyzed:** {len(df):,}")
        st.write(f"**Age Groups:** {len(df['age_group'].unique())}")
        st.write(f"**Lifestyle Factors:** {len(['smoking', 'exercise', 'diet', 'alcohol', 'sleep', 'stress', 'social_connections'])}")
        st.write(f"**Maximum Years Lost:** {df['years_lost'].max():.1f}")
        st.write(f"**Maximum Years Gained:** {df['years_gained'].max():.1f}")
    
    with col2:
        st.markdown("### 🎯 Key Findings")
        st.write("• **Smoking** has the largest negative impact on life expectancy")
        st.write("• **Exercise and diet** choices can add 2-3 years to your life")
        st.write("• **Social connections** significantly affect longevity")
        st.write("• **Lifestyle impact increases with age**")
        st.write("• **Small changes can make a big difference**")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>📊 Data Visualization Portfolio Project | Inspired by FlowingData</p>
            <p>⚠️ This analysis is for educational purposes only and should not replace medical advice</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
