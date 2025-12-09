"""
app.py - Football Prediction App
Simple interface using PPG from FootyStats
"""

import streamlit as st
import pandas as pd
from engine import PredictionEngine, TeamMetrics

# Page configuration
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Initialize prediction engine
engine = PredictionEngine()

# App title
st.title("⚽ Football Match Predictor")
st.markdown("### Using FootyStats Data")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # League selection
    league = st.selectbox(
        "Select League",
        ["Liga NOS", "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Other"]
    )
    
    # Example button
    if st.button("Load Example (Vitória vs Gil Vicente)"):
        st.session_state.load_example = True
    
    st.markdown("---")
    st.info("""
    **Data Source:** FootyStats
    
    **What to enter:**
    1. Goals scored/conceded per game
    2. PPG from last 5 matches
    3. Clean sheet percentage
    
    All from venue-specific stats!
    """)

# Main content area
st.header("📊 Enter Match Details")

# Two columns for team inputs
col1, col2 = st.columns(2)

# Home team inputs
with col1:
    st.subheader("🏠 Home Team")
    
    # Team name
    home_name = st.text_input(
        "Team Name",
        value="Vitória Guimarães" if st.session_state.get('load_example', False) else "Home Team",
        key="home_name"
    )
    
    # Attack/defense stats
    st.write("**Attack & Defense**")
    home_goals_scored = st.number_input(
        "Goals Scored per Game (Home)",
        min_value=0.0,
        max_value=5.0,
        value=1.83 if st.session_state.get('load_example', False) else 1.5,
        step=0.1,
        key="home_scored"
    )
    
    home_goals_conceded = st.number_input(
        "Goals Conceded per Game (Home)",
        min_value=0.0,
        max_value=5.0,
        value=1.33 if st.session_state.get('load_example', False) else 1.2,
        step=0.1,
        key="home_conceded"
    )
    
    # Recent form (PPG)
    st.write("**Recent Form**")
    home_ppg = st.number_input(
        "Points per Game (last 5 home matches)",
        min_value=0.0,
        max_value=3.0,
        value=1.83 if st.session_state.get('load_example', False) else 1.5,
        step=0.1,
        key="home_ppg",
        help="From FootyStats 'FormResultsPPG' → 'Home'"
    )
    
    # Performance metrics
    st.write("**Performance Metrics**")
    home_clean_sheets = st.slider(
        "Clean Sheet % (Home)",
        0, 100,
        17 if st.session_state.get('load_example', False) else 30,
        key="home_cs"
    )
    
    home_failed_score = st.slider(
        "Failed to Score % (Home)",
        0, 100,
        17 if st.session_state.get('load_example', False) else 30,
        key="home_fts"
    )

# Away team inputs
with col2:
    st.subheader("🚗 Away Team")
    
    # Team name
    away_name = st.text_input(
        "Team Name",
        value="Gil Vicente" if st.session_state.get('load_example', False) else "Away Team",
        key="away_name"
    )
    
    # Attack/defense stats
    st.write("**Attack & Defense**")
    away_goals_scored = st.number_input(
        "Goals Scored per Game (Away)",
        min_value=0.0,
        max_value=5.0,
        value=1.50 if st.session_state.get('load_example', False) else 1.2,
        step=0.1,
        key="away_scored"
    )
    
    away_goals_conceded = st.number_input(
        "Goals Conceded per Game (Away)",
        min_value=0.0,
        max_value=5.0,
        value=0.50 if st.session_state.get('load_example', False) else 1.5,
        step=0.1,
        key="away_conceded"
    )
    
    # Recent form (PPG)
    st.write("**Recent Form**")
    away_ppg = st.number_input(
        "Points per Game (last 5 away matches)",
        min_value=0.0,
        max_value=3.0,
        value=1.83 if st.session_state.get('load_example', False) else 1.5,
        step=0.1,
        key="away_ppg",
        help="From FootyStats 'FormResultsPPG' → 'Away'"
    )
    
    # Performance metrics
    st.write("**Performance Metrics**")
    away_clean_sheets = st.slider(
        "Clean Sheet % (Away)",
        0, 100,
        67 if st.session_state.get('load_example', False) else 30,
        key="away_cs"
    )
    
    away_failed_score = st.slider(
        "Failed to Score % (Away)",
        0, 100,
        17 if st.session_state.get('load_example', False) else 30,
        key="away_fts"
    )

# Create team objects
home_team = TeamMetrics(
    goals_scored=home_goals_scored,
    goals_conceded=home_goals_conceded,
    ppg=home_ppg,
    clean_sheet_pct=home_clean_sheets / 100,
    failed_score_pct=home_failed_score / 100
)

away_team = TeamMetrics(
    goals_scored=away_goals_scored,
    goals_conceded=away_goals_conceded,
    ppg=away_ppg,
    clean_sheet_pct=away_clean_sheets / 100,
    failed_score_pct=away_failed_score / 100
)

# Generate predictions button
st.markdown("---")
if st.button("🎯 Generate Predictions", type="primary", use_container_width=True):
    
    with st.spinner("Calculating predictions..."):
        # Get predictions from engine
        predictions = engine.predict(home_team, away_team)
    
    # Display results
    st.success("✅ Predictions Generated!")
    
    # Match header
    st.header(f"🎮 {home_name} vs {away_name}")
    
    # Main predictions in columns
    col1, col2, col3 = st.columns(3)
    
    # Match Result
    with col1:
        st.subheader("🏆 Match Result")
        result = predictions['match_result']
        
        st.metric(
            "Prediction", 
            result['prediction'],
            f"Confidence: {result['confidence']}"
        )
        
        # Probability bars
        st.progress(
            result['home_win'], 
            text=f"Home Win: {result['home_win']:.1%}"
        )
        st.progress(
            result['draw'], 
            text=f"Draw: {result['draw']:.1%}"
        )
        st.progress(
            result['away_win'], 
            text=f"Away Win: {result['away_win']:.1%}"
        )
    
    # Over/Under 2.5
    with col2:
        st.subheader("📈 Over/Under 2.5")
        over_under = predictions['over_under']
        
        st.metric(
            "Prediction",
            over_under['prediction'],
            f"Confidence: {over_under['confidence']}"
        )
        
        st.metric(
            "Expected Total Goals",
            f"{predictions['expected_goals']['total']}"
        )
        
        # Probability bars
        st.progress(
            over_under['over'],
            text=f"Over 2.5: {over_under['over']:.1%}"
        )
        st.progress(
            over_under['under'],
            text=f"Under 2.5: {over_under['under']:.1%}"
        )
    
    # Both Teams to Score
    with col3:
        st.subheader("⚔️ Both Teams to Score")
        btts = predictions['btts']
        
        st.metric(
            "Prediction",
            btts['prediction'],
            f"Confidence: {btts['confidence']}"
        )
        
        # Probability bars
        st.progress(
            btts['yes'],
            text=f"Yes: {btts['yes']:.1%}"
        )
        st.progress(
            btts['no'],
            text=f"No: {btts['no']:.1%}"
        )
    
    # Expected Goals Breakdown
    st.subheader("🎯 Expected Goals Analysis")
    eg_col1, eg_col2, eg_col3 = st.columns(3)
    
    with eg_col1:
        st.metric(
            f"{home_name} Expected Goals",
            f"{predictions['expected_goals']['home']}"
        )
    
    with eg_col2:
        st.metric(
            f"{away_name} Expected Goals",
            f"{predictions['expected_goals']['away']}"
        )
    
    with eg_col3:
        st.metric(
            "Total Expected Goals",
            f"{predictions['expected_goals']['total']}",
            f"League Avg: {engine.league_avg_goals}"
        )
    
    # Form Analysis
    st.subheader("📊 Form Analysis")
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        st.metric(
            f"{home_name} Recent Form",
            f"{predictions['form_analysis']['home_ppg']} PPG",
            f"Form Factor: {predictions['form_analysis']['home_form']}"
        )
    
    with form_col2:
        st.metric(
            f"{away_name} Recent Form",
            f"{predictions['form_analysis']['away_ppg']} PPG",
            f"Form Factor: {predictions['form_analysis']['away_form']}"
        )
    
    # Statistical Summary
    st.subheader("📋 Statistical Summary")
    
    summary_data = {
        "Metric": [
            "Goals Scored/Game",
            "Goals Conceded/Game", 
            "PPG (last 5)",
            "Clean Sheet %",
            "Failed to Score %"
        ],
        home_name: [
            f"{home_goals_scored:.2f}",
            f"{home_goals_conceded:.2f}",
            f"{home_ppg:.2f}",
            f"{home_clean_sheets}%",
            f"{home_failed_score}%"
        ],
        away_name: [
            f"{away_goals_scored:.2f}",
            f"{away_goals_conceded:.2f}",
            f"{away_ppg:.2f}",
            f"{away_clean_sheets}%",
            f"{away_failed_score}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights & Recommendations")
    
    insights = []
    
    # Defense insight
    if away_goals_conceded < 0.8:
        insights.append(
            f"**Strong Away Defense**: {away_name} concedes only {away_goals_conceded:.2f} goals per game away"
        )
    
    # Clean sheet insight
    if away_clean_sheets > 50:
        insights.append(
            f"**High Clean Sheet Rate**: {away_name} keeps clean sheets in {away_clean_sheets}% of away matches"
        )
    
    # Expected goals insight
    if predictions['expected_goals']['total'] < 2.0:
        insights.append(
            f"**Very Low Scoring Expected**: Only {predictions['expected_goals']['total']:.2f} total goals expected"
        )
    elif predictions['expected_goals']['total'] < 2.5:
        insights.append(
            f"**Low Scoring Expected**: {predictions['expected_goals']['total']:.2f} total goals expected (below league average)"
        )
    
    # Form comparison insight
    form_diff = home_ppg - away_ppg
    if form_diff > 0.3:
        insights.append(
            f"**Home Form Advantage**: {home_name} has better recent form ({home_ppg} vs {away_ppg} PPG)"
        )
    elif form_diff < -0.3:
        insights.append(
            f"**Away Form Advantage**: {away_name} has better recent form ({away_ppg} vs {home_ppg} PPG)"
        )
    else:
        insights.append(
            f"**Equal Recent Form**: Both teams have similar recent form ({home_ppg} vs {away_ppg} PPG)"
        )
    
    # Failed to score insight
    if home_failed_score > 40 or away_failed_score > 40:
        insights.append(
            f"**Scoring Struggles**: One or both teams frequently fail to score"
        )
    
    # Display insights
    for insight in insights:
        st.info(insight)
    
    # Betting recommendations
    st.subheader("💰 Recommended Bets")
    
    recommendations = []
    
    # Based on predictions
    if predictions['over_under']['under'] > 0.65:
        recommendations.append("**Under 2.5 Goals** (High confidence)")
    
    if predictions['btts']['no'] > 0.65:
        recommendations.append("**BTTS: No** (High confidence)")
    
    if predictions['match_result']['draw'] > 0.35:
        recommendations.append("**Draw or Double Chance** (Medium confidence)")
    
    if away_goals_conceded < 0.8 and away_clean_sheets > 50:
        recommendations.append("**Away Team Clean Sheet** (Based on defensive stats)")
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            st.success(f"• {rec}")
    else:
        st.warning("No strong betting opportunities identified")

# Footer
st.markdown("---")
st.caption("""
**Data Entry Guide:**
- Use venue-specific stats (Home stats for home team, Away stats for away team)
- PPG = Points per game from last 5 matches at this venue
- All percentages should be entered as whole numbers (e.g., 17 for 17%)
""")

# Clear example flag
if st.session_state.get('load_example', False):
    st.session_state.load_example = False
