"""
app.py - Streamlit Football Predictor App
Simple, working version
"""

import streamlit as st
import pandas as pd
from engine import PredictionEngine, TeamMetrics, MatchContext

# Set page config FIRST
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

def main():
    st.title("⚽ Football Match Predictor")
    st.markdown("Predict match outcomes based on team statistics")
    
    # Initialize engine
    engine = PredictionEngine()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        league = st.selectbox(
            "Select League",
            ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Liga NOS", "Other"]
        )
        
        # Set league context
        league_contexts = {
            "Premier League": MatchContext(league_avg_goals=2.7, home_advantage=1.15),
            "La Liga": MatchContext(league_avg_goals=2.5, home_advantage=1.18),
            "Bundesliga": MatchContext(league_avg_goals=3.0, home_advantage=1.10),
            "Serie A": MatchContext(league_avg_goals=2.6, home_advantage=1.16),
            "Ligue 1": MatchContext(league_avg_goals=2.4, home_advantage=1.17),
            "Liga NOS": MatchContext(league_avg_goals=2.68, home_advantage=1.15),
        }
        
        engine.context = league_contexts.get(league, MatchContext())
        
        st.info(f"Selected: {league}")
        st.write(f"Avg Goals: {engine.context.league_avg_goals}")
    
    # Main content
    st.header("📊 Match Details")
    
    # Two columns for team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team_name = st.text_input("Team Name", value="Vitória Guimarães", key="home_name")
        
        st.write("**Attack/Defense Stats**")
        home_attack = st.number_input("Goals Scored per Game", 
                                     min_value=0.0, max_value=5.0, 
                                     value=1.83, step=0.1, key="home_attack")
        home_defense = st.number_input("Goals Conceded per Game", 
                                      min_value=0.0, max_value=5.0, 
                                      value=1.33, step=0.1, key="home_defense")
        home_ppg = st.number_input("Points per Game", 
                                  min_value=0.0, max_value=3.0, 
                                  value=1.83, step=0.1, key="home_ppg")
        
        st.write("**Performance Metrics**")
        home_clean_sheet = st.slider("Clean Sheet %", 0, 100, 17, key="home_cs") / 100
        home_failed_score = st.slider("Failed to Score %", 0, 100, 17, key="home_fts") / 100
        home_form = st.slider("Recent Form", 0.8, 1.2, 1.0, step=0.05, key="home_form")
    
    with col2:
        st.subheader("🚗 Away Team")
        away_team_name = st.text_input("Team Name", value="Gil Vicente", key="away_name")
        
        st.write("**Attack/Defense Stats**")
        away_attack = st.number_input("Goals Scored per Game", 
                                     min_value=0.0, max_value=5.0, 
                                     value=1.50, step=0.1, key="away_attack")
        away_defense = st.number_input("Goals Conceded per Game", 
                                      min_value=0.0, max_value=5.0, 
                                      value=0.50, step=0.1, key="away_defense")
        away_ppg = st.number_input("Points per Game", 
                                  min_value=0.0, max_value=3.0, 
                                  value=1.83, step=0.1, key="away_ppg")
        
        st.write("**Performance Metrics**")
        away_clean_sheet = st.slider("Clean Sheet %", 0, 100, 67, key="away_cs") / 100
        away_failed_score = st.slider("Failed to Score %", 0, 100, 17, key="away_fts") / 100
        away_form = st.slider("Recent Form", 0.8, 1.2, 1.0, step=0.05, key="away_form")
    
    # H2H Section
    st.subheader("🤝 Head-to-Head")
    h2h_btts = st.slider("H2H BTTS %", 0, 100, 62) / 100
    
    # Create TeamMetrics objects
    home_metrics = TeamMetrics(
        attack_strength=home_attack,
        defense_strength=home_defense,
        form_factor=home_form,
        clean_sheet_pct=home_clean_sheet,
        failed_to_score_pct=home_failed_score,
        ppg=home_ppg,
        btts_pct=0.5  # Default
    )
    
    away_metrics = TeamMetrics(
        attack_strength=away_attack,
        defense_strength=away_defense,
        form_factor=away_form,
        clean_sheet_pct=away_clean_sheet,
        failed_to_score_pct=away_failed_score,
        ppg=away_ppg,
        btts_pct=0.5  # Default
    )
    
    # Prediction Button
    if st.button("🎯 Generate Predictions", type="primary"):
        
        # Calculate all predictions
        result_pred = engine.predict_match_result(home_metrics, away_metrics)
        over_under_pred = engine.predict_over_under(home_metrics, away_metrics)
        btts_pred = engine.predict_btts(home_metrics, away_metrics, h2h_btts)
        expected_goals = engine.calculate_expected_goals(home_metrics, away_metrics)
        patterns = engine.analyze_matchup_patterns(home_metrics, away_metrics)
        
        # Display results
        st.success("✅ Predictions Generated!")
        
        # Matchup info
        st.subheader(f"🎮 {home_team_name} vs {away_team_name}")
        
        # Display in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Match Result**")
            pred = result_pred['prediction'].value
            st.metric("Prediction", pred)
            
            # Progress bars for probabilities
            home_prob = result_pred['probabilities']['home_win']
            draw_prob = result_pred['probabilities']['draw']
            away_prob = result_pred['probabilities']['away_win']
            
            st.progress(home_prob, text=f"Home: {home_prob:.1%}")
            st.progress(draw_prob, text=f"Draw: {draw_prob:.1%}")
            st.progress(away_prob, text=f"Away: {away_prob:.1%}")
        
        with col2:
            st.markdown("**📈 Over/Under 2.5**")
            pred = over_under_pred['prediction'].value
            conf = over_under_pred['confidence']
            st.metric("Prediction", f"{pred}")
            st.metric("Confidence", f"{conf}")
            st.metric("Expected Goals", f"{over_under_pred['expected_goals']}")
            
            # Progress bars
            over_prob = over_under_pred['probabilities']['over']
            under_prob = over_under_pred['probabilities']['under']
            st.progress(over_prob, text=f"Over: {over_prob:.1%}")
            st.progress(under_prob, text=f"Under: {under_prob:.1%}")
        
        with col3:
            st.markdown("**⚔️ Both Teams to Score**")
            pred = btts_pred['prediction'].value
            conf = btts_pred['confidence']
            st.metric("Prediction", f"{pred}")
            st.metric("Confidence", f"{conf}")
            
            # Progress bars
            yes_prob = btts_pred['probabilities']['btts_yes']
            no_prob = btts_pred['probabilities']['btts_no']
            st.progress(yes_prob, text=f"Yes: {yes_prob:.1%}")
            st.progress(no_prob, text=f"No: {no_prob:.1%}")
        
        # Expected Goals Breakdown
        st.subheader("🎯 Expected Goals Analysis")
        eg_col1, eg_col2, eg_col3 = st.columns(3)
        
        with eg_col1:
            st.metric("Home Expected Goals", expected_goals['home_goals'])
        with eg_col2:
            st.metric("Away Expected Goals", expected_goals['away_goals'])
        with eg_col3:
            st.metric("Total Expected Goals", expected_goals['total_goals'])
        
        # Matchup Patterns
        if patterns:
            st.subheader("🔍 Detected Patterns")
            for pattern in patterns:
                st.info(f"• {pattern}")
        
        # Statistical Summary Table
        st.subheader("📋 Statistical Summary")
        
        summary_data = {
            "Metric": ["Goals Scored/Game", "Goals Conceded/Game", "Clean Sheet %", 
                      "Failed to Score %", "Points/Game", "Recent Form"],
            home_team_name: [
                f"{home_attack:.2f}",
                f"{home_defense:.2f}",
                f"{home_clean_sheet:.0%}",
                f"{home_failed_score:.0%}",
                f"{home_ppg:.2f}",
                f"{home_form:.2f}"
            ],
            away_team_name: [
                f"{away_attack:.2f}",
                f"{away_defense:.2f}",
                f"{away_clean_sheet:.0%}",
                f"{away_failed_score:.0%}",
                f"{away_ppg:.2f}",
                f"{away_form:.2f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Key Insights
        st.subheader("💡 Key Insights")
        
        # Defense insight
        if away_defense <= 0.6:
            st.write(f"• **Strong Away Defense**: {away_team_name} concedes only {away_defense:.2f} goals per game away")
        
        # Clean sheet insight
        if away_clean_sheet >= 0.5:
            st.write(f"• **High Clean Sheet Rate**: {away_team_name} keeps clean sheets in {away_clean_sheet:.0%} of away matches")
        
        # Expected goals insight
        if expected_goals['total_goals'] < 2.0:
            st.write(f"• **Low Scoring Match**: Only {expected_goals['total_goals']:.2f} total goals expected")
        elif expected_goals['total_goals'] < 2.5:
            st.write(f"• **Below Average Scoring**: {expected_goals['total_goals']:.2f} goals expected (league avg: {engine.context.league_avg_goals})")
        
        # Parity insight
        if abs(home_ppg - away_ppg) <= 0.2:
            st.write(f"• **Close Matchup**: Both teams have similar PPG ({home_ppg:.2f} vs {away_ppg:.2f})")
    
    # Example Button
    if st.sidebar.button("Load Example Match"):
        st.rerun()

if __name__ == "__main__":
    main()
