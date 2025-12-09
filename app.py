"""
app.py - Football Predictor App for Streamlit Cloud
Simplified version that will definitely work
"""

import streamlit as st
import pandas as pd

# Set page config - MUST BE FIRST
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Import the engine
try:
    from engine import PredictionEngine, TeamMetrics, MatchContext
    ENGINE_AVAILABLE = True
except ImportError as e:
    st.error(f"Engine import error: {e}")
    ENGINE_AVAILABLE = False

def main():
    st.title("⚽ Football Match Predictor")
    st.markdown("### Based on FootyStats Data Analysis")
    
    if not ENGINE_AVAILABLE:
        st.error("Prediction engine not available. Please check engine.py file.")
        return
    
    # Initialize engine
    engine = PredictionEngine()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # League selection
        league = st.selectbox(
            "Select League",
            ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Liga NOS", "Other"]
        )
        
        # Set context
        if league == "Premier League":
            engine.context = MatchContext(league_avg_goals=2.7, home_advantage=1.15)
        elif league == "La Liga":
            engine.context = MatchContext(league_avg_goals=2.5, home_advantage=1.18)
        elif league == "Bundesliga":
            engine.context = MatchContext(league_avg_goals=3.0, home_advantage=1.10)
        elif league == "Serie A":
            engine.context = MatchContext(league_avg_goals=2.6, home_advantage=1.16)
        elif league == "Ligue 1":
            engine.context = MatchContext(league_avg_goals=2.4, home_advantage=1.17)
        elif league == "Liga NOS":
            engine.context = MatchContext(league_avg_goals=2.68, home_advantage=1.15)
        else:
            engine.context = MatchContext()
        
        st.info(f"League: {league}")
        st.caption(f"Avg Goals: {engine.context.league_avg_goals}")
        
        # Example button
        if st.button("Load Example Match"):
            st.session_state.example_loaded = True
    
    # Main content
    st.header("📊 Enter Match Details")
    
    # Two columns for team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        
        # Use session state for example data
        if st.session_state.get('example_loaded', False):
            home_attack = 1.83
            home_defense = 1.33
            home_ppg = 1.83
            home_cs = 17
            home_fts = 17
            home_form = 1.0
            home_name = "Vitória Guimarães"
        else:
            home_name = st.text_input("Team Name", value="Team A", key="home_name")
            home_attack = st.number_input("Goals Scored/Game", 0.0, 5.0, 1.5, 0.1, key="home_attack")
            home_defense = st.number_input("Goals Conceded/Game", 0.0, 5.0, 1.3, 0.1, key="home_defense")
            home_ppg = st.number_input("Points/Game", 0.0, 3.0, 1.8, 0.1, key="home_ppg")
            home_cs = st.slider("Clean Sheet %", 0, 100, 20, key="home_cs")
            home_fts = st.slider("Failed to Score %", 0, 100, 20, key="home_fts")
            home_form = st.slider("Recent Form", 0.8, 1.2, 1.0, 0.05, key="home_form")
    
    with col2:
        st.subheader("🚗 Away Team")
        
        # Use session state for example data
        if st.session_state.get('example_loaded', False):
            away_attack = 1.50
            away_defense = 0.50
            away_ppg = 1.83
            away_cs = 67
            away_fts = 17
            away_form = 1.0
            away_name = "Gil Vicente"
        else:
            away_name = st.text_input("Team Name", value="Team B", key="away_name")
            away_attack = st.number_input("Goals Scored/Game", 0.0, 5.0, 1.3, 0.1, key="away_attack")
            away_defense = st.number_input("Goals Conceded/Game", 0.0, 5.0, 1.0, 0.1, key="away_defense")
            away_ppg = st.number_input("Points/Game", 0.0, 3.0, 1.8, 0.1, key="away_ppg")
            away_cs = st.slider("Clean Sheet %", 0, 100, 20, key="away_cs")
            away_fts = st.slider("Failed to Score %", 0, 100, 20, key="away_fts")
            away_form = st.slider("Recent Form", 0.8, 1.2, 1.0, 0.05, key="away_form")
    
    # H2H section
    st.subheader("🤝 Head-to-Head")
    h2h_btts = st.slider("H2H BTTS %", 0, 100, 50) / 100
    
    # Create metrics objects
    home_metrics = TeamMetrics(
        attack_strength=home_attack,
        defense_strength=home_defense,
        form_factor=home_form,
        clean_sheet_pct=home_cs/100,
        failed_to_score_pct=home_fts/100,
        ppg=home_ppg,
        btts_pct=0.5
    )
    
    away_metrics = TeamMetrics(
        attack_strength=away_attack,
        defense_strength=away_defense,
        form_factor=away_form,
        clean_sheet_pct=away_cs/100,
        failed_to_score_pct=away_fts/100,
        ppg=away_ppg,
        btts_pct=0.5
    )
    
    # Prediction button
    if st.button("🎯 Generate Predictions", type="primary"):
        with st.spinner("Calculating predictions..."):
            
            # Get predictions
            result_pred = engine.predict_match_result(home_metrics, away_metrics)
            over_under_pred = engine.predict_over_under(home_metrics, away_metrics)
            btts_pred = engine.predict_btts(home_metrics, away_metrics, h2h_btts)
            expected_goals = engine.calculate_expected_goals(home_metrics, away_metrics)
            patterns = engine.analyze_matchup_patterns(home_metrics, away_metrics)
            
            # Display results
            st.success("✅ Predictions Generated!")
            
            # Results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🏆 Match Result")
                pred = result_pred['prediction'].value
                st.metric("Prediction", pred)
                
                # Show probabilities
                st.write("Probabilities:")
                home_prob = result_pred['probabilities']['home_win']
                draw_prob = result_pred['probabilities']['draw']
                away_prob = result_pred['probabilities']['away_win']
                
                st.progress(home_prob, f"Home: {home_prob:.1%}")
                st.progress(draw_prob, f"Draw: {draw_prob:.1%}")
                st.progress(away_prob, f"Away: {away_prob:.1%}")
            
            with col2:
                st.markdown("### 📈 Over/Under 2.5")
                pred = over_under_pred['prediction'].value
                conf = over_under_pred['confidence']
                st.metric("Prediction", f"{pred}")
                st.metric("Confidence", f"{conf}")
                st.metric("Expected Goals", f"{over_under_pred['expected_goals']}")
                
                # Show probabilities
                over_prob = over_under_pred['probabilities']['over']
                under_prob = over_under_pred['probabilities']['under']
                st.progress(over_prob, f"Over: {over_prob:.1%}")
                st.progress(under_prob, f"Under: {under_prob:.1%}")
            
            with col3:
                st.markdown("### ⚔️ Both Teams to Score")
                pred = btts_pred['prediction'].value
                conf = btts_pred['confidence']
                st.metric("Prediction", f"{pred}")
                st.metric("Confidence", f"{conf}")
                
                # Show probabilities
                yes_prob = btts_pred['probabilities']['btts_yes']
                no_prob = btts_pred['probabilities']['btts_no']
                st.progress(yes_prob, f"Yes: {yes_prob:.1%}")
                st.progress(no_prob, f"No: {no_prob:.1%}")
            
            # Expected goals
            st.markdown("### 🎯 Expected Goals")
            eg_col1, eg_col2, eg_col3 = st.columns(3)
            
            with eg_col1:
                st.metric("Home", f"{expected_goals['home_goals']}")
            with eg_col2:
                st.metric("Away", f"{expected_goals['away_goals']}")
            with eg_col3:
                st.metric("Total", f"{expected_goals['total_goals']}")
            
            # Patterns
            if patterns:
                st.markdown("### 🔍 Matchup Patterns")
                for pattern in patterns:
                    st.info(f"• {pattern}")
            
            # Statistical summary
            st.markdown("### 📊 Statistical Summary")
            
            summary_df = pd.DataFrame({
                'Metric': ['Goals Scored', 'Goals Conceded', 'Clean Sheet %', 
                          'Failed to Score %', 'Points/Game', 'Recent Form'],
                home_name: [
                    f"{home_attack:.2f}",
                    f"{home_defense:.2f}",
                    f"{home_cs}%",
                    f"{home_fts}%",
                    f"{home_ppg:.2f}",
                    f"{home_form:.2f}"
                ],
                away_name: [
                    f"{away_attack:.2f}",
                    f"{away_defense:.2f}",
                    f"{away_cs}%",
                    f"{away_fts}%",
                    f"{away_ppg:.2f}",
                    f"{away_form:.2f}"
                ]
            })
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Key insights
            st.markdown("### 💡 Key Insights")
            
            insights = []
            
            # Defense insight
            if away_defense <= 0.6:
                insights.append(f"**Strong Away Defense**: {away_name} concedes only {away_defense:.2f} goals per game")
            
            # Clean sheet insight
            if away_cs >= 50:
                insights.append(f"**High Clean Sheet Rate**: {away_name} keeps clean sheets in {away_cs}% of away matches")
            
            # Expected goals insight
            if expected_goals['total_goals'] < 2.0:
                insights.append(f"**Very Low Scoring**: Only {expected_goals['total_goals']:.2f} total goals expected")
            elif expected_goals['total_goals'] < 2.5:
                insights.append(f"**Below Average Scoring**: {expected_goals['total_goals']:.2f} goals expected")
            
            # Parity insight
            if abs(home_ppg - away_ppg) <= 0.2:
                insights.append(f"**Close Matchup**: Both teams have similar PPG ({home_ppg:.2f} vs {away_ppg:.2f})")
            
            for insight in insights:
                st.write(f"• {insight}")

if __name__ == "__main__":
    main()
