"""
app.py - Football Predictor App with Automated Form Calculation
"""

import streamlit as st
import pandas as pd
from engine import PredictionEngine, TeamMetrics, MatchContext

# Set page config - MUST BE FIRST
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

def main():
    st.title("⚽ Football Match Predictor")
    st.markdown("### Based on FootyStats Data Analysis")
    
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
            home_name = "Vitória Guimarães"
            home_attack = 1.83
            home_defense = 1.33
            home_ppg = 1.83
            home_cs = 17
            home_fts = 17
            home_goals_last5 = 7
            home_conceded_last5 = 6
        else:
            home_name = st.text_input("Team Name", value="Team A", key="home_name")
            home_attack = st.number_input("Goals Scored/Game", 0.0, 5.0, 1.5, 0.1, key="home_attack")
            home_defense = st.number_input("Goals Conceded/Game", 0.0, 5.0, 1.3, 0.1, key="home_defense")
            home_ppg = st.number_input("Points/Game", 0.0, 3.0, 1.8, 0.1, key="home_ppg")
            home_cs = st.slider("Clean Sheet %", 0, 100, 20, key="home_cs")
            home_fts = st.slider("Failed to Score %", 0, 100, 20, key="home_fts")
            
            # Recent Form Inputs (Last 5 matches)
            st.write("**Recent Form (Last 5 Matches)**")
            col1a, col1b = st.columns(2)
            with col1a:
                home_goals_last5 = st.number_input("Goals Scored", 0, 30, 7, key="home_goals_last5")
            with col1b:
                home_conceded_last5 = st.number_input("Goals Conceded", 0, 30, 6, key="home_conceded_last5")
    
    with col2:
        st.subheader("🚗 Away Team")
        
        # Use session state for example data
        if st.session_state.get('example_loaded', False):
            away_name = "Gil Vicente"
            away_attack = 1.50
            away_defense = 0.50
            away_ppg = 1.83
            away_cs = 67
            away_fts = 17
            away_goals_last5 = 6
            away_conceded_last5 = 3
        else:
            away_name = st.text_input("Team Name", value="Team B", key="away_name")
            away_attack = st.number_input("Goals Scored/Game", 0.0, 5.0, 1.3, 0.1, key="away_attack")
            away_defense = st.number_input("Goals Conceded/Game", 0.0, 5.0, 1.0, 0.1, key="away_defense")
            away_ppg = st.number_input("Points/Game", 0.0, 3.0, 1.8, 0.1, key="away_ppg")
            away_cs = st.slider("Clean Sheet %", 0, 100, 20, key="away_cs")
            away_fts = st.slider("Failed to Score %", 0, 100, 20, key="away_fts")
            
            # Recent Form Inputs (Last 5 matches)
            st.write("**Recent Form (Last 5 Matches)**")
            col2a, col2b = st.columns(2)
            with col2a:
                away_goals_last5 = st.number_input("Goals Scored", 0, 30, 6, key="away_goals_last5")
            with col2b:
                away_conceded_last5 = st.number_input("Goals Conceded", 0, 30, 3, key="away_conceded_last5")
    
    # H2H section
    st.subheader("🤝 Head-to-Head")
    h2h_btts = st.slider("H2H BTTS %", 0, 100, 50) / 100
    
    # Create metrics objects
    home_metrics = TeamMetrics(
        attack_strength=home_attack,
        defense_strength=home_defense,
        goals_scored_last_5=home_goals_last5,
        goals_conceded_last_5=home_conceded_last5,
        clean_sheet_pct=home_cs/100,
        failed_to_score_pct=home_fts/100,
        ppg=home_ppg,
        btts_pct=0.5
    )
    
    away_metrics = TeamMetrics(
        attack_strength=away_attack,
        defense_strength=away_defense,
        goals_scored_last_5=away_goals_last5,
        goals_conceded_last_5=away_conceded_last5,
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
            
            # Show form factors
            st.info(f"📊 **Automated Form Analysis**: Home: {result_pred['form_factors']['home']}x | Away: {result_pred['form_factors']['away']}x")
            
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
                'Metric': ['Goals Scored/Game', 'Goals Conceded/Game', 'Clean Sheet %', 
                          'Failed to Score %', 'Points/Game', 'Goals Last 5', 'Conceded Last 5'],
                home_name: [
                    f"{home_attack:.2f}",
                    f"{home_defense:.2f}",
                    f"{home_cs}%",
                    f"{home_fts}%",
                    f"{home_ppg:.2f}",
                    f"{home_goals_last5}",
                    f"{home_conceded_last5}"
                ],
                away_name: [
                    f"{away_attack:.2f}",
                    f"{away_defense:.2f}",
                    f"{away_cs}%",
                    f"{away_fts}%",
                    f"{away_ppg:.2f}",
                    f"{away_goals_last5}",
                    f"{away_conceded_last5}"
                ]
            })
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Key insights
            st.markdown("### 💡 Key Insights")
            
            insights = []
            
            # Form insights
            home_form = result_pred['form_factors']['home']
            away_form = result_pred['form_factors']['away']
            
            if home_form >= 1.1:
                insights.append(f"**Home team in good form**: Scoring {home_goals_last5} goals in last 5 matches")
            elif home_form <= 0.9:
                insights.append(f"**Home team in poor form**: Only {home_goals_last5} goals in last 5 matches")
            
            if away_form >= 1.1:
                insights.append(f"**Away team in good form**: Scoring {away_goals_last5} goals in last 5 matches")
            elif away_form <= 0.9:
                insights.append(f"**Away team in poor form**: Only {away_goals_last5} goals in last 5 matches")
            
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
