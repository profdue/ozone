"""
app.py - Clean Football Predictor App
Simplified UI, professional, focused on predictions
"""

import streamlit as st
import pandas as pd
from engine import PredictionEngine, TeamMetrics, MatchContext

# Set page config
st.set_page_config(
    page_title="Football Predictor Pro",
    page_icon="⚽",
    layout="wide"
)

def main():
    st.title("⚽ Football Match Predictor")
    
    # Initialize engine
    engine = PredictionEngine()
    
    # Simple sidebar
    with st.sidebar:
        st.header("Settings")
        league = st.selectbox(
            "League",
            ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Liga NOS", "Other"]
        )
        
        # Set context
        contexts = {
            "Premier League": MatchContext(league_avg_goals=2.7),
            "La Liga": MatchContext(league_avg_goals=2.5),
            "Bundesliga": MatchContext(league_avg_goals=3.0),
            "Serie A": MatchContext(league_avg_goals=2.6),
            "Ligue 1": MatchContext(league_avg_goals=2.4),
            "Liga NOS": MatchContext(league_avg_goals=2.68),
        }
        engine.context = contexts.get(league, MatchContext())
        
        st.caption(f"League Avg: {engine.context.league_avg_goals} goals/game")
        
        # Quick examples
        if st.button("Example: Vitória vs Gil Vicente"):
            st.session_state.home_name = "Vitória Guimarães"
            st.session_state.home_attack = 1.83
            st.session_state.home_defense = 1.33
            st.session_state.home_ppg = 1.83
            st.session_state.home_games = 6
            st.session_state.home_cs = 17
            st.session_state.home_fts = 17
            st.session_state.home_goals5 = 7
            st.session_state.home_conceded5 = 6
            
            st.session_state.away_name = "Gil Vicente"
            st.session_state.away_attack = 1.50
            st.session_state.away_defense = 0.50
            st.session_state.away_ppg = 1.83
            st.session_state.away_games = 6
            st.session_state.away_cs = 67
            st.session_state.away_fts = 17
            st.session_state.away_goals5 = 6
            st.session_state.away_conceded5 = 3
            
            st.session_state.h2h_btts = 62
            st.rerun()
    
    # Main interface - Clean and simple
    st.header("Match Details")
    
    # Team names
    col_names = st.columns(2)
    with col_names[0]:
        home_name = st.text_input("Home Team", 
                                 value=st.session_state.get('home_name', 'Team A'))
    with col_names[1]:
        away_name = st.text_input("Away Team", 
                                 value=st.session_state.get('away_name', 'Team B'))
    
    # Two columns for stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_name}")
        
        # Basic stats in compact format
        attack_def_col = st.columns(2)
        with attack_def_col[0]:
            home_attack = st.number_input("Goals/Game", 0.0, 5.0, 
                                         value=st.session_state.get('home_attack', 1.5), 0.1,
                                         key="home_attack")
        with attack_def_col[1]:
            home_defense = st.number_input("Conceded/Game", 0.0, 5.0,
                                          value=st.session_state.get('home_defense', 1.0), 0.1,
                                          key="home_defense")
        
        # Points and games
        points_col = st.columns(2)
        with points_col[0]:
            home_ppg = st.number_input("Points/Game", 0.0, 3.0,
                                      value=st.session_state.get('home_ppg', 1.5), 0.1,
                                      key="home_ppg")
        with points_col[1]:
            home_games = st.number_input("Games Played", 1, 40,
                                        value=st.session_state.get('home_games', 10),
                                        key="home_games")
        
        # Performance metrics
        st.write("**Performance**")
        perf_col = st.columns(2)
        with perf_col[0]:
            home_cs = st.number_input("Clean Sheet %", 0, 100,
                                     value=st.session_state.get('home_cs', 30),
                                     key="home_cs")
        with perf_col[1]:
            home_fts = st.number_input("Fail to Score %", 0, 100,
                                      value=st.session_state.get('home_fts', 30),
                                      key="home_fts")
        
        # Recent form
        st.write("**Last 5 Matches**")
        form_col = st.columns(2)
        with form_col[0]:
            home_goals5 = st.number_input("Goals Scored", 0, 30,
                                         value=st.session_state.get('home_goals5', 6),
                                         key="home_goals5")
        with form_col[1]:
            home_conceded5 = st.number_input("Goals Conceded", 0, 30,
                                            value=st.session_state.get('home_conceded5', 4),
                                            key="home_conceded5")
    
    with col2:
        st.subheader(f"🚗 {away_name}")
        
        # Basic stats in compact format
        attack_def_col = st.columns(2)
        with attack_def_col[0]:
            away_attack = st.number_input("Goals/Game", 0.0, 5.0,
                                         value=st.session_state.get('away_attack', 1.5), 0.1,
                                         key="away_attack")
        with attack_def_col[1]:
            away_defense = st.number_input("Conceded/Game", 0.0, 5.0,
                                          value=st.session_state.get('away_defense', 1.0), 0.1,
                                          key="away_defense")
        
        # Points and games
        points_col = st.columns(2)
        with points_col[0]:
            away_ppg = st.number_input("Points/Game", 0.0, 3.0,
                                      value=st.session_state.get('away_ppg', 1.5), 0.1,
                                      key="away_ppg")
        with points_col[1]:
            away_games = st.number_input("Games Played", 1, 40,
                                        value=st.session_state.get('away_games', 10),
                                        key="away_games")
        
        # Performance metrics
        st.write("**Performance**")
        perf_col = st.columns(2)
        with perf_col[0]:
            away_cs = st.number_input("Clean Sheet %", 0, 100,
                                     value=st.session_state.get('away_cs', 30),
                                     key="away_cs")
        with perf_col[1]:
            away_fts = st.number_input("Fail to Score %", 0, 100,
                                      value=st.session_state.get('away_fts', 30),
                                      key="away_fts")
        
        # Recent form
        st.write("**Last 5 Matches**")
        form_col = st.columns(2)
        with form_col[0]:
            away_goals5 = st.number_input("Goals Scored", 0, 30,
                                         value=st.session_state.get('away_goals5', 6),
                                         key="away_goals5")
        with form_col[1]:
            away_conceded5 = st.number_input("Goals Conceded", 0, 30,
                                            value=st.session_state.get('away_conceded5', 4),
                                            key="away_conceded5")
    
    # H2H section
    st.subheader("Head-to-Head")
    h2h_btts = st.number_input("H2H BTTS % (all meetings)", 0, 100,
                              value=st.session_state.get('h2h_btts', 50),
                              key="h2h_btts")
    
    # Create metrics
    home_metrics = TeamMetrics(
        attack_strength=home_attack,
        defense_strength=home_defense,
        goals_scored_last_5=home_goals5,
        goals_conceded_last_5=home_conceded5,
        clean_sheet_pct=home_cs/100,
        failed_to_score_pct=home_fts/100,
        ppg=home_ppg,
        btts_pct=0.5,
        games_played=home_games
    )
    
    away_metrics = TeamMetrics(
        attack_strength=away_attack,
        defense_strength=away_defense,
        goals_scored_last_5=away_goals5,
        goals_conceded_last_5=away_conceded5,
        clean_sheet_pct=away_cs/100,
        failed_to_score_pct=away_fts/100,
        ppg=away_ppg,
        btts_pct=0.5,
        games_played=away_games
    )
    
    # Prediction button
    if st.button("Generate Predictions", type="primary", use_container_width=True):
        
        # Get predictions
        result_pred = engine.predict_match_result(home_metrics, away_metrics)
        over_under_pred = engine.predict_over_under(home_metrics, away_metrics)
        btts_pred = engine.predict_btts(home_metrics, away_metrics, h2h_btts/100)
        expected_goals = engine.calculate_expected_goals(home_metrics, away_metrics)
        patterns = engine.analyze_matchup_patterns(home_metrics, away_metrics)
        
        # Display results - Clean and focused
        st.success("Predictions Generated")
        
        # Main predictions in cards
        st.header("📊 Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Match Result")
            pred = result_pred['prediction'].value
            st.metric("Prediction", pred)
            
            # Simple probability display
            home_prob = result_pred['probabilities']['home_win']
            draw_prob = result_pred['probabilities']['draw']
            away_prob = result_pred['probabilities']['away_win']
            
            st.write(f"Home: **{home_prob:.1%}**")
            st.write(f"Draw: **{draw_prob:.1%}**")
            st.write(f"Away: **{away_prob:.1%}**")
        
        with col2:
            st.subheader("Over/Under 2.5")
            pred = over_under_pred['prediction'].value
            conf = over_under_pred['confidence']
            st.metric("Prediction", pred)
            st.metric("Confidence", conf)
            st.metric("Expected Goals", over_under_pred['expected_goals'])
            
            over_prob = over_under_pred['probabilities']['over']
            under_prob = over_under_pred['probabilities']['under']
            st.write(f"Over: **{over_prob:.1%}**")
            st.write(f"Under: **{under_prob:.1%}**")
        
        with col3:
            st.subheader("Both Teams to Score")
            pred = btts_pred['prediction'].value
            conf = btts_pred['confidence']
            st.metric("Prediction", pred)
            st.metric("Confidence", conf)
            
            yes_prob = btts_pred['probabilities']['btts_yes']
            no_prob = btts_pred['probabilities']['btts_no']
            st.write(f"Yes: **{yes_prob:.1%}**")
            st.write(f"No: **{no_prob:.1%}**")
        
        # Expected goals
        st.subheader("Expected Goals")
        eg_col1, eg_col2, eg_col3 = st.columns(3)
        
        with eg_col1:
            st.metric(f"{home_name}", f"{expected_goals['home_goals']}")
        with eg_col2:
            st.metric(f"{away_name}", f"{expected_goals['away_goals']}")
        with eg_col3:
            st.metric("Total", f"{expected_goals['total_goals']}")
        
        # Patterns
        if patterns:
            st.subheader("Key Patterns")
            for pattern in patterns:
                st.info(pattern)
        
        # Team comparison table
        st.subheader("Team Comparison")
        
        comparison_data = {
            '': ['Goals/Game', 'Conceded/Game', 'Clean Sheet %', 
                'Fail to Score %', 'Points/Game', 'Form (Last 5)'],
            home_name: [
                f"{home_attack:.2f}",
                f"{home_defense:.2f}",
                f"{home_cs}%",
                f"{home_fts}%",
                f"{home_ppg:.2f}",
                f"{home_goals5/5:.2f} goals/game"
            ],
            away_name: [
                f"{away_attack:.2f}",
                f"{away_defense:.2f}",
                f"{away_cs}%",
                f"{away_fts}%",
                f"{away_ppg:.2f}",
                f"{away_goals5/5:.2f} goals/game"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        insights = []
        
        # Form insights
        home_form = result_pred['form_factors']['home']
        away_form = result_pred['form_factors']['away']
        
        if home_form >= 1.1:
            insights.append(f"**{home_name} in good form**")
        elif home_form <= 0.9:
            insights.append(f"**{home_name} in poor form**")
        
        if away_form >= 1.1:
            insights.append(f"**{away_name} in good form**")
        elif away_form <= 0.9:
            insights.append(f"**{away_name} in poor form**")
        
        # Defense insights
        if away_defense <= 0.6:
            insights.append(f"**{away_name} has strong defense** ({away_defense:.2f} conceded/game)")
        if home_defense <= 0.6:
            insights.append(f"**{home_name} has strong defense** ({home_defense:.2f} conceded/game)")
        
        # Clean sheet insights
        if away_cs >= 50:
            insights.append(f"**{away_name} keeps clean sheets often** ({away_cs}%)")
        if home_cs >= 50:
            insights.append(f"**{home_name} keeps clean sheets often** ({home_cs}%)")
        
        # Expected goals insights
        if expected_goals['total_goals'] < engine.context.league_avg_goals:
            insights.append(f"**Below average scoring expected** ({expected_goals['total_goals']:.2f} vs {engine.context.league_avg_goals} avg)")
        elif expected_goals['total_goals'] > engine.context.league_avg_goals * 1.2:
            insights.append(f"**High scoring expected** ({expected_goals['total_goals']:.2f} vs {engine.context.league_avg_goals} avg)")
        
        # H2H insights
        if h2h_btts > 70:
            insights.append(f"**Strong H2H BTTS trend** ({h2h_btts}% of meetings)")
        elif h2h_btts < 30:
            insights.append(f"**Weak H2H BTTS trend** ({h2h_btts}% of meetings)")
        
        # Display insights
        for insight in insights:
            st.write(f"• {insight}")

if __name__ == "__main__":
    main()
