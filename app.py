"""
app_v2.py - Football Predictor Pro with xG Integration
Enhanced UI with xG inputs and advanced insights
"""

import streamlit as st
import pandas as pd
from engine_v2 import PredictionEngineV2, TeamMetrics, MatchContext

# Set page config
st.set_page_config(
    page_title="Football Predictor Pro v2.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("⚽ Football Predictor Pro v2.0")
    st.caption("Enhanced with xG Integration for Improved Accuracy")
    
    # Initialize engine
    engine = PredictionEngineV2()
    
    # League contexts presets
    LEAGUE_CONTEXTS = {
        "Premier League": MatchContext(
            league_avg_goals=2.7,
            league_avg_xg=1.35,
            home_advantage=1.15
        ),
        "La Liga": MatchContext(
            league_avg_goals=2.5,
            league_avg_xg=1.25,
            home_advantage=1.18
        ),
        "Bundesliga": MatchContext(
            league_avg_goals=3.0,
            league_avg_xg=1.50,
            home_advantage=1.10
        ),
        "Serie A": MatchContext(
            league_avg_goals=2.6,
            league_avg_xg=1.30,
            home_advantage=1.16
        ),
        "Ligue 1": MatchContext(
            league_avg_goals=2.4,
            league_avg_xg=1.20,
            home_advantage=1.17
        ),
        "Liga NOS": MatchContext(
            league_avg_goals=2.68,
            league_avg_xg=1.34,
            home_advantage=1.15
        ),
        "Other": MatchContext()
    }
    
    # Sidebar with advanced settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # League selection
        league = st.selectbox(
            "Select League",
            list(LEAGUE_CONTEXTS.keys()),
            index=5  # Default to Liga NOS
        )
        engine.context = LEAGUE_CONTEXTS[league]
        
        # Show league stats
        with st.expander("League Statistics"):
            st.write(f"**Average Goals:** {engine.context.league_avg_goals:.2f}")
            st.write(f"**Average xG per team:** {engine.context.league_avg_xg:.2f}")
            st.write(f"**Home Advantage:** {engine.context.home_advantage:.2f}x")
            st.write(f"**Away Penalty:** {engine.context.away_penalty:.2f}x")
        
        # Quick examples
        st.header("📋 Examples")
        
        if st.button("Example 1: Vitória vs Gil Vicente (with xG)"):
            set_example_1()
            st.rerun()
        
        if st.button("Example 2: Strong Defense vs Weak Attack"):
            set_example_2()
            st.rerun()
        
        if st.button("Example 3: xG Mismatch Case"):
            set_example_3()
            st.rerun()
        
        # Info
        st.info("""
        **Version 2.0 Features:**
        - xG Integration for better accuracy
        - Sample-size aware predictions
        - Advanced defensive analysis
        - xG pattern detection
        """)
    
    # Main interface
    st.header("📊 Enter Match Data")
    
    # Team names
    col_names = st.columns(2)
    with col_names[0]:
        home_name = st.text_input(
            "🏠 Home Team",
            value=st.session_state.get('home_name', 'Vitória Guimarães')
        )
    with col_names[1]:
        away_name = st.text_input(
            "🚗 Away Team",
            value=st.session_state.get('away_name', 'Gil Vicente')
        )
    
    # Main tabs for organized input
    tab1, tab2, tab3 = st.tabs(["⚽ Core Stats", "📈 xG Stats", "📊 Recent Form"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{home_name}")
            
            # Attack/Defense
            col_attack = st.columns(2)
            with col_attack[0]:
                home_attack = st.number_input(
                    "Goals/Game", 
                    0.0, 5.0, 
                    value=st.session_state.get('home_attack', 1.83), 
                    step=0.1,
                    key="home_attack"
                )
            with col_attack[1]:
                home_defense = st.number_input(
                    "Conceded/Game", 
                    0.0, 5.0,
                    value=st.session_state.get('home_defense', 1.33), 
                    step=0.1,
                    key="home_defense"
                )
            
            # Points and games
            col_points = st.columns(2)
            with col_points[0]:
                home_ppg = st.number_input(
                    "Points/Game", 
                    0.0, 3.0,
                    value=st.session_state.get('home_ppg', 1.83), 
                    step=0.1,
                    key="home_ppg"
                )
            with col_points[1]:
                home_games = st.number_input(
                    "Games Played", 
                    1, 40,
                    value=st.session_state.get('home_games', 6),
                    key="home_games"
                )
            
            # Performance metrics
            col_perf = st.columns(2)
            with col_perf[0]:
                home_cs = st.slider(
                    "Clean Sheet %", 
                    0, 100,
                    value=st.session_state.get('home_cs', 17),
                    key="home_cs"
                )
            with col_perf[1]:
                home_fts = st.slider(
                    "Fail to Score %", 
                    0, 100,
                    value=st.session_state.get('home_fts', 17),
                    key="home_fts"
                )
        
        with col2:
            st.subheader(f"{away_name}")
            
            # Attack/Defense
            col_attack = st.columns(2)
            with col_attack[0]:
                away_attack = st.number_input(
                    "Goals/Game", 
                    0.0, 5.0,
                    value=st.session_state.get('away_attack', 1.50), 
                    step=0.1,
                    key="away_attack"
                )
            with col_attack[1]:
                away_defense = st.number_input(
                    "Conceded/Game", 
                    0.0, 5.0,
                    value=st.session_state.get('away_defense', 0.50), 
                    step=0.1,
                    key="away_defense"
                )
            
            # Points and games
            col_points = st.columns(2)
            with col_points[0]:
                away_ppg = st.number_input(
                    "Points/Game", 
                    0.0, 3.0,
                    value=st.session_state.get('away_ppg', 1.83), 
                    step=0.1,
                    key="away_ppg"
                )
            with col_points[1]:
                away_games = st.number_input(
                    "Games Played", 
                    1, 40,
                    value=st.session_state.get('away_games', 6),
                    key="away_games"
                )
            
            # Performance metrics
            col_perf = st.columns(2)
            with col_perf[0]:
                away_cs = st.slider(
                    "Clean Sheet %", 
                    0, 100,
                    value=st.session_state.get('away_cs', 67),
                    key="away_cs"
                )
            with col_perf[1]:
                away_fts = st.slider(
                    "Fail to Score %", 
                    0, 100,
                    value=st.session_state.get('away_fts', 17),
                    key="away_fts"
                )
    
    with tab2:
        st.info("xG (Expected Goals) measures the quality of chances created/conceded")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{home_name} xG")
            
            col_xg = st.columns(2)
            with col_xg[0]:
                home_xg_for = st.number_input(
                    "xG Created/Game", 
                    0.0, 5.0,
                    value=st.session_state.get('home_xg_for', 1.95), 
                    step=0.1,
                    key="home_xg_for"
                )
            with col_xg[1]:
                home_xg_against = st.number_input(
                    "xG Conceded/Game", 
                    0.0, 5.0,
                    value=st.session_state.get('home_xg_against', 1.45), 
                    step=0.1,
                    key="home_xg_against"
                )
            
            # xG insights
            if home_xg_for > 0 and home_attack > 0:
                xg_diff = home_xg_for - home_attack
                if xg_diff > 0.3:
                    st.warning(f"Creates more chances than scores (+{xg_diff:.2f} xG difference)")
                elif xg_diff < -0.3:
                    st.success(f"Scores more than creates ({abs(xg_diff):.2f} xG difference)")
        
        with col2:
            st.subheader(f"{away_name} xG")
            
            col_xg = st.columns(2)
            with col_xg[0]:
                away_xg_for = st.number_input(
                    "xG Created/Game", 
                    0.0, 5.0,
                    value=st.session_state.get('away_xg_for', 1.30), 
                    step=0.1,
                    key="away_xg_for"
                )
            with col_xg[1]:
                away_xg_against = st.number_input(
                    "xG Conceded/Game", 
                    0.0, 5.0,
                    value=st.session_state.get('away_xg_against', 0.80), 
                    step=0.1,
                    key="away_xg_against"
                )
            
            # xG insights
            if away_xg_against > 0 and away_defense > 0:
                xga_diff = away_xg_against - away_defense
                if xga_diff > 0.3:
                    st.success(f"Concedes less than expected ({abs(xga_diff):.2f} xG difference)")
                elif xga_diff < -0.3:
                    st.warning(f"Concedes more than expected (+{abs(xga_diff):.2f} xG difference)")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{home_name} Recent Form")
            
            col_form = st.columns(2)
            with col_form[0]:
                home_goals5 = st.number_input(
                    "Goals (Last 5)", 
                    0, 30,
                    value=st.session_state.get('home_goals5', 7),
                    key="home_goals5"
                )
            with col_form[1]:
                home_conceded5 = st.number_input(
                    "Conceded (Last 5)", 
                    0, 30,
                    value=st.session_state.get('home_conceded5', 6),
                    key="home_conceded5"
                )
            
            # Form insights
            home_avg_5 = home_goals5 / 5
            if home_avg_5 > home_attack * 1.2:
                st.success(f"Good recent form: {home_avg_5:.2f} goals/game (↑ from {home_attack:.2f})")
            elif home_avg_5 < home_attack * 0.8:
                st.error(f"Poor recent form: {home_avg_5:.2f} goals/game (↓ from {home_attack:.2f})")
        
        with col2:
            st.subheader(f"{away_name} Recent Form")
            
            col_form = st.columns(2)
            with col_form[0]:
                away_goals5 = st.number_input(
                    "Goals (Last 5)", 
                    0, 30,
                    value=st.session_state.get('away_goals5', 6),
                    key="away_goals5"
                )
            with col_form[1]:
                away_conceded5 = st.number_input(
                    "Conceded (Last 5)", 
                    0, 30,
                    value=st.session_state.get('away_conceded5', 3),
                    key="away_conceded5"
                )
            
            # Form insights
            away_avg_5 = away_goals5 / 5
            if away_avg_5 > away_attack * 1.2:
                st.success(f"Good recent form: {away_avg_5:.2f} goals/game (↑ from {away_attack:.2f})")
            elif away_avg_5 < away_attack * 0.8:
                st.error(f"Poor recent form: {away_avg_5:.2f} goals/game (↓ from {away_attack:.2f})")
    
    # H2H Section
    with st.expander("Head-to-Head Data (Optional)"):
        col_h2h = st.columns(2)
        with col_h2h[0]:
            h2h_btts = st.slider(
                "H2H BTTS %", 
                0, 100,
                value=st.session_state.get('h2h_btts', 62),
                key="h2h_btts"
            )
        with col_h2h[1]:
            h2h_meetings = st.number_input(
                "Total H2H Meetings",
                0, 100,
                value=st.session_state.get('h2h_meetings', 5),
                key="h2h_meetings"
            )
    
    # Create team metrics objects
    home_metrics = TeamMetrics(
        name=home_name,
        attack_strength=home_attack,
        defense_strength=home_defense,
        ppg=home_ppg,
        xg_for=home_xg_for,
        xg_against=home_xg_against,
        clean_sheet_pct=home_cs/100,
        failed_to_score_pct=home_fts/100,
        btts_pct=0.5,  # Default, not used in calculations
        goals_scored_last_5=home_goals5,
        goals_conceded_last_5=home_conceded5,
        games_played=home_games
    )
    
    away_metrics = TeamMetrics(
        name=away_name,
        attack_strength=away_attack,
        defense_strength=away_defense,
        ppg=away_ppg,
        xg_for=away_xg_for,
        xg_against=away_xg_against,
        clean_sheet_pct=away_cs/100,
        failed_to_score_pct=away_fts/100,
        btts_pct=0.5,  # Default, not used in calculations
        goals_scored_last_5=away_goals5,
        goals_conceded_last_5=away_conceded5,
        games_played=away_games
    )
    
    # Generate predictions button
    if st.button("🚀 Generate Advanced Predictions", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing match with xG integration..."):
            # Get predictions
            result_pred = engine.predict_match_result(home_metrics, away_metrics)
            over_under_pred = engine.predict_over_under(home_metrics, away_metrics)
            btts_pred = engine.predict_btts(
                home_metrics, away_metrics, 
                h2h_btts=h2h_btts/100 if h2h_btts else None,
                h2h_meetings=h2h_meetings if h2h_meetings > 0 else None
            )
            expected_goals = engine.calculate_xg_adjusted_goals(home_metrics, away_metrics)
            patterns = engine.analyze_matchup_patterns(home_metrics, away_metrics)
            
            # Display results
            st.success("✅ Advanced Predictions Generated")
            
            # Main predictions in cards
            st.header("🎯 Core Predictions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🏆 Match Result")
                pred = result_pred['prediction'].value
                st.metric("Prediction", pred)
                
                # Probability bars
                home_prob = result_pred['probabilities']['home_win']
                draw_prob = result_pred['probabilities']['draw']
                away_prob = result_pred['probabilities']['away_win']
                
                st.progress(home_prob, text=f"Home: {home_prob:.1%}")
                st.progress(draw_prob, text=f"Draw: {draw_prob:.1%}")
                st.progress(away_prob, text=f"Away: {away_prob:.1%}")
            
            with col2:
                st.subheader("⚖️ Over/Under 2.5")
                pred = over_under_pred['prediction'].value
                conf = over_under_pred['confidence']
                st.metric("Prediction", pred)
                st.metric("Confidence", conf)
                st.metric("Expected Goals", over_under_pred['expected_goals'])
                
                over_prob = over_under_pred['probabilities']['over']
                under_prob = over_under_pred['probabilities']['under']
                
                st.progress(over_prob, text=f"Over: {over_prob:.1%}")
                st.progress(under_prob, text=f"Under: {under_prob:.1%}")
            
            with col3:
                st.subheader("🎯 Both Teams to Score")
                pred = btts_pred['prediction'].value
                conf = btts_pred['confidence']
                st.metric("Prediction", pred)
                st.metric("Confidence", conf)
                
                yes_prob = btts_pred['probabilities']['btts_yes']
                no_prob = btts_pred['probabilities']['btts_no']
                
                st.progress(yes_prob, text=f"Yes: {yes_prob:.1%}")
                st.progress(no_prob, text=f"No: {no_prob:.1%}")
            
            # Expected goals breakdown
            st.header("📊 Expected Goals Analysis")
            
            eg_home, eg_away = expected_goals
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"🏠 {home_name}", f"{eg_home:.2f}")
            with col2:
                st.metric(f"🚗 {away_name}", f"{eg_away:.2f}")
            with col3:
                st.metric("Total Expected", f"{eg_home + eg_away:.2f}")
            
            # xG adjustments
            if over_under_pred.get('xg_adjustments'):
                st.subheader("🔍 xG-Based Adjustments")
                for adjustment in over_under_pred['xg_adjustments']:
                    st.info(adjustment)
            
            # Match patterns and insights
            if patterns:
                st.header("🧠 Key Insights")
                
                col1, col2 = st.columns(2)
                with col1:
                    for i, pattern in enumerate(patterns[:len(patterns)//2]):
                        st.info(f"• {pattern}")
                with col2:
                    for i, pattern in enumerate(patterns[len(patterns)//2:]):
                        st.info(f"• {pattern}")
            
            # Team comparison table
            st.header("📈 Team Comparison")
            
            comparison_data = {
                'Metric': [
                    'Goals/Game', 'xG Created/Game', 
                    'Conceded/Game', 'xG Conceded/Game',
                    'Clean Sheet %', 'Fail to Score %', 
                    'Points/Game', 'Form (Last 5)',
                    'Games Played'
                ],
                home_name: [
                    f"{home_attack:.2f}",
                    f"{home_xg_for:.2f}",
                    f"{home_defense:.2f}",
                    f"{home_xg_against:.2f}",
                    f"{home_cs}%",
                    f"{home_fts}%",
                    f"{home_ppg:.2f}",
                    f"{home_goals5/5:.2f} goals/game",
                    f"{home_games}"
                ],
                away_name: [
                    f"{away_attack:.2f}",
                    f"{away_xg_for:.2f}",
                    f"{away_defense:.2f}",
                    f"{away_xg_against:.2f}",
                    f"{away_cs}%",
                    f"{away_fts}%",
                    f"{away_ppg:.2f}",
                    f"{away_goals5/5:.2f} goals/game",
                    f"{away_games}"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Defensive analysis
            st.header("🛡️ Defensive Analysis")
            
            home_def_analysis = engine.analyze_defensive_strength(home_metrics)
            away_def_analysis = engine.analyze_defensive_strength(away_metrics)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{home_name} Defense")
                def get_defense_strength(analysis):
                    if analysis['is_very_strong']:
                        return "🎯 **VERY STRONG**"
                    elif analysis['is_strong']:
                        return "✅ **Strong**"
                    elif analysis['is_weak']:
                        return "⚠️ **Weak**"
                    else:
                        return "📊 **Average**"
                
                st.markdown(f"**Strength:** {get_defense_strength(home_def_analysis)}")
                
                if home_def_analysis['xg_better_than_actual']:
                    st.success(f"Defense better than stats show (xGA: {home_xg_against:.2f} vs Actual: {home_defense:.2f})")
                elif home_def_analysis['xg_worse_than_actual']:
                    st.warning(f"Defense worse than stats show (xGA: {home_xg_against:.2f} vs Actual: {home_defense:.2f})")
                
                if home_def_analysis['clean_sheet_likely']:
                    st.info(f"High clean sheet probability ({home_cs}%)")
            
            with col2:
                st.subheader(f"{away_name} Defense")
                st.markdown(f"**Strength:** {get_defense_strength(away_def_analysis)}")
                
                if away_def_analysis['xg_better_than_actual']:
                    st.success(f"Defense better than stats show (xGA: {away_xg_against:.2f} vs Actual: {away_defense:.2f})")
                elif away_def_analysis['xg_worse_than_actual']:
                    st.warning(f"Defense worse than stats show (xGA: {away_xg_against:.2f} vs Actual: {away_defense:.2f})")
                
                if away_def_analysis['clean_sheet_likely']:
                    st.info(f"High clean sheet probability ({away_cs}%)")
            
            # League context
            st.header("🏆 League Context")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("League Avg Goals", f"{engine.context.league_avg_goals:.2f}")
            with col2:
                st.metric("League Avg xG/Team", f"{engine.context.league_avg_xg:.2f}")
            with col3:
                st.metric("Predicted Total", f"{eg_home + eg_away:.2f}")

def set_example_1():
    """Example: Vitória vs Gil Vicente with xG"""
    st.session_state.home_name = "Vitória Guimarães"
    st.session_state.home_attack = 1.83
    st.session_state.home_defense = 1.33
    st.session_state.home_ppg = 1.83
    st.session_state.home_games = 6
    st.session_state.home_cs = 17
    st.session_state.home_fts = 17
    st.session_state.home_xg_for = 1.95
    st.session_state.home_xg_against = 1.45
    st.session_state.home_goals5 = 7
    st.session_state.home_conceded5 = 6
    
    st.session_state.away_name = "Gil Vicente"
    st.session_state.away_attack = 1.50
    st.session_state.away_defense = 0.50
    st.session_state.away_ppg = 1.83
    st.session_state.away_games = 6
    st.session_state.away_cs = 67
    st.session_state.away_fts = 17
    st.session_state.away_xg_for = 1.30
    st.session_state.away_xg_against = 0.80
    st.session_state.away_goals5 = 6
    st.session_state.away_conceded5 = 3
    
    st.session_state.h2h_btts = 62
    st.session_state.h2h_meetings = 5

def set_example_2():
    """Example: Strong Defense vs Weak Attack"""
    st.session_state.home_name = "Team A"
    st.session_state.home_attack = 0.8
    st.session_state.home_defense = 0.6
    st.session_state.home_ppg = 1.5
    st.session_state.home_games = 10
    st.session_state.home_cs = 60
    st.session_state.home_fts = 40
    st.session_state.home_xg_for = 0.9
    st.session_state.home_xg_against = 0.7
    st.session_state.home_goals5 = 3
    st.session_state.home_conceded5 = 3
    
    st.session_state.away_name = "Team B"
    st.session_state.away_attack = 1.1
    st.session_state.away_defense = 1.8
    st.session_state.away_ppg = 1.2
    st.session_state.away_games = 10
    st.session_state.away_cs = 20
    st.session_state.away_fts = 30
    st.session_state.away_xg_for = 1.3
    st.session_state.away_xg_against = 2.0
    st.session_state.away_goals5 = 8
    st.session_state.away_conceded5 = 9
    
    st.session_state.h2h_btts = 40
    st.session_state.h2h_meetings = 3

def set_example_3():
    """Example: xG Mismatch Case"""
    st.session_state.home_name = "Team X"
    st.session_state.home_attack = 1.0
    st.session_state.home_defense = 1.5
    st.session_state.home_ppg = 1.3
    st.session_state.home_games = 12
    st.session_state.home_cs = 25
    st.session_state.home_fts = 35
    st.session_state.home_xg_for = 1.8  # High xG, low actual goals
    st.session_state.home_xg_against = 1.6
    st.session_state.home_goals5 = 7
    st.session_state.home_conceded5 = 8
    
    st.session_state.away_name = "Team Y"
    st.session_state.away_attack = 1.6
    st.session_state.away_defense = 0.7
    st.session_state.away_ppg = 1.8
    st.session_state.away_games = 12
    st.session_state.away_cs = 50
    st.session_state.away_fts = 20
    st.session_state.away_xg_for = 1.5
    st.session_state.away_xg_against = 0.5  # Low xGA, moderate actual
    st.session_state.away_goals5 = 8
    st.session_state.away_conceded5 = 4
    
    st.session_state.h2h_btts = 50
    st.session_state.h2h_meetings = 4

if __name__ == "__main__":
    main()