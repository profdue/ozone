"""
app.py - Football Predictor App with NO SLIDERS
Professional data input with context and validation
"""

import streamlit as st
import pandas as pd
from engine import PredictionEngine, TeamMetrics, MatchContext

# Set page config - MUST BE FIRST
st.set_page_config(
    page_title="Football Match Predictor Pro",
    page_icon="⚽",
    layout="wide"
)

def get_league_context(league_name: str) -> MatchContext:
    """Get league-specific context"""
    contexts = {
        "Premier League": MatchContext(
            league_avg_goals=2.7,
            league_avg_clean_sheets=0.30,
            league_avg_failed_to_score=0.30,
            home_advantage=1.15
        ),
        "La Liga": MatchContext(
            league_avg_goals=2.5,
            league_avg_clean_sheets=0.35,
            league_avg_failed_to_score=0.35,
            home_advantage=1.18
        ),
        "Bundesliga": MatchContext(
            league_avg_goals=3.0,
            league_avg_clean_sheets=0.25,
            league_avg_failed_to_score=0.25,
            home_advantage=1.10
        ),
        "Serie A": MatchContext(
            league_avg_goals=2.6,
            league_avg_clean_sheets=0.40,
            league_avg_failed_to_score=0.35,
            home_advantage=1.16
        ),
        "Ligue 1": MatchContext(
            league_avg_goals=2.4,
            league_avg_clean_sheets=0.35,
            league_avg_failed_to_score=0.40,
            home_advantage=1.17
        ),
        "Liga NOS": MatchContext(
            league_avg_goals=2.68,
            league_avg_clean_sheets=0.30,
            league_avg_failed_to_score=0.35,
            home_advantage=1.15
        ),
    }
    return contexts.get(league_name, MatchContext())

def team_input_section(team_name: str, is_home: bool, example_data: dict = None) -> dict:
    """
    Professional team input section with context and validation
    Returns dictionary of team metrics
    """
    st.subheader(f"{'🏠' if is_home else '🚗'} {team_name}")
    
    # Initialize with example data if provided
    if example_data:
        default_attack = example_data.get('attack', 1.5)
        default_defense = example_data.get('defense', 1.0)
        default_ppg = example_data.get('ppg', 1.5)
        default_games = example_data.get('games_played', 10)
        default_cs = example_data.get('clean_sheet_pct', 30)
        default_fts = example_data.get('failed_to_score_pct', 30)
        default_goals_last5 = example_data.get('goals_last5', 6)
        default_conceded_last5 = example_data.get('conceded_last5', 4)
    else:
        default_attack = 1.5
        default_defense = 1.0
        default_ppg = 1.5
        default_games = 10
        default_cs = 30
        default_fts = 30
        default_goals_last5 = 6
        default_conceded_last5 = 4
    
    # Basic stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        attack = st.number_input(
            "Goals Scored/Game",
            min_value=0.0,
            max_value=5.0,
            value=default_attack,
            step=0.1,
            key=f"{team_name}_attack"
        )
        st.caption("Higher = better attack")
    
    with col2:
        defense = st.number_input(
            "Goals Conceded/Game",
            min_value=0.0,
            max_value=5.0,
            value=default_defense,
            step=0.1,
            key=f"{team_name}_defense"
        )
        st.caption("Lower = better defense")
    
    with col3:
        ppg = st.number_input(
            "Points/Game",
            min_value=0.0,
            max_value=3.0,
            value=default_ppg,
            step=0.1,
            key=f"{team_name}_ppg"
        )
        st.caption("3.0 = perfect, 1.0 = average")
    
    with col4:
        games_played = st.number_input(
            "Games Played",
            min_value=1,
            max_value=40,
            value=default_games,
            key=f"{team_name}_games"
        )
        st.caption("For sample size adjustment")
    
    # Performance metrics with league context
    st.write("**📊 Performance Metrics**")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        clean_sheet = st.number_input(
            "Clean Sheet %",
            min_value=0,
            max_value=100,
            value=default_cs,
            key=f"{team_name}_cs"
        )
        # Context indicator
        if clean_sheet > 50:
            st.success(f"Excellent (>50%)")
        elif clean_sheet > 30:
            st.info(f"Good (30-50%)")
        else:
            st.warning(f"Below average (<30%)")
    
    with col6:
        failed_score = st.number_input(
            "Failed to Score %",
            min_value=0,
            max_value=100,
            value=default_fts,
            key=f"{team_name}_fts"
        )
        # Context indicator
        if failed_score < 20:
            st.success(f"Rarely fails (<20%)")
        elif failed_score < 40:
            st.info(f"Average (20-40%)")
        else:
            st.warning(f"Often fails (>40%)")
    
    with col7:
        btts_pct = st.number_input(
            "BTTS %",
            min_value=0,
            max_value=100,
            value=50,
            key=f"{team_name}_btts"
        )
        # Context indicator
        if btts_pct > 60:
            st.success(f"High BTTS rate")
        elif btts_pct > 40:
            st.info(f"Average BTTS rate")
        else:
            st.warning(f"Low BTTS rate")
    
    # Recent form (Last 5 matches)
    st.write("**📈 Recent Form (Last 5 Matches)**")
    col8, col9 = st.columns(2)
    
    with col8:
        goals_last5 = st.number_input(
            "Goals Scored",
            min_value=0,
            max_value=30,
            value=default_goals_last5,
            key=f"{team_name}_goals_last5"
        )
        avg_last5 = goals_last5 / 5
        st.caption(f"Average: {avg_last5:.2f}/game")
    
    with col9:
        conceded_last5 = st.number_input(
            "Goals Conceded",
            min_value=0,
            max_value=30,
            value=default_conceded_last5,
            key=f"{team_name}_conceded_last5"
        )
        avg_conceded_last5 = conceded_last5 / 5
        st.caption(f"Average: {avg_conceded_last5:.2f}/game")
    
    return {
        'attack': attack,
        'defense': defense,
        'ppg': ppg,
        'games_played': games_played,
        'clean_sheet': clean_sheet,
        'failed_score': failed_score,
        'btts_pct': btts_pct,
        'goals_last5': goals_last5,
        'conceded_last5': conceded_last5,
        'team_name': team_name
    }

def main():
    st.title("⚽ Football Match Predictor Pro")
    st.markdown("### Professional football prediction with data-driven insights")
    
    # Initialize engine
    engine = PredictionEngine()
    
    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ League Settings")
        
        # League selection
        league = st.selectbox(
            "Select League",
            ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Liga NOS", "Other"]
        )
        
        # Set league context
        engine.context = get_league_context(league)
        
        # Display league info
        st.info(f"**League:** {league}")
        st.caption(f"• Avg Goals: {engine.context.league_avg_goals}")
        st.caption(f"• Avg Clean Sheets: {engine.context.league_avg_clean_sheets:.0%}")
        st.caption(f"• Avg Failed to Score: {engine.context.league_avg_failed_to_score:.0%}")
        
        # Example matches
        st.header("📋 Example Matches")
        
        if st.button("Load Vitória vs Gil Vicente"):
            st.session_state.home_data = {
                'attack': 1.83, 'defense': 1.33, 'ppg': 1.83,
                'games_played': 6, 'clean_sheet': 17, 'failed_score': 17,
                'goals_last5': 7, 'conceded_last5': 6
            }
            st.session_state.away_data = {
                'attack': 1.50, 'defense': 0.50, 'ppg': 1.83,
                'games_played': 6, 'clean_sheet': 67, 'failed_score': 17,
                'goals_last5': 6, 'conceded_last5': 3
            }
            st.session_state.h2h_btts = 62
            st.rerun()
        
        if st.button("Load Pisa vs Parma"):
            st.session_state.home_data = {
                'attack': 0.14, 'defense': 0.57, 'ppg': 0.86,
                'games_played': 7, 'clean_sheet': 57, 'failed_score': 14,
                'goals_last5': 3, 'conceded_last5': 1
            }
            st.session_state.away_data = {
                'attack': 0.50, 'defense': 1.17, 'ppg': 0.83,
                'games_played': 6, 'clean_sheet': 33, 'failed_score': 50,
                'goals_last5': 3, 'conceded_last5': 5
            }
            st.session_state.h2h_btts = 100
            st.rerun()
        
        if st.button("Load Modena vs Catanzaro"):
            st.session_state.home_data = {
                'attack': 1.86, 'defense': 0.43, 'ppg': 2.43,
                'games_played': 7, 'clean_sheet': 57, 'failed_score': 14,
                'goals_last5': 9, 'conceded_last5': 2
            }
            st.session_state.away_data = {
                'attack': 1.00, 'defense': 1.00, 'ppg': 1.00,
                'games_played': 6, 'clean_sheet': 33, 'failed_score': 50,
                'goals_last5': 6, 'conceded_last5': 6
            }
            st.session_state.h2h_btts = 100
            st.rerun()
    
    # Main prediction interface
    st.header("📊 Match Prediction")
    
    # Two columns for team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Home team input
        if 'home_data' in st.session_state:
            home_data = team_input_section(
                "Home Team",
                is_home=True,
                example_data=st.session_state.home_data
            )
        else:
            home_data = team_input_section("Home Team", is_home=True)
    
    with col2:
        # Away team input
        if 'away_data' in st.session_state:
            away_data = team_input_section(
                "Away Team",
                is_home=False,
                example_data=st.session_state.away_data
            )
        else:
            away_data = team_input_section("Away Team", is_home=False)
    
    # H2H and additional inputs
    st.header("🤝 Head-to-Head History")
    col_h2h1, col_h2h2 = st.columns(2)
    
    with col_h2h1:
        h2h_btts = st.number_input(
            "H2H BTTS %",
            min_value=0,
            max_value=100,
            value=st.session_state.get('h2h_btts', 50),
            key="h2h_btts"
        )
        if h2h_btts > 70:
            st.success("Very high BTTS rate in H2H")
        elif h2h_btts < 30:
            st.warning("Very low BTTS rate in H2H")
    
    with col_h2h2:
        h2h_meetings = st.number_input(
            "Number of H2H Meetings",
            min_value=0,
            max_value=50,
            value=5,
            key="h2h_meetings"
        )
        if h2h_meetings < 3:
            st.warning("Limited H2H data")
    
    # Create TeamMetrics objects
    home_metrics = TeamMetrics(
        attack_strength=home_data['attack'],
        defense_strength=home_data['defense'],
        goals_scored_last_5=home_data['goals_last5'],
        goals_conceded_last_5=home_data['conceded_last5'],
        clean_sheet_pct=home_data['clean_sheet']/100,
        failed_to_score_pct=home_data['failed_score']/100,
        ppg=home_data['ppg'],
        btts_pct=home_data['btts_pct']/100,
        games_played=home_data['games_played']
    )
    
    away_metrics = TeamMetrics(
        attack_strength=away_data['attack'],
        defense_strength=away_data['defense'],
        goals_scored_last_5=away_data['goals_last5'],
        goals_conceded_last_5=away_data['conceded_last5'],
        clean_sheet_pct=away_data['clean_sheet']/100,
        failed_to_score_pct=away_data['failed_score']/100,
        ppg=away_data['ppg'],
        btts_pct=away_data['btts_pct']/100,
        games_played=away_data['games_played']
    )
    
    # Prediction button
    if st.button("🎯 Generate Professional Predictions", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and generating predictions..."):
            
            # Get all predictions
            result_pred = engine.predict_match_result(home_metrics, away_metrics)
            over_under_pred = engine.predict_over_under(home_metrics, away_metrics)
            btts_pred = engine.predict_btts(home_metrics, away_metrics, h2h_btts/100)
            expected_goals = engine.calculate_expected_goals(home_metrics, away_metrics)
            patterns = engine.analyze_matchup_patterns(home_metrics, away_metrics)
            
            # Display professional results
            st.success("✅ Professional Analysis Complete!")
            
            # Sample size warning if needed
            if (home_data['games_played'] < 6 or away_data['games_played'] < 6):
                st.warning("⚠️ **Sample Size Alert**: One or both teams have limited data (<6 games). Predictions may be less reliable.")
            
            # Show form analysis
            st.info(f"📊 **Form Analysis**: Home: {result_pred['form_factors']['home']}x | Away: {result_pred['form_factors']['away']}x")
            
            # Results dashboard
            st.header("📈 Prediction Dashboard")
            
            # Row 1: Main predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🏆 Match Result")
                pred = result_pred['prediction'].value
                st.metric("Prediction", pred, delta=None)
                
                # Probability bars
                home_prob = result_pred['probabilities']['home_win']
                draw_prob = result_pred['probabilities']['draw']
                away_prob = result_pred['probabilities']['away_win']
                
                st.progress(home_prob, f"Home: {home_prob:.1%}")
                st.progress(draw_prob, f"Draw: {draw_prob:.1%}")
                st.progress(away_prob, f"Away: {away_prob:.1%}")
            
            with col2:
                st.subheader("📊 Over/Under 2.5")
                pred = over_under_pred['prediction'].value
                conf = over_under_pred['confidence']
                st.metric("Prediction", f"{pred}")
                st.metric("Confidence", f"{conf}")
                st.metric("Expected Goals", f"{over_under_pred['expected_goals']}")
                
                # Probability bars
                over_prob = over_under_pred['probabilities']['over']
                under_prob = over_under_pred['probabilities']['under']
                
                st.progress(over_prob, f"Over: {over_prob:.1%}")
                st.progress(under_prob, f"Under: {under_prob:.1%}")
            
            with col3:
                st.subheader("⚔️ Both Teams to Score")
                pred = btts_pred['prediction'].value
                conf = btts_pred['confidence']
                st.metric("Prediction", f"{pred}")
                st.metric("Confidence", f"{conf}")
                
                # Probability bars
                yes_prob = btts_pred['probabilities']['btts_yes']
                no_prob = btts_pred['probabilities']['btts_no']
                
                st.progress(yes_prob, f"Yes: {yes_prob:.1%}")
                st.progress(no_prob, f"No: {no_prob:.1%}")
            
            # Row 2: Expected goals breakdown
            st.subheader("🎯 Expected Goals Analysis")
            eg_col1, eg_col2, eg_col3, eg_col4 = st.columns(4)
            
            with eg_col1:
                st.metric("Home Expected", f"{expected_goals['home_goals']}", 
                         delta=f"vs {home_data['attack']:.2f} avg")
            with eg_col2:
                st.metric("Away Expected", f"{expected_goals['away_goals']}", 
                         delta=f"vs {away_data['attack']:.2f} avg")
            with eg_col3:
                st.metric("Total Expected", f"{expected_goals['total_goals']}")
            with eg_col4:
                league_diff = expected_goals['total_goals'] - engine.context.league_avg_goals
                st.metric("vs League Avg", f"{league_diff:+.2f}")
            
            # Patterns and insights
            if patterns:
                st.subheader("🔍 Detected Patterns")
                cols = st.columns(2)
                for idx, pattern in enumerate(patterns):
                    with cols[idx % 2]:
                        st.info(f"• {pattern}")
            
            # Statistical comparison
            st.subheader("📋 Team Comparison")
            
            comparison_data = {
                'Metric': [
                    'Goals Scored/Game', 'Goals Conceded/Game', 'Clean Sheet %',
                    'Failed to Score %', 'Points/Game', 'Recent Goals/Game',
                    'Games Played'
                ],
                home_data['team_name']: [
                    f"{home_data['attack']:.2f}",
                    f"{home_data['defense']:.2f}",
                    f"{home_data['clean_sheet']}%",
                    f"{home_data['failed_score']}%",
                    f"{home_data['ppg']:.2f}",
                    f"{home_data['goals_last5']/5:.2f}",
                    f"{home_data['games_played']}"
                ],
                away_data['team_name']: [
                    f"{away_data['attack']:.2f}",
                    f"{away_data['defense']:.2f}",
                    f"{away_data['clean_sheet']}%",
                    f"{away_data['failed_score']}%",
                    f"{away_data['ppg']:.2f}",
                    f"{away_data['goals_last5']/5:.2f}",
                    f"{away_data['games_played']}"
                ],
                'League Average': [
                    f"{engine.context.league_avg_goals/2:.2f}",
                    f"{engine.context.league_avg_goals/2:.2f}",
                    f"{engine.context.league_avg_clean_sheets*100:.0f}%",
                    f"{engine.context.league_avg_failed_to_score*100:.0f}%",
                    "~1.50",
                    "N/A",
                    "N/A"
                ]
            }
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Professional insights
            st.subheader("💡 Professional Insights")
            
            insights = []
            
            # Form insights
            home_form = result_pred['form_factors']['home']
            away_form = result_pred['form_factors']['away']
            
            if home_form >= 1.1:
                insights.append(f"**{home_data['team_name']} in strong form**: Scoring {home_data['goals_last5']/5:.2f}/game recently vs {home_data['attack']:.2f} season average")
            elif home_form <= 0.9:
                insights.append(f"**{home_data['team_name']} in poor form**: Scoring {home_data['goals_last5']/5:.2f}/game recently vs {home_data['attack']:.2f} season average")
            
            if away_form >= 1.1:
                insights.append(f"**{away_data['team_name']} in strong form**: Scoring {away_data['goals_last5']/5:.2f}/game recently vs {away_data['attack']:.2f} season average")
            elif away_form <= 0.9:
                insights.append(f"**{away_data['team_name']} in poor form**: Scoring {away_data['goals_last5']/5:.2f}/game recently vs {away_data['attack']:.2f} season average")
            
            # Defense insights
            if away_data['defense'] <= 0.6:
                insights.append(f"**Strong away defense**: {away_data['team_name']} concedes only {away_data['defense']:.2f} goals/game away")
            if home_data['defense'] <= 0.6:
                insights.append(f"**Strong home defense**: {home_data['team_name']} concedes only {home_data['defense']:.2f} goals/game at home")
            
            # Clean sheet insights
            if away_data['clean_sheet'] >= 50:
                insights.append(f"**High away clean sheet rate**: {away_data['team_name']} keeps clean sheets in {away_data['clean_sheet']}% of away matches")
            if home_data['clean_sheet'] >= 50:
                insights.append(f"**High home clean sheet rate**: {home_data['team_name']} keeps clean sheets in {home_data['clean_sheet']}% of home matches")
            
            # Expected goals insights
            if expected_goals['total_goals'] < engine.context.league_avg_goals * 0.7:
                insights.append(f"**Very low scoring expected**: {expected_goals['total_goals']:.2f} total goals vs {engine.context.league_avg_goals} league average")
            elif expected_goals['total_goals'] < engine.context.league_avg_goals:
                insights.append(f"**Below average scoring**: {expected_goals['total_goals']:.2f} total goals vs {engine.context.league_avg_goals} league average")
            elif expected_goals['total_goals'] > engine.context.league_avg_goals * 1.3:
                insights.append(f"**High scoring expected**: {expected_goals['total_goals']:.2f} total goals vs {engine.context.league_avg_goals} league average")
            
            # H2H insights
            if h2h_btts > 70:
                insights.append(f"**Strong H2H BTTS trend**: Both teams scored in {h2h_btts}% of past meetings")
            elif h2h_btts < 30:
                insights.append(f"**Weak H2H BTTS trend**: Both teams scored in only {h2h_btts}% of past meetings")
            
            # Display insights in columns
            if insights:
                insight_cols = st.columns(2)
                for idx, insight in enumerate(insights):
                    with insight_cols[idx % 2]:
                        st.write(f"• {insight}")

if __name__ == "__main__":
    main()
