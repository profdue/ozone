"""
app.py - User Interface and Data Input
Handles data collection, display, and user interaction
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional
from engine import PredictionEngine, TeamMetrics, MatchContext

class FootballPredictorApp:
    """
    Streamlit app for football predictions
    Handles UI, data input, and displays predictions
    """
    
    def __init__(self):
        st.set_page_config(
            page_title="Football Match Predictor",
            page_icon="⚽",
            layout="wide"
        )
        self.engine = PredictionEngine()
        
    def run(self):
        """Main app runner"""
        st.title("⚽ Football Match Predictor")
        st.markdown("### Based on FootyStats Data Analysis")
        
        # Sidebar for league settings
        with st.sidebar:
            st.header("⚙️ League Settings")
            self._setup_league_settings()
        
        # Main prediction interface
        st.header("📊 Match Prediction")
        
        # Two-column layout for team inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_metrics = self._get_team_inputs(is_home=True)
        
        with col2:
            st.subheader("🚗 Away Team")
            away_metrics = self._get_team_inputs(is_home=False)
        
        # H2H and additional inputs
        st.subheader("🤝 Head-to-Head History")
        h2h_btts = st.slider("H2H BTTS Percentage", 0, 100, 50) / 100
        
        # Prediction button
        if st.button("🎯 Generate Prediction", type="primary"):
            if home_metrics and away_metrics:
                self._display_predictions(home_metrics, away_metrics, h2h_btts)
            else:
                st.error("Please fill in all required fields for both teams.")
    
    def _setup_league_settings(self):
        """Configure league parameters in sidebar"""
        league = st.selectbox(
            "Select League",
            ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Liga NOS", "Other"]
        )
        
        # Set league context based on selection
        league_contexts = {
            "Premier League": MatchContext(league_avg_goals=2.7, home_advantage=1.15),
            "La Liga": MatchContext(league_avg_goals=2.5, home_advantage=1.18),
            "Bundesliga": MatchContext(league_avg_goals=3.0, home_advantage=1.10),
            "Serie A": MatchContext(league_avg_goals=2.6, home_advantage=1.16),
            "Ligue 1": MatchContext(league_avg_goals=2.4, home_advantage=1.17),
            "Liga NOS": MatchContext(league_avg_goals=2.68, home_advantage=1.15),
        }
        
        self.engine.context = league_contexts.get(league, MatchContext())
        
        st.info(f"League: {league}")
        st.caption(f"Avg Goals: {self.engine.context.league_avg_goals}")
    
    def _get_team_inputs(self, is_home: bool = True) -> Optional[TeamMetrics]:
        """Get team statistics inputs from user"""
        
        # Team name
        prefix = "Home" if is_home else "Away"
        team_name = st.text_input(f"{prefix} Team Name", value="Team A" if is_home else "Team B")
        
        # Create expander for detailed stats
        with st.expander(f"📈 {prefix} Team Statistics", expanded=True):
            # Attack and defense
            col1, col2 = st.columns(2)
            with col1:
                attack = st.number_input(
                    f"Goals Scored per Game",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.5 if is_home else 1.3,
                    step=0.1,
                    key=f"{prefix}_attack"
                )
            with col2:
                defense = st.number_input(
                    f"Goals Conceded per Game",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.3 if is_home else 1.0,
                    step=0.1,
                    key=f"{prefix}_defense"
                )
            
            # Form and PPG
            col3, col4 = st.columns(2)
            with col3:
                ppg = st.number_input(
                    f"Points per Game",
                    min_value=0.0,
                    max_value=3.0,
                    value=1.8 if is_home else 1.8,
                    step=0.1,
                    key=f"{prefix}_ppg"
                )
            with col4:
                form = st.slider(
                    f"Recent Form (1 = poor, 1.2 = excellent)",
                    min_value=0.8,
                    max_value=1.2,
                    value=1.0,
                    step=0.05,
                    key=f"{prefix}_form"
                )
            
            # Clean sheets and failed to score
            col5, col6 = st.columns(2)
            with col5:
                clean_sheet = st.slider(
                    f"Clean Sheet %",
                    min_value=0,
                    max_value=100,
                    value=20 if is_home else 60,
                    key=f"{prefix}_cs"
                ) / 100
            with col6:
                failed_score = st.slider(
                    f"Failed to Score %",
                    min_value=0,
                    max_value=100,
                    value=20 if is_home else 20,
                    key=f"{prefix}_fts"
                ) / 100
            
            # BTTS percentage
            btts_pct = st.slider(
                f"BTTS % in matches",
                min_value=0,
                max_value=100,
                value=60 if is_home else 40,
                key=f"{prefix}_btts"
            ) / 100
        
        # Return TeamMetrics object
        return TeamMetrics(
            attack_strength=attack,
            defense_strength=defense,
            form_factor=form,
            clean_sheet_pct=clean_sheet,
            failed_to_score_pct=failed_score,
            ppg=ppg,
            btts_pct=btts_pct
        )
    
    def _display_predictions(self, home: TeamMetrics, away: TeamMetrics, h2h_btts: float):
        """Display all predictions in a nice format"""
        
        # Calculate predictions
        result_pred = self.engine.predict_match_result(home, away)
        over_under_pred = self.engine.predict_over_under(home, away)
        btts_pred = self.engine.predict_btts(home, away, h2h_btts)
        expected_goals = self.engine.calculate_expected_goals(home, away)
        patterns = self.engine.analyze_matchup_patterns(home, away)
        
        # Header
        st.success("✅ Predictions Generated!")
        
        # Display in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Match Result")
            pred = result_pred['prediction'].value
            home_prob = result_pred['probabilities']['home_win']
            draw_prob = result_pred['probabilities']['draw']
            away_prob = result_pred['probabilities']['away_win']
            
            st.metric("Prediction", pred)
            st.progress(home_prob, text=f"Home: {home_prob:.1%}")
            st.progress(draw_prob, text=f"Draw: {draw_prob:.1%}")
            st.progress(away_prob, text=f"Away: {away_prob:.1%}")
        
        with col2:
            st.subheader("📈 Over/Under 2.5")
            pred = over_under_pred['prediction'].value
            conf = over_under_pred['confidence']
            over_prob = over_under_pred['probabilities']['over']
            under_prob = over_under_pred['probabilities']['under']
            exp_goals = over_under_pred['expected_goals']
            
            st.metric("Prediction", f"{pred} ({conf})")
            st.metric("Expected Goals", exp_goals)
            st.progress(over_prob, text=f"Over: {over_prob:.1%}")
            st.progress(under_prob, text=f"Under: {under_prob:.1%}")
        
        with col3:
            st.subheader("⚔️ Both Teams to Score")
            pred = btts_pred['prediction'].value
            conf = btts_pred['confidence']
            yes_prob = btts_pred['probabilities']['btts_yes']
            no_prob = btts_pred['probabilities']['btts_no']
            
            st.metric("Prediction", f"{pred} ({conf})")
            st.progress(yes_prob, text=f"Yes: {yes_prob:.1%}")
            st.progress(no_prob, text=f"No: {no_prob:.1%}")
        
        # Expected goals breakdown
        st.subheader("🎯 Expected Goals Breakdown")
        eg_col1, eg_col2, eg_col3 = st.columns(3)
        
        with eg_col1:
            st.metric("Home Expected Goals", expected_goals['home_goals'])
        with eg_col2:
            st.metric("Away Expected Goals", expected_goals['away_goals'])
        with eg_col3:
            st.metric("Total Expected Goals", expected_goals['total_goals'])
        
        # Matchup patterns
        if patterns:
            st.subheader("🔍 Matchup Patterns Detected")
            for pattern in patterns:
                st.info(f"• {pattern}")
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        summary_data = {
            "Metric": ["Defense Quality", "Attack Strength", "Clean Sheet %", "Failed to Score %", "PPG"],
            "Home": [
                f"{home.defense_strength:.2f}",
                f"{home.attack_strength:.2f}",
                f"{home.clean_sheet_pct:.0%}",
                f"{home.failed_to_score_pct:.0%}",
                f"{home.ppg:.2f}"
            ],
            "Away": [
                f"{away.defense_strength:.2f}",
                f"{away.attack_strength:.2f}",
                f"{away.clean_sheet_pct:.0%}",
                f"{away.failed_to_score_pct:.0%}",
                f"{away.ppg:.2f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Key Insights
        st.subheader("💡 Key Insights")
        
        insights = []
        
        # Defensive strength insight
        if away.defense_strength <= 0.6:
            insights.append("Away team has **very strong defense** (≤ 0.6 goals conceded/game)")
        
        # Clean sheet insight
        if away.clean_sheet_pct >= 0.5:
            insights.append(f"Away team keeps clean sheets in **{away.clean_sheet_pct:.0%}** of matches")
        
        # Parity insight
        if abs(home.ppg - away.ppg) <= 0.2:
            insights.append("Close matchup - teams have similar points per game")
        
        # Expected goals insight
        if expected_goals['total_goals'] < 2.0:
            insights.append("**Very low scoring** match expected")
        elif expected_goals['total_goals'] < 2.5:
            insights.append("**Low scoring** match expected")
        
        for insight in insights:
            st.write(f"• {insight}")
    
    def load_example_data(self):
        """Load example data for demonstration"""
        st.sidebar.subheader("📋 Example Data")
        
        if st.sidebar.button("Load Vitória vs Gil Vicente Example"):
            # This would load the actual data from our analysis
            st.session_state.home_attack = 1.83
            st.session_state.home_defense = 1.33
            st.session_state.home_ppg = 1.83
            st.session_state.home_cs = 17
            st.session_state.home_fts = 17
            
            st.session_state.away_attack = 1.50
            st.session_state.away_defense = 0.50
            st.session_state.away_ppg = 1.83
            st.session_state.away_cs = 67
            st.session_state.away_fts = 17
            
            st.rerun()

def main():
    """Main entry point"""
    app = FootballPredictorApp()
    app.run()

if __name__ == "__main__":
    main()
