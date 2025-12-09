"""
Football Match Predictor - Single File App
No separate engine.py needed
Works on Streamlit Cloud
"""

import streamlit as st
import pandas as pd

# ========== PREDICTION ENGINE ==========
def predict_match(home_stats, away_stats):
    """
    Simple prediction engine that works for any team
    """
    # Calculate expected goals
    home_expected = (home_stats['goals_scored'] + away_stats['goals_conceded']) / 2
    away_expected = (away_stats['goals_scored'] + home_stats['goals_conceded']) / 2
    total_expected = home_expected + away_expected
    
    # 1. Match Result
    home_strength = home_stats['ppg'] + (home_stats['goals_scored'] * 0.3)
    away_strength = away_stats['ppg'] + (away_stats['goals_scored'] * 0.3)
    
    if home_strength - away_strength > 0.5:
        result = "Home Win"
    elif away_strength - home_strength > 0.5:
        result = "Away Win"
    else:
        result = "Draw"
    
    # 2. Over/Under
    if total_expected > 2.7:
        over_under = "Over 2.5"
    elif total_expected < 2.2:
        over_under = "Under 2.5"
    else:
        over_under = "Around 2.5"
    
    # 3. BTTS
    if (home_stats['goals_scored'] > 1.2 and away_stats['goals_scored'] > 1.2 and
        (home_stats['goals_conceded'] > 1.0 or away_stats['goals_conceded'] > 1.0)):
        btts = "Yes"
    elif home_stats['clean_sheets'] > 55 or away_stats['clean_sheets'] > 55:
        btts = "No"
    else:
        btts = "No (Leaning)"
    
    # 4. Confidence
    confidence = 50
    ppg_diff = abs(home_stats['ppg'] - away_stats['ppg'])
    if ppg_diff > 0.7:
        confidence += 20
    elif ppg_diff > 0.3:
        confidence += 10
    
    if home_stats['goals_conceded'] < 0.9 or away_stats['goals_conceded'] < 0.9:
        confidence += 10
    
    confidence = min(85, max(45, confidence))
    
    return {
        'result': result,
        'over_under': over_under,
        'btts': btts,
        'confidence': confidence,
        'total_goals': round(total_expected, 1),
        'home_goals': round(home_expected, 1),
        'away_goals': round(away_expected, 1)
    }

# ========== STREAMLIT APP ==========
def main():
    st.set_page_config(
        page_title="Football Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    # Title
    st.title("⚽ Football Match Predictor")
    st.markdown("Enter team statistics to get predictions for any league")
    
    # Two columns for team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_name = st.text_input("Team Name", "Vitória Guimarães", key="home")
        
        st.markdown("**Home Form Stats**")
        home_ppg = st.slider("Points Per Game", 0.0, 3.0, 1.83, 0.1, key="h_ppg")
        home_scored = st.slider("Avg Goals Scored", 0.0, 4.0, 1.83, 0.1, key="h_gs")
        home_conceded = st.slider("Avg Goals Conceded", 0.0, 4.0, 1.33, 0.1, key="h_gc")
        home_cs = st.slider("Clean Sheet %", 0, 100, 17, key="h_cs")
    
    with col2:
        st.subheader("✈️ Away Team")
        away_name = st.text_input("Team Name", "Gil Vicente", key="away")
        
        st.markdown("**Away Form Stats**")
        away_ppg = st.slider("Points Per Game", 0.0, 3.0, 1.83, 0.1, key="a_ppg")
        away_scored = st.slider("Avg Goals Scored", 0.0, 4.0, 1.50, 0.1, key="a_gs")
        away_conceded = st.slider("Avg Goals Conceded", 0.0, 4.0, 0.50, 0.1, key="a_gc")
        away_cs = st.slider("Clean Sheet %", 0, 100, 67, key="a_cs")
    
    st.divider()
    
    # Stats summary
    if st.button("📊 Show Stats Summary", type="secondary"):
        stats_df = pd.DataFrame({
            'Statistic': ['PPG', 'Goals Scored', 'Goals Conceded', 'Clean Sheet %'],
            home_name: [home_ppg, home_scored, home_conceded, f"{home_cs}%"],
            away_name: [away_ppg, away_scored, away_conceded, f"{away_cs}%"],
            'Difference': [
                f"{home_ppg - away_ppg:+.2f}",
                f"{home_scored - away_scored:+.2f}",
                f"{home_conceded - away_conceded:+.2f}",
                f"{home_cs - away_cs:+d}%"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    # Prediction button
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        # Prepare data
        home_stats = {
            'ppg': home_ppg,
            'goals_scored': home_scored,
            'goals_conceded': home_conceded,
            'clean_sheets': home_cs
        }
        
        away_stats = {
            'ppg': away_ppg,
            'goals_scored': away_scored,
            'goals_conceded': away_conceded,
            'clean_sheets': away_cs
        }
        
        # Get prediction
        pred = predict_match(home_stats, away_stats)
        
        # Display results
        st.success("✅ Prediction Generated!")
        
        # Results in columns
        r1, r2, r3, r4 = st.columns(4)
        
        with r1:
            st.metric("Match Result", pred['result'], f"{pred['confidence']}% conf")
        
        with r2:
            st.metric("Total Goals", pred['over_under'], f"Exp: {pred['total_goals']}")
        
        with r3:
            st.metric("Both Teams Score", pred['btts'])
        
        with r4:
            st.metric("Expected Score", f"{pred['home_goals']}-{pred['away_goals']}")
        
        # Analysis
        st.divider()
        st.subheader("📈 Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown(f"""
            **Key Factors:**
            
            • **Home Advantage:** {home_name} averages **{home_ppg} PPG** at home
            
            • **Away Defense:** {away_name} concedes only **{away_conceded}** goals away
            
            • **Clean Sheets:** {away_name} keeps clean sheets in **{away_cs}%** of away games
            
            • **Goal Expectancy:** {pred['home_goals']} - {pred['away_goals']}
            """)
        
        with analysis_col2:
            # Betting recommendation
            if pred['confidence'] > 70:
                rec = "✅ Strong Bet"
                color = "green"
            elif pred['confidence'] > 55:
                rec = "⚠️ Moderate Bet"
                color = "orange"
            else:
                rec = "❌ Avoid/Low Stake"
                color = "red"
            
            st.markdown(f"""
            **Betting Advice:**
            
            **{rec}**
            
            Consider betting on:
            • **{pred['result']}**
            • **{pred['over_under']} goals**
            
            Confidence: **{pred['confidence']}%**
            """)
        
        # Create a simple chart
        chart_data = pd.DataFrame({
            'Team': [home_name, away_name],
            'Attack': [home_scored, away_scored],
            'Defense': [3 - home_conceded, 3 - away_conceded]  # Invert so higher = better
        })
        
        st.bar_chart(chart_data.set_index('Team'))
    
    # Sidebar with examples
    with st.sidebar:
        st.header("📚 Quick Examples")
        
        st.markdown("""
        **Copy these examples:**
        
        **Defensive Battle:**
        ```
        Home: PPG 1.8, Scored 1.4, Conceded 0.9, CS 35%
        Away: PPG 1.6, Scored 1.1, Conceded 0.8, CS 45%
        ```
        
        **High Scoring:**
        ```
        Home: PPG 2.1, Scored 2.3, Conceded 1.6, CS 15%
        Away: PPG 1.7, Scored 1.9, Conceded 1.8, CS 10%
        ```
        
        **Even Match:**
        ```
        Home: PPG 1.5, Scored 1.3, Conceded 1.2, CS 25%
        Away: PPG 1.4, Scored 1.2, Conceded 1.3, CS 20%
        ```
        """)
        
        st.divider()
        
        st.markdown("""
        **Where to find stats:**
        
        1. **FootyStats.org** (free)
        2. **SofaScore.com**
        3. **WhoScored.com**
        4. **League websites**
        
        Use **HOME** stats for home team  
        Use **AWAY** stats for away team
        """)
    
    # Footer
    st.divider()
    st.caption("Ozone Predictor • No API required • Works for any football league")

# Run the app
if __name__ == "__main__":
    main()
