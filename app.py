"""
Football Match Predictor - No pandas version
Works on Streamlit Cloud with Python 3.13
"""

import streamlit as st

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
    
    # Stats summary (without pandas)
    if st.button("📊 Show Stats Summary", type="secondary"):
        st.markdown(f"""
        ### 📈 Stats Comparison
        
        | Statistic | {home_name} | {away_name} | Difference |
        |-----------|-------------|-------------|------------|
        | **PPG** | {home_ppg} | {away_ppg} | **{home_ppg - away_ppg:+.2f}** |
        | **Goals Scored** | {home_scored} | {away_scored} | **{home_scored - away_scored:+.2f}** |
        | **Goals Conceded** | {home_conceded} | {away_conceded} | **{home_conceded - away_conceded:+.2f}** |
        | **Clean Sheet %** | {home_cs}% | {away_cs}% | **{home_cs - away_cs:+d}%** |
        """)
    
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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Match Result", pred['result'], f"{pred['confidence']}% conf")
        
        with col2:
            st.metric("Total Goals", pred['over_under'], f"Exp: {pred['total_goals']}")
        
        with col3:
            st.metric("Both Teams Score", pred['btts'])
        
        with col4:
            st.metric("Expected Score", f"{pred['home_goals']}-{pred['away_goals']}")
        
        # Analysis
        st.divider()
        st.subheader("📈 Analysis")
        
        # Create a simple visual comparison
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            **Key Factors:**
            
            • **Home Advantage:** {home_name} averages **{home_ppg} PPG** at home
            
            • **Away Defense:** {away_name} concedes only **{away_conceded}** goals away
            
            • **Clean Sheets:** {away_name} keeps clean sheets in **{away_cs}%** of away games
            
            • **Goal Expectancy:** {pred['home_goals']} - {pred['away_goals']}
            
            **Prediction Logic:**
            - Home Strength: {home_stats['ppg']} + ({home_stats['goals_scored']} × 0.3) = {home_stats['ppg'] + home_stats['goals_scored'] * 0.3:.2f}
            - Away Strength: {away_stats['ppg']} + ({away_stats['goals_scored']} × 0.3) = {away_stats['ppg'] + away_stats['goals_scored'] * 0.3:.2f}
            - Strength Difference: {home_stats['ppg'] + home_stats['goals_scored'] * 0.3 - (away_stats['ppg'] + away_stats['goals_scored'] * 0.3):.2f}
            """)
        
        with col_b:
            # Betting recommendation
            if pred['confidence'] > 70:
                rec = "✅ Strong Bet"
                color = "green"
                emoji = "💰"
            elif pred['confidence'] > 55:
                rec = "⚠️ Moderate Bet"
                color = "orange"
                emoji = "💡"
            else:
                rec = "❌ Avoid/Low Stake"
                color = "red"
                emoji = "⚡"
            
            st.markdown(f"""
            **{emoji} Betting Advice**
            
            **{rec}**
            
            **Recommended Bets:**
            1. **Match Result:** {pred['result']}
            2. **Total Goals:** {pred['over_under']}
            3. **BTTS:** {pred['btts']}
            
            **Confidence Level:** {pred['confidence']}%
            
            **Expected Value Bets:**
            - Home Win odds should be > 2.0
            - Under 2.5 odds should be > 1.8
            - BTTS No odds should be > 1.7
            """)
        
        # Simple bar chart using Streamlit's native chart
        st.divider()
        st.subheader("📊 Team Comparison")
        
        # Create comparison data
        comparison_data = {
            'Attack Strength': [home_scored, away_scored],
            'Defense Strength': [3 - home_conceded, 3 - away_conceded],
            'Clean Sheet %': [home_cs / 100, away_cs / 100]
        }
        
        # Show comparison
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            st.progress(home_scored / 4, text=f"{home_name}: {home_scored} goals")
            st.progress(away_scored / 4, text=f"{away_name}: {away_scored} goals")
            st.caption("Attack (Goals/Game)")
        
        with col_y:
            st.progress((3 - home_conceded) / 3, text=f"{home_name}: {home_conceded} conceded")
            st.progress((3 - away_conceded) / 3, text=f"{away_name}: {away_conceded} conceded")
            st.caption("Defense (Lower = Better)")
        
        with col_z:
            st.progress(home_cs / 100, text=f"{home_name}: {home_cs}%")
            st.progress(away_cs / 100, text=f"{away_name}: {away_cs}%")
            st.caption("Clean Sheet %")
    
    # Sidebar with examples
    with st.sidebar:
        st.header("📚 Quick Examples")
        
        example = st.selectbox(
            "Load Example:",
            ["Select example", "Defensive Battle", "High Scoring Match", "Even Matchup", "Your Example Match"]
        )
        
        if example == "Defensive Battle":
            st.session_state['home_ppg'] = 1.8
            st.session_state['h_gs'] = 1.4
            st.session_state['h_gc'] = 0.9
            st.session_state['h_cs'] = 35
            
            st.session_state['away_ppg'] = 1.6
            st.session_state['a_gs'] = 1.1
            st.session_state['a_gc'] = 0.8
            st.session_state['a_cs'] = 45
            
            st.rerun()
        
        elif example == "High Scoring Match":
            st.session_state['home_ppg'] = 2.1
            st.session_state['h_gs'] = 2.3
            st.session_state['h_gc'] = 1.6
            st.session_state['h_cs'] = 15
            
            st.session_state['away_ppg'] = 1.7
            st.session_state['a_gs'] = 1.9
            st.session_state['a_gc'] = 1.8
            st.session_state['a_cs'] = 10
            
            st.rerun()
        
        elif example == "Even Matchup":
            st.session_state['home_ppg'] = 1.5
            st.session_state['h_gs'] = 1.3
            st.session_state['h_gc'] = 1.2
            st.session_state['h_cs'] = 25
            
            st.session_state['away_ppg'] = 1.4
            st.session_state['a_gs'] = 1.2
            st.session_state['a_gc'] = 1.3
            st.session_state['a_cs'] = 20
            
            st.rerun()
        
        st.divider()
        
        st.markdown("""
        **How to Use:**
        1. Enter team names
        2. Set stats for each team
        3. Click "Generate Prediction"
        4. Get betting advice
        
        **Stats Guide:**
        - **PPG:** Points per game (3=win, 1=draw, 0=loss)
        - **Clean Sheet %:** Games without conceding
        - Use **HOME** stats for home team
        - Use **AWAY** stats for away team
        """)
        
        st.divider()
        
        st.markdown("""
        **Data Sources:**
        - FootyStats.org (free)
        - SofaScore.com
        - WhoScored.com
        - League websites
        """)
    
    # Footer
    st.divider()
    st.caption("⚽ Ozone Predictor • No dependencies • Works on any device")

# Run the app
if __name__ == "__main__":
    main()
