import streamlit as st
from engine import PredictionEngine

st.set_page_config(page_title="⚽ Ozone Predictor", layout="wide")

st.title("⚽ Football Match Predictor")
st.markdown("Enter team stats to get predictions for ANY league")

# Create two columns for team inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_name = st.text_input("Team Name", "Vitória Guimarães", key="home_name")
    
    st.markdown("**Home Stats (when playing at home)**")
    home_ppg = st.number_input("Points Per Game", min_value=0.0, max_value=3.0, value=1.83, step=0.1, key="home_ppg")
    home_goals_scored = st.number_input("Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.83, step=0.1, key="home_gs")
    home_goals_conceded = st.number_input("Avg Goals Conceded", min_value=0.0, max_value=5.0, value=1.33, step=0.1, key="home_gc")
    home_clean_sheets = st.slider("Clean Sheet %", 0, 100, 17, key="home_cs")

with col2:
    st.subheader("✈️ Away Team")
    away_name = st.text_input("Team Name", "Gil Vicente", key="away_name")
    
    st.markdown("**Away Stats (when playing away)**")
    away_ppg = st.number_input("Points Per Game", min_value=0.0, max_value=3.0, value=1.83, step=0.1, key="away_ppg")
    away_goals_scored = st.number_input("Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.50, step=0.1, key="away_gs")
    away_goals_conceded = st.number_input("Avg Goals Conceded", min_value=0.0, max_value=5.0, value=0.50, step=0.1, key="away_gc")
    away_clean_sheets = st.slider("Clean Sheet %", 0, 100, 67, key="away_cs")

# H2H Section (optional)
with st.expander("⚔️ Head-to-Head History (Optional)"):
    col_a, col_b = st.columns(2)
    with col_a:
        h2h_meetings = st.number_input("Total Meetings", min_value=0, value=21, step=1)
        h2h_home_wins = st.number_input(f"{home_name} Wins", min_value=0, value=9, step=1)
    with col_b:
        h2h_away_wins = st.number_input(f"{away_name} Wins", min_value=0, value=7, step=1)
        h2h_avg_goals = st.number_input("Avg Goals per Match", min_value=0.0, value=2.95, step=0.1)

# Predict Button
if st.button("🎯 Generate Prediction", type="primary"):
    # Prepare stats
    home_stats = {
        'ppg': home_ppg,
        'goals_scored': home_goals_scored,
        'goals_conceded': home_goals_conceded,
        'clean_sheets': home_clean_sheets
    }
    
    away_stats = {
        'ppg': away_ppg,
        'goals_scored': away_goals_scored,
        'goals_conceded': away_goals_conceded,
        'clean_sheets': away_clean_sheets
    }
    
    # Optional H2H stats
    h2h_stats = None
    if h2h_meetings > 0:
        h2h_stats = {
            'avg_goals': h2h_avg_goals,
            'home_win_rate': (h2h_home_wins / h2h_meetings) * 100,
            'away_win_rate': (h2h_away_wins / h2h_meetings) * 100
        }
    
    # Make prediction
    engine = PredictionEngine()
    prediction = engine.predict_match(home_stats, away_stats, h2h_stats)
    
    # Display results
    st.divider()
    st.header(f"📊 Prediction: {home_name} vs {away_name}")
    
    # Confidence indicator
    conf_color = "green" if prediction['confidence'] > 70 else "orange" if prediction['confidence'] > 55 else "red"
    st.markdown(f"**Confidence:** :{conf_color}[{prediction['confidence']}%]")
    
    # Results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏆 Match Result",
            value=prediction['result'],
            delta=f"{prediction['expected_goals']} expected goals"
        )
    
    with col2:
        st.metric(
            label="⚽ Goals",
            value=prediction['over_under']
        )
    
    with col3:
        st.metric(
            label="🎯 Both Teams Score",
            value=prediction['btts'].split(" - ")[0]
        )
    
    # Explanation
    st.divider()
    st.subheader("🤔 How this prediction was made")
    
    st.markdown(f"""
    **Key Factors Considered:**
    
    1. **Home Advantage:** {home_name} averages **{home_ppg} points per game** at home
    2. **Away Performance:** {away_name} averages **{away_ppg} points per game** away
    3. **Attack vs Defense:** 
       - {home_name} scores {home_goals_scored} goals but concedes {home_goals_conceded} at home
       - {away_name} scores {away_goals_scored} goals but concedes only {away_goals_conceded} away
    4. **Clean Sheets:** {away_name} keeps clean sheets in **{away_clean_sheets}%** of away games
    """)
    
    # Simple betting advice
    st.divider()
    st.subheader("💰 Betting Recommendation")
    
    if prediction['confidence'] > 70:
        st.success(f"**STRONG BET** - High confidence prediction")
        st.info("Consider: {prediction['result']} & {prediction['over_under']}")
    elif prediction['confidence'] > 55:
        st.warning(f"**MODERATE BET** - Reasonable confidence")
        st.info("Consider: {prediction['over_under']}")
    else:
        st.error(f"**AVOID OR SMALL BET** - Low confidence")
        st.info("Match too close to call. Consider watching instead.")

# Sidebar with examples
with st.sidebar:
    st.header("📚 Examples")
    
    st.markdown("""
    **Quick Examples:**
    
    **Defensive Away Team:**
    - Home: PPG 1.8, Scored 1.8, Conceded 1.3
    - Away: PPG 1.8, Scored 1.5, Conceded 0.5
    
    **High-Scoring Match:**
    - Home: PPG 2.0, Scored 2.5, Conceded 1.5  
    - Away: PPG 1.5, Scored 2.0, Conceded 2.0
    
    **Even Matchup:**
    - Home: PPG 1.6, Scored 1.4, Conceded 1.2
    - Away: PPG 1.5, Scored 1.3, Conceded 1.3
    """)
    
    st.divider()
    st.markdown("""
    **Stats Guide:**
    - **PPG:** Points per game (Win=3, Draw=1, Loss=0)
    - **Clean Sheet %:** Games where team didn't concede
    - Use team's **HOME** stats for home team
    - Use team's **AWAY** stats for away team
    """)

# Footer
st.divider()
st.caption("Ozone Predictor v1.0 • Works for any league • Based on FootyStats methodology")
