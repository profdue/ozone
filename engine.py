"""
Simple football prediction engine
No external dependencies - pure Python
"""

def predict_match(home_stats, away_stats, h2h_stats=None):
    """
    Predict match outcome with simple rules
    home_stats: dict with 'ppg', 'goals_scored', 'goals_conceded', 'clean_sheets'
    away_stats: same structure for away team
    """
    
    # 1. Calculate expected goals
    expected_home_goals = (home_stats['goals_scored'] + away_stats['goals_conceded']) / 2
    expected_away_goals = (away_stats['goals_scored'] + home_stats['goals_conceded']) / 2
    expected_total = expected_home_goals + expected_away_goals
    
    # 2. Match result prediction
    home_strength = home_stats['ppg'] + (home_stats['goals_scored'] / 3)
    away_strength = away_stats['ppg'] + (away_stats['goals_scored'] / 3)
    
    if home_strength - away_strength > 0.5:
        result = "Home Win"
    elif away_strength - home_strength > 0.5:
        result = "Away Win"
    else:
        result = "Draw"
    
    # 3. Over/Under prediction
    if expected_total > 2.8:
        over_under = "Over 2.5 (High)"
    elif expected_total > 2.4:
        over_under = "Over 2.5"
    elif expected_total < 2.0:
        over_under = "Under 2.5 (High)"
    elif expected_total < 2.3:
        over_under = "Under 2.5"
    else:
        over_under = "Around 2.5 goals"
    
    # 4. BTTS prediction
    if (home_stats['goals_scored'] > 1.0 and away_stats['goals_scored'] > 1.0 and
        (home_stats['goals_conceded'] > 1.0 or away_stats['goals_conceded'] > 1.0)):
        btts = "Yes"
    elif home_stats['clean_sheets'] > 60 or away_stats['clean_sheets'] > 60:
        btts = "No (Clean Sheet likely)"
    else:
        btts = "No"
    
    # 5. Confidence
    confidence = 50
    ppg_diff = abs(home_stats['ppg'] - away_stats['ppg'])
    if ppg_diff > 0.8:
        confidence += 25
    elif ppg_diff > 0.4:
        confidence += 15
    
    if home_stats['goals_conceded'] < 0.8 or away_stats['goals_conceded'] < 0.8:
        confidence += 10
    
    confidence = min(90, max(40, confidence))
    
    return {
        'result': result,
        'over_under': over_under,
        'btts': btts,
        'confidence': confidence,
        'expected_total_goals': round(expected_total, 1),
        'expected_home_goals': round(expected_home_goals, 1),
        'expected_away_goals': round(expected_away_goals, 1)
    }


# Example test function
def test_prediction():
    """Test with example data"""
    home = {'ppg': 1.83, 'goals_scored': 1.83, 'goals_conceded': 1.33, 'clean_sheets': 17}
    away = {'ppg': 1.83, 'goals_scored': 1.50, 'goals_conceded': 0.50, 'clean_sheets': 67}
    
    return predict_match(home, away)
