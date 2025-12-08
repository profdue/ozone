class PredictionEngine:
    """
    Simple prediction engine that works for ANY team/league
    Just give it the stats, it gives predictions
    """
    
    @staticmethod
    def predict_match(home_stats, away_stats, h2h_stats=None):
        """
        home_stats: dict with home team's HOME stats
        away_stats: dict with away team's AWAY stats
        h2h_stats: optional dict with head-to-head history
        
        Returns: dict with predictions
        """
        
        # 1. MATCH RESULT (1X2) - Simple logic
        result = PredictionEngine._predict_result(home_stats, away_stats)
        
        # 2. OVER/UNDER 2.5 GOALS
        over_under = PredictionEngine._predict_over_under(home_stats, away_stats)
        
        # 3. BOTH TEAMS TO SCORE
        btts = PredictionEngine._predict_btts(home_stats, away_stats, h2h_stats)
        
        # 4. CONFIDENCE SCORE
        confidence = PredictionEngine._calculate_confidence(home_stats, away_stats, result)
        
        return {
            'result': result,
            'over_under': over_under,
            'btts': btts,
            'confidence': confidence,
            'expected_goals': PredictionEngine._calculate_expected_goals(home_stats, away_stats)
        }
    
    @staticmethod
    def _predict_result(home, away):
        """Simple: Compare home strength vs away strength"""
        home_strength = home.get('ppg', 0) + (home.get('goals_scored', 0) / 2)
        away_strength = away.get('ppg', 0) + (away.get('goals_scored', 0) / 2)
        
        diff = home_strength - away_strength
        
        if diff > 0.5:
            return "Home Win"
        elif diff < -0.5:
            return "Away Win"
        else:
            return "Draw"
    
    @staticmethod
    def _predict_over_under(home, away):
        """Will there be more or less than 2.5 goals?"""
        home_attack = home.get('goals_scored', 0)
        away_attack = away.get('goals_scored', 0)
        home_defense = home.get('goals_conceded', 0)
        away_defense = away.get('goals_conceded', 0)
        
        # Simple average
        expected_total = (home_attack + away_attack + home_defense + away_defense) / 2
        
        if expected_total > 2.8:
            return "Over 2.5 (High)"
        elif expected_total > 2.4:
            return "Over 2.5"
        elif expected_total < 2.0:
            return "Under 2.5 (High)"
        elif expected_total < 2.3:
            return "Under 2.5"
        else:
            return "Around 2.5 goals"
    
    @staticmethod
    def _predict_btts(home, away, h2h=None):
        """Will both teams score? Yes/No"""
        # Check if teams are likely to score
        home_likely_to_score = home.get('goals_scored', 0) > 1.0
        away_likely_to_score = away.get('goals_scored', 0) > 1.0
        
        # Check if teams are likely to concede
        home_likely_to_concede = home.get('goals_conceded', 0) > 1.0
        away_likely_to_concede = away.get('goals_conceded', 0) > 1.0
        
        # Simple rules
        if home_likely_to_score and away_likely_to_score:
            if home_likely_to_concede or away_likely_to_concede:
                return "Yes - Both teams likely to score"
        
        # Check clean sheets (if provided)
        home_clean_sheets = home.get('clean_sheets', 50)  # default 50%
        away_clean_sheets = away.get('clean_sheets', 50)
        
        if home_clean_sheets > 60 or away_clean_sheets > 60:
            return "No - One team keeps clean sheets"
        
        return "No - Low scoring expected"
    
    @staticmethod
    def _calculate_confidence(home, away, result_prediction):
        """How confident are we? 0-100%"""
        confidence = 50  # Start at average
        
        # Add confidence for clear differences
        ppg_diff = abs(home.get('ppg', 0) - away.get('ppg', 0))
        if ppg_diff > 0.8:
            confidence += 20
        elif ppg_diff > 0.4:
            confidence += 10
            
        # Add for defensive strength
        if home.get('goals_conceded', 0) < 0.8 or away.get('goals_conceded', 0) < 0.8:
            confidence += 10
            
        # Cap at 90, minimum 30
        return min(90, max(30, confidence))
    
    @staticmethod
    def _calculate_expected_goals(home, away):
        """Calculate expected total goals"""
        return round(
            (home.get('goals_scored', 0) + away.get('goals_conceded', 0) +
             away.get('goals_scored', 0) + home.get('goals_conceded', 0)) / 2,
            1
        )

# Example usage (for testing)
if __name__ == "__main__":
    # Sample data - this is what you'll input in Streamlit
    vitória_home_stats = {
        'ppg': 1.83,
        'goals_scored': 1.83,
        'goals_conceded': 1.33,
        'clean_sheets': 17  # percentage
    }
    
    gil_away_stats = {
        'ppg': 1.83,
        'goals_scored': 1.50,
        'goals_conceded': 0.50,
        'clean_sheets': 67
    }
    
    engine = PredictionEngine()
    prediction = engine.predict_match(vitória_home_stats, gil_away_stats)
    
    print("Prediction:", prediction)
