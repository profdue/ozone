"""
engine.py - Football Prediction Engine
Uses PPG directly for form calculations
"""

from typing import Dict
import math
from dataclasses import dataclass

@dataclass
class TeamMetrics:
    """Team statistics - simple and clean"""
    goals_scored: float      # Goals scored per game
    goals_conceded: float    # Goals conceded per game  
    ppg: float               # Points per game (last 5 matches)
    clean_sheet_pct: float   # Clean sheet percentage (0-1)
    failed_score_pct: float = 0.0  # Failed to score percentage (0-1)

class PredictionEngine:
    """
    Main prediction engine
    Converts PPG to form factor automatically
    """
    
    def __init__(self):
        # League context
        self.league_avg_ppg = 1.33  # Average points per game
        self.league_avg_goals = 2.68  # Average goals per match
        
    def predict(self, home: TeamMetrics, away: TeamMetrics) -> Dict:
        """
        Generate all predictions for a match
        """
        # Convert PPG to form factors
        home_form = self._calculate_form_factor(home.ppg)
        away_form = self._calculate_form_factor(away.ppg)
        
        # Calculate expected goals
        expected_goals = self._calculate_expected_goals(home, away, home_form, away_form)
        
        # Generate predictions
        predictions = {
            'expected_goals': expected_goals,
            'over_under': self._predict_over_under(expected_goals['total']),
            'btts': self._predict_btts(expected_goals['home'], expected_goals['away']),
            'match_result': self._predict_match_result(expected_goals['home'], expected_goals['away']),
            'form_analysis': {
                'home_form': round(home_form, 2),
                'away_form': round(away_form, 2),
                'home_ppg': home.ppg,
                'away_ppg': away.ppg
            }
        }
        
        return predictions
    
    def _calculate_form_factor(self, ppg: float) -> float:
        """
        Convert PPG to form factor (0.8-1.2 scale)
        Automatically calculated - user doesn't need to know
        """
        if ppg <= 0:
            return 0.8
        
        # Scale PPG to 0.8-1.2 range
        # Average PPG (1.33) maps to 1.0
        # 0 PPG maps to 0.8, 3.0 PPG maps to 1.2
        
        if ppg <= self.league_avg_ppg:
            # Below average: scale from 0.8 to 1.0
            ratio = ppg / self.league_avg_ppg
            form_factor = 0.8 + (ratio * 0.2)
        else:
            # Above average: scale from 1.0 to 1.2
            ratio = (ppg - self.league_avg_ppg) / (3.0 - self.league_avg_ppg)
            form_factor = 1.0 + (ratio * 0.2)
        
        # Ensure within bounds
        return max(0.8, min(1.2, round(form_factor, 2)))
    
    def _calculate_expected_goals(self, home: TeamMetrics, away: TeamMetrics, 
                                 home_form: float, away_form: float) -> Dict:
        """
        Calculate expected goals for both teams
        """
        # Base expected goals
        home_expected = (home.goals_scored + away.goals_conceded) / 2
        away_expected = (away.goals_scored + home.goals_conceded) / 2
        
        # Apply home advantage (15% boost for home team)
        home_expected *= 1.15
        
        # Apply form factors
        home_expected *= home_form
        away_expected *= away_form
        
        # Apply defensive strength adjustments
        if away.goals_conceded < 0.8:  # Strong defense
            home_expected *= 0.8
        
        if home.goals_conceded < 0.8:  # Strong defense
            away_expected *= 0.8
        
        # Ensure reasonable bounds
        home_expected = max(0.2, min(3.0, home_expected))
        away_expected = max(0.2, min(3.0, away_expected))
        
        return {
            'home': round(home_expected, 2),
            'away': round(away_expected, 2),
            'total': round(home_expected + away_expected, 2)
        }
    
    def _predict_over_under(self, total_goals: float) -> Dict:
        """
        Predict Over/Under 2.5 goals
        """
        # Simple probability calculation
        if total_goals > 3.0:
            over_prob = 0.75
        elif total_goals > 2.5:
            over_prob = 0.65
        elif total_goals > 2.0:
            over_prob = 0.45
        elif total_goals > 1.5:
            over_prob = 0.35
        else:
            over_prob = 0.25
        
        # Determine prediction
        prediction = "Over 2.5" if over_prob > 0.5 else "Under 2.5"
        
        return {
            'over': round(over_prob, 3),
            'under': round(1 - over_prob, 3),
            'prediction': prediction,
            'confidence': self._get_confidence_level(max(over_prob, 1 - over_prob))
        }
    
    def _predict_btts(self, home_goals: float, away_goals: float) -> Dict:
        """
        Predict Both Teams to Score
        """
        # Probability each team scores (using Poisson)
        prob_home_scores = 1 - math.exp(-home_goals)
        prob_away_scores = 1 - math.exp(-away_goals)
        
        # Combined probability for BTTS
        btts_prob = prob_home_scores * prob_away_scores
        
        # Determine prediction
        prediction = "BTTS Yes" if btts_prob > 0.5 else "BTTS No"
        
        return {
            'yes': round(btts_prob, 3),
            'no': round(1 - btts_prob, 3),
            'prediction': prediction,
            'confidence': self._get_confidence_level(max(btts_prob, 1 - btts_prob))
        }
    
    def _predict_match_result(self, home_goals: float, away_goals: float) -> Dict:
        """
        Predict match result probabilities
        """
        goal_diff = home_goals - away_goals
        
        # Simple probability assignment based on goal difference
        if goal_diff > 1.0:
            home_win, draw, away_win = 0.55, 0.25, 0.20
        elif goal_diff > 0.5:
            home_win, draw, away_win = 0.45, 0.30, 0.25
        elif goal_diff > 0:
            home_win, draw, away_win = 0.40, 0.35, 0.25
        elif goal_diff > -0.5:
            home_win, draw, away_win = 0.35, 0.35, 0.30
        elif goal_diff > -1.0:
            home_win, draw, away_win = 0.25, 0.30, 0.45
        else:
            home_win, draw, away_win = 0.20, 0.25, 0.55
        
        # Determine prediction
        if home_win > away_win and home_win > draw:
            prediction = "Home Win"
        elif away_win > home_win and away_win > draw:
            prediction = "Away Win"
        else:
            prediction = "Draw"
        
        return {
            'home_win': round(home_win, 3),
            'draw': round(draw, 3),
            'away_win': round(away_win, 3),
            'prediction': prediction,
            'confidence': self._get_confidence_level(max(home_win, draw, away_win))
        }
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        Convert probability to confidence level
        """
        if probability >= 0.70:
            return "High"
        elif probability >= 0.60:
            return "Medium"
        elif probability >= 0.55:
            return "Low"
        else:
            return "Very Low"
