"""
engine.py - Pure prediction logic engine
No UI, no data input - just mathematical calculations
"""

from typing import Dict, Tuple, List, Optional
import math
from dataclasses import dataclass
from enum import Enum

class MarketType(Enum):
    MATCH_RESULT = "1X2"
    OVER_UNDER_25 = "Over/Under 2.5"
    BTTS = "Both Teams to Score"

class Prediction(Enum):
    HOME_WIN = "Home Win"
    AWAY_WIN = "Away Win"
    DRAW = "Draw"
    OVER_25 = "Over 2.5"
    UNDER_25 = "Under 2.5"
    BTTS_YES = "BTTS Yes"
    BTTS_NO = "BTTS No"

@dataclass
class TeamMetrics:
    """Pure statistical metrics for calculations"""
    # Attack metrics (goals per game)
    attack_strength: float
    # Defense metrics (goals conceded per game, lower = better)
    defense_strength: float
    # Form metrics (0-1, higher = better recent form)
    form_factor: float
    # Clean sheet percentage (0-1)
    clean_sheet_pct: float
    # Failed to score percentage (0-1)
    failed_to_score_pct: float
    # Points per game
    ppg: float
    # BTTS percentage (0-1)
    btts_pct: float

@dataclass
class MatchContext:
    """Context for the match prediction"""
    # League averages for normalization
    league_avg_goals: float = 2.68
    league_avg_btts: float = 0.46
    league_avg_home_win: float = 0.45
    # Home advantage multiplier
    home_advantage: float = 1.15
    # Away penalty multiplier
    away_penalty: float = 0.92

class PredictionEngine:
    """
    PURE PREDICTION ENGINE
    Only contains mathematical logic - no UI, no data collection
    """
    
    def __init__(self, context: Optional[MatchContext] = None):
        self.context = context or MatchContext()
        
        # Universal thresholds
        self.THRESHOLDS = {
            'STRONG_DEFENSE': 0.8,  # ≤ 0.8 goals conceded
            'VERY_STRONG_DEFENSE': 0.6,
            'WEAK_DEFENSE': 1.4,
            'STRONG_ATTACK': 1.6,
            'WEAK_ATTACK': 1.0,
            'HIGH_CLEAN_SHEET': 0.5,
            'HIGH_FAILED_TO_SCORE': 0.4,
            'SIGNIFICANT_PPG_DIFF': 0.5,
            'CLOSE_PPG': 0.2,
        }
    
    def predict_match_result(self, home: TeamMetrics, away: TeamMetrics) -> Dict:
        """
        Predict 1X2 outcome with probabilities
        """
        # Base probabilities from PPG comparison
        home_win_base = self._calculate_win_probability_from_ppg(home.ppg, away.ppg, is_home=True)
        away_win_base = self._calculate_win_probability_from_ppg(away.ppg, home.ppg, is_home=False)
        draw_base = 0.35  # League average draw rate
        
        # Adjust for defensive strength
        if away.defense_strength <= self.THRESHOLDS['VERY_STRONG_DEFENSE']:
            home_win_base *= 0.7  # Reduce home win chance by 30%
            draw_base *= 1.2
            away_win_base *= 1.1
        
        # Adjust for clean sheet probability
        if away.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            home_win_base *= 0.8
            draw_base += 0.1
        
        # Normalize probabilities
        total = home_win_base + draw_base + away_win_base
        home_prob = home_win_base / total
        draw_prob = draw_base / total
        away_prob = away_win_base / total
        
        # Determine prediction
        if abs(home_prob - away_prob) < 0.1 or draw_prob > 0.4:
            prediction = Prediction.DRAW
        elif home_prob > away_prob + 0.15:
            prediction = Prediction.HOME_WIN
        elif away_prob > home_prob + 0.15:
            prediction = Prediction.AWAY_WIN
        else:
            prediction = Prediction.DRAW
        
        return {
            'prediction': prediction,
            'probabilities': {
                'home_win': round(home_prob, 3),
                'draw': round(draw_prob, 3),
                'away_win': round(away_prob, 3)
            }
        }
    
    def predict_over_under(self, home: TeamMetrics, away: TeamMetrics) -> Dict:
        """
        Predict Over/Under 2.5 goals
        """
        # Calculate expected goals
        expected_home_goals = (home.attack_strength + away.defense_strength) / 2
        expected_away_goals = (away.attack_strength + home.defense_strength) / 2
        expected_total = expected_home_goals + expected_away_goals
        
        # Adjust for venue
        expected_home_goals *= self.context.home_advantage
        expected_away_goals *= self.context.away_penalty
        expected_total = expected_home_goals + expected_away_goals
        
        # Calculate Poisson probability
        prob_over = self._poisson_over_25(expected_total)
        prob_under = 1 - prob_over
        
        # Defensive adjustment
        if away.defense_strength <= self.THRESHOLDS['VERY_STRONG_DEFENSE']:
            prob_over *= 0.6
            prob_under = 1 - prob_over
        
        # Offensive weakness adjustment
        if home.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']:
            prob_over *= 0.8
            prob_under = 1 - prob_over
        
        # Determine prediction
        if prob_under > 0.65:
            prediction = Prediction.UNDER_25
            confidence = self._calculate_confidence(prob_under)
        elif prob_over > 0.65:
            prediction = Prediction.OVER_25
            confidence = self._calculate_confidence(prob_over)
        else:
            prediction = Prediction.UNDER_25 if prob_under > prob_over else Prediction.OVER_25
            confidence = 'Low'
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'over': round(prob_over, 3),
                'under': round(prob_under, 3)
            },
            'expected_goals': round(expected_total, 2)
        }
    
    def predict_btts(self, home: TeamMetrics, away: TeamMetrics, h2h_btts: Optional[float] = None) -> Dict:
        """
        Predict Both Teams to Score
        """
        # Base probability from team stats
        prob_home_scores = 1 - away.clean_sheet_pct
        prob_away_scores = 1 - home.clean_sheet_pct
        prob_btts = prob_home_scores * prob_away_scores
        
        # Adjust for failed to score
        if home.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            prob_btts *= 0.7
        if away.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            prob_btts *= 0.7
        
        # Defensive adjustment
        if away.defense_strength <= self.THRESHOLDS['VERY_STRONG_DEFENSE']:
            prob_btts *= 0.6
        
        # H2H adjustment if available
        if h2h_btts is not None:
            prob_btts = (prob_btts * 0.7) + (h2h_btts * 0.3)
        
        prob_no_btts = 1 - prob_btts
        
        # Determine prediction
        if prob_btts > 0.65:
            prediction = Prediction.BTTS_YES
            confidence = self._calculate_confidence(prob_btts)
        elif prob_no_btts > 0.65:
            prediction = Prediction.BTTS_NO
            confidence = self._calculate_confidence(prob_no_btts)
        else:
            prediction = Prediction.BTTS_NO if prob_no_btts > prob_btts else Prediction.BTTS_YES
            confidence = 'Low'
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'btts_yes': round(prob_btts, 3),
                'btts_no': round(prob_no_btts, 3)
            }
        }
    
    def calculate_expected_goals(self, home: TeamMetrics, away: TeamMetrics) -> Dict:
        """
        Calculate detailed expected goals
        """
        # Base calculations
        expected_home = (home.attack_strength + away.defense_strength) / 2
        expected_away = (away.attack_strength + home.defense_strength) / 2
        
        # Apply venue adjustments
        expected_home *= self.context.home_advantage
        expected_away *= self.context.away_penalty
        
        # Form adjustments
        expected_home *= home.form_factor
        expected_away *= away.form_factor
        
        # Final adjustments
        expected_home = max(0.2, min(3.0, expected_home))
        expected_away = max(0.2, min(3.0, expected_away))
        
        return {
            'home_goals': round(expected_home, 2),
            'away_goals': round(expected_away, 2),
            'total_goals': round(expected_home + expected_away, 2)
        }
    
    def analyze_matchup_patterns(self, home: TeamMetrics, away: TeamMetrics) -> List[str]:
        """
        Identify key matchup patterns
        """
        patterns = []
        
        # Defensive battle pattern
        if (away.defense_strength <= self.THRESHOLDS['VERY_STRONG_DEFENSE'] and 
            home.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']):
            patterns.append("Defensive battle - Low scoring expected")
        
        # Clean sheet pattern
        if away.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            patterns.append(f"Away team high clean sheet rate ({away.clean_sheet_pct:.0%})")
        
        # Offensive struggle pattern
        if (home.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE'] and
            away.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']):
            patterns.append("Both teams struggle to score")
        
        # Parity pattern
        if abs(home.ppg - away.ppg) <= self.THRESHOLDS['CLOSE_PPG']:
            patterns.append("Close matchup - Draw likely")
        
        return patterns
    
    # ========== PRIVATE HELPER METHODS ==========
    
    def _calculate_win_probability_from_ppg(self, team_ppg: float, opponent_ppg: float, is_home: bool) -> float:
        """Convert PPG differential to win probability"""
        base_diff = team_ppg - opponent_ppg
        
        if is_home:
            base_diff += 0.3  # Home advantage
        
        # Convert to probability (simplified)
        if base_diff > 1.0:
            return 0.55
        elif base_diff > 0.5:
            return 0.45
        elif base_diff > 0:
            return 0.40
        elif base_diff > -0.5:
            return 0.35
        else:
            return 0.30
    
    def _poisson_over_25(self, lambda_total: float) -> float:
        """Poisson probability for Over 2.5 goals"""
        if lambda_total <= 0:
            return 0.05
        
        # Simple Poisson calculation
        prob_0 = math.exp(-lambda_total)
        prob_1 = lambda_total * math.exp(-lambda_total)
        prob_2 = (lambda_total ** 2) * math.exp(-lambda_total) / 2
        
        prob_under = prob_0 + prob_1 + prob_2
        prob_over = 1 - prob_under
        
        return max(0.05, min(0.95, prob_over))
    
    def _calculate_confidence(self, probability: float) -> str:
        """Convert probability to confidence level"""
        if probability >= 0.75:
            return 'High'
        elif probability >= 0.65:
            return 'Medium'
        elif probability >= 0.55:
            return 'Low'
        else:
            return 'Very Low'
