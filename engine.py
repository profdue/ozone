"""
engine_v2.py - Football Prediction Engine v2.0 with xG Integration
Enhanced with xG blending, sample-size aware weights, and defensive analysis
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
    """Team metrics with xG integration"""
    # Core Performance Stats
    attack_strength: float           # Actual goals scored/game
    defense_strength: float          # Actual goals conceded/game
    ppg: float                       # Points per game
    
    # xG Stats (NEW - Major Enhancement)
    xg_for: float                    # Expected goals created/game
    xg_against: float                # Expected goals conceded/game
    
    # Performance Metrics
    clean_sheet_pct: float           # Clean sheet percentage (0-1)
    failed_to_score_pct: float       # Failed to score percentage (0-1)
    btts_pct: float                  # BTTS percentage (0-1)
    
    # Recent Form (Last 5 Matches)
    goals_scored_last_5: int         # Goals scored in last 5
    goals_conceded_last_5: int       # Goals conceded in last 5
    
    # Sample Size Awareness
    games_played: int                # Total games played
    
    # Optional xG splits (for future enhancement)
    home_xg_for: Optional[float] = None
    away_xg_for: Optional[float] = None
    home_xg_against: Optional[float] = None
    away_xg_against: Optional[float] = None
    
    # Team name for debugging/display
    name: Optional[str] = None

@dataclass
class MatchContext:
    """Match context with league-specific parameters"""
    # League Averages for Normalization
    league_avg_goals: float = 2.68
    league_avg_btts: float = 0.46
    league_avg_clean_sheets: float = 0.30
    league_avg_failed_to_score: float = 0.30
    league_avg_xg: float = 1.34           # Per team (league_avg_goals/2)
    
    # Venue Multipliers
    home_advantage: float = 1.15
    away_penalty: float = 0.92

class PredictionEngineV2:
    """
    Prediction Engine v2.0 with xG Integration
    """
    
    def __init__(self, context: Optional[MatchContext] = None):
        self.context = context or MatchContext()
        
        # Enhanced thresholds with xG awareness
        self.THRESHOLDS = {
            # Defense
            'VERY_STRONG_DEFENSE': 0.6,
            'STRONG_DEFENSE': 0.8,
            'WEAK_DEFENSE': 1.4,
            
            # Attack
            'STRONG_ATTACK': 1.6,
            'WEAK_ATTACK': 1.0,
            
            # Performance
            'HIGH_CLEAN_SHEET': 0.5,
            'HIGH_FAILED_TO_SCORE': 0.4,
            
            # Form
            'GOOD_FORM': 1.1,
            'POOR_FORM': 0.9,
            
            # Sample Size
            'MIN_GAMES_RELIABLE': 6,
            
            # xG thresholds
            'XG_LUCKY_DEFENSE': 0.85,    # xGA < actual * 0.85
            'XG_UNLUCKY_ATTACK': 1.15,   # xG > actual * 1.15
        }
    
    def validate_and_adjust_metrics(self, team: TeamMetrics) -> TeamMetrics:
        """
        Validate and adjust metrics for small sample sizes
        """
        # Ensure percentages are in 0-1 range
        team.clean_sheet_pct = max(0.0, min(1.0, team.clean_sheet_pct))
        team.failed_to_score_pct = max(0.0, min(1.0, team.failed_to_score_pct))
        team.btts_pct = max(0.0, min(1.0, team.btts_pct))
        
        # Bayesian adjustment for small samples
        if team.games_played < self.THRESHOLDS['MIN_GAMES_RELIABLE']:
            adjustment_weight = team.games_played / self.THRESHOLDS['MIN_GAMES_RELIABLE']
            
            # Adjust attack strength toward league average
            team.attack_strength = (
                team.attack_strength * adjustment_weight + 
                (self.context.league_avg_goals/2) * (1 - adjustment_weight)
            )
            
            # Adjust defense strength
            team.defense_strength = (
                team.defense_strength * adjustment_weight + 
                (self.context.league_avg_goals/2) * (1 - adjustment_weight)
            )
            
            # Adjust xG stats if they exist
            if team.xg_for > 0:
                team.xg_for = (
                    team.xg_for * adjustment_weight + 
                    self.context.league_avg_xg * (1 - adjustment_weight)
                )
            if team.xg_against > 0:
                team.xg_against = (
                    team.xg_against * adjustment_weight + 
                    self.context.league_avg_xg * (1 - adjustment_weight)
                )
        
        return team
    
    def calculate_form_factor(self, team: TeamMetrics) -> float:
        """
        Compare recent performance to season average
        Returns multiplier between 0.8 and 1.2
        """
        recent_goals_pg = team.goals_scored_last_5 / 5
        season_avg = team.attack_strength
        
        if season_avg <= 0.1:
            return 1.0
        
        form_ratio = recent_goals_pg / season_avg
        bounded_form = max(0.8, min(1.2, form_ratio))
        
        # Weight based on sample size reliability
        if team.games_played < self.THRESHOLDS['MIN_GAMES_RELIABLE']:
            weight_recent = 0.6  # Small sample: trust recent form more
        else:
            weight_recent = 0.4  # Good sample: balanced approach
        
        return (1 - weight_recent) * 1.0 + weight_recent * bounded_form
    
    def _get_xg_weight(self, games_played: int) -> float:
        """Determine xG weight based on sample size"""
        if games_played < 8: return 0.3   # Small sample, trust xG less
        if games_played < 15: return 0.5  # Medium sample, equal weight
        return 0.7                         # Large sample, trust xG more
    
    def calculate_xg_adjusted_goals(self, home: TeamMetrics, away: TeamMetrics) -> Tuple[float, float]:
        """
        Blend actual goals with xG data
        """
        # Validate and adjust metrics first
        home = self.validate_and_adjust_metrics(home)
        away = self.validate_and_adjust_metrics(away)
        
        # Determine xG weight based on sample size
        xg_weight_home = self._get_xg_weight(home.games_played)
        xg_weight_away = self._get_xg_weight(away.games_played)
        
        # Home team expected to score
        home_attack_quality = (
            home.attack_strength * (1 - xg_weight_home) + 
            home.xg_for * xg_weight_home
        )
        
        away_defense_quality = (
            away.defense_strength * (1 - xg_weight_away) + 
            away.xg_against * xg_weight_away
        )
        
        expected_home = (home_attack_quality + away_defense_quality) / 2
        
        # Away team expected to score
        away_attack_quality = (
            away.attack_strength * (1 - xg_weight_away) + 
            away.xg_for * xg_weight_away
        )
        
        home_defense_quality = (
            home.defense_strength * (1 - xg_weight_home) + 
            home.xg_against * xg_weight_home
        )
        
        expected_away = (away_attack_quality + home_defense_quality) / 2
        
        # Apply form adjustments
        home_form = self.calculate_form_factor(home)
        away_form = self.calculate_form_factor(away)
        
        expected_home *= home_form
        expected_away *= away_form
        
        # Apply venue adjustments
        expected_home *= self.context.home_advantage
        expected_away *= self.context.away_penalty
        
        # Reasonable bounds
        expected_home = max(0.2, min(3.0, expected_home))
        expected_away = max(0.2, min(3.0, expected_away))
        
        return expected_home, expected_away
    
    def analyze_defensive_strength(self, team: TeamMetrics) -> Dict:
        """
        Key defensive analysis that drives many adjustments
        """
        analysis = {
            'is_very_strong': False,
            'is_strong': False,
            'is_weak': False,
            'clean_sheet_likely': False,
            'xg_better_than_actual': False,
            'xg_worse_than_actual': False
        }
        
        # Actual goals analysis
        if team.defense_strength <= self.THRESHOLDS['VERY_STRONG_DEFENSE']:
            analysis['is_very_strong'] = True
        elif team.defense_strength <= self.THRESHOLDS['STRONG_DEFENSE']:
            analysis['is_strong'] = True
        elif team.defense_strength >= self.THRESHOLDS['WEAK_DEFENSE']:
            analysis['is_weak'] = True
        
        # xG enhancement: Is defense actually better/worse than stats show?
        if team.xg_against < team.defense_strength * self.THRESHOLDS['XG_LUCKY_DEFENSE']:
            analysis['xg_better_than_actual'] = True  # Conceding less than expected
        elif team.xg_against > team.defense_strength * (1/self.THRESHOLDS['XG_LUCKY_DEFENSE']):
            analysis['xg_worse_than_actual'] = True   # Conceding more than expected
        
        # Clean sheet analysis
        if team.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            analysis['clean_sheet_likely'] = True
        
        return analysis
    
    def predict_match_result(self, home: TeamMetrics, away: TeamMetrics) -> Dict:
        """
        Predict Home Win, Draw, or Away Win with probabilities
        """
        # Validate and adjust metrics
        home = self.validate_and_adjust_metrics(home)
        away = self.validate_and_adjust_metrics(away)
        
        # 1. Calculate base probabilities from PPG comparison
        home_win_base = self._ppg_to_win_probability(home.ppg, away.ppg, is_home=True)
        away_win_base = self._ppg_to_win_probability(away.ppg, home.ppg, is_home=False)
        draw_base = 0.35  # League average draw rate
        
        # 2. Apply form adjustments
        home_form = self.calculate_form_factor(home)
        away_form = self.calculate_form_factor(away)
        
        home_win_base *= home_form
        away_win_base *= away_form
        
        # 3. Defensive strength adjustments (CRITICAL)
        away_defense_analysis = self.analyze_defensive_strength(away)
        
        if away_defense_analysis['is_very_strong']:
            # Strong away defense reduces home win chance
            reduction = 0.7 if away.games_played >= 8 else 0.85
            home_win_base *= reduction
            draw_base *= 1.2
            away_win_base *= 1.1
        
        # 4. Clean sheet adjustments
        if away.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            home_win_base *= 0.85
            draw_base += 0.08
        
        # 5. Failed to score adjustments
        if home.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            home_win_base *= 0.9
        
        # 6. xG-based finishing adjustments (NEW)
        # If home team creates chances but doesn't finish (due for goals)
        if home.xg_for > home.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            home_win_base *= 1.1
        
        # If away team lucky in defense (due to concede more)
        if away.xg_against < away.defense_strength * self.THRESHOLDS['XG_LUCKY_DEFENSE']:
            away_win_base *= 0.9
        
        # 7. Normalize probabilities
        total = home_win_base + draw_base + away_win_base
        home_prob = home_win_base / total
        draw_prob = draw_base / total
        away_prob = away_win_base / total
        
        # 8. Determine prediction
        if draw_prob > 0.4 or abs(home_prob - away_prob) < 0.1:
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
            },
            'form_factors': {
                'home': round(home_form, 2),
                'away': round(away_form, 2)
            },
            'defensive_analysis': {
                'away': away_defense_analysis
            }
        }
    
    def predict_over_under(self, home: TeamMetrics, away: TeamMetrics) -> Dict:
        """
        Predict Over or Under 2.5 total goals
        """
        # Validate and adjust metrics
        home = self.validate_and_adjust_metrics(home)
        away = self.validate_and_adjust_metrics(away)
        
        # Calculate xG-adjusted expected goals
        expected_home, expected_away = self.calculate_xg_adjusted_goals(home, away)
        total_expected = expected_home + expected_away
        
        # Base Poisson probability
        prob_over = self._poisson_over_25(total_expected)
        prob_under = 1 - prob_over
        
        # Defensive adjustments
        away_defense = self.analyze_defensive_strength(away)
        
        if away_defense['is_very_strong']:
            reduction = 0.6 if away.games_played >= 8 else 0.8
            prob_over *= reduction
            prob_under = 1 - prob_over
        
        # xG-specific adjustments (NEW)
        xg_adjustments = []
        
        # Home team creates chances but doesn't finish
        if home.xg_for > home.attack_strength * 1.2:
            prob_over *= 1.15
            if home.name:
                xg_adjustments.append(
                    f"{home.name} creates {home.xg_for:.2f} xG but scores {home.attack_strength:.2f} - due for goals"
                )
        
        # Away team allows chances but gets lucky
        if away.xg_against < away.defense_strength * 0.8:
            prob_over *= 1.1
            if away.name:
                xg_adjustments.append(
                    f"{away.name} allows {away.xg_against:.2f} xG but concedes {away.defense_strength:.2f} - defensive luck may end"
                )
        
        # Re-normalize
        prob_under = 1 - prob_over
        
        # Determine prediction and confidence
        if prob_under > 0.65:
            prediction = Prediction.UNDER_25
            confidence = self._probability_to_confidence(prob_under)
        elif prob_over > 0.65:
            prediction = Prediction.OVER_25
            confidence = self._probability_to_confidence(prob_over)
        else:
            prediction = Prediction.UNDER_25 if prob_under > prob_over else Prediction.OVER_25
            confidence = "Low"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'over': round(prob_over, 3),
                'under': round(prob_under, 3)
            },
            'expected_goals': round(total_expected, 2),
            'xg_adjustments': xg_adjustments,
            'detailed_expected': {
                'home': round(expected_home, 2),
                'away': round(expected_away, 2)
            }
        }
    
    def predict_btts(self, home: TeamMetrics, away: TeamMetrics, 
                    h2h_btts: Optional[float] = None,
                    h2h_meetings: Optional[int] = None) -> Dict:
        """
        Predict if Both Teams will Score
        """
        # Validate and adjust metrics
        home = self.validate_and_adjust_metrics(home)
        away = self.validate_and_adjust_metrics(away)
        
        # Base probability from team stats
        prob_home_scores = 1 - away.clean_sheet_pct
        prob_away_scores = 1 - home.clean_sheet_pct
        
        # Apply form adjustments
        home_form = self.calculate_form_factor(home)
        away_form = self.calculate_form_factor(away)
        
        prob_home_scores *= home_form
        prob_away_scores *= away_form
        
        # Combine for BTTS probability
        prob_btts = prob_home_scores * prob_away_scores
        
        # Adjust for failed to score
        if home.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            prob_btts *= 0.8
        if away.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            prob_btts *= 0.8
        
        # Defensive adjustment
        away_defense = self.analyze_defensive_strength(away)
        if away_defense['is_very_strong']:
            prob_btts *= 0.7
        
        # xG adjustments (NEW)
        # Home team finishing issues
        if home.xg_for > home.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 0.9  # Creates chances but doesn't finish
        
        # Away team defensive luck
        if away.xg_against < away.defense_strength * self.THRESHOLDS['XG_LUCKY_DEFENSE']:
            prob_btts *= 1.1  # Allows chances, luck may run out
        
        # H2H adjustment (MODERATE weight: 15-25%)
        if h2h_btts is not None:
            # Weight based on recency and sample size
            h2h_weight = min(0.25, 0.15 + (h2h_meetings / 20)) if h2h_meetings else 0.2
            prob_btts = (prob_btts * (1 - h2h_weight)) + (h2h_btts * h2h_weight)
        
        prob_no_btts = 1 - prob_btts
        
        # Determine prediction
        if prob_btts > 0.65:
            prediction = Prediction.BTTS_YES
            confidence = self._probability_to_confidence(prob_btts)
        elif prob_no_btts > 0.65:
            prediction = Prediction.BTTS_NO
            confidence = self._probability_to_confidence(prob_no_btts)
        else:
            prediction = Prediction.BTTS_NO if prob_no_btts > prob_btts else Prediction.BTTS_YES
            confidence = "Low"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'btts_yes': round(prob_btts, 3),
                'btts_no': round(prob_no_btts, 3)
            },
            'form_factors': {
                'home': round(home_form, 2),
                'away': round(away_form, 2)
            }
        }
    
    def calculate_expected_goals(self, home: TeamMetrics, away: TeamMetrics) -> Dict:
        """Calculate expected goals (alias for compatibility)"""
        expected_home, expected_away = self.calculate_xg_adjusted_goals(home, away)
        return {
            'home_goals': round(expected_home, 2),
            'away_goals': round(expected_away, 2),
            'total_goals': round(expected_home + expected_away, 2)
        }
    
    def analyze_matchup_patterns(self, home: TeamMetrics, away: TeamMetrics) -> List[str]:
        """
        Generate key insights about the matchup
        """
        patterns = []
        
        # Add team names for better insights
        home_name = home.name or "Home team"
        away_name = away.name or "Away team"
        
        # Form patterns
        home_form = self.calculate_form_factor(home)
        away_form = self.calculate_form_factor(away)
        
        if home_form >= self.THRESHOLDS['GOOD_FORM']:
            patterns.append(f"{home_name} in good form (scoring {home.goals_scored_last_5} in last 5)")
        elif home_form <= self.THRESHOLDS['POOR_FORM']:
            patterns.append(f"{home_name} in poor form (scoring {home.goals_scored_last_5} in last 5)")
        
        if away_form >= self.THRESHOLDS['GOOD_FORM']:
            patterns.append(f"{away_name} in good form (scoring {away.goals_scored_last_5} in last 5)")
        elif away_form <= self.THRESHOLDS['POOR_FORM']:
            patterns.append(f"{away_name} in poor form (scoring {away.goals_scored_last_5} in last 5)")
        
        # Defensive battle detection
        away_defense = self.analyze_defensive_strength(away)
        if away_defense['is_very_strong'] and home.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']:
            patterns.append("Defensive battle expected - low scoring likely")
        
        # Clean sheet patterns
        if away.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            patterns.append(f"{away_name} high clean sheet rate ({away.clean_sheet_pct:.0%})")
        
        # xG patterns (NEW)
        if home.xg_for > home.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            patterns.append(f"{home_name} creates chances ({home.xg_for:.2f} xG) but doesn't finish ({home.attack_strength:.2f} goals)")
        
        if away.xg_against < away.defense_strength * self.THRESHOLDS['XG_LUCKY_DEFENSE']:
            patterns.append(f"{away_name} defensive luck may end ({away.xg_against:.2f} xGA vs {away.defense_strength:.2f} actual)")
        
        # Parity pattern
        if abs(home.ppg - away.ppg) <= 0.2:
            patterns.append("Close matchup - draw possible")
        
        # Offensive strength patterns
        if home.attack_strength >= self.THRESHOLDS['STRONG_ATTACK']:
            patterns.append(f"{home_name} has strong attack ({home.attack_strength:.2f} goals/game)")
        
        if away.attack_strength >= self.THRESHOLDS['STRONG_ATTACK']:
            patterns.append(f"{away_name} has strong attack ({away.attack_strength:.2f} goals/game)")
        
        return patterns
    
    # ========== HELPER METHODS ==========
    
    def _ppg_to_win_probability(self, team_ppg: float, opponent_ppg: float, is_home: bool) -> float:
        """
        Convert PPG differential to win probability
        """
        base_diff = team_ppg - opponent_ppg
        
        if is_home:
            base_diff += 0.3  # Home advantage
        
        # Simplified conversion
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
        """
        Poisson probability for Over 2.5 goals
        """
        if lambda_total <= 0:
            return 0.05
        
        # P(0) + P(1) + P(2) = probability of Under 2.5
        prob_0 = math.exp(-lambda_total)
        prob_1 = lambda_total * math.exp(-lambda_total)
        prob_2 = (lambda_total ** 2) * math.exp(-lambda_total) / 2
        
        prob_under = prob_0 + prob_1 + prob_2
        prob_over = 1 - prob_under
        
        return max(0.05, min(0.95, prob_over))
    
    def _probability_to_confidence(self, probability: float) -> str:
        """
        Map probability to confidence level
        """
        if probability >= 0.75:
            return "High"
        elif probability >= 0.65:
            return "Medium"
        elif probability >= 0.55:
            return "Low"
        else:
            return "Very Low"