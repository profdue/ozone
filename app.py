"""
Football Predictor Pro v2.0 with SESSION STATE TRACKING
Streamlit Cloud Compatible - No File System Access Needed
"""

import streamlit as st
import pandas as pd
import math
import json
from datetime import datetime
import uuid
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# ========== ENGINE CLASSES ==========

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

# ========== SESSION STATE TRACKING SYSTEM ==========

class PredictionTracker:
    """
    Session-based tracking system for predictions and results
    (No file system access needed - Streamlit Cloud compatible)
    """
    
    def __init__(self):
        # Initialize session state if not exists
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        
        if 'results' not in st.session_state:
            st.session_state.results = []
    
    def save_prediction(self, 
                       home_team: str, 
                       away_team: str, 
                       match_date: str,
                       predictions: Dict,
                       inputs: Dict) -> str:
        """
        Save a prediction to session state
        Returns prediction_id for later reference
        """
        # Create unique ID
        prediction_id = str(uuid.uuid4())[:8]
        
        # Convert predictions to serializable format
        serializable_predictions = self._convert_predictions_for_json(predictions)
        
        # Prepare data
        prediction_data = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "match_info": {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "prediction_date": datetime.now().strftime("%Y-%m-%d")
            },
            "inputs": inputs,
            "predictions": serializable_predictions,
            "result": None,
            "result_added": False
        }
        
        # Save to session state
        st.session_state.predictions.append(prediction_data)
        
        # Also add to results tracker
        self._add_to_results_tracker(prediction_id, home_team, away_team, match_date)
        
        return prediction_id
    
    def _convert_predictions_for_json(self, predictions: Dict) -> Dict:
        """Convert prediction objects to serializable format"""
        serializable = {}
        
        for key, value in predictions.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts
                serializable[key] = self._convert_dict_for_json(value)
            elif isinstance(value, (Prediction, Enum)):
                # Convert enums to strings
                serializable[key] = value.value
            elif hasattr(value, '__dict__'):
                # Convert objects to dicts
                serializable[key] = self._convert_dict_for_json(value.__dict__)
            else:
                serializable[key] = value
        
        return serializable
    
    def _convert_dict_for_json(self, data: Dict) -> Dict:
        """Recursively convert dict to JSON-serializable format"""
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._convert_dict_for_json(value)
            elif isinstance(value, list):
                result[key] = [self._convert_for_json(item) for item in value]
            elif isinstance(value, (Prediction, Enum)):
                result[key] = value.value
            elif hasattr(value, '__dict__'):
                result[key] = self._convert_dict_for_json(value.__dict__)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        
        return result
    
    def _convert_for_json(self, item):
        """Convert any item to JSON-serializable format"""
        if isinstance(item, dict):
            return self._convert_dict_for_json(item)
        elif isinstance(item, list):
            return [self._convert_for_json(i) for i in item]
        elif isinstance(item, (Prediction, Enum)):
            return item.value
        elif hasattr(item, '__dict__'):
            return self._convert_dict_for_json(item.__dict__)
        elif isinstance(item, datetime):
            return item.isoformat()
        else:
            return item
    
    def _add_to_results_tracker(self, prediction_id: str, home: str, away: str, match_date: str):
        """Add prediction to results list in session state"""
        new_entry = {
            'prediction_id': prediction_id,
            'match_date': match_date,
            'home_team': home,
            'away_team': away,
            'result_entered': 'NO',
            'actual_result': '',
            'actual_score': '',
            'actual_btts': '',
            'actual_over_under': '',
            'notes': ''
        }
        
        # Add to results list
        st.session_state.results.append(new_entry)
    
    def enter_result(self, 
                    prediction_id: str,
                    result: str,  # 1/X/2
                    score: str,   # "2-1"
                    btts: str,    # "Yes"/"No"
                    over_under: str,  # "Over"/"Under"
                    notes: str = "") -> bool:
        """
        Enter match result for a prediction
        Returns success status
        """
        try:
            # Update results in session state
            for res in st.session_state.results:
                if res['prediction_id'] == prediction_id:
                    res['result_entered'] = 'YES'
                    res['actual_result'] = result
                    res['actual_score'] = score
                    res['actual_btts'] = btts
                    res['actual_over_under'] = over_under
                    res['notes'] = notes
                    break
            
            # Also update prediction in session state
            for pred in st.session_state.predictions:
                if pred['prediction_id'] == prediction_id:
                    pred['result'] = {
                        'actual_result': result,
                        'actual_score': score,
                        'actual_btts': btts,
                        'actual_over_under': over_under,
                        'notes': notes,
                        'result_date': datetime.now().isoformat()
                    }
                    pred['result_added'] = True
                    break
            
            return True
            
        except Exception as e:
            st.error(f"Error saving result: {e}")
            return False
    
    def get_pending_results(self) -> pd.DataFrame:
        """Get predictions waiting for results"""
        if 'results' not in st.session_state or not st.session_state.results:
            return pd.DataFrame()
        
        pending = [r for r in st.session_state.results if r['result_entered'] == 'NO']
        
        if not pending:
            return pd.DataFrame()
        
        df = pd.DataFrame(pending)
        
        # Add days until match
        if 'match_date' in df.columns:
            df['days_until_match'] = df['match_date'].apply(
                lambda x: (pd.to_datetime(x) - pd.Timestamp.today()).days
                if pd.notna(x) and str(x) != 'nan' else None
            )
        
        return df
    
    def get_completed_matches(self) -> pd.DataFrame:
        """Get matches with results entered"""
        if 'results' not in st.session_state or not st.session_state.results:
            return pd.DataFrame()
        
        completed = [r for r in st.session_state.results if r['result_entered'] == 'YES']
        
        if not completed:
            return pd.DataFrame()
        
        return pd.DataFrame(completed)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if 'results' not in st.session_state or not st.session_state.results:
            return {}
        
        completed = [r for r in st.session_state.results if r['result_entered'] == 'YES']
        
        if not completed:
            return {}
        
        # Initialize stats
        stats = {
            'total_matches': len(completed),
            'last_updated': datetime.now().isoformat(),
            'by_market': {},
            'by_pattern': {},
            'overall': {}
        }
        
        # Calculate results distribution
        result_counts = {}
        for res in completed:
            result = res.get('actual_result', '')
            if result and result != '':
                result_counts[result] = result_counts.get(result, 0) + 1
        
        if result_counts:
            stats['by_market']['result'] = {
                'results_distribution': result_counts,
                'total': len(completed)
            }
        
        return stats
    
    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict]:
        """Get a specific prediction by ID"""
        if 'predictions' not in st.session_state or not st.session_state.predictions:
            return None
        
        for pred in st.session_state.predictions:
            if pred['prediction_id'] == prediction_id:
                return pred
        
        return None
    
    def get_recent_predictions(self, limit: int = 10) -> pd.DataFrame:
        """Get recent predictions"""
        if 'predictions' not in st.session_state or not st.session_state.predictions:
            return pd.DataFrame()
        
        predictions = []
        
        # Sort by timestamp (most recent first)
        sorted_predictions = sorted(
            st.session_state.predictions, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )[:limit]
        
        for pred in sorted_predictions:
            predictions.append({
                'prediction_id': pred['prediction_id'],
                'match': f"{pred['match_info']['home_team']} vs {pred['match_info']['away_team']}",
                'date': pred['match_info']['match_date'],
                'prediction_date': pred['match_info'].get('prediction_date', ''),
                'result_added': pred.get('result_added', False)
            })
        
        return pd.DataFrame(predictions)

# ========== PREDICTION ENGINE v2.0 ==========

class PredictionEngineV2:
    """
    Prediction Engine v2.0 with xG Integration - COMPLETE FIXED VERSION
    Now with Pattern Detection Signals
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
            'HIGH_CLEAN_SHEET': 0.45,      # Lowered from 0.50 for better detection
            'HIGH_FAILED_TO_SCORE': 0.35,  # Lowered from 0.40 for better detection
            
            # Form
            'EXCELLENT_FORM': 1.15,        # Added for better form detection
            'GOOD_FORM': 1.05,
            'AVERAGE_FORM': 0.95,
            'POOR_FORM': 0.85,
            'VERY_POOR_FORM': 0.70,
            
            # Sample Size
            'MIN_GAMES_RELIABLE': 6,
            
            # xG thresholds
            'XG_LUCKY_DEFENSE': 0.85,      # xGA < actual * 0.85
            'XG_UNLUCKY_ATTACK': 1.15,     # xG > actual * 1.15
            'XG_OVERPERFORMER': 0.85,      # actual > xG * 1.15 (scores more than creates)
            
            # Pattern thresholds
            'XG_SIGNIFICANT_DIFF': 0.3,    # Significant xG difference
            'WEAK_DEFENSE_THRESHOLD': 1.4, # What constitutes a weak defense
        }
    
    # ========== PATTERN DETECTION SYSTEM ==========
    
    def detect_high_confidence_patterns(self, home: TeamMetrics, away: TeamMetrics) -> List[Dict]:
        """
        Identify our 3 HIGH ACCURACY PATTERNS from the betting strategy
        Returns list of detected patterns with visual signals
        """
        patterns_detected = []
        
        # Get defensive analysis
        home_def = self.analyze_defensive_strength(home)
        away_def = self.analyze_defensive_strength(away)
        
        # Calculate xG differences
        home_xg_diff = home.xg_for - home.attack_strength
        away_xg_diff = away.xg_for - away.attack_strength
        
        # PATTERN 1: DEFENSIVE BATTLE 🔥
        # Both teams have "Defense BETTER than stats show"
        if home_def['xg_better_than_actual'] and away_def['xg_better_than_actual']:
            patterns_detected.append({
                'name': 'DEFENSIVE BATTLE 🔥',
                'pattern': 1,
                'bet': 'UNDER 2.5 & BTTS NO',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'Both {home.name} and {away.name} conceding LESS than expected',
                'validation': '100% accuracy in sample',
                'pattern_type': 'defensive_battle',
                'stake': 'MAX BET (2x normal)',
                'additional_bets': ['0-0 or 1-0 Correct Score', 'Draw']
            })
        
        # PATTERN 2: REGRESSION EXPLOSION 🔥
        # Team "due for goals" + opponent has "weak defense"
        
        # Home due for goals + Away weak defense
        if (home_xg_diff > self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and away_def['is_weak']) or \
           (home.xg_for > home.attack_strength * 1.2 and away.defense_strength >= self.THRESHOLDS['WEAK_DEFENSE_THRESHOLD']):
            patterns_detected.append({
                'name': 'REGRESSION EXPLOSION 🔥',
                'pattern': 2,
                'bet': 'OVER 2.5 & BTTS YES',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{home.name} due for goals (+{home_xg_diff:.2f} xG diff) + {away.name} weak defense ({away.defense_strength:.2f} conceded)',
                'validation': '100% accuracy in sample',
                'pattern_type': 'regression_explosion',
                'stake': 'MAX BET (2x normal)',
                'additional_bets': [f'{home.name} Over 1.5 Team Goals', f'{home.name} to Win']
            })
        
        # Away due for goals + Home weak defense
        if (away_xg_diff > self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and home_def['is_weak']) or \
           (away.xg_for > away.attack_strength * 1.2 and home.defense_strength >= self.THRESHOLDS['WEAK_DEFENSE_THRESHOLD']):
            patterns_detected.append({
                'name': 'REGRESSION EXPLOSION 🔥',
                'pattern': 2,
                'bet': 'OVER 2.5 & BTTS YES',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{away.name} due for goals (+{away_xg_diff:.2f} xG diff) + {home.name} weak defense ({home.defense_strength:.2f} conceded)',
                'validation': '100% accuracy in sample',
                'pattern_type': 'regression_explosion',
                'stake': 'MAX BET (2x normal)',
                'additional_bets': [f'{away.name} Over 1.5 Team Goals', f'{away.name} to Win']
            })
        
        # PATTERN 3: REGRESSION SUPPRESSION 🔥
        # Team "overperforming xG" + opponent has "defense better than stats"
        
        # Home overperforming + Away strong defense
        if (home.attack_strength > home.xg_for * 1.2 and away_def['xg_better_than_actual']) or \
           (home_xg_diff < -self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and away_def['xg_better_than_actual']):
            patterns_detected.append({
                'name': 'REGRESSION SUPPRESSION 🔥',
                'pattern': 3,
                'bet': 'UNDER 2.5 + Team Under 1.5',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{home.name} overperforming xG (-{abs(home_xg_diff):.2f} xG diff) + {away.name} strong defense',
                'validation': '100% accuracy in sample',
                'pattern_type': 'regression_suppression',
                'stake': 'STRONG BET (1.5x normal)',
                'additional_bets': [f'{home.name} Under 1.5 Goals', 'Draw']
            })
        
        # Away overperforming + Home strong defense
        if (away.attack_strength > away.xg_for * 1.2 and home_def['xg_better_than_actual']) or \
           (away_xg_diff < -self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and home_def['xg_better_than_actual']):
            patterns_detected.append({
                'name': 'REGRESSION SUPPRESSION 🔥',
                'pattern': 3,
                'bet': 'UNDER 2.5 + Team Under 1.5',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{away.name} overperforming xG (-{abs(away_xg_diff):.2f} xG diff) + {home.name} strong defense',
                'validation': '100% accuracy in sample',
                'pattern_type': 'regression_suppression',
                'stake': 'STRONG BET (1.5x normal)',
                'additional_bets': [f'{away.name} Under 1.5 Goals', 'Draw']
            })
        
        return patterns_detected
    
    def get_pattern_based_advice(self, patterns: List[Dict]) -> Dict:
        """
        Generate betting advice based on detected patterns
        """
        if not patterns:
            return {
                'primary_bet': None,
                'stake': 'REDUCED (0.5x normal)',
                'confidence': 'Low - No patterns detected',
                'advice': 'Proceed with caution. Use standard engine predictions.'
            }
        
        # Count pattern types
        pattern_counts = {}
        for pattern in patterns:
            p_type = pattern['pattern_type']
            pattern_counts[p_type] = pattern_counts.get(p_type, 0) + 1
        
        # Determine strongest pattern
        if 'defensive_battle' in pattern_counts:
            return {
                'primary_bet': 'UNDER 2.5 & BTTS NO',
                'stake': 'MAX BET (2x normal)',
                'confidence': 'HIGHEST - Defensive Battle',
                'advice': 'Bet heavily on low scoring. Both defenses stronger than stats show.'
            }
        elif 'regression_explosion' in pattern_counts:
            return {
                'primary_bet': 'OVER 2.5 & BTTS YES',
                'stake': 'MAX BET (2x normal)',
                'confidence': 'HIGH - Regression Explosion',
                'advice': 'Attack due to regress upwards + weak defense = high scoring.'
            }
        elif 'regression_suppression' in pattern_counts:
            return {
                'primary_bet': 'UNDER 2.5',
                'stake': 'STRONG BET (1.5x normal)',
                'confidence': 'HIGH - Regression Suppression',
                'advice': 'Overperforming attack meets strong defense = regression down.'
            }
        
        return {
            'primary_bet': None,
            'stake': 'NORMAL (1x)',
            'confidence': 'Medium - Mixed patterns',
            'advice': 'Multiple patterns detected. Review carefully.'
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
        Returns multiplier between 0.7 and 1.3 (wider range for extreme cases)
        """
        recent_goals_pg = team.goals_scored_last_5 / 5
        season_avg = team.attack_strength
        
        if season_avg <= 0.1:
            return 1.0
        
        form_ratio = recent_goals_pg / season_avg
        
        # FIXED: Better handling of extreme form cases
        if form_ratio >= 1.5:      # Scoring 50%+ more recently
            bounded_form = 1.3
        elif form_ratio >= 1.3:    # Scoring 30%+ more recently
            bounded_form = 1.2
        elif form_ratio >= 1.15:   # Scoring 15%+ more recently
            bounded_form = 1.1
        elif form_ratio <= 0.5:    # Scoring 50%+ less recently
            bounded_form = 0.7
        elif form_ratio <= 0.7:    # Scoring 30%+ less recently
            bounded_form = 0.8
        elif form_ratio <= 0.85:   # Scoring 15%+ less recently
            bounded_form = 0.9
        else:
            bounded_form = form_ratio  # Keep as is for normal ranges
        
        # Ensure reasonable bounds
        bounded_form = max(0.7, min(1.3, bounded_form))
        
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
        expected_home = max(0.2, min(3.5, expected_home))
        expected_away = max(0.2, min(3.5, expected_away))
        
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
            'xg_better_than_actual': False,  # Defense is BETTER than stats show (concedes less than expected)
            'xg_worse_than_actual': False    # Defense is WORSE than stats show (concedes more than expected)
        }
        
        # Actual goals analysis
        if team.defense_strength <= self.THRESHOLDS['VERY_STRONG_DEFENSE']:
            analysis['is_very_strong'] = True
        elif team.defense_strength <= self.THRESHOLDS['STRONG_DEFENSE']:
            analysis['is_strong'] = True
        elif team.defense_strength >= self.THRESHOLDS['WEAK_DEFENSE']:
            analysis['is_weak'] = True
        
        # xG enhancement - Compare xGA to actual goals conceded
        # If xGA > Actual: Team concedes LESS than expected = Defense is BETTER than stats show
        # If xGA < Actual: Team concedes MORE than expected = Defense is WORSE than stats show
        if team.xg_against > team.defense_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            analysis['xg_better_than_actual'] = True  # Conceding LESS than expected = BETTER defense
        elif team.xg_against < team.defense_strength * self.THRESHOLDS['XG_LUCKY_DEFENSE']:
            analysis['xg_worse_than_actual'] = True   # Conceding MORE than expected = WORSE defense
        
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
        
        # If away team creates chances but doesn't finish
        if away.xg_for > away.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            away_win_base *= 1.1
        
        # If home team overperforming xG (due for regression)
        if home.attack_strength > home.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            home_win_base *= 0.9
        
        # If away team overperforming xG (due for regression)
        if away.attack_strength > away.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            away_win_base *= 0.9
        
        # If away team defense is worse than stats show (due to concede more)
        if away_defense_analysis['xg_worse_than_actual']:
            away_win_base *= 0.9  # Defensive luck may run out
        
        # 7. Normalize probabilities
        total = home_win_base + draw_base + away_win_base
        home_prob = home_win_base / total
        draw_prob = draw_base / total
        away_prob = away_win_base / total
        
        # 8. Determine prediction with better logic
        max_prob = max(home_prob, draw_prob, away_prob)
        
        if max_prob == draw_prob and draw_prob > 0.35:
            prediction = Prediction.DRAW
        elif home_prob > away_prob + 0.1:
            prediction = Prediction.HOME_WIN
        elif away_prob > home_prob + 0.1:
            prediction = Prediction.AWAY_WIN
        else:
            prediction = Prediction.DRAW
        
        return {
            'prediction': prediction,
            'prediction_value': prediction.value,
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
                'home': self.analyze_defensive_strength(home),
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
        home_defense = self.analyze_defensive_strength(home)
        away_defense = self.analyze_defensive_strength(away)
        
        if away_defense['is_very_strong']:
            reduction = 0.6 if away.games_played >= 8 else 0.8
            prob_over *= reduction
            prob_under = 1 - prob_over
        
        # Offensive adjustments
        if home.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']:
            prob_over *= 0.85
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
        
        # Home team overperforming xG (due for regression)
        if home.attack_strength > home.xg_for * 1.2:
            prob_over *= 0.9
            if home.name:
                xg_adjustments.append(
                    f"{home.name} scores {home.attack_strength:.2f} but creates only {home.xg_for:.2f} xG - regression possible"
                )
        
        # Away team creates chances but doesn't finish
        if away.xg_for > away.attack_strength * 1.2:
            prob_over *= 1.1
            if away.name:
                xg_adjustments.append(
                    f"{away.name} creates {away.xg_for:.2f} xG but scores {away.attack_strength:.2f} - due for goals"
                )
        
        # Away team overperforming xG (due for regression)
        if away.attack_strength > away.xg_for * 1.2:
            prob_over *= 0.9
            if away.name:
                xg_adjustments.append(
                    f"{away.name} scores {away.attack_strength:.2f} but creates only {away.xg_for:.2f} xG - regression possible"
                )
        
        # Away team defense is worse than stats show
        if away_defense['xg_worse_than_actual']:
            prob_over *= 1.1
            if away.name:
                xg_adjustments.append(
                    f"{away.name} defense worse than stats show - could concede more"
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
            'prediction_value': prediction.value,
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
        
        # Home team overperforming xG
        if home.attack_strength > home.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 1.1  # Scores more than creates - regression likely
        
        # Away team finishing issues
        if away.xg_for > away.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 0.9  # Creates chances but doesn't finish
        
        # Away team overperforming xG
        if away.attack_strength > away.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 1.1  # Scores more than creates - regression likely
        
        # Away team defensive luck
        if away_defense['xg_worse_than_actual']:
            prob_btts *= 1.1  # Defense worse than stats show - more likely to concede
        
        # H2H adjustment (MODERATE weight: 15-25%)
        if h2h_btts is not None:
            # Weight based on recency and sample size
            h2h_weight = min(0.25, 0.15 + (h2h_meetings / 20)) if h2h_meetings else 0.2
            prob_btts = (prob_btts * (1 - h2h_weight)) + (h2h_btts * h2h_weight)
        
        prob_no_btts = 1 - prob_btts
        
        # Determine prediction with better threshold
        min_diff_for_prediction = 0.05  # Need at least 5% difference
        
        if prob_btts - prob_no_btts > min_diff_for_prediction:
            prediction = Prediction.BTTS_YES
            confidence = self._probability_to_confidence(prob_btts)
        elif prob_no_btts - prob_btts > min_diff_for_prediction:
            prediction = Prediction.BTTS_NO
            confidence = self._probability_to_confidence(prob_no_btts)
        else:
            # Too close to call - go with statistical favorite
            if prob_btts > 0.5:
                prediction = Prediction.BTTS_YES
            else:
                prediction = Prediction.BTTS_NO
            confidence = "Toss-up"
        
        return {
            'prediction': prediction,
            'prediction_value': prediction.value,
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
    
    def analyze_matchup_patterns(self, home: TeamMetrics, away: TeamMetrics) -> List[str]:
        """
        Generate key insights about the matchup - COMPLETE FIXED VERSION
        """
        patterns = []
        
        # Add team names for better insights
        home_name = home.name or "Home team"
        away_name = away.name or "Away team"
        
        # 1. Form patterns (IMPROVED with better thresholds)
        home_form = self.calculate_form_factor(home)
        away_form = self.calculate_form_factor(away)
        
        if home_form >= self.THRESHOLDS['EXCELLENT_FORM']:
            patterns.append(f"{home_name} in EXCELLENT form ({home.goals_scored_last_5} goals in last 5)")
        elif home_form >= self.THRESHOLDS['GOOD_FORM']:
            patterns.append(f"{home_name} in GOOD form ({home.goals_scored_last_5} goals in last 5)")
        elif home_form <= self.THRESHOLDS['VERY_POOR_FORM']:
            patterns.append(f"{home_name} in VERY POOR form ({home.goals_scored_last_5} goals in last 5)")
        elif home_form <= self.THRESHOLDS['POOR_FORM']:
            patterns.append(f"{home_name} in POOR form ({home.goals_scored_last_5} goals in last 5)")
        elif home_form < self.THRESHOLDS['AVERAGE_FORM']:
            patterns.append(f"{home_name} in BELOW AVERAGE form ({home.goals_scored_last_5} goals in last 5)")
        
        if away_form >= self.THRESHOLDS['EXCELLENT_FORM']:
            patterns.append(f"{away_name} in EXCELLENT form ({away.goals_scored_last_5} goals in last 5)")
        elif away_form >= self.THRESHOLDS['GOOD_FORM']:
            patterns.append(f"{away_name} in GOOD form ({away.goals_scored_last_5} goals in last 5)")
        elif away_form <= self.THRESHOLDS['VERY_POOR_FORM']:
            patterns.append(f"{away_name} in VERY POOR form ({away.goals_scored_last_5} goals in last 5)")
        elif away_form <= self.THRESHOLDS['POOR_FORM']:
            patterns.append(f"{away_name} in POOR form ({away.goals_scored_last_5} goals in last 5)")
        elif away_form < self.THRESHOLDS['AVERAGE_FORM']:
            patterns.append(f"{away_name} in BELOW AVERAGE form ({away.goals_scored_last_5} goals in last 5)")
        
        # 2. Attack strength patterns (ADDED - was missing)
        if home.attack_strength >= self.THRESHOLDS['STRONG_ATTACK']:
            patterns.append(f"{home_name} has STRONG attack ({home.attack_strength:.2f} goals/game)")
        elif home.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']:
            patterns.append(f"{home_name} has WEAK attack ({home.attack_strength:.2f} goals/game)")
        
        if away.attack_strength >= self.THRESHOLDS['STRONG_ATTACK']:
            patterns.append(f"{away_name} has STRONG attack ({away.attack_strength:.2f} goals/game)")
        elif away.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']:
            patterns.append(f"{away_name} has WEAK attack ({away.attack_strength:.2f} goals/game)")
        
        # 3. Defensive analysis
        home_defense = self.analyze_defensive_strength(home)
        away_defense = self.analyze_defensive_strength(away)
        
        if home_defense['is_very_strong']:
            patterns.append(f"{home_name} has VERY STRONG defense ({home.defense_strength:.2f} conceded/game)")
        elif home_defense['is_strong']:
            patterns.append(f"{home_name} has STRONG defense ({home.defense_strength:.2f} conceded/game)")
        elif home_defense['is_weak']:
            patterns.append(f"{home_name} has WEAK defense ({home.defense_strength:.2f} conceded/game)")
        
        if away_defense['is_very_strong']:
            patterns.append(f"{away_name} has VERY STRONG defense ({away.defense_strength:.2f} conceded/game)")
        elif away_defense['is_strong']:
            patterns.append(f"{away_name} has STRONG defense ({away.defense_strength:.2f} conceded/game)")
        elif away_defense['is_weak']:
            patterns.append(f"{away_name} has WEAK defense ({away.defense_strength:.2f} conceded/game)")
        
        # 4. Defensive battle detection (ENHANCED for pattern signals)
        if (away_defense['is_very_strong'] or away_defense['is_strong']) and home.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']:
            patterns.append("🔥 DEFENSIVE BATTLE likely - low scoring expected")
        
        # 5. High scoring potential
        if home.attack_strength >= self.THRESHOLDS['STRONG_ATTACK'] and away_defense['is_weak']:
            patterns.append("🔥 HIGH SCORING POTENTIAL - strong attack vs weak defense")
        
        if away.attack_strength >= self.THRESHOLDS['STRONG_ATTACK'] and home_defense['is_weak']:
            patterns.append("🔥 HIGH SCORING POTENTIAL - strong away attack vs weak home defense")
        
        # 6. Clean sheet patterns (ADDED - was missing)
        if home.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            patterns.append(f"{home_name} keeps clean sheets ({home.clean_sheet_pct:.0%})")
        
        if away.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            patterns.append(f"{away_name} keeps clean sheets ({away.clean_sheet_pct:.0%})")
        
        # 7. Failed to score patterns (ADDED - was missing)
        if home.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            patterns.append(f"{home_name} fails to score often ({home.failed_to_score_pct:.0%})")
        
        if away.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            patterns.append(f"{away_name} fails to score often ({away.failed_to_score_pct:.0%})")
        
        # 8. xG patterns - FIXED to detect BOTH underperformers AND overperformers
        home_xg_diff = home.xg_for - home.attack_strength
        away_xg_diff = away.xg_for - away.attack_strength
        
        if home_xg_diff > 0.2:
            patterns.append(f"🔥 {home_name} creates chances ({home.xg_for:.2f} xG) - due for goals")
        elif home_xg_diff < -0.2:
            patterns.append(f"🔥 {home_name} overperforming xG ({home.attack_strength:.2f} goals vs {home.xg_for:.2f} xG) - regression possible")
        
        if away_xg_diff > 0.2:
            patterns.append(f"🔥 {away_name} creates chances ({away.xg_for:.2f} xG) - due for goals")
        elif away_xg_diff < -0.2:
            patterns.append(f"🔥 {away_name} overperforming xG ({away.attack_strength:.2f} goals vs {away.xg_for:.2f} xG) - regression possible")
        
        # 9. Defensive xG patterns (ENHANCED for pattern signals)
        if home_defense['xg_better_than_actual']:
            patterns.append(f"🔥 {home_name} defense BETTER than stats show")
        elif home_defense['xg_worse_than_actual']:
            patterns.append(f"🔥 {home_name} defense WORSE than stats show")
        
        if away_defense['xg_better_than_actual']:
            patterns.append(f"🔥 {away_name} defense BETTER than stats show")
        elif away_defense['xg_worse_than_actual']:
            patterns.append(f"🔥 {away_name} defense WORSE than stats show")
        
        # 10. Parity pattern
        if abs(home.ppg - away.ppg) <= 0.2:
            patterns.append("CLOSE MATCHUP - draw possible")
        
        # 11. Home dominance
        if home.ppg > away.ppg + 0.5:
            patterns.append(f"{home_name} significantly stronger (+{home.ppg - away.ppg:.2f} PPG)")
        
        # 12. Away dominance
        if away.ppg > home.ppg + 0.5:
            patterns.append(f"{away_name} significantly stronger (+{away.ppg - home.ppg:.2f} PPG)")
        
        return patterns
    
    # ========== HELPER METHODS ==========
    
    def _ppg_to_win_probability(self, team_ppg: float, opponent_ppg: float, is_home: bool) -> float:
        """
        Convert PPG differential to win probability
        """
        base_diff = team_ppg - opponent_ppg
        
        if is_home:
            base_diff += 0.3  # Home advantage
        
        # More nuanced conversion
        if base_diff > 1.0:
            return 0.55
        elif base_diff > 0.5:
            return 0.48
        elif base_diff > 0:
            return 0.42
        elif base_diff > -0.5:
            return 0.38
        elif base_diff > -1.0:
            return 0.32
        else:
            return 0.28
    
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
        elif probability >= 0.60:
            return "Low"
        elif probability >= 0.55:
            return "Very Low"
        else:
            return "Toss-up"

# ========== STREAMLIT TRACKING INTERFACE ==========

def render_tracking_dashboard(tracker: PredictionTracker):
    """Main tracking interface"""
    st.title("📊 Prediction Tracking Dashboard")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Enter Results", 
        "⏳ Pending Matches", 
        "✅ Completed Matches", 
        "📈 Performance"
    ])
    
    with tab1:
        render_enter_results(tracker)
    
    with tab2:
        render_pending_matches(tracker)
    
    with tab3:
        render_completed_matches(tracker)
    
    with tab4:
        render_performance_dashboard(tracker)

def render_enter_results(tracker: PredictionTracker):
    """Interface to enter match results"""
    st.subheader("📝 Enter Match Results")
    
    # Get pending matches
    pending = tracker.get_pending_results()
    
    if len(pending) == 0:
        st.info("No pending matches to enter results for.")
        return
    
    # Select match
    pending['display'] = pending.apply(
        lambda row: f"{row['home_team']} vs {row['away_team']} ({row['match_date']})", 
        axis=1
    )
    
    selected_display = st.selectbox(
        "Select match to enter result:",
        pending['display'].tolist()
    )
    
    if selected_display:
        # Get the selected row
        selected_row = pending[pending['display'] == selected_display].iloc[0]
        prediction_id = selected_row['prediction_id']
        
        st.write(f"**Match:** {selected_row['home_team']} vs {selected_row['away_team']}")
        st.write(f"**Date:** {selected_row['match_date']}")
        
        # Get prediction details
        prediction = tracker.get_prediction_by_id(prediction_id)
        if prediction:
            pred_details = prediction.get('predictions', {})
            if 'match_result' in pred_details:
                # Handle both old and new format
                mr_data = pred_details['match_result']
                if isinstance(mr_data, dict):
                    st.write(f"**Your Prediction:** {mr_data.get('prediction_value', mr_data.get('prediction', 'N/A'))}")
        
        # Result input form
        with st.form(f"result_form_{prediction_id}"):
            col1, col2 = st.columns(2)
            
            with col1:
                result = st.selectbox("Result:", ["1", "X", "2"], 
                                    help="1=Home Win, X=Draw, 2=Away Win")
                score = st.text_input("Score (e.g., 2-1):", "0-0")
            
            with col2:
                btts = st.selectbox("Both Teams Scored:", ["Yes", "No"])
                over_under = st.selectbox("Over/Under 2.5:", ["Over", "Under"])
            
            notes = st.text_area("Notes (optional):", 
                               placeholder="Any important context...")
            
            submitted = st.form_submit_button("✅ Save Result")
            
            if submitted:
                if tracker.enter_result(prediction_id, result, score, btts, over_under, notes):
                    st.success(f"✅ Result saved for {selected_row['home_team']} vs {selected_row['away_team']}")
                    st.rerun()
                else:
                    st.error("Failed to save result. Please try again.")

def render_pending_matches(tracker: PredictionTracker):
    """Show matches waiting for results"""
    st.subheader("⏳ Matches Waiting for Results")
    
    pending = tracker.get_pending_results()
    
    if len(pending) == 0:
        st.success("✅ All matches have results entered!")
        return
    
    st.write(f"**{len(pending)} matches pending results:**")
    
    # Display as a nice table
    display_cols = ['match_date', 'home_team', 'away_team', 'days_until_match']
    display_df = pending[display_cols].copy()
    display_df.columns = ['Date', 'Home', 'Away', 'Days Until Match']
    
    # Color code by days
    def color_days(val):
        if val is None:
            return ''
        if val < 0:
            return 'background-color: #ffcccc'  # Red for past due
        elif val <= 2:
            return 'background-color: #fff3cd'  # Yellow for imminent
        else:
            return ''
    
    styled_df = display_df.style.applymap(color_days, subset=['Days Until Match'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Quick actions
    st.write("**Quick Actions:**")
    if st.button("🔄 Refresh List"):
        st.rerun()

def render_completed_matches(tracker: PredictionTracker):
    """Show matches with results entered"""
    st.subheader("✅ Completed Matches with Results")
    
    completed = tracker.get_completed_matches()
    
    if len(completed) == 0:
        st.info("No completed matches yet. Enter results in the 'Enter Results' tab.")
        return
    
    st.write(f"**{len(completed)} matches with results:**")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        team_filter = st.text_input("Filter by team:", "")
    with col2:
        sort_by = st.selectbox("Sort by:", ["Date (newest)", "Date (oldest)", "Home Team"])
    
    # Apply filters
    display_df = completed.copy()
    
    if team_filter:
        mask = (display_df['home_team'].str.contains(team_filter, case=False)) | \
               (display_df['away_team'].str.contains(team_filter, case=False))
        display_df = display_df[mask]
    
    # Sort
    if sort_by == "Date (newest)":
        display_df = display_df.sort_values('match_date', ascending=False)
    elif sort_by == "Date (oldest)":
        display_df = display_df.sort_values('match_date', ascending=True)
    elif sort_by == "Home Team":
        display_df = display_df.sort_values('home_team')
    
    # Display
    display_cols = ['match_date', 'home_team', 'away_team', 'actual_result', 
                   'actual_score', 'actual_btts', 'actual_over_under', 'notes']
    
    display_df = display_df[display_cols].copy()
    display_df.columns = ['Date', 'Home', 'Away', 'Result', 'Score', 'BTTS', 'O/U', 'Notes']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Export option
    if st.button("📥 Export to CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def render_performance_dashboard(tracker: PredictionTracker):
    """Show performance metrics"""
    st.subheader("📈 Prediction Performance")
    
    stats = tracker.get_performance_stats()
    
    if not stats:
        st.info("No performance data yet. Enter some results first.")
        return
    
    # Overall stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Matches", stats['total_matches'])
    
    with col2:
        last_updated = pd.to_datetime(stats['last_updated']).strftime('%Y-%m-%d %H:%M')
        st.metric("Last Updated", last_updated)
    
    # Quick insights
    st.subheader("📊 Results Distribution")
    
    if 'by_market' in stats and 'result' in stats['by_market']:
        result_dist = stats['by_market']['result']['results_distribution']
        
        # Display as metrics
        col_results = st.columns(3)
        with col_results[0]:
            st.metric("Home Wins (1)", result_dist.get('1', 0))
        with col_results[1]:
            st.metric("Draws (X)", result_dist.get('X', 0))
        with col_results[2]:
            st.metric("Away Wins (2)", result_dist.get('2', 0))
    
    # Recent predictions
    st.subheader("🕐 Recent Predictions")
    recent = tracker.get_recent_predictions(limit=20)
    
    if len(recent) > 0:
        # Calculate completion rate
        completed = recent['result_added'].sum()
        completion_rate = (completed / len(recent)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recent Predictions", len(recent))
        with col2:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Show recent table
        display_recent = recent.copy()
        display_recent.columns = ['ID', 'Match', 'Match Date', 'Prediction Date', 'Result Entered']
        st.dataframe(display_recent, use_container_width=True)
    
    # Data management
    st.subheader("🗃️ Data Management")
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    if st.button("🗑️ Clear All Data"):
        if st.checkbox("I'm sure I want to delete ALL prediction data"):
            # Clear session state
            if 'predictions' in st.session_state:
                del st.session_state.predictions
            if 'results' in st.session_state:
                del st.session_state.results
            
            # Reinitialize tracker
            tracker.__init__()
            st.success("All data cleared!")
            st.rerun()

# ========== EXAMPLE DATA FUNCTIONS ==========

def set_example_preston_coventry():
    """Example: Preston vs Coventry with realistic data"""
    st.session_state.home_name = "Preston"
    st.session_state.home_attack = 1.40
    st.session_state.home_defense = 1.00
    st.session_state.home_ppg = 1.80
    st.session_state.home_games = 19
    st.session_state.home_cs = 30
    st.session_state.home_fts = 20
    st.session_state.home_xg_for = 1.36
    st.session_state.home_xg_against = 1.42
    st.session_state.home_goals5 = 7
    st.session_state.home_conceded5 = 7
    
    st.session_state.away_name = "Coventry"
    st.session_state.away_attack = 2.50
    st.session_state.away_defense = 1.40
    st.session_state.away_ppg = 2.00
    st.session_state.away_games = 19
    st.session_state.away_cs = 40
    st.session_state.away_fts = 20
    st.session_state.away_xg_for = 1.86
    st.session_state.away_xg_against = 1.40
    st.session_state.away_goals5 = 9
    st.session_state.away_conceded5 = 9
    
    st.session_state.h2h_btts = 40
    st.session_state.h2h_meetings = 5

def set_example_southampton():
    """Example: Southampton vs West Brom with realistic data"""
    st.session_state.home_name = "Southampton"
    st.session_state.home_attack = 1.44
    st.session_state.home_defense = 0.89
    st.session_state.home_ppg = 1.67
    st.session_state.home_games = 19
    st.session_state.home_cs = 33
    st.session_state.home_fts = 33
    st.session_state.home_xg_for = 1.79
    st.session_state.home_xg_against = 1.14
    st.session_state.home_goals5 = 9
    st.session_state.home_conceded5 = 4
    
    st.session_state.away_name = "West Brom"
    st.session_state.away_attack = 1.00
    st.session_state.away_defense = 1.70
    st.session_state.away_ppg = 0.90
    st.session_state.away_games = 19
    st.session_state.away_cs = 20
    st.session_state.away_fts = 30
    st.session_state.away_xg_for = 1.20
    st.session_state.away_xg_against = 1.90
    st.session_state.away_goals5 = 5
    st.session_state.away_conceded5 = 10
    
    st.session_state.h2h_btts = 60
    st.session_state.h2h_meetings = 5

def set_example_defensive_battle():
    """Example: Defensive Battle pattern"""
    st.session_state.home_name = "Norwich"
    st.session_state.home_attack = 0.90
    st.session_state.home_defense = 1.10
    st.session_state.home_ppg = 1.10
    st.session_state.home_games = 19
    st.session_state.home_cs = 30
    st.session_state.home_fts = 40
    st.session_state.home_xg_for = 1.52
    st.session_state.home_xg_against = 1.39  # xGA > Actual = Defense BETTER than stats
    st.session_state.home_goals5 = 7
    st.session_state.home_conceded5 = 6
    
    st.session_state.away_name = "Sheffield United"
    st.session_state.away_attack = 0.89
    st.session_state.away_defense = 1.33
    st.session_state.away_ppg = 0.89
    st.session_state.away_games = 18
    st.session_state.away_cs = 22
    st.session_state.away_fts = 44
    st.session_state.away_xg_for = 1.19
    st.session_state.away_xg_against = 1.48  # xGA > Actual = Defense BETTER than stats
    st.session_state.away_goals5 = 5
    st.session_state.away_conceded5 = 8
    
    st.session_state.h2h_btts = 60
    st.session_state.h2h_meetings = 5

def clear_session_state():
    """Clear all session state data"""
    keys_to_clear = [
        'home_name', 'home_attack', 'home_defense', 'home_ppg', 'home_games',
        'home_cs', 'home_fts', 'home_xg_for', 'home_xg_against',
        'home_goals5', 'home_conceded5',
        'away_name', 'away_attack', 'away_defense', 'away_ppg', 'away_games',
        'away_cs', 'away_fts', 'away_xg_for', 'away_xg_against',
        'away_goals5', 'away_conceded5',
        'h2h_btts', 'h2h_meetings'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# ========== MAIN APP ==========

def main():
    st.set_page_config(
        page_title="Football Predictor Pro v2.0 with Tracking",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize tracker and engine - NO FILE SYSTEM ACCESS
    tracker = PredictionTracker()  # Uses session state only
    engine = PredictionEngineV2()
    
    # League contexts presets
    LEAGUE_CONTEXTS = {
        "Premier League": MatchContext(
            league_avg_goals=2.7,
            league_avg_xg=1.35,
            home_advantage=1.15
        ),
        "La Liga": MatchContext(
            league_avg_goals=2.5,
            league_avg_xg=1.25,
            home_advantage=1.18
        ),
        "Bundesliga": MatchContext(
            league_avg_goals=3.0,
            league_avg_xg=1.50,
            home_advantage=1.10
        ),
        "Serie A": MatchContext(
            league_avg_goals=2.6,
            league_avg_xg=1.30,
            home_advantage=1.16
        ),
        "Ligue 1": MatchContext(
            league_avg_goals=2.4,
            league_avg_xg=1.20,
            home_advantage=1.17
        ),
        "Liga NOS": MatchContext(
            league_avg_goals=2.68,
            league_avg_xg=1.34,
            home_advantage=1.15
        ),
        "Championship": MatchContext(
            league_avg_goals=2.55,
            league_avg_xg=1.28,
            home_advantage=1.16
        ),
        "Other": MatchContext()
    }
    
    # Check if we should show tracking dashboard
    if st.session_state.get('show_tracking', False):
        render_tracking_dashboard(tracker)
        if st.button("← Back to Predictor"):
            st.session_state.show_tracking = False
            st.rerun()
        return
    
    # ========== MAIN PREDICTOR INTERFACE ==========
    
    st.title("⚽ Football Predictor Pro v2.0 with Tracking")
    st.caption("Enhanced with xG Integration + Session State Tracking 📊")
    
    # Sidebar with advanced settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # League selection
        league = st.selectbox(
            "Select League",
            list(LEAGUE_CONTEXTS.keys()),
            index=6  # Default to Championship
        )
        engine.context = LEAGUE_CONTEXTS[league]
        
        # TRACKING SYSTEM BUTTON
        st.markdown("---")
        st.header("📊 Tracking System")
        if st.button("📈 Open Tracking Dashboard", use_container_width=True):
            st.session_state.show_tracking = True
            st.rerun()
        
        # Data management in sidebar
        st.markdown("---")
        st.header("🗃️ Data Management")
        
        if st.button("Initialize Session Data"):
            # Reinitialize tracker
            tracker.__init__()
            st.success("Session data initialized!")
            st.rerun()
        
        if st.button("Clear All Predictions", type="secondary"):
            if st.checkbox("Confirm delete ALL predictions"):
                # Clear session state
                if 'predictions' in st.session_state:
                    del st.session_state.predictions
                if 'results' in st.session_state:
                    del st.session_state.results
                
                # Reinitialize tracker
                tracker.__init__()
                st.success("All predictions cleared!")
                st.rerun()
        
        # Quick examples
        st.header("📋 Examples")
        
        if st.button("Example: Preston vs Coventry"):
            set_example_preston_coventry()
            st.rerun()
        
        if st.button("Example: Southampton vs West Brom"):
            set_example_southampton()
            st.rerun()
        
        if st.button("Example: Defensive Battle"):
            set_example_defensive_battle()
            st.rerun()
        
        if st.button("Clear Inputs", type="secondary"):
            clear_session_state()
            st.rerun()
        
        # Info
        st.info("""
        **Version 2.0 with Session Tracking:**
        - xG Integration for better accuracy
        - Pattern detection signals
        - **NEW:** Session-based tracking
        - **NEW:** No file system access needed
        - **NEW:** Works on Streamlit Cloud
        """)
    
    # Main interface
    st.header("📊 Enter Match Data")
    
    # Match date
    match_date = st.date_input(
        "📅 Match Date",
        datetime.now(),
        help="Required for tracking predictions"
    )
    
    # Team names
    col_names = st.columns(2)
    with col_names[0]:
        home_name = st.text_input(
            "🏠 Home Team",
            value=st.session_state.get('home_name', 'Preston'),
            key="home_name_input"
        )
    with col_names[1]:
        away_name = st.text_input(
            "🚗 Away Team",
            value=st.session_state.get('away_name', 'Coventry'),
            key="away_name_input"
        )
    
    # Main tabs for organized input
    tab1, tab2, tab3 = st.tabs(["⚽ Core Stats", "📈 xG Stats", "📊 Recent Form"])
    
    with tab1:
        st.info("Enter basic team statistics for the season")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{home_name}")
            
            # Attack/Defense
            col_attack = st.columns(2)
            with col_attack[0]:
                home_attack = st.number_input(
                    "Goals/Game", 
                    0.0, 5.0, 
                    value=float(st.session_state.get('home_attack', 1.40)), 
                    step=0.01,
                    key="home_attack_input"
                )
            with col_attack[1]:
                home_defense = st.number_input(
                    "Conceded/Game", 
                    0.0, 5.0,
                    value=float(st.session_state.get('home_defense', 1.00)), 
                    step=0.01,
                    key="home_defense_input"
                )
            
            # Points and games
            col_points = st.columns(2)
            with col_points[0]:
                home_ppg = st.number_input(
                    "Points/Game", 
                    0.0, 3.0,
                    value=float(st.session_state.get('home_ppg', 1.80)), 
                    step=0.01,
                    key="home_ppg_input"
                )
            with col_points[1]:
                home_games = st.number_input(
                    "Games Played", 
                    1, 40,
                    value=int(st.session_state.get('home_games', 19)),
                    key="home_games_input"
                )
            
            # Performance metrics
            st.write("**Performance Metrics**")
            col_perf = st.columns(2)
            with col_perf[0]:
                home_cs = st.number_input(
                    "Clean Sheet %", 
                    0, 100,
                    value=int(st.session_state.get('home_cs', 30)),
                    key="home_cs_input"
                )
            with col_perf[1]:
                home_fts = st.number_input(
                    "Fail to Score %", 
                    0, 100,
                    value=int(st.session_state.get('home_fts', 20)),
                    key="home_fts_input"
                )
        
        with col2:
            st.subheader(f"{away_name}")
            
            # Attack/Defense
            col_attack = st.columns(2)
            with col_attack[0]:
                away_attack = st.number_input(
                    "Goals/Game", 
                    0.0, 5.0,
                    value=float(st.session_state.get('away_attack', 2.50)), 
                    step=0.01,
                    key="away_attack_input"
                )
            with col_attack[1]:
                away_defense = st.number_input(
                    "Conceded/Game", 
                    0.0, 5.0,
                    value=float(st.session_state.get('away_defense', 1.40)), 
                    step=0.01,
                    key="away_defense_input"
                )
            
            # Points and games
            col_points = st.columns(2)
            with col_points[0]:
                away_ppg = st.number_input(
                    "Points/Game", 
                    0.0, 3.0,
                    value=float(st.session_state.get('away_ppg', 2.00)), 
                    step=0.01,
                    key="away_ppg_input"
                )
            with col_points[1]:
                away_games = st.number_input(
                    "Games Played", 
                    1, 40,
                    value=int(st.session_state.get('away_games', 19)),
                    key="away_games_input"
                )
            
            # Performance metrics
            st.write("**Performance Metrics**")
            col_perf = st.columns(2)
            with col_perf[0]:
                away_cs = st.number_input(
                    "Clean Sheet %", 
                    0, 100,
                    value=int(st.session_state.get('away_cs', 40)),
                    key="away_cs_input"
                )
            with col_perf[1]:
                away_fts = st.number_input(
                    "Fail to Score %", 
                    0, 100,
                    value=int(st.session_state.get('away_fts', 20)),
                    key="away_fts_input"
                )
    
    with tab2:
        st.info("xG (Expected Goals) measures the quality of chances created/conceded")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{home_name} xG")
            
            col_xg = st.columns(2)
            with col_xg[0]:
                home_xg_for = st.number_input(
                    "xG Created/Game", 
                    0.0, 5.0,
                    value=float(st.session_state.get('home_xg_for', 1.36)), 
                    step=0.01,
                    key="home_xg_for_input"
                )
            with col_xg[1]:
                home_xg_against = st.number_input(
                    "xG Conceded/Game", 
                    0.0, 5.0,
                    value=float(st.session_state.get('home_xg_against', 1.42)), 
                    step=0.01,
                    key="home_xg_against_input"
                )
        
        with col2:
            st.subheader(f"{away_name} xG")
            
            col_xg = st.columns(2)
            with col_xg[0]:
                away_xg_for = st.number_input(
                    "xG Created/Game", 
                    0.0, 5.0,
                    value=float(st.session_state.get('away_xg_for', 1.86)), 
                    step=0.01,
                    key="away_xg_for_input"
                )
            with col_xg[1]:
                away_xg_against = st.number_input(
                    "xG Conceded/Game", 
                    0.0, 5.0,
                    value=float(st.session_state.get('away_xg_against', 1.40)), 
                    step=0.01,
                    key="away_xg_against_input"
                )
    
    with tab3:
        st.info("Enter goals from the last 5 matches for each team")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{home_name} Recent Form")
            
            col_form = st.columns(2)
            with col_form[0]:
                home_goals5 = st.number_input(
                    "Goals Scored (Last 5)", 
                    0, 30,
                    value=int(st.session_state.get('home_goals5', 7)),
                    key="home_goals5_input"
                )
            with col_form[1]:
                home_conceded5 = st.number_input(
                    "Goals Conceded (Last 5)", 
                    0, 30,
                    value=int(st.session_state.get('home_conceded5', 7)),
                    key="home_conceded5_input"
                )
        
        with col2:
            st.subheader(f"{away_name} Recent Form")
            
            col_form = st.columns(2)
            with col_form[0]:
                away_goals5 = st.number_input(
                    "Goals Scored (Last 5)", 
                    0, 30,
                    value=int(st.session_state.get('away_goals5', 9)),
                    key="away_goals5_input"
                )
            with col_form[1]:
                away_conceded5 = st.number_input(
                    "Goals Conceded (Last 5)", 
                    0, 30,
                    value=int(st.session_state.get('away_conceded5', 9)),
                    key="away_conceded5_input"
                )
    
    # H2H Section
    with st.expander("Head-to-Head Data (Optional)", expanded=True):
        st.info("Enter historical head-to-head statistics for additional context")
        col_h2h = st.columns(2)
        with col_h2h[0]:
            h2h_btts = st.number_input(
                "H2H BTTS %", 
                0, 100,
                value=int(st.session_state.get('h2h_btts', 40)),
                key="h2h_btts_input",
                help="Percentage of previous meetings where both teams scored"
            )
        with col_h2h[1]:
            h2h_meetings = st.number_input(
                "Total H2H Meetings",
                0, 100,
                value=int(st.session_state.get('h2h_meetings', 5)),
                key="h2h_meetings_input",
                help="Total number of previous meetings between these teams"
            )
    
    # Create team metrics objects
    home_metrics = TeamMetrics(
        name=home_name,
        attack_strength=home_attack,
        defense_strength=home_defense,
        ppg=home_ppg,
        xg_for=home_xg_for,
        xg_against=home_xg_against,
        clean_sheet_pct=home_cs/100,
        failed_to_score_pct=home_fts/100,
        btts_pct=0.5,
        goals_scored_last_5=home_goals5,
        goals_conceded_last_5=home_conceded5,
        games_played=home_games
    )
    
    away_metrics = TeamMetrics(
        name=away_name,
        attack_strength=away_attack,
        defense_strength=away_defense,
        ppg=away_ppg,
        xg_for=away_xg_for,
        xg_against=away_xg_against,
        clean_sheet_pct=away_cs/100,
        failed_to_score_pct=away_fts/100,
        btts_pct=0.5,
        goals_scored_last_5=away_goals5,
        goals_conceded_last_5=away_conceded5,
        games_played=away_games
    )
    
    # Generate predictions button
    st.divider()
    if st.button("🚀 Generate Advanced Predictions", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing match with xG integration and pattern detection..."):
            # Get predictions
            result_pred = engine.predict_match_result(home_metrics, away_metrics)
            over_under_pred = engine.predict_over_under(home_metrics, away_metrics)
            btts_pred = engine.predict_btts(
                home_metrics, away_metrics, 
                h2h_btts=h2h_btts/100 if h2h_btts else None,
                h2h_meetings=h2h_meetings if h2h_meetings > 0 else None
            )
            expected_goals = engine.calculate_xg_adjusted_goals(home_metrics, away_metrics)
            patterns = engine.analyze_matchup_patterns(home_metrics, away_metrics)
            
            # NEW: Pattern Detection
            patterns_detected = engine.detect_high_confidence_patterns(home_metrics, away_metrics)
            pattern_advice = engine.get_pattern_based_advice(patterns_detected)
            
            # ========== SAVE PREDICTION TO SESSION STATE ==========
            # Convert predictions to serializable format
            serializable_predictions = {
                'match_result': {
                    'prediction': result_pred['prediction'].value,
                    'prediction_value': result_pred['prediction'].value,
                    'probabilities': result_pred['probabilities'],
                    'form_factors': result_pred['form_factors'],
                    'defensive_analysis': result_pred['defensive_analysis']
                },
                'over_under': {
                    'prediction': over_under_pred['prediction'].value,
                    'prediction_value': over_under_pred['prediction'].value,
                    'confidence': over_under_pred['confidence'],
                    'probabilities': over_under_pred['probabilities'],
                    'expected_goals': over_under_pred['expected_goals'],
                    'xg_adjustments': over_under_pred.get('xg_adjustments', []),
                    'detailed_expected': over_under_pred['detailed_expected']
                },
                'btts': {
                    'prediction': btts_pred['prediction'].value,
                    'prediction_value': btts_pred['prediction'].value,
                    'confidence': btts_pred['confidence'],
                    'probabilities': btts_pred['probabilities'],
                    'form_factors': btts_pred['form_factors']
                },
                'expected_goals': {'home': expected_goals[0], 'away': expected_goals[1]},
                'patterns': patterns_detected,
                'pattern_advice': pattern_advice,
                'matchup_patterns': patterns
            }
            
            prediction_id = tracker.save_prediction(
                home_team=home_name,
                away_team=away_name,
                match_date=str(match_date),
                predictions=serializable_predictions,
                inputs={
                    'home_metrics': {
                        'attack_strength': home_attack,
                        'defense_strength': home_defense,
                        'ppg': home_ppg,
                        'xg_for': home_xg_for,
                        'xg_against': home_xg_against,
                        'clean_sheet_pct': home_cs/100,
                        'failed_to_score_pct': home_fts/100,
                        'goals_scored_last_5': home_goals5,
                        'goals_conceded_last_5': home_conceded5,
                        'games_played': home_games
                    },
                    'away_metrics': {
                        'attack_strength': away_attack,
                        'defense_strength': away_defense,
                        'ppg': away_ppg,
                        'xg_for': away_xg_for,
                        'xg_against': away_xg_against,
                        'clean_sheet_pct': away_cs/100,
                        'failed_to_score_pct': away_fts/100,
                        'goals_scored_last_5': away_goals5,
                        'goals_conceded_last_5': away_conceded5,
                        'games_played': away_games
                    },
                    'league': league,
                    'h2h_data': {
                        'btts': h2h_btts,
                        'meetings': h2h_meetings
                    }
                }
            )
            
            # Display results
            st.success(f"✅ Advanced Predictions Generated (Saved as ID: {prediction_id})")
            
            # Display the rest of your predictions (your existing display code)
            # ... [Your existing display code for predictions goes here] ...
            # I've removed it to save space, but it should be your original display code

if __name__ == "__main__":
    main()
