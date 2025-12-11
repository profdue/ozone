"""
Football Predictor Pro v2.0 - WITH GOOGLE SHEETS TRACKING
Enhanced with Google Sheets integration for saving predictions and tracking performance
"""

import streamlit as st
import pandas as pd
import math
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz

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

# ========== GOOGLE SHEETS INTEGRATION ==========

class GoogleSheetsTracker:
    """Handles Google Sheets integration for saving predictions"""
    
    def __init__(self):
        self.sheet_url = "https://docs.google.com/spreadsheets/d/13Zw4TksoH9P1PWv1HuNpapZOOnT_dIm_Pr85qRof4yE/edit#gid=0"
        self.sheet_name = "Football Predictions"
        
    def connect(self):
        """Connect to Google Sheets using credentials from secrets"""
        try:
            # Get credentials from secrets
            if 'google_sheets' not in st.secrets:
                st.error("Google Sheets credentials not found in secrets.toml")
                return None
            
            # Create credentials from secrets
            creds_dict = dict(st.secrets['google_sheets'])
            credentials = Credentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            # Connect to Google Sheets
            client = gspread.authorize(credentials)
            return client
            
        except Exception as e:
            st.error(f"Error connecting to Google Sheets: {str(e)}")
            return None
    
    def save_prediction(self, prediction_data: Dict):
        """Save a prediction to Google Sheets"""
        try:
            client = self.connect()
            if not client:
                return {'success': False, 'error': 'Failed to connect'}
            
            # Open the spreadsheet
            spreadsheet = client.open_by_url(self.sheet_url)
            worksheet = spreadsheet.sheet1
            
            # Prepare row data
            row_data = [
                datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S"),
                prediction_data['home_team'],
                prediction_data['away_team'],
                prediction_data.get('pattern', ''),
                prediction_data.get('primary_bet', ''),
                prediction_data.get('stake', ''),
                prediction_data.get('confidence', ''),
                'PENDING',  # Status
                '',  # Actual Score
                '',  # Bet Outcome
                prediction_data.get('notes', '')
            ]
            
            # Append to sheet
            worksheet.append_row(row_data)
            
            # Get the row number for reference
            all_values = worksheet.get_all_values()
            row_number = len(all_values)  # 1-indexed
            
            return {
                'success': True,
                'row_number': row_number,
                'sheet_url': self.sheet_url
            }
            
        except Exception as e:
            st.error(f"Error saving to Google Sheets: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_predictions(self, limit: int = 50):
        """Get predictions from Google Sheets"""
        try:
            client = self.connect()
            if not client:
                return pd.DataFrame()
            
            spreadsheet = client.open_by_url(self.sheet_url)
            worksheet = spreadsheet.sheet1
            
            # Get all data
            data = worksheet.get_all_records()
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(data)
            
            # Sort by timestamp (newest first)
            if 'Timestamp' in df.columns and not df.empty:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df = df.sort_values('Timestamp', ascending=False)
            
            # Limit results
            if limit and len(df) > limit:
                df = df.head(limit)
            
            return df
            
        except Exception as e:
            st.error(f"Error reading from Google Sheets: {str(e)}")
            return pd.DataFrame()
    
    def update_result(self, row_index: int, actual_score: str, bet_outcome: str, notes: str = ""):
        """Update a prediction with actual result"""
        try:
            client = self.connect()
            if not client:
                return False
            
            spreadsheet = client.open_by_url(self.sheet_url)
            worksheet = spreadsheet.sheet1
            
            # Convert row_index to 1-based for Google Sheets
            # row_index is the DataFrame index, need to add 2 (1 for header, 1 for 0-based)
            sheet_row = row_index + 2
            
            # Update the row
            worksheet.update(f'I{sheet_row}', [[actual_score]])  # Actual Score column
            worksheet.update(f'J{sheet_row}', [[bet_outcome]])   # Bet Outcome column
            worksheet.update(f'K{sheet_row}', [[notes]])         # Notes column
            
            # Update Status based on bet outcome
            if bet_outcome:
                status = 'COMPLETED'
                worksheet.update(f'H{sheet_row}', [[status]])    # Status column
            
            return True
            
        except Exception as e:
            st.error(f"Error updating Google Sheets: {str(e)}")
            return False

# ========== PREDICTION ENGINE ==========

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
            'HIGH_CLEAN_SHEET': 0.45,
            'HIGH_FAILED_TO_SCORE': 0.35,
            
            # Form
            'EXCELLENT_FORM': 1.15,
            'GOOD_FORM': 1.05,
            'AVERAGE_FORM': 0.95,
            'POOR_FORM': 0.85,
            'VERY_POOR_FORM': 0.70,
            
            # Sample Size
            'MIN_GAMES_RELIABLE': 6,
            
            # xG thresholds
            'XG_LUCKY_DEFENSE': 0.85,
            'XG_UNLUCKY_ATTACK': 1.15,
            'XG_OVERPERFORMER': 0.85,
            
            # Pattern thresholds
            'XG_SIGNIFICANT_DIFF': 0.3,
            'WEAK_DEFENSE_THRESHOLD': 1.4,
        }
    
    def detect_high_confidence_patterns(self, home: TeamMetrics, away: TeamMetrics) -> List[Dict]:
        """Identify HIGH ACCURACY PATTERNS"""
        patterns_detected = []
        
        # Get defensive analysis
        home_def = self.analyze_defensive_strength(home)
        away_def = self.analyze_defensive_strength(away)
        
        # Calculate xG differences
        home_xg_diff = home.xg_for - home.attack_strength
        away_xg_diff = away.xg_for - away.attack_strength
        
        # PATTERN 1: DEFENSIVE BATTLE
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
        
        # PATTERN 2: REGRESSION EXPLOSION
        if (home_xg_diff > self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and away_def['is_weak']) or \
           (home.xg_for > home.attack_strength * 1.2 and away.defense_strength >= self.THRESHOLDS['WEAK_DEFENSE_THRESHOLD']):
            patterns_detected.append({
                'name': 'REGRESSION EXPLOSION 🔥',
                'pattern': 2,
                'bet': 'OVER 2.5 & BTTS YES',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{home.name} due for goals (+{home_xg_diff:.2f} xG diff) + {away.name} weak defense',
                'validation': '100% accuracy in sample',
                'pattern_type': 'regression_explosion',
                'stake': 'MAX BET (2x normal)',
                'additional_bets': [f'{home.name} Over 1.5 Team Goals', f'{home.name} to Win']
            })
        
        if (away_xg_diff > self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and home_def['is_weak']) or \
           (away.xg_for > away.attack_strength * 1.2 and home.defense_strength >= self.THRESHOLDS['WEAK_DEFENSE_THRESHOLD']):
            patterns_detected.append({
                'name': 'REGRESSION EXPLOSION 🔥',
                'pattern': 2,
                'bet': 'OVER 2.5 & BTTS YES',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{away.name} due for goals (+{away_xg_diff:.2f} xG diff) + {home.name} weak defense',
                'validation': '100% accuracy in sample',
                'pattern_type': 'regression_explosion',
                'stake': 'MAX BET (2x normal)',
                'additional_bets': [f'{away.name} Over 1.5 Team Goals', f'{away.name} to Win']
            })
        
        # PATTERN 3: REGRESSION SUPPRESSION
        if (home.attack_strength > home.xg_for * 1.2 and away_def['xg_better_than_actual']) or \
           (home_xg_diff < -self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and away_def['xg_better_than_actual']):
            patterns_detected.append({
                'name': 'REGRESSION SUPPRESSION 🔥',
                'pattern': 3,
                'bet': 'UNDER 2.5 + Team Under 1.5',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{home.name} overperforming xG + {away.name} strong defense',
                'validation': '100% accuracy in sample',
                'pattern_type': 'regression_suppression',
                'stake': 'STRONG BET (1.5x normal)',
                'additional_bets': [f'{home.name} Under 1.5 Goals', 'Draw']
            })
        
        if (away.attack_strength > away.xg_for * 1.2 and home_def['xg_better_than_actual']) or \
           (away_xg_diff < -self.THRESHOLDS['XG_SIGNIFICANT_DIFF'] and home_def['xg_better_than_actual']):
            patterns_detected.append({
                'name': 'REGRESSION SUPPRESSION 🔥',
                'pattern': 3,
                'bet': 'UNDER 2.5 + Team Under 1.5',
                'confidence': 'HIGH',
                'signal': 'GREEN LIGHT 🟢',
                'description': f'{away.name} overperforming xG + {home.name} strong defense',
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
        
        # xG enhancement
        if team.xg_against > team.defense_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            analysis['xg_better_than_actual'] = True
        elif team.xg_against < team.defense_strength * self.THRESHOLDS['XG_LUCKY_DEFENSE']:
            analysis['xg_worse_than_actual'] = True
        
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
        draw_base = 0.35
        
        # 2. Apply form adjustments
        home_form = self.calculate_form_factor(home)
        away_form = self.calculate_form_factor(away)
        
        home_win_base *= home_form
        away_win_base *= away_form
        
        # 3. Defensive strength adjustments
        away_defense_analysis = self.analyze_defensive_strength(away)
        
        if away_defense_analysis['is_very_strong']:
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
        
        # 6. xG-based finishing adjustments
        if home.xg_for > home.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            home_win_base *= 1.1
        
        if away.xg_for > away.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            away_win_base *= 1.1
        
        if home.attack_strength > home.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            home_win_base *= 0.9
        
        if away.attack_strength > away.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            away_win_base *= 0.9
        
        if away_defense_analysis['xg_worse_than_actual']:
            away_win_base *= 0.9
        
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
        
        # xG-specific adjustments
        xg_adjustments = []
        
        if home.xg_for > home.attack_strength * 1.2:
            prob_over *= 1.15
            if home.name:
                xg_adjustments.append(
                    f"{home.name} creates {home.xg_for:.2f} xG but scores {home.attack_strength:.2f} - due for goals"
                )
        
        if home.attack_strength > home.xg_for * 1.2:
            prob_over *= 0.9
            if home.name:
                xg_adjustments.append(
                    f"{home.name} scores {home.attack_strength:.2f} but creates only {home.xg_for:.2f} xG - regression possible"
                )
        
        if away.xg_for > away.attack_strength * 1.2:
            prob_over *= 1.1
            if away.name:
                xg_adjustments.append(
                    f"{away.name} creates {away.xg_for:.2f} xG but scores {away.attack_strength:.2f} - due for goals"
                )
        
        if away.attack_strength > away.xg_for * 1.2:
            prob_over *= 0.9
            if away.name:
                xg_adjustments.append(
                    f"{away.name} scores {away.attack_strength:.2f} but creates only {away.xg_for:.2f} xG - regression possible"
                )
        
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
        
        # xG adjustments
        if home.xg_for > home.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 0.9
        
        if home.attack_strength > home.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 1.1
        
        if away.xg_for > away.attack_strength * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 0.9
        
        if away.attack_strength > away.xg_for * self.THRESHOLDS['XG_UNLUCKY_ATTACK']:
            prob_btts *= 1.1
        
        if away_defense['xg_worse_than_actual']:
            prob_btts *= 1.1
        
        # H2H adjustment
        if h2h_btts is not None:
            h2h_weight = min(0.25, 0.15 + (h2h_meetings / 20)) if h2h_meetings else 0.2
            prob_btts = (prob_btts * (1 - h2h_weight)) + (h2h_btts * h2h_weight)
        
        prob_no_btts = 1 - prob_btts
        
        # Determine prediction
        min_diff_for_prediction = 0.05
        
        if prob_btts - prob_no_btts > min_diff_for_prediction:
            prediction = Prediction.BTTS_YES
            confidence = self._probability_to_confidence(prob_btts)
        elif prob_no_btts - prob_btts > min_diff_for_prediction:
            prediction = Prediction.BTTS_NO
            confidence = self._probability_to_confidence(prob_no_btts)
        else:
            if prob_btts > 0.5:
                prediction = Prediction.BTTS_YES
            else:
                prediction = Prediction.BTTS_NO
            confidence = "Toss-up"
        
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
    
    def analyze_matchup_patterns(self, home: TeamMetrics, away: TeamMetrics) -> List[str]:
        """
        Generate key insights about the matchup
        """
        patterns = []
        
        # Add team names for better insights
        home_name = home.name or "Home team"
        away_name = away.name or "Away team"
        
        # 1. Form patterns
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
        
        # 2. Attack strength patterns
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
        
        # 4. Defensive battle detection
        if (away_defense['is_very_strong'] or away_defense['is_strong']) and home.attack_strength <= self.THRESHOLDS['WEAK_ATTACK']:
            patterns.append("🔥 DEFENSIVE BATTLE likely - low scoring expected")
        
        # 5. High scoring potential
        if home.attack_strength >= self.THRESHOLDS['STRONG_ATTACK'] and away_defense['is_weak']:
            patterns.append("🔥 HIGH SCORING POTENTIAL - strong attack vs weak defense")
        
        if away.attack_strength >= self.THRESHOLDS['STRONG_ATTACK'] and home_defense['is_weak']:
            patterns.append("🔥 HIGH SCORING POTENTIAL - strong away attack vs weak home defense")
        
        # 6. Clean sheet patterns
        if home.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            patterns.append(f"{home_name} keeps clean sheets ({home.clean_sheet_pct:.0%})")
        
        if away.clean_sheet_pct >= self.THRESHOLDS['HIGH_CLEAN_SHEET']:
            patterns.append(f"{away_name} keeps clean sheets ({away.clean_sheet_pct:.0%})")
        
        # 7. Failed to score patterns
        if home.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            patterns.append(f"{home_name} fails to score often ({home.failed_to_score_pct:.0%})")
        
        if away.failed_to_score_pct >= self.THRESHOLDS['HIGH_FAILED_TO_SCORE']:
            patterns.append(f"{away_name} fails to score often ({away.failed_to_score_pct:.0%})")
        
        # 8. xG patterns
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
        
        # 9. Defensive xG patterns
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

# ========== STREAMLIT APP ==========

def main():
    def main():
    # ========== DIAGNOSTIC TEST ==========
    st.title("🔍 Google Sheets Connection Diagnostic")
    
    # Test 1: Check if secrets are loaded
    st.subheader("Test 1: Secrets Loading")
    if 'google_sheets' in st.secrets:
        st.success("✅ 'google_sheets' key found in secrets")
        
        # Test 2: Check specific keys
        required_keys = ['type', 'project_id', 'private_key_id', 'private_key', 
                        'client_email', 'client_id', 'auth_uri', 'token_uri']
        missing_keys = []
        for key in required_keys:
            if key not in st.secrets['google_sheets']:
                missing_keys.append(key)
        
        if missing_keys:
            st.error(f"❌ Missing keys: {missing_keys}")
        else:
            st.success("✅ All required keys present")
            
            # Test 3: Test connection to Google Sheets
            st.subheader("Test 3: Google Sheets Connection")
            try:
                import gspread
                from google.oauth2.service_account import Credentials
                
                # Create credentials
                creds_dict = dict(st.secrets['google_sheets'])
                credentials = Credentials.from_service_account_info(
                    creds_dict,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
                
                # Test connection
                client = gspread.authorize(credentials)
                
                # Try to open the spreadsheet
                spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/13Zw4TksoH9P1PWv1HuNpapZOOnT_dIm_Pr85qRof4yE/edit#gid=0")
                worksheet = spreadsheet.sheet1
                
                # Test read access
                test_data = worksheet.get_all_values()
                st.success(f"✅ Successfully connected! Found {len(test_data)} rows in sheet")
                
                # Test write access
                test_timestamp = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
                test_row = [test_timestamp, "Test Home", "Test Away", "Test Pattern", 
                           "Test Bet", "1x", "Medium", "PENDING", "", "", "Diagnostic Test"]
                worksheet.append_row(test_row)
                st.success("✅ Successfully wrote test row to Google Sheet!")
                
            except gspread.exceptions.APIError as e:
                st.error(f"❌ Google Sheets API Error: {str(e)}")
                if "PERMISSION_DENIED" in str(e):
                    st.info("📋 **Solution**: Make sure you've shared the Google Sheet with the service account email: ozone-football@football-prediction-tracker.iam.gserviceaccount.com")
                elif "invalid_grant" in str(e).lower():
                    st.info("🔑 **Solution**: The private key format may still be incorrect. Try regenerating the service account credentials.")
            except Exception as e:
                st.error(f"❌ Connection Error: {str(e)}")
    else:
        st.error("❌ 'google_sheets' key NOT found in secrets")
        st.info("📋 **Solution**: Make sure your .streamlit/secrets.toml file is in the correct location and contains the [google_sheets] section")
    
    st.divider()
    # ========== END DIAGNOSTIC ==========
    st.set_page_config(
        page_title="Football Predictor Pro v2.0 - WITH TRACKING",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize Google Sheets tracker
    sheets_tracker = GoogleSheetsTracker()
    
    # Initialize prediction engine
    engine = PredictionEngineV2()
    
    # League contexts
    LEAGUE_CONTEXTS = {
        "Premier League": MatchContext(league_avg_goals=2.7, league_avg_xg=1.35, home_advantage=1.15),
        "La Liga": MatchContext(league_avg_goals=2.5, league_avg_xg=1.25, home_advantage=1.18),
        "Bundesliga": MatchContext(league_avg_goals=3.0, league_avg_xg=1.50, home_advantage=1.10),
        "Serie A": MatchContext(league_avg_goals=2.6, league_avg_xg=1.30, home_advantage=1.16),
        "Ligue 1": MatchContext(league_avg_goals=2.4, league_avg_xg=1.20, home_advantage=1.17),
        "Liga NOS": MatchContext(league_avg_goals=2.68, league_avg_xg=1.34, home_advantage=1.15),
        "Championship": MatchContext(league_avg_goals=2.55, league_avg_xg=1.28, home_advantage=1.16),
        "Other": MatchContext()
    }
    
    # Main app with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 PREDICT", 
        "📋 PREDICTION HISTORY", 
        "📈 PERFORMANCE DASHBOARD",
        "🔍 UPDATE RESULTS"
    ])
    
    with tab1:
        # ========== YOUR EXISTING PREDICTION INTERFACE ==========
        st.title("⚽ Football Predictor Pro v2.0")
        st.caption("Enhanced with xG Integration + Google Sheets Tracking 📊")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            league = st.selectbox("Select League", list(LEAGUE_CONTEXTS.keys()), index=6)
            engine.context = LEAGUE_CONTEXTS[league]
            
            # Show league stats
            with st.expander("League Statistics"):
                st.write(f"**Average Goals:** {engine.context.league_avg_goals:.2f}")
                st.write(f"**Average xG per team:** {engine.context.league_avg_xg:.2f}")
                st.write(f"**Home Advantage:** {engine.context.home_advantage:.2f}x")
                st.write(f"**Away Penalty:** {engine.context.away_penalty:.2f}x")
            
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
            if st.button("Clear All Data"):
                clear_session_state()
                st.rerun()
            
            st.info("""
            **Version 2.0 Features:**
            - xG Integration for better accuracy
            - Sample-size aware predictions
            - Advanced defensive analysis
            - xG pattern detection
            - **NEW: Google Sheets Tracking**
            """)
        
        # Your existing input form
        st.header("📊 Enter Match Data")
        
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
        tab1a, tab1b, tab1c = st.tabs(["⚽ Core Stats", "📈 xG Stats", "📊 Recent Form"])
        
        with tab1a:
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
        
        with tab1b:
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
        
        with tab1c:
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
                
                # Pattern detection
                patterns_detected = engine.detect_high_confidence_patterns(home_metrics, away_metrics)
                pattern_advice = engine.get_pattern_based_advice(patterns_detected)
                
                # Store predictions in session state for saving
                st.session_state['last_prediction'] = {
                    'home_team': home_name,
                    'away_team': away_name,
                    'result_pred': result_pred,
                    'over_under_pred': over_under_pred,
                    'btts_pred': btts_pred,
                    'patterns_detected': patterns_detected,
                    'pattern_advice': pattern_advice,
                    'expected_goals': expected_goals,
                    'timestamp': datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Display results
                st.success("✅ Advanced Predictions Generated")
                
                # ========== PATTERN SIGNALS SECTION ==========
                if patterns_detected:
                    st.header("🎯 TRUST THE PATTERN Signals 🔥")
                    
                    # Display each detected pattern with FIRE emoji
                    for pattern in patterns_detected:
                        with st.container():
                            st.markdown(f"### {pattern['signal']} {pattern['name']}")
                            
                            col1, col2 = st.columns([3, 2])
                            with col1:
                                st.info(pattern['description'])
                            with col2:
                                st.success(f"**PRIMARY BET:** {pattern['bet']}")
                                st.caption(f"**Stake:** {pattern['stake']}")
                            
                            # Additional bets
                            if pattern.get('additional_bets'):
                                with st.expander("Additional Betting Options"):
                                    for bet in pattern['additional_bets']:
                                        st.write(f"• {bet}")
                            
                            st.caption(f"**Validation:** {pattern['validation']}")
                            st.divider()
                    
                    # Pattern-based betting advice
                    st.subheader("💰 PATTERN-BASED BETTING ADVICE")
                    
                    advice_container = st.container()
                    with advice_container:
                        col_adv1, col_adv2, col_adv3 = st.columns(3)
                        
                        with col_adv1:
                            if pattern_advice['primary_bet']:
                                st.success(f"**Primary Bet:**\n{pattern_advice['primary_bet']}")
                        
                        with col_adv2:
                            st.warning(f"**Stake Level:**\n{pattern_advice['stake']}")
                        
                        with col_adv3:
                            st.info(f"**Confidence:**\n{pattern_advice['confidence']}")
                        
                        st.markdown(f"**Advice:** {pattern_advice['advice']}")
                        
                    # Check for engine contradictions
                    engine_ou_pred = over_under_pred['prediction'].value
                    engine_btts_pred = btts_pred['prediction'].value
                    
                    # Determine pattern suggestion
                    if patterns_detected[0]['pattern'] == 1:  # Defensive Battle
                        pattern_suggests = "UNDER 2.5 & BTTS NO"
                    elif patterns_detected[0]['pattern'] == 2:  # Regression Explosion
                        pattern_suggests = "OVER 2.5 & BTTS YES"
                    elif patterns_detected[0]['pattern'] == 3:  # Regression Suppression
                        pattern_suggests = "UNDER 2.5"
                    else:
                        pattern_suggests = ""
                    
                    # Check for contradictions
                    if pattern_suggests:
                        if ("UNDER" in pattern_suggests and "OVER" in engine_ou_pred) or \
                           ("OVER" in pattern_suggests and "UNDER" in engine_ou_pred):
                            st.error("⚠️ **ENGINE CONTRADICTION DETECTED**")
                            st.warning(f"Pattern suggests: {pattern_suggests}")
                            st.warning(f"Engine predicts: {engine_ou_pred} & {engine_btts_pred}")
                            st.info("**TRUST THE PATTERN** - Our high-accuracy patterns have proven more reliable!")
                else:
                    st.warning("⚠️ No high-confidence patterns detected")
                    st.info("Proceed with standard betting strategy using engine predictions below.")
                    st.caption("Consider reduced stakes (0.5x normal) due to lack of clear patterns.")
                
                # Main predictions in cards
                st.header("🎯 Core Predictions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🏆 Match Result")
                    pred = result_pred['prediction'].value
                    st.metric("Prediction", pred)
                    
                    # Probability bars with better formatting
                    home_prob = result_pred['probabilities']['home_win']
                    draw_prob = result_pred['probabilities']['draw']
                    away_prob = result_pred['probabilities']['away_win']
                    
                    # Color-coded probabilities
                    col_prob = st.columns(3)
                    with col_prob[0]:
                        st.metric("Home", f"{home_prob:.1%}")
                    with col_prob[1]:
                        st.metric("Draw", f"{draw_prob:.1%}")
                    with col_prob[2]:
                        st.metric("Away", f"{away_prob:.1%}")
                    
                    # Form factors
                    with st.expander("Form Factors"):
                        home_form = result_pred['form_factors']['home']
                        away_form = result_pred['form_factors']['away']
                        
                        if home_form >= 1.15:
                            st.success(f"**{home_name}:** {home_form:.2f}x (Excellent form)")
                        elif home_form >= 1.05:
                            st.success(f"**{home_name}:** {home_form:.2f}x (Good form)")
                        elif home_form <= 0.70:
                            st.error(f"**{home_name}:** {home_form:.2f}x (Very poor form)")
                        elif home_form <= 0.85:
                            st.error(f"**{home_name}:** {home_form:.2f}x (Poor form)")
                        elif home_form < 0.95:
                            st.warning(f"**{home_name}:** {home_form:.2f}x (Below average form)")
                        else:
                            st.info(f"**{home_name}:** {home_form:.2f}x (Average form)")
                        
                        if away_form >= 1.15:
                            st.success(f"**{away_name}:** {away_form:.2f}x (Excellent form)")
                        elif away_form >= 1.05:
                            st.success(f"**{away_name}:** {away_form:.2f}x (Good form)")
                        elif away_form <= 0.70:
                            st.error(f"**{away_name}:** {away_form:.2f}x (Very poor form)")
                        elif away_form <= 0.85:
                            st.error(f"**{away_name}:** {away_form:.2f}x (Poor form)")
                        elif away_form < 0.95:
                            st.warning(f"**{away_name}:** {away_form:.2f}x (Below average form)")
                        else:
                            st.info(f"**{away_name}:** {away_form:.2f}x (Average form)")
                
                with col2:
                    st.subheader("⚖️ Over/Under 2.5")
                    pred = over_under_pred['prediction'].value
                    conf = over_under_pred['confidence']
                    st.metric("Prediction", pred)
                    st.metric("Confidence", conf)
                    st.metric("Expected Goals", over_under_pred['expected_goals'])
                    
                    # Probability comparison
                    over_prob = over_under_pred['probabilities']['over']
                    under_prob = over_under_pred['probabilities']['under']
                    
                    col_ou = st.columns(2)
                    with col_ou[0]:
                        st.metric("Over", f"{over_prob:.1%}")
                    with col_ou[1]:
                        st.metric("Under", f"{under_prob:.1%}")
                
                with col3:
                    st.subheader("🎯 Both Teams to Score")
                    pred = btts_pred['prediction'].value
                    conf = btts_pred['confidence']
                    st.metric("Prediction", pred)
                    st.metric("Confidence", conf)
                    
                    # Probability comparison
                    yes_prob = btts_pred['probabilities']['btts_yes']
                    no_prob = btts_pred['probabilities']['btts_no']
                    
                    col_btts = st.columns(2)
                    with col_btts[0]:
                        st.metric("Yes", f"{yes_prob:.1%}")
                    with col_btts[1]:
                        st.metric("No", f"{no_prob:.1%}")
                    
                    # H2H influence
                    if h2h_btts:
                        with st.expander("H2H Influence"):
                            st.write(f"H2H BTTS: {h2h_btts}%")
                            st.write(f"Meetings: {h2h_meetings}")
                
                # Expected goals breakdown
                st.header("📊 Expected Goals Analysis")
                
                eg_home, eg_away = expected_goals
                total_expected = eg_home + eg_away
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    delta_home = eg_home - home_attack
                    st.metric(f"🏠 {home_name}", f"{eg_home:.2f}", f"{delta_home:+.2f} vs avg")
                with col2:
                    delta_away = eg_away - away_attack
                    st.metric(f"🚗 {away_name}", f"{eg_away:.2f}", f"{delta_away:+.2f} vs avg")
                with col3:
                    delta_total = total_expected - engine.context.league_avg_goals
                    st.metric("Total Expected", f"{total_expected:.2f}", f"{delta_total:+.2f} vs league avg")
                
                # xG adjustments
                if over_under_pred.get('xg_adjustments'):
                    st.header("🔍 xG-Based Adjustments")
                    for adjustment in over_under_pred['xg_adjustments']:
                        if "regression" in adjustment.lower():
                            st.warning(adjustment)
                        else:
                            st.info(adjustment)
                
                # Match patterns and insights
                if patterns:
                    st.header("🧠 Key Insights & Patterns")
                    
                    # Remove duplicates while preserving order
                    unique_patterns = []
                    seen_patterns = set()
                    for pattern in patterns:
                        # Create a simplified key for deduplication
                        key = ''.join(filter(str.isalpha, pattern.lower()))
                        if key not in seen_patterns:
                            seen_patterns.add(key)
                            unique_patterns.append(pattern)
                    
                    # Display unique patterns in columns
                    if unique_patterns:
                        col1, col2 = st.columns(2)
                        mid_point = len(unique_patterns) // 2
                        
                        with col1:
                            for pattern in unique_patterns[:mid_point]:
                                if "🔥" in pattern:
                                    st.error(f"• {pattern}")
                                elif "EXCELLENT" in pattern or "STRONG attack" in pattern:
                                    st.success(f"• {pattern}")
                                elif "VERY POOR" in pattern or "POOR form" in pattern or "WEAK" in pattern or "struggle" in pattern:
                                    st.error(f"• {pattern}")
                                elif "due for goals" in pattern.lower():
                                    st.warning(f"• {pattern}")
                                elif "regression possible" in pattern.lower():
                                    st.warning(f"• {pattern}")
                                else:
                                    st.info(f"• {pattern}")
                        
                        with col2:
                            for pattern in unique_patterns[mid_point:]:
                                if "🔥" in pattern:
                                    st.error(f"• {pattern}")
                                elif "EXCELLENT" in pattern or "STRONG attack" in pattern:
                                    st.success(f"• {pattern}")
                                elif "VERY POOR" in pattern or "POOR form" in pattern or "WEAK" in pattern or "struggle" in pattern:
                                    st.error(f"• {pattern}")
                                elif "due for goals" in pattern.lower():
                                    st.warning(f"• {pattern}")
                                elif "regression possible" in pattern.lower():
                                    st.warning(f"• {pattern}")
                                else:
                                    st.info(f"• {pattern}")
                
                # Defensive analysis
                st.header("🛡️ Defensive Analysis")
                
                home_def_analysis = result_pred['defensive_analysis']['home']
                away_def_analysis = result_pred['defensive_analysis']['away']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{home_name} Defense")
                    
                    # Strength indicator
                    if home_def_analysis['is_very_strong']:
                        strength = "🎯 **VERY STRONG**"
                    elif home_def_analysis['is_strong']:
                        strength = "✅ **Strong**"
                    elif home_def_analysis['is_weak']:
                        strength = "⚠️ **Weak**"
                    else:
                        strength = "📊 **Average**"
                    
                    st.markdown(f"**Strength:** {strength}")
                    st.write(f"**Goals Conceded/Game:** {home_defense:.2f}")
                    st.write(f"**xG Conceded/Game:** {home_xg_against:.2f}")
                    
                    if home_def_analysis['xg_better_than_actual']:
                        st.success(f"✅ Defense BETTER than stats show")
                        st.caption(f"(Concedes {home_defense:.2f} but xGA suggests {home_xg_against:.2f} - conceding LESS than expected)")
                    elif home_def_analysis['xg_worse_than_actual']:
                        st.error(f"⚠️ Defense WORSE than stats show")
                        st.caption(f"(Concedes {home_defense:.2f} but xGA suggests {home_xg_against:.2f} - conceding MORE than expected)")
                    else:
                        st.info(f"📊 Defense matches expected performance")
                    
                    if home_def_analysis['clean_sheet_likely']:
                        st.success(f"✅ High clean sheet probability ({home_cs}%)")
                
                with col2:
                    st.subheader(f"{away_name} Defense")
                    
                    # Strength indicator
                    if away_def_analysis['is_very_strong']:
                        strength = "🎯 **VERY STRONG**"
                    elif away_def_analysis['is_strong']:
                        strength = "✅ **Strong**"
                    elif away_def_analysis['is_weak']:
                        strength = "⚠️ **Weak**"
                    else:
                        strength = "📊 **Average**"
                    
                    st.markdown(f"**Strength:** {strength}")
                    st.write(f"**Goals Conceded/Game:** {away_defense:.2f}")
                    st.write(f"**xG Conceded/Game:** {away_xg_against:.2f}")
                    
                    if away_def_analysis['xg_better_than_actual']:
                        st.success(f"✅ Defense BETTER than stats show")
                        st.caption(f"(Concedes {away_defense:.2f} but xGA suggests {away_xg_against:.2f} - conceding LESS than expected)")
                    elif away_def_analysis['xg_worse_than_actual']:
                        st.error(f"⚠️ Defense WORSE than stats show")
                        st.caption(f"(Concedes {away_defense:.2f} but xGA suggests {away_xg_against:.2f} - conceding MORE than expected)")
                    else:
                        st.info(f"📊 Defense matches expected performance")
                    
                    if away_def_analysis['clean_sheet_likely']:
                        st.success(f"✅ High clean sheet probability ({away_cs}%)")
                
                # League context and final summary
                st.header("🏆 Final Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("League Avg Goals", f"{engine.context.league_avg_goals:.2f}")
                with col2:
                    st.metric("League Avg xG/Team", f"{engine.context.league_avg_xg:.2f}")
                with col3:
                    st.metric("Predicted Total", f"{total_expected:.2f}")
                
                # Final recommendation
                st.subheader("🎯 Final Recommendation")
                
                # Create a summary based on all predictions
                recommendations = []
                
                # Match result
                if result_pred['prediction'] == Prediction.HOME_WIN:
                    recommendations.append(f"**Home Win** ({result_pred['probabilities']['home_win']:.1%} probability)")
                elif result_pred['prediction'] == Prediction.AWAY_WIN:
                    recommendations.append(f"**Away Win** ({result_pred['probabilities']['away_win']:.1%} probability)")
                else:
                    recommendations.append(f"**Draw** ({result_pred['probabilities']['draw']:.1%} probability)")
                
                # Over/Under
                recommendations.append(f"**{over_under_pred['prediction'].value}** ({over_under_pred['probabilities']['over' if 'Over' in over_under_pred['prediction'].value else 'under']:.1%} probability)")
                
                # BTTS
                recommendations.append(f"**{btts_pred['prediction'].value}** ({btts_pred['probabilities']['btts_yes' if 'Yes' in btts_pred['prediction'].value else 'btts_no']:.1%} probability)")
                
                # Display recommendations
                st.info(" | ".join(recommendations))
                
                # Risk assessment
                st.subheader("⚠️ Risk Assessment")
                
                risks = []
                if over_under_pred['confidence'] in ["Low", "Very Low", "Toss-up"]:
                    risks.append("Over/Under prediction has low confidence")
                if btts_pred['confidence'] in ["Low", "Very Low", "Toss-up"]:
                    risks.append("BTTS prediction is very close to 50/50")
                if abs(result_pred['probabilities']['home_win'] - result_pred['probabilities']['away_win']) < 0.15:
                    risks.append("Match result prediction is close")
                
                if risks:
                    for risk in risks:
                        st.warning(risk)
                else:
                    st.success("All predictions have reasonable confidence levels")
        
        # NEW: Save to Google Sheets button (only shows after prediction)
        if 'last_prediction' in st.session_state:
            st.divider()
            st.subheader("💾 Save Prediction")
            
            col_save1, col_save2 = st.columns([3, 1])
            
            with col_save1:
                if st.button("💾 Save Prediction to Google Sheets", type="secondary", use_container_width=True):
                    with st.spinner("Saving to Google Sheets..."):
                        # Prepare data for saving
                        pred_data = st.session_state['last_prediction']
                        pattern_info = pred_data['patterns_detected'][0] if pred_data['patterns_detected'] else {}
                        
                        save_data = {
                            'home_team': pred_data['home_team'],
                            'away_team': pred_data['away_team'],
                            'pattern': pattern_info.get('name', 'No pattern detected'),
                            'primary_bet': pattern_info.get('bet', 'No pattern bet'),
                            'stake': pattern_info.get('stake', 'NORMAL (1x)'),
                            'confidence': pattern_info.get('confidence', 'Medium'),
                            'notes': f"Expected: {pred_data['expected_goals'][0]:.1f}-{pred_data['expected_goals'][1]:.1f}"
                        }
                        
                        # Save to Google Sheets
                        result = sheets_tracker.save_prediction(save_data)
                        
                        if result and result['success']:
                            st.success(f"✅ Prediction saved to Google Sheets! (Row {result['row_number']})")
                            st.markdown(f"[📊 Open Google Sheet]({result['sheet_url']})")
                        else:
                            st.error("Failed to save to Google Sheets. Check credentials.")
            
            with col_save2:
                st.markdown("""
                <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>
                📊 Save to track performance
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        # ========== PREDICTION HISTORY TAB ==========
        st.title("📋 Prediction History")
        st.caption("View all saved predictions from Google Sheets")
        
        try:
            # Fetch predictions from Google Sheets
            with st.spinner("Loading predictions from Google Sheets..."):
                predictions_df = sheets_tracker.get_predictions(limit=100)
            
            if predictions_df.empty:
                st.info("No predictions saved yet. Generate and save predictions in the PREDICT tab.")
            else:
                # Display statistics
                total_predictions = len(predictions_df)
                pending = len(predictions_df[predictions_df['Status'] == 'PENDING'])
                completed = total_predictions - pending
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", total_predictions)
                with col2:
                    st.metric("Pending", pending)
                with col3:
                    st.metric("Completed", completed)
                
                # Filter options
                st.subheader("Filter Predictions")
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                with col_filter1:
                    status_filter = st.selectbox(
                        "Status",
                        ["All", "PENDING", "COMPLETED"],
                        key="status_filter"
                    )
                
                with col_filter2:
                    team_filter = st.text_input(
                        "Team (optional)",
                        placeholder="Enter team name",
                        key="team_filter"
                    )
                
                with col_filter3:
                    pattern_filter = st.selectbox(
                        "Pattern",
                        ["All"] + list(predictions_df['Pattern'].unique()) if 'Pattern' in predictions_df.columns else ["All"],
                        key="pattern_filter"
                    )
                
                # Apply filters
                filtered_df = predictions_df.copy()
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['Status'] == status_filter]
                if team_filter:
                    filtered_df = filtered_df[
                        filtered_df['Home Team'].str.contains(team_filter, case=False, na=False) | 
                        filtered_df['Away Team'].str.contains(team_filter, case=False, na=False)
                    ]
                if pattern_filter != "All" and 'Pattern' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Pattern'] == pattern_filter]
                
                # Display table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    column_config={
                        "Timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                        "Home Team": st.column_config.TextColumn("Home"),
                        "Away Team": st.column_config.TextColumn("Away"),
                        "Pattern": st.column_config.TextColumn("Pattern"),
                        "Primary Bet": st.column_config.TextColumn("Bet"),
                        "Stake": st.column_config.TextColumn("Stake"),
                        "Confidence": st.column_config.TextColumn("Conf"),
                        "Status": st.column_config.TextColumn("Status"),
                        "Actual Score": st.column_config.TextColumn("Score"),
                        "Bet Outcome": st.column_config.TextColumn("Outcome"),
                        "Notes": st.column_config.TextColumn("Notes", width="large")
                    },
                    hide_index=True
                )
                
                # Download option
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="football_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
            st.info("Make sure Google Sheets credentials are set up correctly.")
    
    with tab3:
        # ========== PERFORMANCE DASHBOARD TAB ==========
        st.title("📈 Performance Dashboard")
        st.caption("Track your prediction performance over time")
        
        try:
            # Fetch predictions
            predictions_df = sheets_tracker.get_predictions(limit=200)
            
            if predictions_df.empty:
                st.info("No predictions saved yet. Save some predictions first.")
            else:
                # Filter only completed predictions with outcomes
                completed_df = predictions_df[
                    (predictions_df['Status'] == 'COMPLETED') & 
                    (predictions_df['Bet Outcome'].isin(['WIN', 'LOSS', 'PUSH']))
                ]
                
                if completed_df.empty:
                    st.info("No completed predictions with outcomes yet. Update results in the UPDATE RESULTS tab.")
                else:
                    # Calculate metrics
                    total_bets = len(completed_df)
                    wins = len(completed_df[completed_df['Bet Outcome'] == 'WIN'])
                    losses = len(completed_df[completed_df['Bet Outcome'] == 'LOSS'])
                    pushes = len(completed_df[completed_df['Bet Outcome'] == 'PUSH'])
                    
                    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
                    
                    # Display KPIs
                    st.subheader("📊 Overall Performance")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Bets", total_bets)
                    with col2:
                        st.metric("Wins", wins)
                    with col3:
                        st.metric("Losses", losses)
                    with col4:
                        st.metric("Pushes", pushes)
                    with col5:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    
                    # Pattern performance
                    st.subheader("🎯 Pattern Performance")
                    
                    if 'Pattern' in completed_df.columns:
                        pattern_stats = []
                        for pattern in completed_df['Pattern'].unique():
                            pattern_df = completed_df[completed_df['Pattern'] == pattern]
                            pattern_wins = len(pattern_df[pattern_df['Bet Outcome'] == 'WIN'])
                            pattern_losses = len(pattern_df[pattern_df['Bet Outcome'] == 'LOSS'])
                            pattern_total = pattern_wins + pattern_losses
                            pattern_win_rate = (pattern_wins / pattern_total * 100) if pattern_total > 0 else 0
                            
                            pattern_stats.append({
                                'Pattern': pattern,
                                'Total Bets': len(pattern_df),
                                'Wins': pattern_wins,
                                'Losses': pattern_losses,
                                'Win Rate %': round(pattern_win_rate, 1)
                            })
                        
                        if pattern_stats:
                            pattern_df = pd.DataFrame(pattern_stats)
                            pattern_df = pattern_df.sort_values('Win Rate %', ascending=False)
                            st.dataframe(pattern_df, use_container_width=True, hide_index=True)
                    
                    # Monthly performance
                    st.subheader("📅 Monthly Trend")
                    
                    if 'Timestamp' in completed_df.columns:
                        completed_df['Month'] = pd.to_datetime(completed_df['Timestamp']).dt.to_period('M').astype(str)
                        monthly_stats = completed_df.groupby('Month').agg({
                            'Bet Outcome': ['count', lambda x: (x == 'WIN').sum()]
                        }).round(1)
                        
                        if not monthly_stats.empty:
                            monthly_stats.columns = ['Total Bets', 'Wins']
                            monthly_stats['Win Rate %'] = (monthly_stats['Wins'] / monthly_stats['Total Bets'] * 100).round(1)
                            st.dataframe(monthly_stats, use_container_width=True)
                    
                    # Stake performance
                    st.subheader("💰 Stake Performance")
                    
                    if 'Stake' in completed_df.columns:
                        stake_stats = []
                        for stake in completed_df['Stake'].unique():
                            stake_df = completed_df[completed_df['Stake'] == stake]
                            stake_wins = len(stake_df[stake_df['Bet Outcome'] == 'WIN'])
                            stake_losses = len(stake_df[stake_df['Bet Outcome'] == 'LOSS'])
                            stake_total = stake_wins + stake_losses
                            stake_win_rate = (stake_wins / stake_total * 100) if stake_total > 0 else 0
                            
                            stake_stats.append({
                                'Stake': stake,
                                'Total Bets': len(stake_df),
                                'Wins': stake_wins,
                                'Losses': stake_losses,
                                'Win Rate %': round(stake_win_rate, 1)
                            })
                        
                        if stake_stats:
                            stake_df = pd.DataFrame(stake_stats)
                            stake_df = stake_df.sort_values('Win Rate %', ascending=False)
                            st.dataframe(stake_df, use_container_width=True, hide_index=True)
                    
                    # Recent performance
                    st.subheader("📈 Recent Performance (Last 10 Bets)")
                    
                    recent_df = completed_df.head(10)
                    if not recent_df.empty:
                        recent_wins = len(recent_df[recent_df['Bet Outcome'] == 'WIN'])
                        recent_losses = len(recent_df[recent_df['Bet Outcome'] == 'LOSS'])
                        recent_total = recent_wins + recent_losses
                        recent_win_rate = (recent_wins / recent_total * 100) if recent_total > 0 else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Recent Win Rate", f"{recent_win_rate:.1f}%")
                        with col2:
                            trend = "📈 Improving" if recent_win_rate > win_rate else "📉 Declining" if recent_win_rate < win_rate else "➡️ Stable"
                            st.metric("Trend", trend)
        
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
    
    with tab4:
        # ========== UPDATE RESULTS TAB ==========
        st.title("🔍 Update Match Results")
        st.caption("Enter actual match results to track prediction performance")
        
        try:
            # Fetch predictions from Google Sheets
            with st.spinner("Loading predictions from Google Sheets..."):
                predictions_df = sheets_tracker.get_predictions(limit=100)
            
            if predictions_df.empty:
                st.info("No predictions found. Save some predictions first.")
            else:
                # Get pending predictions
                pending_df = predictions_df[predictions_df['Status'] == 'PENDING']
                
                if pending_df.empty:
                    st.success("🎉 All predictions are up to date!")
                    st.info("No pending matches to update.")
                else:
                    # Select match to update
                    st.subheader("Select Match to Update")
                    
                    # Create display options
                    match_options = []
                    for idx, row in pending_df.iterrows():
                        match_str = f"{row.get('Home Team', 'Home')} vs {row.get('Away Team', 'Away')} - {row.get('Pattern', 'No pattern')} - {row.get('Primary Bet', 'No bet')}"
                        match_options.append((idx, match_str, row))
                    
                    if match_options:
                        selected_option = st.selectbox(
                            "Choose a match:",
                            options=[opt[1] for opt in match_options],
                            index=0
                        )
                        
                        # Get the selected row
                        selected_idx = [opt[0] for opt in match_options if opt[1] == selected_option][0]
                        selected_row = [opt[2] for opt in match_options if opt[1] == selected_option][0]
                        
                        # Display current prediction
                        with st.expander("📋 View Prediction Details", expanded=True):
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.write(f"**Home Team:** {selected_row.get('Home Team', '')}")
                                st.write(f"**Away Team:** {selected_row.get('Away Team', '')}")
                                st.write(f"**Pattern:** {selected_row.get('Pattern', '')}")
                                st.write(f"**Prediction Date:** {selected_row.get('Timestamp', '')}")
                            with col_info2:
                                st.write(f"**Primary Bet:** {selected_row.get('Primary Bet', '')}")
                                st.write(f"**Stake:** {selected_row.get('Stake', '')}")
                                st.write(f"**Confidence:** {selected_row.get('Confidence', '')}")
                                st.write(f"**Status:** {selected_row.get('Status', 'PENDING')}")
                        
                        # Result input form
                        st.subheader("⚽ Enter Match Results")
                        
                        col_score1, col_score2, col_score3 = st.columns(3)
                        with col_score1:
                            home_score = st.number_input("Home Score", 0, 20, 0, key="home_score_input")
                        with col_score2:
                            away_score = st.number_input("Away Score", 0, 20, 0, key="away_score_input")
                        with col_score3:
                            total_goals = home_score + away_score
                            st.metric("Total Goals", total_goals)
                        
                        # Calculate BTTS
                        btts_actual = "YES" if home_score > 0 and away_score > 0 else "NO"
                        st.write(f"**Both Teams Scored:** {btts_actual}")
                        
                        # Match result
                        match_result = "Home Win" if home_score > away_score else "Away Win" if away_score > home_score else "Draw"
                        st.write(f"**Match Result:** {match_result}")
                        
                        # Determine bet outcome
                        st.subheader("💰 Determine Bet Outcome")
                        
                        # Get the predicted bet from the selected row
                        predicted_bet = selected_row.get('Primary Bet', '')
                        bet_outcome = "PUSH"  # Default
                        
                        if "UNDER 2.5" in predicted_bet.upper():
                            if total_goals < 2.5:
                                bet_outcome = "WIN"
                            elif total_goals > 2.5:
                                bet_outcome = "LOSS"
                            else:
                                bet_outcome = "PUSH"
                        elif "OVER 2.5" in predicted_bet.upper():
                            if total_goals > 2.5:
                                bet_outcome = "WIN"
                            elif total_goals < 2.5:
                                bet_outcome = "LOSS"
                            else:
                                bet_outcome = "PUSH"
                        elif "BTTS NO" in predicted_bet.upper():
                            if btts_actual == "NO":
                                bet_outcome = "WIN"
                            else:
                                bet_outcome = "LOSS"
                        elif "BTTS YES" in predicted_bet.upper():
                            if btts_actual == "YES":
                                bet_outcome = "WIN"
                            else:
                                bet_outcome = "LOSS"
                        elif "WIN" in predicted_bet.upper():
                            # For match winner bets
                            predicted_winner = "Home" if "HOME" in predicted_bet.upper() else "Away" if "AWAY" in predicted_bet.upper() else ""
                            if predicted_winner:
                                if predicted_winner == "Home" and home_score > away_score:
                                    bet_outcome = "WIN"
                                elif predicted_winner == "Away" and away_score > home_score:
                                    bet_outcome = "WIN"
                                else:
                                    bet_outcome = "LOSS"
                        
                        # Display outcome
                        outcome_col1, outcome_col2 = st.columns(2)
                        with outcome_col1:
                            if bet_outcome == "WIN":
                                st.success(f"✅ **Bet Outcome:** {bet_outcome}")
                            elif bet_outcome == "LOSS":
                                st.error(f"❌ **Bet Outcome:** {bet_outcome}")
                            else:
                                st.warning(f"⚪ **Bet Outcome:** {bet_outcome}")
                        
                        with outcome_col2:
                            notes = st.text_area("Notes (optional)", placeholder="Key match events, injuries, weather conditions, etc.", height=100)
                        
                        # Update button
                        if st.button("💾 Save Results to Google Sheets", type="primary", use_container_width=True):
                            with st.spinner("Updating Google Sheets..."):
                                success = sheets_tracker.update_result(
                                    row_index=selected_idx,
                                    actual_score=f"{home_score}-{away_score}",
                                    bet_outcome=bet_outcome,
                                    notes=notes
                                )
                                
                                if success:
                                    st.success("✅ Results saved successfully!")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error("Failed to save results. Please try again.")
                    else:
                        st.info("No pending matches found.")
        
        except Exception as e:
            st.error(f"Error loading pending matches: {str(e)}")

# ========== YOUR EXISTING HELPER FUNCTIONS ==========

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
    st.session_state.home_xg_against = 1.39
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
    st.session_state.away_xg_against = 1.48
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
        'h2h_btts', 'h2h_meetings',
        'last_prediction'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

if __name__ == "__main__":
    main()
