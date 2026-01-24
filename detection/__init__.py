"""Detection module initialization"""
from .sleeper_scanner import SleeperScanner, get_sleeper_scanner
from .sleeper_scanner_v3 import SleeperScannerV3, get_sleeper_scanner_v3
from .ob_scanner import OBScanner, get_ob_scanner
from .signal_merger import SignalMerger, get_signal_merger
from .trend_analyzer import TrendAnalyzer, TrendRegime, TrendScore, get_trend_analyzer

__all__ = [
    # Sleeper Scanner (legacy)
    'SleeperScanner', 'get_sleeper_scanner',
    # Sleeper Scanner v3 (5-day strategy)
    'SleeperScannerV3', 'get_sleeper_scanner_v3',
    # Other scanners
    'OBScanner', 'get_ob_scanner',
    'SignalMerger', 'get_signal_merger',
    'TrendAnalyzer', 'TrendRegime', 'TrendScore', 'get_trend_analyzer'
]
