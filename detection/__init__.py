"""Detection module initialization"""
from .sleeper_scanner import SleeperScanner, get_sleeper_scanner
from .sleeper_scanner_v3 import SleeperScannerV3, get_sleeper_scanner_v3
from .ob_scanner import OBScanner, get_ob_scanner
from .signal_merger import SignalMerger, get_signal_merger
from .trend_analyzer import TrendAnalyzer, TrendRegime, TrendScore, get_trend_analyzer
from .direction_engine import DirectionEngine, Direction, DirectionResult, get_direction_engine, resolve_direction

__all__ = [
    # Sleeper Scanner (legacy)
    'SleeperScanner', 'get_sleeper_scanner',
    # Sleeper Scanner v3 (5-day strategy)
    'SleeperScannerV3', 'get_sleeper_scanner_v3',
    # Direction Engine v1 (professional direction model)
    'DirectionEngine', 'Direction', 'DirectionResult', 'get_direction_engine', 'resolve_direction',
    # Other scanners
    'OBScanner', 'get_ob_scanner',
    'SignalMerger', 'get_signal_merger',
    'TrendAnalyzer', 'TrendRegime', 'TrendScore', 'get_trend_analyzer'
]
