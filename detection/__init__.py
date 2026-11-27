"""Detection module initialization"""
from .sleeper_scanner import SleeperScanner, get_sleeper_scanner
from .sleeper_scanner_v3 import SleeperScannerV3, get_sleeper_scanner_v3
from .ob_scanner import OBScanner, get_ob_scanner
from .signal_merger import SignalMerger, get_signal_merger
from .trend_analyzer import TrendAnalyzer, TrendRegime, TrendScore, get_trend_analyzer
from .direction_engine import DirectionEngine, Direction, DirectionResult, get_direction_engine, resolve_direction
from .direction_engine_v7 import DirectionEngineV7, DirectionResultV7, get_direction_engine_v7
from .direction_engine_v8 import DirectionEngineV8, DirectionResultV8, get_direction_engine_v8
from .smc_analyzer import SMCAnalyzer, SMCAnalysisResult, get_smc_analyzer, StructureSignal, MarketBias, PriceZone
from .entry_manager import EntryManager, EntrySetup, EntryState, StopLossMode, get_entry_manager
from .smc_signal_processor import SMCSignalProcessor, SMCSignalResult, get_smc_processor

__all__ = [
    # Sleeper Scanner (legacy)
    'SleeperScanner', 'get_sleeper_scanner',
    # Sleeper Scanner v3 (5-day strategy)
    'SleeperScannerV3', 'get_sleeper_scanner_v3',
    # Direction Engine v1 (professional direction model)
    'DirectionEngine', 'Direction', 'DirectionResult', 'get_direction_engine', 'resolve_direction',
    # Direction Engine v7 (Sleeper optimized)
    'DirectionEngineV7', 'DirectionResultV7', 'get_direction_engine_v7',
    # Direction Engine v8 (SMC integrated)
    'DirectionEngineV8', 'DirectionResultV8', 'get_direction_engine_v8',
    # SMC Analyzer
    'SMCAnalyzer', 'SMCAnalysisResult', 'get_smc_analyzer', 'StructureSignal', 'MarkerBias', 'PriceZone',
    # Entry Manager v8
    'EntryManager', 'EntrySetup', 'EntryState', 'StopLossMode', 'get_entry_manager',
    # SMC Signal Processor v8.1
    'SMCSignalProcessor', 'SMCSignalResult', 'get_smc_processor',
    # Other scanners
    'OBScanner', 'get_ob_scanner',
    'SignalMerger', 'get_signal_merger',
    'TrendAnalyzer', 'TrendRegime', 'TrendScore', 'get_trend_analyzer'
]
