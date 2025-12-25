#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SQUEEZE DETECTOR v1.0                                    ║
║                                                                              ║
║  Система виявлення прихованого накопичення перед памп/дамп                   ║
║                                                                              ║
║  Компоненти:                                                                 ║
║  • Recorder - збір даних через WebSocket + REST                              ║
║  • Analyzer - розрахунок K = ΔOI / ΔPrice                                    ║
║  • Models - SQLAlchemy моделі                                                ║
║  • Routes - Flask API                                                        ║
║  • Worker - Background процес                                                ║
║                                                                              ║
║  Автор: SVV Webhook Bot                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "1.0.0"
__author__ = "SVV Webhook Bot"

from .models import (
    MarketSnapshot,
    SqueezeSignal,
    SqueezeWatchlist,
    SqueezeConfig,
    create_squeeze_tables,
    run_migrations,
)

from .routes import (
    register_squeeze_detector_routes,
    get_detector_manager,
    set_bot_instance,
    SqueezeDetectorManager,
)

__all__ = [
    # Models
    'MarketSnapshot',
    'SqueezeSignal', 
    'SqueezeWatchlist',
    'SqueezeConfig',
    'create_squeeze_tables',
    'run_migrations',
    # Routes
    'register_squeeze_detector_routes',
    'get_detector_manager',
    'set_bot_instance',
    'SqueezeDetectorManager',
]
