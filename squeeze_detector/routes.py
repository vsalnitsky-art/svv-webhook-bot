#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SQUEEZE DETECTOR v1.0 - ROUTES                           ║
║                                                                              ║
║  Flask routes для UI та API:                                                 ║
║  • /squeeze - головна сторінка з heatmap                                     ║
║  • /squeeze/api/* - JSON API endpoints                                       ║
║                                                                              ║
║  Автор: SVV Webhook Bot                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Blueprint, render_template, jsonify, request

logger = logging.getLogger(__name__)

# Singleton instances
_detector_instance: Optional['SqueezeDetectorManager'] = None

# === SAFE IMPORTS ===
# Спробуємо відносні імпорти, якщо не працює - абсолютні

try:
    from .analyzer import DEFAULT_CONFIG
except ImportError:
    try:
        from squeeze_detector.analyzer import DEFAULT_CONFIG
    except ImportError:
        DEFAULT_CONFIG = {
            'sd_price_change_threshold': 2.0,
            'sd_oi_change_threshold': 5.0,
            'sd_k_coefficient_threshold': 3.0,
            'sd_lookback_4h': True,
            'sd_lookback_8h': True,
            'sd_lookback_24h': True,
            'sd_funding_extreme_positive': 0.0003,
            'sd_funding_extreme_negative': -0.0003,
            'sd_min_consecutive_signals': 2,
            'sd_ready_consecutive_signals': 4,
            'sd_watchlist_timeout_hours': 48,
            'sd_breakout_threshold': 3.0,
            'sd_base_confidence': 50,
            'sd_k_confidence_multiplier': 5,
            'sd_funding_confidence_bonus': 15,
        }
        logger.warning("Using fallback DEFAULT_CONFIG")


class SqueezeDetectorManager:
    """
    Головний менеджер Squeeze Detector.
    Координує Recorder та Analyzer.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_session_factory=None, bot_instance=None):
        if self._initialized:
            return
        
        self.db_session_factory = db_session_factory
        self.bot_instance = bot_instance
        
        self.recorder = None
        self.analyzer = None
        
        self.config = self._load_config()
        self.monitored_symbols = []
        
        # Status
        self.status = {
            'initialized': False,
            'recording': False,
            'analyzing': False,
            'last_error': None,
        }
        
        self._initialize()
        self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """Завантажує конфігурацію"""
        # Базові дефолти (Aggressive mode)
        config = {
            'sd_enabled': True,
            'sd_top_coins': 400,              # Aggressive default
            'sd_snapshot_interval': 60,       # 1 min
            'sd_analysis_interval': 60,       # 1 min
            'sd_min_volume_24h': 1_000_000,   # $1M
            # Analysis Method
            'sd_analysis_method': 'combined', # net_change | volatility_range | combined
            'sd_price_change_threshold': 2.5,
            'sd_volatility_threshold': 4.0,   # Range threshold
            'sd_oi_change_threshold': 4.0,
            'sd_k_coefficient_threshold': 2.5,
            # Lookback
            'sd_lookback_4h': True,
            'sd_lookback_8h': True,
            'sd_lookback_24h': True,
            # Funding
            'sd_funding_extreme_positive': 0.0003,
            'sd_funding_extreme_negative': -0.0003,
            # Watchlist
            'sd_min_consecutive_signals': 2,
            'sd_ready_consecutive_signals': 4,
            'sd_watchlist_timeout_hours': 48,
            'sd_breakout_threshold': 3.0,
            # Auto-trade
            'sd_execute_mode': 'off',
            'sd_auto_trade_size_usdt': 100,
            'sd_auto_trade_leverage': 5,
            'sd_auto_trade_tp_percent': 10,
            'sd_auto_trade_sl_percent': 3,
            # Alerts
            'sd_telegram_alerts': False,
            'sd_ui_alerts': True,
            # Auto-start
            'sd_auto_start': True,  # Auto-start Recording & Analyzing
        }
        
        # Завантажуємо з settings_manager - ПЕРЕЗАПИСУЄ дефолти
        try:
            from settings_manager import settings
            if settings:
                saved = settings.get_all()
                for key in list(config.keys()):
                    if key in saved and saved[key] is not None:
                        config[key] = saved[key]
                        logger.debug(f"Config loaded from DB: {key}={saved[key]}")
        except ImportError:
            logger.warning("settings_manager not available")
        
        logger.info(f"📋 Config loaded: top_coins={config.get('sd_top_coins')}, interval={config.get('sd_snapshot_interval')}, min_vol={config.get('sd_min_volume_24h')}, method={config.get('sd_analysis_method')}")
        return config
    
    def _initialize(self):
        """Ініціалізує компоненти"""
        if not self.db_session_factory:
            logger.warning("No DB session factory provided")
            return
        
        try:
            # Створюємо таблиці - безпечний імпорт
            try:
                from .models import create_squeeze_tables, run_migrations
            except ImportError:
                from squeeze_detector.models import create_squeeze_tables, run_migrations
            
            # Отримуємо engine з db_manager
            engine = None
            try:
                from models import db_manager
                if db_manager:
                    engine = db_manager.engine
            except ImportError:
                logger.warning("models.db_manager not available for table creation")
            
            if engine:
                create_squeeze_tables(engine)
                run_migrations(engine)
            
            # API credentials
            api_key = None
            api_secret = None
            
            if self.bot_instance:
                api_key = getattr(self.bot_instance, 'api_key', None)
                api_secret = getattr(self.bot_instance, 'api_secret', None)
            
            # Ініціалізуємо Recorder - безпечний імпорт
            try:
                from .recorder import DataRecorder
            except ImportError:
                from squeeze_detector.recorder import DataRecorder
                
            self.recorder = DataRecorder(
                self.db_session_factory,
                api_key=api_key,
                api_secret=api_secret,
                use_websocket=False,  # Поки без WebSocket
            )
            
            # Ініціалізуємо Analyzer - безпечний імпорт
            try:
                from .analyzer import SqueezeAnalyzer
            except ImportError:
                from squeeze_detector.analyzer import SqueezeAnalyzer
                
            self.analyzer = SqueezeAnalyzer(
                self.db_session_factory,
                config=self.config,
            )
            
            self.status['initialized'] = True
            logger.info("✅ SqueezeDetectorManager initialized")
            
            # === AUTO-START Recording & Analyzing ===
            # Автоматично запускаємо при ініціалізації
            if self.config.get('sd_auto_start', True):
                logger.info("🚀 Auto-starting Recording & Analyzing...")
                self.start_recording()
                self.start_analyzing()
            
        except Exception as e:
            logger.error(f"❌ Initialize error: {e}")
            import traceback
            traceback.print_exc()
            self.status['last_error'] = str(e)
    
    def update_config(self, new_config: Dict):
        """Оновлює конфігурацію"""
        old_top_coins = self.config.get('sd_top_coins')
        old_min_volume = self.config.get('sd_min_volume_24h')
        old_snapshot_interval = self.config.get('sd_snapshot_interval')
        old_analysis_interval = self.config.get('sd_analysis_interval')
        
        self.config.update(new_config)
        
        if self.analyzer:
            self.analyzer.update_config(new_config)
        
        # Зберігаємо в settings_manager
        try:
            from settings_manager import settings
            if settings:
                settings.save_settings(new_config)
                logger.info(f"💾 Settings saved to DB: {list(new_config.keys())}")
        except ImportError:
            pass
        
        # Якщо змінились параметри фільтрації - оновлюємо список символів
        new_top_coins = new_config.get('sd_top_coins', old_top_coins)
        new_min_volume = new_config.get('sd_min_volume_24h', old_min_volume)
        
        if new_top_coins != old_top_coins or new_min_volume != old_min_volume:
            logger.info(f"🔄 Updating symbols: top_coins {old_top_coins} -> {new_top_coins}, min_vol {old_min_volume} -> {new_min_volume}")
            self.update_symbols()
            
            # Якщо recording/analyzing активні - перезапускаємо з новим списком
            if self.status.get('recording'):
                self.stop_recording()
                self.start_recording()
            if self.status.get('analyzing'):
                self.stop_analyzing()
                self.start_analyzing()
        
        # Якщо змінились інтервали - перезапускаємо
        new_snapshot = new_config.get('sd_snapshot_interval', old_snapshot_interval)
        new_analysis = new_config.get('sd_analysis_interval', old_analysis_interval)
        
        if new_snapshot != old_snapshot_interval and self.status.get('recording'):
            logger.info(f"🔄 Restarting recording with new interval: {new_snapshot}s")
            self.stop_recording()
            self.start_recording()
            
        if new_analysis != old_analysis_interval and self.status.get('analyzing'):
            logger.info(f"🔄 Restarting analyzing with new interval: {new_analysis}s")
            self.stop_analyzing()
            self.start_analyzing()
        
        logger.info(f"🔧 Config updated: top_coins={self.config.get('sd_top_coins')}, min_vol={self.config.get('sd_min_volume_24h')}, interval={self.config.get('sd_snapshot_interval')}")
    
    def update_symbols(self):
        """Оновлює список монет з API"""
        if not self.recorder:
            return []
        
        min_volume = self.config.get('sd_min_volume_24h', 5_000_000)
        limit = self.config.get('sd_top_coins', 100)
        
        self.monitored_symbols = self.recorder.update_symbols_from_api(min_volume, limit)
        
        logger.info(f"📋 Updated symbols: {len(self.monitored_symbols)}")
        return self.monitored_symbols
    
    def start_recording(self):
        """Запускає запис даних"""
        if not self.recorder:
            return False
        
        if not self.monitored_symbols:
            self.update_symbols()
        
        interval = self.config.get('sd_snapshot_interval', 300)
        self.recorder.start_periodic_recording(interval)
        self.status['recording'] = True
        
        return True
    
    def stop_recording(self):
        """Зупиняє запис"""
        if self.recorder:
            self.recorder.stop_periodic_recording()
        self.status['recording'] = False
    
    def start_analyzing(self):
        """Запускає аналіз"""
        if not self.analyzer:
            return False
        
        if not self.monitored_symbols:
            self.update_symbols()
        
        interval = self.config.get('sd_analysis_interval', 300)
        self.analyzer.start_periodic_analysis(self.monitored_symbols, interval)
        self.status['analyzing'] = True
        
        return True
    
    def stop_analyzing(self):
        """Зупиняє аналіз"""
        if self.analyzer:
            self.analyzer.stop_periodic_analysis()
        self.status['analyzing'] = False
    
    def run_single_scan(self) -> Dict:
        """Запускає один цикл сканування"""
        if not self.recorder or not self.analyzer:
            return {'error': 'Not initialized'}
        
        if not self.monitored_symbols:
            self.update_symbols()
        
        # Записуємо snapshot
        recorded = self.recorder.record_snapshot()
        
        # Аналізуємо
        results = self.analyzer.run_full_analysis(self.monitored_symbols)
        results['snapshots_recorded'] = recorded
        
        return results
    
    def get_heatmap(self) -> list:
        """Повертає теплову карту"""
        if not self.analyzer:
            return []
        
        if not self.monitored_symbols:
            self.update_symbols()
        
        return [e.to_dict() for e in self.analyzer.generate_heatmap(self.monitored_symbols)]
    
    def get_watchlist(self) -> list:
        """Повертає watchlist"""
        if not self.analyzer:
            return []
        return self.analyzer.get_watchlist()
    
    def get_signals(self, limit: int = 50) -> list:
        """Повертає сигнали"""
        if not self.analyzer:
            return []
        return self.analyzer.get_recent_signals(limit)
    
    def get_status(self) -> Dict:
        """Повертає статус системи"""
        status = {
            **self.status,
            'monitored_symbols': len(self.monitored_symbols),
            'config': {k: v for k, v in self.config.items() if k.startswith('sd_')},
        }
        
        if self.recorder:
            status['recorder'] = self.recorder.get_stats()
        
        if self.analyzer:
            status['analyzer'] = self.analyzer.get_stats()
        
        return status
    
    def clear_watchlist(self):
        """Очищає watchlist"""
        if self.analyzer:
            self.analyzer.clear_watchlist()
    
    def remove_from_watchlist(self, symbol: str) -> bool:
        """Видаляє символ з watchlist"""
        if self.analyzer:
            return self.analyzer.remove_from_watchlist(symbol)
        return False
    
    def shutdown(self):
        """Завершує роботу"""
        self.stop_recording()
        self.stop_analyzing()
        
        if self.recorder:
            self.recorder.shutdown()
        
        logger.info("📊 SqueezeDetectorManager shutdown")


def get_detector_manager() -> SqueezeDetectorManager:
    """Отримує singleton instance"""
    global _detector_instance
    
    if _detector_instance is None:
        # Спроба отримати залежності
        db_session_factory = None
        bot_instance = None
        
        try:
            from models import db_manager
            if db_manager:
                db_session_factory = db_manager.get_session
        except ImportError as e:
            logger.warning(f"models.db_manager not available: {e}")
        
        # НЕ імпортуємо main_app щоб уникнути circular import
        # bot_instance буде встановлено через set_bot_instance()
        
        _detector_instance = SqueezeDetectorManager(
            db_session_factory=db_session_factory,
            bot_instance=bot_instance,
        )
    
    return _detector_instance


def set_bot_instance(bot):
    """Встановлює bot instance (викликається з main_app після ініціалізації)"""
    global _detector_instance
    
    if _detector_instance:
        _detector_instance.bot_instance = bot
        # Реініціалізуємо recorder з новими credentials
        if bot and _detector_instance.recorder:
            api_key = getattr(bot, 'api_key', None)
            api_secret = getattr(bot, 'api_secret', None)
            if api_key and api_secret:
                _detector_instance.recorder.rest_client.api_key = api_key
                _detector_instance.recorder.rest_client.api_secret = api_secret
                logger.info("✅ Squeeze Detector: bot instance set")


# ============================================================================
#                              FLASK ROUTES
# ============================================================================

def register_squeeze_detector_routes(app):
    """Реєструє Flask routes"""
    
    # Ініціалізуємо manager при старті (auto-start recording/analyzing)
    try:
        manager = get_detector_manager()
        logger.info(f"📊 Squeeze Detector initialized at startup: recording={manager.status.get('recording')}, analyzing={manager.status.get('analyzing')}")
    except Exception as e:
        logger.error(f"Failed to initialize Squeeze Detector at startup: {e}")
    
    @app.route('/squeeze')
    def squeeze_detector_page():
        """Головна сторінка Squeeze Detector"""
        manager = get_detector_manager()
        
        return render_template(
            'squeeze_detector.html',
            status=manager.get_status(),
            config=manager.config,
        )
    
    # === API ENDPOINTS ===
    
    @app.route('/squeeze/api/status')
    def squeeze_api_status():
        """Статус системи"""
        manager = get_detector_manager()
        return jsonify(manager.get_status())
    
    @app.route('/squeeze/api/scan', methods=['POST'])
    def squeeze_api_scan():
        """Запускає один скан"""
        manager = get_detector_manager()
        results = manager.run_single_scan()
        return jsonify(results)
    
    @app.route('/squeeze/api/heatmap')
    def squeeze_api_heatmap():
        """Теплова карта"""
        manager = get_detector_manager()
        return jsonify(manager.get_heatmap())
    
    @app.route('/squeeze/api/watchlist')
    def squeeze_api_watchlist():
        """Watchlist"""
        manager = get_detector_manager()
        return jsonify(manager.get_watchlist())
    
    @app.route('/squeeze/api/signals')
    def squeeze_api_signals():
        """Сигнали"""
        manager = get_detector_manager()
        limit = request.args.get('limit', 50, type=int)
        return jsonify(manager.get_signals(limit))
    
    @app.route('/squeeze/api/symbols')
    def squeeze_api_symbols():
        """Список монет"""
        manager = get_detector_manager()
        symbols = manager.update_symbols()
        return jsonify({
            'count': len(symbols),
            'symbols': symbols,
        })
    
    @app.route('/squeeze/api/config', methods=['GET', 'POST'])
    def squeeze_api_config():
        """Конфігурація"""
        manager = get_detector_manager()
        
        if request.method == 'POST':
            data = request.json or {}
            manager.update_config(data)
            return jsonify({'status': 'ok'})
        
        return jsonify(manager.config)
    
    @app.route('/squeeze/api/recording', methods=['POST'])
    def squeeze_api_recording():
        """Керування записом"""
        manager = get_detector_manager()
        data = request.json or {}
        
        if data.get('start'):
            manager.start_recording()
        elif data.get('stop'):
            manager.stop_recording()
        
        return jsonify({'recording': manager.status['recording']})
    
    @app.route('/squeeze/api/analyzing', methods=['POST'])
    def squeeze_api_analyzing():
        """Керування аналізом"""
        manager = get_detector_manager()
        data = request.json or {}
        
        if data.get('start'):
            manager.start_analyzing()
        elif data.get('stop'):
            manager.stop_analyzing()
        
        return jsonify({'analyzing': manager.status['analyzing']})
    
    @app.route('/squeeze/api/clear', methods=['POST'])
    def squeeze_api_clear():
        """Очищення watchlist"""
        manager = get_detector_manager()
        manager.clear_watchlist()
        return jsonify({'status': 'cleared'})
    
    @app.route('/squeeze/api/execute', methods=['POST'])
    def squeeze_api_execute():
        """
        Виконує угоду через RSI Sniper PRO.
        
        Request:
            {
                'symbol': 'SOLUSDT',
                'direction': 'SHORT',
                'entry_price': 180.50,  # optional, якщо не вказано - береться поточна
                'tp_percent': 10,
                'sl_percent': 3,
                'leverage': 5,
                'size_usdt': 100,
                'bias_confidence': 89,
                'k_coefficient': 5.1,
                'remove_from_watchlist': True
            }
        """
        manager = get_detector_manager()
        data = request.json or {}
        
        symbol = data.get('symbol')
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        direction = data.get('direction')
        if direction not in ['LONG', 'SHORT']:
            return jsonify({'success': False, 'error': 'Invalid direction'}), 400
        
        try:
            # Import RSI Sniper
            from rsi_sniper_pro import RSISniperPro
            sniper = RSISniperPro()
            
            # Get current price if not provided
            entry_price = data.get('entry_price')
            if not entry_price:
                # Try to get from recorder
                if manager.recorder:
                    snapshots = manager.recorder.get_snapshots(symbol, hours=1)
                    if snapshots:
                        entry_price = snapshots[-1].mark_price or snapshots[-1].last_price
                
                # Fallback - try from Bybit API
                if not entry_price:
                    try:
                        from bot import bot_instance
                        if bot_instance and bot_instance.exchange:
                            ticker = bot_instance.exchange.fetch_ticker(symbol)
                            entry_price = ticker.get('last', 0)
                    except:
                        pass
            
            if not entry_price:
                return jsonify({'success': False, 'error': 'Could not get entry price'}), 400
            
            # Prepare squeeze data for RSI Sniper
            squeeze_data = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': float(entry_price),
                'tp_percent': float(data.get('tp_percent', manager.config.get('sd_auto_trade_tp_percent', 10))),
                'sl_percent': float(data.get('sl_percent', manager.config.get('sd_auto_trade_sl_percent', 3))),
                'leverage': int(data.get('leverage', manager.config.get('sd_auto_trade_leverage', 5))),
                'size_usdt': float(data.get('size_usdt', manager.config.get('sd_auto_trade_size_usdt', 100))),
                'bias_confidence': data.get('bias_confidence', 50),
                'k_coefficient': data.get('k_coefficient', 0),
                'source': 'SQUEEZE_DETECTOR'
            }
            
            # Execute via RSI Sniper
            result = sniper.execute_from_squeeze(squeeze_data)
            
            if result.get('success'):
                # Remove from watchlist if requested
                if data.get('remove_from_watchlist', True):
                    manager.remove_from_watchlist(symbol)
                    result['removed_from_watchlist'] = True
                
                logger.info(f"🎯 Squeeze executed: {symbol} {direction} @ {entry_price}")
            
            return jsonify(result)
            
        except ImportError as e:
            logger.error(f"RSI Sniper import error: {e}")
            return jsonify({'success': False, 'error': 'RSI Sniper module not available'}), 500
        except Exception as e:
            logger.error(f"Execute error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/squeeze/api/remove_watchlist', methods=['POST'])
    def squeeze_api_remove_watchlist():
        """Видаляє символ з watchlist"""
        manager = get_detector_manager()
        data = request.json or {}
        
        symbol = data.get('symbol')
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        success = manager.remove_from_watchlist(symbol)
        return jsonify({'success': success, 'symbol': symbol})
    
    @app.route('/squeeze/api/load_history', methods=['POST'])
    def squeeze_api_load_history():
        """Завантажує історичні дані"""
        manager = get_detector_manager()
        data = request.json or {}
        
        symbol = data.get('symbol')
        hours = data.get('hours', 24)
        
        if not symbol:
            return jsonify({'error': 'Symbol required'}), 400
        
        if manager.recorder:
            success = manager.recorder.load_historical_data(symbol, hours)
            return jsonify({'success': success})
        
        return jsonify({'error': 'Recorder not initialized'}), 500
    
    logger.info("✅ Squeeze Detector routes registered")
