"""
Configuration Persistence Service v2.5
======================================
Збереження та завантаження всіх параметрів в БД
"""

import json
import logging
from datetime import datetime
from models import db_manager, ScannerConfig as ScannerConfigModel

logger = logging.getLogger(__name__)


class ConfigPersistenceService:
    """Сервіс для збереження конфігурації в БД"""
    
    def __init__(self):
        self.db = db_manager
        logger.info("✅ Config Persistence Service initialized")
    
    def save_config(self, scanner_config_obj):
        """
        Зберегти всю конфігурацію в БД
        
        Args:
            scanner_config_obj: Об'єкт scanner_config
        """
        try:
            session = self.db.get_session()
            
            # Збираємо всі параметри в один JSON
            config_data = {
                # Базові параметри
                'trading_style': scanner_config_obj.trading_style,
                'aggressiveness': scanner_config_obj.aggressiveness,
                'automation_mode': scanner_config_obj.automation_mode,
                'indicator_timeframe': scanner_config_obj.indicator_timeframe,
                
                # Keep-Alive
                'keep_alive_enabled': scanner_config_obj.keep_alive_enabled,
                'keep_alive_interval': scanner_config_obj.keep_alive_interval,
                
                # Параметри індикатора
                'indicator_params': scanner_config_obj.indicator_params,
                
                # Risk Management
                'risk_params': scanner_config_obj.risk_params,
                
                # Auto Close
                'auto_close_params': scanner_config_obj.auto_close_params,
                'auto_open_enabled': scanner_config_obj.auto_open_enabled,
                
                # Scanner параметри
                'scanner_params': scanner_config_obj.scanner_params if hasattr(scanner_config_obj, 'scanner_params') else {},
                
                # Timestamp
                'saved_at': datetime.utcnow().isoformat()
            }
            
            # Деактивувати старі конфігурації
            session.query(ScannerConfigModel).filter_by(is_active=True).update({'is_active': False})
            
            # Створити нову активну конфігурацію
            new_config = ScannerConfigModel(
                timestamp=datetime.utcnow(),
                trading_style=scanner_config_obj.trading_style,
                aggressiveness=scanner_config_obj.aggressiveness,
                automation_mode=scanner_config_obj.automation_mode,
                params_json=json.dumps(config_data, ensure_ascii=False),
                is_active=True
            )
            
            session.add(new_config)
            session.commit()
            
            logger.info(f"✅ Configuration saved to DB (ID: {new_config.id})")
            logger.info(f"   Trading Style: {scanner_config_obj.trading_style}")
            logger.info(f"   Aggressiveness: {scanner_config_obj.aggressiveness}")
            logger.info(f"   Automation: {scanner_config_obj.automation_mode}")
            
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving config to DB: {e}")
            return False
    
    def load_config(self, scanner_config_obj):
        """
        Завантажити конфігурацію з БД
        
        Args:
            scanner_config_obj: Об'єкт scanner_config для заповнення
        """
        try:
            session = self.db.get_session()
            
            # Знайти активну конфігурацію
            active_config = session.query(ScannerConfigModel)\
                .filter_by(is_active=True)\
                .order_by(ScannerConfigModel.timestamp.desc())\
                .first()
            
            if not active_config:
                logger.info("ℹ️ No saved config found, using defaults")
                session.close()
                return False
            
            # Розпакувати JSON
            config_data = json.loads(active_config.params_json)
            
            # Застосувати параметри
            scanner_config_obj.trading_style = config_data.get('trading_style', 'daytrading')
            scanner_config_obj.aggressiveness = config_data.get('aggressiveness', 'auto')
            scanner_config_obj.automation_mode = config_data.get('automation_mode', 'semi_auto')
            scanner_config_obj.indicator_timeframe = config_data.get('indicator_timeframe', '240')
            
            scanner_config_obj.keep_alive_enabled = config_data.get('keep_alive_enabled', True)
            scanner_config_obj.keep_alive_interval = config_data.get('keep_alive_interval', 300)
            
            scanner_config_obj.indicator_params = config_data.get('indicator_params', {})
            scanner_config_obj.risk_params = config_data.get('risk_params', {})
            scanner_config_obj.auto_close_params = config_data.get('auto_close_params', {})
            scanner_config_obj.auto_open_enabled = config_data.get('auto_open_enabled', True)
            
            # Scanner параметри (якщо є)
            if hasattr(scanner_config_obj, 'scanner_params'):
                scanner_config_obj.scanner_params = config_data.get('scanner_params', {})
            
            logger.info(f"✅ Configuration loaded from DB (ID: {active_config.id})")
            logger.info(f"   Saved at: {active_config.timestamp}")
            logger.info(f"   Trading Style: {scanner_config_obj.trading_style}")
            logger.info(f"   Aggressiveness: {scanner_config_obj.aggressiveness}")
            logger.info(f"   Automation: {scanner_config_obj.automation_mode}")
            
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading config from DB: {e}")
            return False
    
    def get_config_history(self, limit=10):
        """Отримати історію збережених конфігурацій"""
        try:
            session = self.db.get_session()
            
            configs = session.query(ScannerConfigModel)\
                .order_by(ScannerConfigModel.timestamp.desc())\
                .limit(limit)\
                .all()
            
            history = []
            for config in configs:
                config_data = json.loads(config.params_json)
                history.append({
                    'id': config.id,
                    'timestamp': config.timestamp.isoformat(),
                    'trading_style': config.trading_style,
                    'aggressiveness': config.aggressiveness,
                    'automation_mode': config.automation_mode,
                    'is_active': config.is_active,
                    'full_data': config_data
                })
            
            session.close()
            return history
            
        except Exception as e:
            logger.error(f"❌ Error getting config history: {e}")
            return []
    
    def restore_config(self, config_id: int, scanner_config_obj):
        """
        Відновити конфігурацію з історії
        
        Args:
            config_id: ID конфігурації
            scanner_config_obj: Об'єкт scanner_config
        """
        try:
            session = self.db.get_session()
            
            # Знайти конфігурацію
            config = session.query(ScannerConfigModel).filter_by(id=config_id).first()
            
            if not config:
                logger.error(f"❌ Config ID {config_id} not found")
                session.close()
                return False
            
            # Деактивувати всі
            session.query(ScannerConfigModel).filter_by(is_active=True).update({'is_active': False})
            
            # Активувати вибрану
            config.is_active = True
            session.commit()
            
            # Завантажити
            self.load_config(scanner_config_obj)
            
            logger.info(f"✅ Config ID {config_id} restored")
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error restoring config: {e}")
            return False
    
    def export_config(self, scanner_config_obj):
        """Експорт конфігурації в JSON (для backup)"""
        config_data = {
            'trading_style': scanner_config_obj.trading_style,
            'aggressiveness': scanner_config_obj.aggressiveness,
            'automation_mode': scanner_config_obj.automation_mode,
            'indicator_timeframe': scanner_config_obj.indicator_timeframe,
            'keep_alive_enabled': scanner_config_obj.keep_alive_enabled,
            'keep_alive_interval': scanner_config_obj.keep_alive_interval,
            'indicator_params': scanner_config_obj.indicator_params,
            'risk_params': scanner_config_obj.risk_params,
            'auto_close_params': scanner_config_obj.auto_close_params,
            'auto_open_enabled': scanner_config_obj.auto_open_enabled,
            'scanner_params': scanner_config_obj.scanner_params if hasattr(scanner_config_obj, 'scanner_params') else {},
            'exported_at': datetime.utcnow().isoformat()
        }
        return json.dumps(config_data, indent=2, ensure_ascii=False)
    
    def import_config(self, json_str: str, scanner_config_obj):
        """Імпорт конфігурації з JSON"""
        try:
            config_data = json.loads(json_str)
            
            # Застосувати параметри
            scanner_config_obj.trading_style = config_data.get('trading_style')
            scanner_config_obj.aggressiveness = config_data.get('aggressiveness')
            scanner_config_obj.automation_mode = config_data.get('automation_mode')
            scanner_config_obj.indicator_timeframe = config_data.get('indicator_timeframe')
            scanner_config_obj.keep_alive_enabled = config_data.get('keep_alive_enabled')
            scanner_config_obj.keep_alive_interval = config_data.get('keep_alive_interval')
            scanner_config_obj.indicator_params = config_data.get('indicator_params')
            scanner_config_obj.risk_params = config_data.get('risk_params')
            scanner_config_obj.auto_close_params = config_data.get('auto_close_params')
            scanner_config_obj.auto_open_enabled = config_data.get('auto_open_enabled')
            
            if hasattr(scanner_config_obj, 'scanner_params'):
                scanner_config_obj.scanner_params = config_data.get('scanner_params', {})
            
            # Зберегти в БД
            self.save_config(scanner_config_obj)
            
            logger.info("✅ Config imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error importing config: {e}")
            return False


# Глобальний екземпляр
config_persistence = ConfigPersistenceService()
