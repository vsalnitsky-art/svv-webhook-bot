"""
Database Operations - CRUD operations for Sleeper OB Bot
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy import desc, and_, or_
from storage.db_models import (
    get_session, init_db,
    SleeperCandidate, OrderBlock, Trade, PerformanceStats, BotSetting, EventLog,
    SymbolBlacklist
)
from config import DEFAULT_SETTINGS

class DBOperations:
    """Database operations handler"""
    
    def __init__(self):
        init_db()
        self._init_default_settings()
    
    def _init_default_settings(self):
        """Initialize default settings if not present"""
        session = get_session()
        try:
            for key, value in DEFAULT_SETTINGS.items():
                existing = session.query(BotSetting).filter_by(key=key).first()
                if not existing:
                    setting = BotSetting(
                        key=key,
                        value=json.dumps(value) if not isinstance(value, str) else value
                    )
                    session.add(setting)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error initializing settings: {e}")
        finally:
            session.close()
    
    # === SETTINGS ===
    
    def get_setting(self, key: str, default=None):
        """Get a setting value"""
        session = get_session()
        try:
            setting = session.query(BotSetting).filter_by(key=key).first()
            if setting:
                try:
                    return json.loads(setting.value)
                except:
                    return setting.value
            return default
        finally:
            session.close()
    
    def set_setting(self, key: str, value):
        """Set a setting value"""
        session = get_session()
        try:
            setting = session.query(BotSetting).filter_by(key=key).first()
            # Always use json.dumps for consistency
            str_value = json.dumps(value)
            
            if setting:
                setting.value = str_value
            else:
                setting = BotSetting(key=key, value=str_value)
                session.add(setting)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error setting {key}: {e}")
            return False
        finally:
            session.close()
    
    def get_all_settings(self) -> Dict:
        """Get all settings"""
        session = get_session()
        try:
            settings = session.query(BotSetting).all()
            result = {}
            for s in settings:
                try:
                    result[s.key] = json.loads(s.value)
                except:
                    result[s.key] = s.value
            return result
        finally:
            session.close()
    
    # === SLEEPER CANDIDATES ===
    
    def upsert_sleeper(self, data: Dict) -> Optional[SleeperCandidate]:
        """Insert or update sleeper candidate"""
        # Convert numpy types to Python native types
        clean_data = {}
        for key, value in data.items():
            if hasattr(value, 'item'):  # numpy scalar
                clean_data[key] = value.item()
            elif isinstance(value, (int, float, str, bool, type(None), datetime)):
                clean_data[key] = value
            else:
                try:
                    clean_data[key] = float(value)
                except (TypeError, ValueError):
                    clean_data[key] = str(value)
        
        # Get valid column names from model
        valid_columns = {c.name for c in SleeperCandidate.__table__.columns}
        
        # Filter to only valid columns (prevents "invalid keyword argument" errors)
        filtered_data = {k: v for k, v in clean_data.items() if k in valid_columns}
        
        session = get_session()
        try:
            sleeper = session.query(SleeperCandidate).filter_by(
                symbol=filtered_data['symbol']
            ).first()
            
            if sleeper:
                for key, value in filtered_data.items():
                    if hasattr(sleeper, key):
                        setattr(sleeper, key, value)
                sleeper.checks_count += 1
            else:
                sleeper = SleeperCandidate(**filtered_data)
                session.add(sleeper)
            
            session.commit()
            session.refresh(sleeper)
            return sleeper
        except Exception as e:
            session.rollback()
            print(f"Error upserting sleeper: {e}")
            return None
        finally:
            session.close()
    
    def get_sleepers(self, state: str = None, min_score: float = None, 
                     limit: int = 100) -> List[Dict]:
        """Get sleeper candidates"""
        session = get_session()
        try:
            query = session.query(SleeperCandidate)
            
            if state:
                query = query.filter(SleeperCandidate.state == state)
            if min_score:
                query = query.filter(SleeperCandidate.total_score >= min_score)
            
            query = query.order_by(desc(SleeperCandidate.total_score))
            sleepers = query.limit(limit).all()
            return [s.to_dict() for s in sleepers]
        finally:
            session.close()
    
    def get_ready_sleepers(self, min_score: float = 80) -> List[Dict]:
        """Get sleepers ready for OB scan"""
        return self.get_sleepers(state='READY', min_score=min_score)
    
    def update_sleeper_state(self, symbol: str, state: str, 
                             direction: str = None, hp_change: int = 0) -> bool:
        """Update sleeper state"""
        session = get_session()
        try:
            sleeper = session.query(SleeperCandidate).filter_by(symbol=symbol).first()
            if sleeper:
                sleeper.state = state
                if direction:
                    sleeper.direction = direction
                if hp_change:
                    sleeper.hp = max(0, min(10, sleeper.hp + hp_change))
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error updating sleeper state: {e}")
            return False
        finally:
            session.close()
    
    def remove_dead_sleepers(self) -> int:
        """Remove sleepers with HP = 0"""
        session = get_session()
        try:
            count = session.query(SleeperCandidate).filter(
                SleeperCandidate.hp <= 0
            ).delete()
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            return 0
        finally:
            session.close()
    
    def clear_all_sleepers(self) -> int:
        """Remove ALL sleepers (for fresh scan)"""
        session = get_session()
        try:
            count = session.query(SleeperCandidate).delete()
            session.commit()
            print(f"[DB] Cleared {count} sleepers")
            return count
        except Exception as e:
            session.rollback()
            print(f"[DB] Error clearing sleepers: {e}")
            return 0
        finally:
            session.close()
    
    def remove_low_quality_sleepers(self) -> int:
        """Remove sleepers with poor data quality (funding=0, oi=0, bb=0)"""
        session = get_session()
        try:
            from sqlalchemy import and_
            count = session.query(SleeperCandidate).filter(
                and_(
                    SleeperCandidate.funding_rate == 0,
                    SleeperCandidate.oi_change_4h == 0,
                    SleeperCandidate.bb_width == 0
                )
            ).delete()
            session.commit()
            print(f"[DB] Removed {count} low-quality sleepers")
            return count
        except Exception as e:
            session.rollback()
            print(f"[DB] Error removing low-quality sleepers: {e}")
            return 0
        finally:
            session.close()
    
    # === ORDER BLOCKS ===
    
    def add_orderblock(self, data: Dict) -> Optional[OrderBlock]:
        """Add new order block"""
        # Convert numpy types and map fields
        clean_data = {}
        
        # Field mapping from new OB scanner format to DB model
        field_map = {
            'symbol': 'symbol',
            'timeframe': 'timeframe',
            'ob_type': 'ob_type',
            'top': 'ob_high',
            'bottom': 'ob_low',
            'quality': 'quality_score',
            'volume_ratio': 'volume_ratio',   # Правильне відношення (1.5x, 2.3x)
            'impulse_pct': 'impulse_pct',     # Відсоток імпульсного руху
        }
        
        for src, dst in field_map.items():
            if src in data:
                value = data[src]
                # Convert numpy types
                if hasattr(value, 'item'):
                    clean_data[dst] = value.item()
                else:
                    clean_data[dst] = value
        
        # Calculate mid point
        if 'ob_high' in clean_data and 'ob_low' in clean_data:
            clean_data['ob_mid'] = (clean_data['ob_high'] + clean_data['ob_low']) / 2
        
        # ob_type is already LONG or SHORT from scanner
        # (Legacy support: Bull -> LONG, Bear -> SHORT)
        if 'ob_type' in clean_data:
            ob_type = clean_data['ob_type']
            if ob_type == 'Bull':
                clean_data['ob_type'] = 'LONG'
            elif ob_type == 'Bear':
                clean_data['ob_type'] = 'SHORT'
            # LONG and SHORT pass through as-is
        
        # Status based on breaker
        clean_data['status'] = 'MITIGATED' if data.get('breaker', False) else 'ACTIVE'
        
        # Expiration
        clean_data['expires_at'] = datetime.utcnow() + timedelta(hours=48)
        
        session = get_session()
        try:
            # Check for duplicate - similar zone
            if all(k in clean_data for k in ['symbol', 'timeframe', 'ob_type', 'ob_mid']):
                zone_tolerance = abs(clean_data.get('ob_high', 0) - clean_data.get('ob_low', 0)) * 0.5
                existing = session.query(OrderBlock).filter(
                    and_(
                        OrderBlock.symbol == clean_data['symbol'],
                        OrderBlock.timeframe == clean_data['timeframe'],
                        OrderBlock.status == 'ACTIVE',
                        OrderBlock.ob_type == clean_data['ob_type'],
                        OrderBlock.ob_mid.between(
                            clean_data['ob_mid'] - zone_tolerance,
                            clean_data['ob_mid'] + zone_tolerance
                        )
                    )
                ).first()
                
                if existing:
                    return existing
            
            ob = OrderBlock(**clean_data)
            session.add(ob)
            session.commit()
            session.refresh(ob)
            return ob
        except Exception as e:
            session.rollback()
            print(f"Error adding OB: {e}")
            return None
        finally:
            session.close()
    
    def get_orderblocks(self, symbol: str = None, status: str = 'ACTIVE', 
                        timeframe: str = None, limit: int = 50) -> List[Dict]:
        """Get order blocks"""
        session = get_session()
        try:
            query = session.query(OrderBlock)
            
            if symbol:
                query = query.filter(OrderBlock.symbol == symbol)
            if status:
                query = query.filter(OrderBlock.status == status)
            if timeframe:
                query = query.filter(OrderBlock.timeframe == timeframe)
            
            query = query.order_by(desc(OrderBlock.quality_score))
            obs = query.limit(limit).all()
            return [ob.to_dict() for ob in obs]
        finally:
            session.close()
    
    def update_ob_status(self, ob_id: int, status: str, touch_count: int = None) -> bool:
        """Update order block status"""
        session = get_session()
        try:
            ob = session.query(OrderBlock).filter_by(id=ob_id).first()
            if ob:
                ob.status = status
                if status == 'TOUCHED':
                    ob.touched_at = datetime.utcnow()
                if touch_count is not None:
                    ob.touch_count = touch_count
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()
    
    def expire_old_orderblocks(self, max_age_minutes: int = 60) -> int:
        """Expire old order blocks"""
        session = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
            count = session.query(OrderBlock).filter(
                and_(
                    OrderBlock.status == 'ACTIVE',
                    OrderBlock.created_at < cutoff
                )
            ).update({'status': 'EXPIRED'})
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            return 0
        finally:
            session.close()
    
    # === TRADES ===
    
    def add_trade(self, data: Dict) -> Optional[Trade]:
        """Add new trade"""
        session = get_session()
        try:
            trade = Trade(**data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
        except Exception as e:
            session.rollback()
            print(f"Error adding trade: {e}")
            return None
        finally:
            session.close()
    
    def get_trades(self, status: str = None, symbol: str = None, 
                   limit: int = 50) -> List[Dict]:
        """Get trades"""
        session = get_session()
        try:
            query = session.query(Trade)
            
            if status:
                query = query.filter(Trade.status == status)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            query = query.order_by(desc(Trade.opened_at))
            trades = query.limit(limit).all()
            return [t.to_dict() for t in trades]
        finally:
            session.close()
    
    def get_open_trades(self) -> List[Dict]:
        """Get open trades"""
        return self.get_trades(status='OPEN')
    
    def update_trade(self, trade_id: int, data: Dict) -> bool:
        """Update trade fields"""
        session = get_session()
        try:
            trade = session.query(Trade).filter_by(id=trade_id).first()
            if trade:
                for key, value in data.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error updating trade: {e}")
            return False
        finally:
            session.close()
    
    def close_trade(self, trade_id: int, exit_price: float, exit_reason: str,
                    pnl_usdt: float, pnl_percent: float = 0, fees: float = 0) -> bool:
        """Close a trade"""
        session = get_session()
        try:
            trade = session.query(Trade).filter_by(id=trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.closed_at = datetime.utcnow()
                trade.exit_reason = exit_reason
                trade.realized_pnl = pnl_usdt
                trade.fee = fees
                trade.status = 'CLOSED'
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_trade_stats(self, days: int = 30) -> Dict:
        """Get trade statistics"""
        session = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = session.query(Trade).filter(
                and_(
                    Trade.status == 'CLOSED',
                    Trade.opened_at >= cutoff
                )
            ).all()
            
            if not trades:
                return {
                    'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                    'win_rate': 0, 'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
                    'profit_factor': 0
                }
            
            # Use realized_pnl field (correct field name from model)
            winners = [t for t in trades if (t.realized_pnl or 0) > 0]
            losers = [t for t in trades if (t.realized_pnl or 0) <= 0]
            
            total_wins = sum((t.realized_pnl or 0) for t in winners)
            total_losses = abs(sum((t.realized_pnl or 0) for t in losers))
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winners),
                'losing_trades': len(losers),
                'win_rate': (len(winners) / len(trades) * 100) if trades else 0,
                'total_pnl': sum((t.realized_pnl or 0) for t in trades),
                'avg_win': total_wins / len(winners) if winners else 0,
                'avg_loss': total_losses / len(losers) if losers else 0,
                'profit_factor': total_wins / total_losses if total_losses > 0 else 0
            }
        finally:
            session.close()
    
    # === EVENT LOG ===
    
    def log_event(self, message: str, level: str = 'INFO', 
                  category: str = 'SYSTEM', symbol: str = None):
        """Log an event"""
        session = get_session()
        try:
            event = EventLog(
                message=message,
                level=level,
                category=category,
                symbol=symbol
            )
            session.add(event)
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()
    
    def get_recent_events(self, limit: int = 50, level: str = None, category: str = None) -> List[Dict]:
        """Get recent events"""
        session = get_session()
        try:
            query = session.query(EventLog)
            if level:
                query = query.filter(EventLog.level == level)
            if category:
                query = query.filter(EventLog.category == category)
            
            events = query.order_by(desc(EventLog.timestamp)).limit(limit).all()
            return [e.to_dict() for e in events]
        finally:
            session.close()
    
    def clear_old_events(self, days: int = 7) -> int:
        """Clear old events"""
        session = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            count = session.query(EventLog).filter(
                EventLog.timestamp < cutoff
            ).delete()
            session.commit()
            return count
        except:
            session.rollback()
            return 0
        finally:
            session.close()
    
    # ==========================================
    # BLACKLIST OPERATIONS (v8.2.2)
    # ==========================================
    
    def add_to_blacklist(self, symbol: str, reason: str = 'MANUAL', 
                         volatility: float = 0, note: str = None) -> bool:
        """
        Додає монету в чорний список
        
        Args:
            symbol: Торгова пара (BTCUSDT)
            reason: LOW_VOLATILITY / STABLECOIN / MANUAL / DELISTED
            volatility: Поточна волатильність (%) коли додавали
            note: Нотатка
        """
        session = get_session()
        try:
            existing = session.query(SymbolBlacklist).filter_by(symbol=symbol).first()
            if existing:
                return False  # Already in blacklist
            
            entry = SymbolBlacklist(
                symbol=symbol,
                reason=reason,
                volatility_24h=volatility,
                note=note
            )
            session.add(entry)
            session.commit()
            print(f"[BLACKLIST] Added {symbol} ({reason})")
            
            # v8.2.3: Auto-remove from sleepers when added to blacklist
            self.remove_sleeper(symbol)
            
            return True
        except Exception as e:
            session.rollback()
            print(f"[BLACKLIST] Error adding {symbol}: {e}")
            return False
        finally:
            session.close()
    
    def remove_sleeper(self, symbol: str) -> bool:
        """Видаляє sleeper за символом"""
        session = get_session()
        try:
            deleted = session.query(SleeperCandidate).filter_by(symbol=symbol).delete()
            session.commit()
            if deleted:
                print(f"[DB] Removed sleeper: {symbol}")
            return deleted > 0
        except:
            session.rollback()
            return False
        finally:
            session.close()
    
    def remove_blacklisted_sleepers(self) -> int:
        """Видаляє всі sleepers які є в blacklist"""
        blacklist = self.get_blacklist()
        removed = 0
        for symbol in blacklist:
            if self.remove_sleeper(symbol):
                removed += 1
        if removed > 0:
            print(f"[DB] Removed {removed} blacklisted sleepers")
        return removed
    
    def remove_duplicate_sleepers(self) -> int:
        """Видаляє дублікати sleepers (залишає найновіший)"""
        session = get_session()
        try:
            # Знаходимо дублікати
            from sqlalchemy import func
            duplicates = session.query(
                SleeperCandidate.symbol,
                func.count(SleeperCandidate.id).label('count')
            ).group_by(SleeperCandidate.symbol).having(func.count(SleeperCandidate.id) > 1).all()
            
            removed = 0
            for symbol, count in duplicates:
                # Залишаємо найновіший (найбільший id)
                sleepers = session.query(SleeperCandidate).filter_by(
                    symbol=symbol
                ).order_by(SleeperCandidate.id.desc()).all()
                
                # Видаляємо всі крім першого (найновішого)
                for s in sleepers[1:]:
                    session.delete(s)
                    removed += 1
            
            session.commit()
            if removed > 0:
                print(f"[DB] Removed {removed} duplicate sleepers")
            return removed
        except Exception as e:
            session.rollback()
            print(f"[DB] Error removing duplicates: {e}")
            return 0
        finally:
            session.close()
    
    def remove_from_blacklist(self, symbol: str) -> bool:
        """Видаляє монету з чорного списку"""
        session = get_session()
        try:
            deleted = session.query(SymbolBlacklist).filter_by(symbol=symbol).delete()
            session.commit()
            if deleted:
                print(f"[BLACKLIST] Removed {symbol}")
            return deleted > 0
        except:
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_blacklist(self) -> List[str]:
        """Повертає список заблокованих символів"""
        session = get_session()
        try:
            entries = session.query(SymbolBlacklist).all()
            return [e.symbol for e in entries]
        except Exception as e:
            # Table might not exist yet
            print(f"[BLACKLIST] Warning: Could not get blacklist: {e}")
            return []
        finally:
            session.close()
    
    def get_blacklist_full(self) -> List[Dict]:
        """Повертає повний список з деталями"""
        session = get_session()
        try:
            entries = session.query(SymbolBlacklist).order_by(
                SymbolBlacklist.added_at.desc()
            ).all()
            return [e.to_dict() for e in entries]
        except Exception as e:
            print(f"[BLACKLIST] Warning: Could not get blacklist: {e}")
            return []
        finally:
            session.close()
    
    def is_blacklisted(self, symbol: str) -> bool:
        """Перевіряє чи монета в чорному списку"""
        session = get_session()
        try:
            exists = session.query(SymbolBlacklist).filter_by(symbol=symbol).first()
            return exists is not None
        finally:
            session.close()
    
    def clear_blacklist(self) -> int:
        """Очищає весь чорний список"""
        session = get_session()
        try:
            count = session.query(SymbolBlacklist).delete()
            session.commit()
            print(f"[BLACKLIST] Cleared {count} entries")
            return count
        except:
            session.rollback()
            return 0
        finally:
            session.close()
    
    def auto_blacklist_low_volatility(self, symbols: List[str], 
                                       volatility_data: Dict[str, float],
                                       min_volatility: float = 3.0) -> int:
        """
        Автоматично додає "важкі" монети в blacklist
        
        Args:
            symbols: Список символів для перевірки
            volatility_data: {symbol: volatility_24h_pct}
            min_volatility: Мінімальна волатильність (%) щоб НЕ потрапити в blacklist
        
        Returns:
            Кількість доданих в blacklist
        """
        added = 0
        for symbol in symbols:
            vol = volatility_data.get(symbol, 0)
            if vol < min_volatility and vol > 0:  # Exclude zero (no data)
                if self.add_to_blacklist(
                    symbol=symbol,
                    reason='LOW_VOLATILITY',
                    volatility=vol,
                    note=f'Auto-blacklisted: {vol:.1f}% < {min_volatility}%'
                ):
                    added += 1
        return added


# Singleton instance
_db_instance = None

def get_db() -> DBOperations:
    """Get database operations instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DBOperations()
    return _db_instance
