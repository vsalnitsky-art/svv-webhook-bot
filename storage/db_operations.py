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
    SymbolBlacklist, SMCOBState,
    Top100OBSnapshot, Top100OBHistory,
    VolumizedRadarMetadata, VolumizedRadarStat, VolumizedRadarSnapshot
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
            
            query = query.order_by(desc(Trade.entry_time))
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
                    pnl_usdt: float, pnl_percent: float, fees: float = 0) -> bool:
        """Close a trade"""
        session = get_session()
        try:
            trade = session.query(Trade).filter_by(id=trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.exit_time = datetime.utcnow()
                trade.exit_reason = exit_reason
                trade.pnl_usdt = pnl_usdt
                trade.pnl_percent = pnl_percent
                trade.fees_paid = fees
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
                    Trade.entry_time >= cutoff
                )
            ).all()
            
            if not trades:
                return {
                    'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                    'win_rate': 0, 'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
                    'profit_factor': 0
                }
            
            winners = [t for t in trades if t.pnl_usdt > 0]
            losers = [t for t in trades if t.pnl_usdt <= 0]
            
            total_wins = sum(t.pnl_usdt for t in winners)
            total_losses = abs(sum(t.pnl_usdt for t in losers))
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winners),
                'losing_trades': len(losers),
                'win_rate': (len(winners) / len(trades) * 100) if trades else 0,
                'total_pnl': sum(t.pnl_usdt for t in trades),
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
    
    # ============================================================
    # SMC Order Block State (Pine SMC_PRO_BOT__47_)
    # ============================================================
    # Single-row-per-(symbol,timeframe) cache. Updated on every scan tick
    # for every watchlist symbol. Used by:
    #   1) Chart panel — to render the OB badge without re-running detection
    #   2) Signal gate — to block opens that don't match the OB direction
    #
    # Both readers and writers share this table so there's no race between
    # "what the user sees" and "what the gate checks".
    
    def upsert_smc_ob_state(self, symbol: str, timeframe: str,
                              ob_data: Optional[Dict]) -> bool:
        """Update (or insert) the SMC OB state for a symbol+timeframe.
        
        Pass ob_data=None when the detector found no valid OB — we still
        write a row with bias=None so the gate sees "we computed and
        nothing exists" (different from "we never computed").
        
        Returns True on successful write, False on DB error.
        """
        session = get_session()
        try:
            row = session.query(SMCOBState).filter_by(
                symbol=symbol, timeframe=timeframe).first()
            if row is None:
                row = SMCOBState(symbol=symbol, timeframe=timeframe)
                session.add(row)
            
            if ob_data:
                row.bias = ob_data.get('bias')
                row.bar_high = ob_data.get('bar_high')
                row.bar_low = ob_data.get('bar_low')
                row.bar_time_ms = ob_data.get('bar_time')
                row.bar_idx = ob_data.get('bar_idx')
                row.created_at_idx = ob_data.get('created_at_idx')
                row.created_at_t = ob_data.get('created_at_t')
                row.created_by_tag = ob_data.get('created_by_tag')
            else:
                # Explicitly clear when no OB exists — important for the gate.
                row.bias = None
                row.bar_high = None
                row.bar_low = None
                row.bar_time_ms = None
                row.bar_idx = None
                row.created_at_idx = None
                row.created_at_t = None
                row.created_by_tag = None
            
            row.computed_at = datetime.utcnow()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"[DB] upsert_smc_ob_state error for {symbol}@{timeframe}: {e}")
            return False
        finally:
            session.close()
    
    def get_smc_ob_state(self, symbol: str,
                          timeframe: str) -> Optional[Dict]:
        """Fetch the cached OB state. Returns None if no row exists yet
        (i.e. scanner hasn't computed for this symbol+TF), or a dict with
        `bias=None` when scanner ran but found no valid OB.
        
        The distinction matters: 'never computed' is reasonable to
        wait-and-retry, while 'computed and empty' is a definitive signal
        for the gate to block.
        """
        session = get_session()
        try:
            row = session.query(SMCOBState).filter_by(
                symbol=symbol, timeframe=timeframe).first()
            if row is None:
                return None
            return row.to_dict()
        except Exception as e:
            print(f"[DB] get_smc_ob_state error for {symbol}@{timeframe}: {e}")
            return None
        finally:
            session.close()
    
    def get_smc_ob_states_bulk(self, symbols: List[str],
                                timeframe: str) -> Dict[str, Dict]:
        """Batch fetcher: returns dict {symbol → ob_state_dict} for all
        rows matching the requested symbols at the given timeframe.
        Symbols with no row are simply absent from the returned dict
        (caller can treat absence as "not yet computed").
        
        Used by SMCScanner.get_state() to efficiently populate per-symbol
        OB markers in the watchlist UI without doing one DB roundtrip
        per symbol (would be 20+ queries on every /api/smc/state poll).
        """
        if not symbols:
            return {}
        session = get_session()
        try:
            rows = session.query(SMCOBState).filter(
                SMCOBState.timeframe == timeframe,
                SMCOBState.symbol.in_(symbols)
            ).all()
            return {r.symbol: r.to_dict() for r in rows}
        except Exception as e:
            print(f"[DB] get_smc_ob_states_bulk error: {e}")
            return {}
        finally:
            session.close()
    
    def delete_smc_ob_state_for_symbol(self, symbol: str) -> int:
        """Wipe all OB state rows for a symbol — called when symbol is
        removed from watchlist so we don't accumulate stale data.
        Returns count of rows deleted.
        """
        session = get_session()
        try:
            n = session.query(SMCOBState).filter_by(symbol=symbol).delete()
            session.commit()
            return n
        except Exception as e:
            session.rollback()
            print(f"[DB] delete_smc_ob_state_for_symbol error for {symbol}: {e}")
            return 0
        finally:
            session.close()
    
    # ============================================================
    # TOP-100 4H OB Radar — snapshot + history
    # ============================================================
    
    def upsert_top100_ob_snapshot(self, symbol: str, market_ctx: Dict,
                                    ob_data: Optional[Dict],
                                    is_fresh_for_symbol: bool = False) -> bool:
        """Upsert a TOP-100 OB snapshot row.
        
        Args:
            symbol: e.g. 'BTCUSDT'
            market_ctx: dict with 'quote_volume_24h', 'last_price',
                'price_change_24h' from the ticker fetch.
            ob_data: dict from ob_detector.detect_last_order_block (may be
                None if no valid OB exists for this symbol right now).
            is_fresh_for_symbol: True if this OB is different from the
                previous snapshot (created_at_t differs). When True,
                discovered_at is set to now; when False, only last_seen_at
                advances. Caller should compute this by comparing with the
                prior snapshot before calling.
        
        Returns True on successful write.
        """
        session = get_session()
        try:
            row = session.query(Top100OBSnapshot).filter_by(symbol=symbol).first()
            now = datetime.utcnow()
            if row is None:
                row = Top100OBSnapshot(symbol=symbol)
                session.add(row)
                # First time we've seen this symbol — if it has an OB now,
                # treat it as freshly discovered.
                if ob_data:
                    row.discovered_at = now
            
            row.quote_volume_24h = market_ctx.get('quote_volume_24h')
            row.last_price = market_ctx.get('last_price')
            row.price_change_24h = market_ctx.get('price_change_24h')
            
            if ob_data:
                row.bias = ob_data.get('bias')
                row.bar_high = ob_data.get('bar_high')
                row.bar_low = ob_data.get('bar_low')
                row.bar_time_ms = ob_data.get('bar_time')
                row.created_at_t = ob_data.get('created_at_t')
                row.created_by_tag = ob_data.get('created_by_tag')
                # Zone fields supplied by scanner via ob_data extension
                row.zone = ob_data.get('zone')
                row.zone_correct = ob_data.get('zone_correct')
                row.zone_pct = ob_data.get('zone_pct')
                if is_fresh_for_symbol:
                    # Caller indicated this is a different OB than before
                    row.discovered_at = now
                row.last_seen_at = now
            else:
                # No valid OB — clear OB fields but keep symbol row alive
                # so we still have market_ctx tracking.
                row.bias = None
                row.bar_high = None
                row.bar_low = None
                row.bar_time_ms = None
                row.created_at_t = None
                row.created_by_tag = None
                row.zone = None
                row.zone_correct = None
                row.zone_pct = None
                # Don't touch discovered_at — preserve "we last saw this
                # OB at X" history. last_seen_at also preserved for the
                # same reason — UI can show "OB lost N hours ago".
            
            row.scanned_at = now
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"[DB] upsert_top100_ob_snapshot error for {symbol}: {e}")
            return False
        finally:
            session.close()
    
    def clear_top100_ob_snapshots(self) -> int:
        """Remove ALL Top100 OB snapshot rows. Returns row count cleared.
        
        Used when the scanner timeframe changes — OBs computed on different
        TFs aren't comparable, so a clean slate is the only safe option.
        We DON'T touch the history table (sob_top100_ob_history); it's
        an audit log of past events and stays intact for analytics.
        """
        session = get_session()
        try:
            count = session.query(Top100OBSnapshot).delete(synchronize_session=False)
            session.commit()
            return int(count or 0)
        except Exception as e:
            session.rollback()
            print(f"[DB] clear_top100_ob_snapshots error: {e}")
            return 0
        finally:
            session.close()
    
    def get_top100_ob_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get a single symbol's current snapshot row, or None."""
        session = get_session()
        try:
            row = session.query(Top100OBSnapshot).filter_by(symbol=symbol).first()
            return row.to_dict() if row else None
        except Exception as e:
            print(f"[DB] get_top100_ob_snapshot error for {symbol}: {e}")
            return None
        finally:
            session.close()
    
    def list_top100_ob_snapshots(self, only_with_ob: bool = False,
                                   min_quote_volume: float = 0.0
                                   ) -> List[Dict]:
        """Return all current snapshots (optionally filtered).
        
        Args:
            only_with_ob: if True, return only rows where bias is set
                (i.e. an OB currently exists). Default False — UI may
                want to show "no OB" symbols too.
            min_quote_volume: filter out symbols with 24h vol below this.
        """
        session = get_session()
        try:
            q = session.query(Top100OBSnapshot)
            if only_with_ob:
                q = q.filter(Top100OBSnapshot.bias.isnot(None))
            if min_quote_volume > 0:
                q = q.filter(Top100OBSnapshot.quote_volume_24h >= min_quote_volume)
            # Sort by quote_volume descending — most-traded first
            q = q.order_by(Top100OBSnapshot.quote_volume_24h.desc())
            return [r.to_dict() for r in q.all()]
        except Exception as e:
            print(f"[DB] list_top100_ob_snapshots error: {e}")
            return []
        finally:
            session.close()
    
    def add_top100_ob_history(self, symbol: str, event_type: str,
                                ob_data: Optional[Dict],
                                price_at_event: Optional[float],
                                quote_volume_24h: Optional[float]) -> bool:
        """Append a history row. event_type: 'created' | 'mitigated' | 'replaced'.
        
        For 'mitigated' events ob_data should be the OB that was mitigated
        (snapshot from the previous scan). For 'created' / 'replaced' it's
        the new OB. Caller is responsible for providing the right snapshot.
        """
        if event_type not in ('created', 'mitigated', 'replaced'):
            print(f"[DB] add_top100_ob_history: invalid event_type {event_type!r}")
            return False
        session = get_session()
        try:
            h = Top100OBHistory(
                symbol=symbol,
                event_type=event_type,
                bias=ob_data.get('bias') if ob_data else None,
                bar_high=ob_data.get('bar_high') if ob_data else None,
                bar_low=ob_data.get('bar_low') if ob_data else None,
                bar_time_ms=ob_data.get('bar_time') if ob_data else None,
                created_by_tag=ob_data.get('created_by_tag') if ob_data else None,
                price_at_event=price_at_event,
                quote_volume_24h=quote_volume_24h,
            )
            session.add(h)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"[DB] add_top100_ob_history error for {symbol}: {e}")
            return False
        finally:
            session.close()
    
    def list_top100_ob_history(self, hours: int = 24,
                                 event_types: Optional[List[str]] = None,
                                 limit: int = 100) -> List[Dict]:
        """Return recent history events.
        
        Args:
            hours: how far back to look (default 24h for the "Recent
                Discoveries" feed).
            event_types: optional filter to specific event types.
            limit: cap result count.
        """
        from datetime import timedelta
        session = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            q = session.query(Top100OBHistory).filter(
                Top100OBHistory.created_at >= cutoff)
            if event_types:
                q = q.filter(Top100OBHistory.event_type.in_(event_types))
            q = q.order_by(Top100OBHistory.created_at.desc()).limit(limit)
            return [r.to_dict() for r in q.all()]
        except Exception as e:
            print(f"[DB] list_top100_ob_history error: {e}")
            return []
        finally:
            session.close()
    
    def cleanup_top100_ob_history(self, retention_days: int = 30) -> int:
        """Delete history rows older than `retention_days`. Called daily by
        the scanner. Returns rows deleted.
        """
        from datetime import timedelta
        session = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=retention_days)
            n = session.query(Top100OBHistory).filter(
                Top100OBHistory.created_at < cutoff).delete()
            session.commit()
            return n
        except Exception as e:
            session.rollback()
            print(f"[DB] cleanup_top100_ob_history error: {e}")
            return 0
        finally:
            session.close()
    
    # ============================================================
    # Volumized OB Radar — metadata / stats / snapshots
    # ============================================================
    
    def volradar_get_metadata(self, symbol: str) -> Optional[Dict]:
        """Return active radar metadata for a symbol, or None if not tracked."""
        session = get_session()
        try:
            row = session.query(VolumizedRadarMetadata).filter_by(symbol=symbol).first()
            return row.to_dict() if row else None
        except Exception as e:
            print(f"[DB] volradar_get_metadata error: {e}")
            return None
        finally:
            session.close()
    
    def volradar_list_metadata(self) -> List[Dict]:
        """All currently tracked radar items (for dashboard view)."""
        session = get_session()
        try:
            rows = session.query(VolumizedRadarMetadata).order_by(
                VolumizedRadarMetadata.added_at.desc()).all()
            return [r.to_dict() for r in rows]
        except Exception as e:
            print(f"[DB] volradar_list_metadata error: {e}")
            return []
        finally:
            session.close()
    
    def volradar_add(self, symbol: str, ob_direction: str,
                    ob_top: float, ob_bottom: float, ob_volume: float,
                    pd_zone_pct: float, scan_tf: str,
                    ttl_hours: int = 24) -> bool:
        """Insert metadata row + bump stats. Returns True if new add, False
        if a row already existed (radar shouldn't re-add the same symbol).
        
        Stats: times_added++, last_added_at=now.
        """
        session = get_session()
        try:
            # Idempotency: refuse to insert if metadata already present.
            existing = session.query(VolumizedRadarMetadata).filter_by(symbol=symbol).first()
            if existing:
                return False
            
            now = datetime.utcnow()
            session.add(VolumizedRadarMetadata(
                symbol=symbol,
                added_at=now,
                expires_at=now + timedelta(hours=ttl_hours),
                ob_direction=ob_direction,
                ob_top=ob_top,
                ob_bottom=ob_bottom,
                ob_volume=ob_volume,
                pd_zone_pct=pd_zone_pct,
                scan_tf=scan_tf,
            ))
            
            # Stats upsert
            stat = session.query(VolumizedRadarStat).filter_by(symbol=symbol).first()
            if stat is None:
                stat = VolumizedRadarStat(symbol=symbol, times_added=1, last_added_at=now)
                session.add(stat)
            else:
                stat.times_added = (stat.times_added or 0) + 1
                stat.last_added_at = now
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"[DB] volradar_add error for {symbol}: {e}")
            return False
        finally:
            session.close()
    
    def volradar_mark_signal_fired(self, symbol: str) -> bool:
        """Called from smc_scanner._send_alert hook. If symbol is tracked by
        radar, latch signal_fired_at and bump times_signal_fired. Deletes
        the metadata row (symbol "graduates" to normal watchlist flow).
        
        Returns True if a row was removed; False if symbol wasn't tracked
        (which is the normal case for most alerts).
        """
        session = get_session()
        try:
            row = session.query(VolumizedRadarMetadata).filter_by(symbol=symbol).first()
            if row is None:
                return False
            
            stat = session.query(VolumizedRadarStat).filter_by(symbol=symbol).first()
            if stat is not None:
                stat.times_signal_fired = (stat.times_signal_fired or 0) + 1
            
            session.delete(row)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"[DB] volradar_mark_signal_fired error: {e}")
            return False
        finally:
            session.close()
    
    def volradar_remove(self, symbol: str, reason: str,
                       cooldown_hours: int = 6) -> bool:
        """Delete metadata row + bump correct removal counter + set cooldown.
        
        Args:
            reason: 'auto_ttl' | 'manual' — drives which counter ++
            cooldown_hours: future re-adds blocked until now+cooldown
        """
        session = get_session()
        try:
            row = session.query(VolumizedRadarMetadata).filter_by(symbol=symbol).first()
            if row is None:
                return False
            
            stat = session.query(VolumizedRadarStat).filter_by(symbol=symbol).first()
            if stat is not None:
                if reason == 'auto_ttl':
                    stat.times_auto_removed = (stat.times_auto_removed or 0) + 1
                elif reason == 'manual':
                    stat.times_manual_removed = (stat.times_manual_removed or 0) + 1
                stat.last_cooldown_until = datetime.utcnow() + timedelta(hours=cooldown_hours)
            
            session.delete(row)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"[DB] volradar_remove error: {e}")
            return False
        finally:
            session.close()
    
    def volradar_find_expired(self) -> List[str]:
        """Return symbols whose TTL has elapsed (cleanup daemon's worklist)."""
        session = get_session()
        try:
            rows = session.query(VolumizedRadarMetadata.symbol).filter(
                VolumizedRadarMetadata.expires_at < datetime.utcnow()).all()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"[DB] volradar_find_expired error: {e}")
            return []
        finally:
            session.close()
    
    def volradar_get_stat(self, symbol: str) -> Optional[Dict]:
        """Lifetime counters for a symbol. None if never tracked."""
        session = get_session()
        try:
            row = session.query(VolumizedRadarStat).filter_by(symbol=symbol).first()
            return row.to_dict() if row else None
        except Exception as e:
            print(f"[DB] volradar_get_stat error: {e}")
            return None
        finally:
            session.close()
    
    def volradar_list_stats(self, limit: int = 100) -> List[Dict]:
        """All stats rows sorted by times_added desc (top-performers first)."""
        session = get_session()
        try:
            rows = session.query(VolumizedRadarStat).order_by(
                VolumizedRadarStat.times_added.desc()).limit(limit).all()
            return [r.to_dict() for r in rows]
        except Exception as e:
            print(f"[DB] volradar_list_stats error: {e}")
            return []
        finally:
            session.close()
    
    def volradar_is_on_cooldown(self, symbol: str) -> bool:
        """True if symbol is still inside its post-removal cooldown window."""
        session = get_session()
        try:
            stat = session.query(VolumizedRadarStat).filter_by(symbol=symbol).first()
            if stat is None or stat.last_cooldown_until is None:
                return False
            return stat.last_cooldown_until > datetime.utcnow()
        except Exception as e:
            print(f"[DB] volradar_is_on_cooldown error: {e}")
            return False
        finally:
            session.close()
    
    def volradar_log_snapshot(self, symbol: str, qualified: bool, action: str,
                              ob_direction: Optional[str] = None,
                              ob_top: Optional[float] = None,
                              ob_bottom: Optional[float] = None,
                              pd_zone_pct: Optional[float] = None,
                              error_msg: Optional[str] = None) -> None:
        """Append one audit-log entry per (scan, symbol). Cheap & idempotent."""
        session = get_session()
        try:
            session.add(VolumizedRadarSnapshot(
                scan_time=datetime.utcnow(),
                symbol=symbol,
                ob_direction=ob_direction,
                ob_top=ob_top,
                ob_bottom=ob_bottom,
                pd_zone_pct=pd_zone_pct,
                qualified=qualified,
                action=action,
                error_msg=error_msg,
            ))
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[DB] volradar_log_snapshot error: {e}")
        finally:
            session.close()
    
    def volradar_list_snapshots(self, hours: int = 24, limit: int = 500) -> List[Dict]:
        """Recent audit log entries (default last 24h, capped at 500 rows)."""
        session = get_session()
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            rows = session.query(VolumizedRadarSnapshot).filter(
                VolumizedRadarSnapshot.scan_time >= since
            ).order_by(VolumizedRadarSnapshot.scan_time.desc()).limit(limit).all()
            return [r.to_dict() for r in rows]
        except Exception as e:
            print(f"[DB] volradar_list_snapshots error: {e}")
            return []
        finally:
            session.close()
    
    def volradar_prune_snapshots(self, retention_days: int = 7) -> int:
        """Daily cleanup — keep last `retention_days` of audit log."""
        session = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=retention_days)
            n = session.query(VolumizedRadarSnapshot).filter(
                VolumizedRadarSnapshot.scan_time < cutoff).delete()
            session.commit()
            return n
        except Exception as e:
            session.rollback()
            print(f"[DB] volradar_prune_snapshots error: {e}")
            return 0
        finally:
            session.close()


# Singleton instance
_db_instance = None

def get_db() -> DBOperations:
    """Get database operations instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DBOperations()
    return _db_instance
