"""
CTR Database Operations - Additional methods for CTR Scanner v8.3.0

Add these methods to the DBOperations class in db_operations.py
"""

# Add to imports at the top of db_operations.py:
# from storage.db_models import CTRWatchlistItem, CTRSignal, CTRKlineCache

# ==========================================
# CTR WATCHLIST OPERATIONS (v8.3.0)
# ==========================================

def get_ctr_watchlist(self) -> List[str]:
    """Get CTR watchlist from database"""
    from storage.db_models import CTRWatchlistItem, get_session
    session = get_session()
    try:
        items = session.query(CTRWatchlistItem).filter_by(is_active=True).all()
        return [item.symbol for item in items]
    except Exception as e:
        print(f"[DB CTR] Error getting watchlist: {e}")
        return []
    finally:
        session.close()

def add_ctr_watchlist_item(self, symbol: str) -> bool:
    """Add symbol to CTR watchlist"""
    from storage.db_models import CTRWatchlistItem, get_session
    session = get_session()
    try:
        symbol = symbol.upper()
        existing = session.query(CTRWatchlistItem).filter_by(symbol=symbol).first()
        if existing:
            existing.is_active = True
        else:
            item = CTRWatchlistItem(symbol=symbol, is_active=True)
            session.add(item)
        session.commit()
        print(f"[DB CTR] Added to watchlist: {symbol}")
        return True
    except Exception as e:
        session.rollback()
        print(f"[DB CTR] Error adding to watchlist: {e}")
        return False
    finally:
        session.close()

def remove_ctr_watchlist_item(self, symbol: str, delete_data: bool = True) -> bool:
    """Remove symbol from CTR watchlist and optionally delete all related data"""
    from storage.db_models import CTRWatchlistItem, CTRSignal, CTRKlineCache, get_session
    session = get_session()
    try:
        symbol = symbol.upper()
        
        # Remove from watchlist
        deleted = session.query(CTRWatchlistItem).filter_by(symbol=symbol).delete()
        
        if delete_data:
            # Delete signals for this symbol
            signals_deleted = session.query(CTRSignal).filter_by(symbol=symbol).delete()
            
            # Delete kline cache for this symbol
            cache_deleted = session.query(CTRKlineCache).filter_by(symbol=symbol).delete()
            
            print(f"[DB CTR] Removed {symbol}: watchlist={deleted}, signals={signals_deleted}, cache={cache_deleted}")
        
        session.commit()
        return deleted > 0
    except Exception as e:
        session.rollback()
        print(f"[DB CTR] Error removing from watchlist: {e}")
        return False
    finally:
        session.close()


# ==========================================
# CTR SIGNALS OPERATIONS (v8.3.0)
# ==========================================

def add_ctr_signal(self, symbol: str, signal_type: str, price: float, 
                   stc: float = None, timeframe: str = None,
                   smc_filtered: bool = False, smc_trend: str = None,
                   zone: str = None, notified: bool = True) -> bool:
    """Add CTR signal to database"""
    from storage.db_models import CTRSignal, get_session
    from datetime import datetime, timezone
    session = get_session()
    try:
        signal = CTRSignal(
            symbol=symbol.upper(),
            signal_type=signal_type,
            price=price,
            stc=stc,
            timeframe=timeframe,
            smc_filtered=smc_filtered,
            smc_trend=smc_trend,
            zone=zone,
            notified=notified,
            timestamp=datetime.now(timezone.utc)
        )
        session.add(signal)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"[DB CTR] Error adding signal: {e}")
        return False
    finally:
        session.close()

def get_last_ctr_signal(self, symbol: str) -> Optional[Dict]:
    """Get last CTR signal for a symbol"""
    from storage.db_models import CTRSignal, get_session
    session = get_session()
    try:
        signal = session.query(CTRSignal).filter_by(
            symbol=symbol.upper()
        ).order_by(CTRSignal.timestamp.desc()).first()
        
        if signal:
            return signal.to_dict()
        return None
    except Exception as e:
        print(f"[DB CTR] Error getting last signal: {e}")
        return None
    finally:
        session.close()

def get_ctr_signals(self, limit: int = 20, symbol: str = None) -> List[Dict]:
    """Get CTR signals"""
    from storage.db_models import CTRSignal, get_session
    session = get_session()
    try:
        query = session.query(CTRSignal)
        if symbol:
            query = query.filter_by(symbol=symbol.upper())
        signals = query.order_by(CTRSignal.timestamp.desc()).limit(limit).all()
        return [s.to_dict() for s in signals]
    except Exception as e:
        print(f"[DB CTR] Error getting signals: {e}")
        return []
    finally:
        session.close()

def delete_ctr_signal(self, signal_id: int) -> bool:
    """Delete a specific CTR signal by ID"""
    from storage.db_models import CTRSignal, get_session
    session = get_session()
    try:
        deleted = session.query(CTRSignal).filter_by(id=signal_id).delete()
        session.commit()
        return deleted > 0
    except Exception as e:
        session.rollback()
        print(f"[DB CTR] Error deleting signal: {e}")
        return False
    finally:
        session.close()

def clear_ctr_signals(self, symbol: str = None) -> int:
    """Clear all CTR signals or for a specific symbol"""
    from storage.db_models import CTRSignal, get_session
    session = get_session()
    try:
        query = session.query(CTRSignal)
        if symbol:
            query = query.filter_by(symbol=symbol.upper())
        count = query.delete()
        session.commit()
        print(f"[DB CTR] Cleared {count} signals")
        return count
    except Exception as e:
        session.rollback()
        print(f"[DB CTR] Error clearing signals: {e}")
        return 0
    finally:
        session.close()

def is_duplicate_ctr_signal(self, symbol: str, signal_type: str) -> bool:
    """
    Check if this signal is a duplicate of the last signal.
    Returns True if the last signal for this symbol has the same direction.
    """
    last_signal = self.get_last_ctr_signal(symbol)
    if last_signal and last_signal.get('type') == signal_type:
        return True
    return False


# ==========================================
# CTR KLINE CACHE OPERATIONS (v8.3.0)
# ==========================================

def save_ctr_klines(self, symbol: str, timeframe: str, klines_data: str, candles_count: int) -> bool:
    """Save klines cache to database"""
    from storage.db_models import CTRKlineCache, get_session
    session = get_session()
    try:
        symbol = symbol.upper()
        cache = session.query(CTRKlineCache).filter_by(
            symbol=symbol, timeframe=timeframe
        ).first()
        
        if cache:
            cache.klines_data = klines_data
            cache.candles_count = candles_count
        else:
            cache = CTRKlineCache(
                symbol=symbol,
                timeframe=timeframe,
                klines_data=klines_data,
                candles_count=candles_count
            )
            session.add(cache)
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"[DB CTR] Error saving klines: {e}")
        return False
    finally:
        session.close()

def get_ctr_klines(self, symbol: str, timeframe: str) -> Optional[str]:
    """Get klines cache from database"""
    from storage.db_models import CTRKlineCache, get_session
    session = get_session()
    try:
        cache = session.query(CTRKlineCache).filter_by(
            symbol=symbol.upper(), timeframe=timeframe
        ).first()
        
        if cache:
            return cache.klines_data
        return None
    except Exception as e:
        print(f"[DB CTR] Error getting klines: {e}")
        return None
    finally:
        session.close()

def delete_ctr_klines(self, symbol: str) -> bool:
    """Delete klines cache for a symbol"""
    from storage.db_models import CTRKlineCache, get_session
    session = get_session()
    try:
        deleted = session.query(CTRKlineCache).filter_by(symbol=symbol.upper()).delete()
        session.commit()
        return deleted > 0
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()

def clear_ctr_klines(self) -> int:
    """Clear all klines cache"""
    from storage.db_models import CTRKlineCache, get_session
    session = get_session()
    try:
        count = session.query(CTRKlineCache).delete()
        session.commit()
        print(f"[DB CTR] Cleared {count} kline caches")
        return count
    except Exception as e:
        session.rollback()
        return 0
    finally:
        session.close()
