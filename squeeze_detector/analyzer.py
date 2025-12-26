#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SQUEEZE DETECTOR v1.0 - ANALYZER                         ║
║                                                                              ║
║  "Мозок" системи - аналізує дані та генерує сигнали:                         ║
║  • K = ΔOI / ΔPrice - коефіцієнт аномалії                                    ║
║  • Фільтр флету (price change < threshold)                                   ║
║  • Фільтр накопичення (OI change > threshold)                                ║
║  • Funding bias для визначення напрямку                                      ║
║                                                                              ║
║  Алгоритм "Мисливець":                                                       ║
║  1. Фільтр волатильності: ціна < 2% за 4h                                    ║
║  2. Фільтр аномалії OI: OI > 5% за 4h                                        ║
║  3. K = ΔOI / ΔPrice - чим вище, тим сильніше накопичення                    ║
║  4. Funding < 0 → LONG bias (Short Squeeze)                                  ║
║  5. Funding > 0 → SHORT bias (Long Squeeze)                                  ║
║                                                                              ║
║  Автор: SVV Webhook Bot                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# === SAFE MODEL IMPORTS ===
try:
    from .models import (
        MarketSnapshot, 
        SqueezeSignal, 
        SqueezeWatchlist,
        get_snapshots_for_period
    )
except ImportError:
    try:
        from squeeze_detector.models import (
            MarketSnapshot, 
            SqueezeSignal, 
            SqueezeWatchlist,
            get_snapshots_for_period
        )
    except ImportError:
        logger.error("Failed to import models - squeeze_detector may not work")
        MarketSnapshot = None
        SqueezeSignal = None
        SqueezeWatchlist = None
        get_snapshots_for_period = None


# ============================================================================
#                              ENUMS & CONSTANTS
# ============================================================================

class SignalType(Enum):
    """Типи сигналів"""
    ACCUMULATION_START = "ACCUMULATION_START"      # Початок накопичення
    ACCUMULATION_CONTINUE = "ACCUMULATION_CONTINUE" # Продовження
    SQUEEZE_READY = "SQUEEZE_READY"                # Готовий до вибуху
    BREAKOUT_UP = "BREAKOUT_UP"                    # Пробій вгору
    BREAKOUT_DOWN = "BREAKOUT_DOWN"                # Пробій вниз


class Direction(Enum):
    """Напрямок очікуваного руху"""
    LONG = "LONG"      # Очікується памп (Short Squeeze)
    SHORT = "SHORT"    # Очікується дамп (Long Squeeze)
    UNKNOWN = "UNKNOWN"


class WatchlistPhase(Enum):
    """Фази watchlist"""
    WATCHING = "WATCHING"           # Спостереження
    ACCUMULATING = "ACCUMULATING"   # Підтверджене накопичення
    SQUEEZE_READY = "SQUEEZE_READY" # Готовий
    TRIGGERED = "TRIGGERED"         # Breakout стався
    EXPIRED = "EXPIRED"             # Timeout


# Default налаштування
DEFAULT_CONFIG = {
    # Фільтри
    'sd_price_change_threshold': 2.0,    # Флет якщо ціна < 2% за період
    'sd_oi_change_threshold': 5.0,       # Аномалія якщо OI > 5% за період
    'sd_k_coefficient_threshold': 3.0,   # K > 3 = сигнал
    
    # Lookback періоди (години)
    'sd_lookback_4h': True,
    'sd_lookback_8h': True,
    'sd_lookback_24h': True,
    
    # Funding bias
    'sd_funding_extreme_positive': 0.0003,  # 0.03% - Long Squeeze territory
    'sd_funding_extreme_negative': -0.0003, # -0.03% - Short Squeeze territory
    
    # Watchlist
    'sd_min_consecutive_signals': 2,     # Мін. сканів підряд для ACCUMULATING
    'sd_ready_consecutive_signals': 4,   # Сканів для SQUEEZE_READY
    'sd_watchlist_timeout_hours': 48,    # Timeout якщо немає breakout
    
    # Breakout detection
    'sd_breakout_threshold': 3.0,        # >3% рух = breakout
    
    # Confidence
    'sd_base_confidence': 50,
    'sd_k_confidence_multiplier': 5,     # +5 confidence за кожен K
    'sd_funding_confidence_bonus': 15,   # +15 якщо funding extreme
}


# ============================================================================
#                              DATA CLASSES
# ============================================================================

@dataclass
class AnalysisResult:
    """Результат аналізу однієї монети"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Зміни за період
    price_change_pct: float = 0.0
    oi_change_pct: float = 0.0
    volume_change_pct: float = 0.0
    
    # K coefficient
    k_coefficient: float = 0.0
    
    # Funding
    funding_rate: float = 0.0
    funding_bias: str = "NEUTRAL"
    
    # Поточні значення
    current_price: float = 0.0
    current_oi: float = 0.0
    
    # Lookback
    lookback_hours: int = 4
    
    # Volatility
    volatility_range_pct: float = 0.0  # (max - min) / min * 100
    
    # Фільтри
    is_flat: bool = False           # Net Change < threshold
    is_low_volatility: bool = False # Range < threshold
    is_accumulating: bool = False   # OI росте
    
    # Сигнал
    has_signal: bool = False
    signal_type: str = None
    direction: str = "UNKNOWN"
    confidence: int = 0
    
    # Додаткові дані
    raw_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price_change_pct': round(self.price_change_pct, 2),
            'volatility_range_pct': round(self.volatility_range_pct, 2),
            'oi_change_pct': round(self.oi_change_pct, 2),
            'k_coefficient': round(self.k_coefficient, 2),
            'funding_rate': self.funding_rate,
            'funding_bias': self.funding_bias,
            'current_price': self.current_price,
            'current_oi': self.current_oi,
            'lookback_hours': self.lookback_hours,
            'is_flat': self.is_flat,
            'is_low_volatility': self.is_low_volatility,
            'is_accumulating': self.is_accumulating,
            'has_signal': self.has_signal,
            'signal_type': self.signal_type,
            'direction': self.direction,
            'confidence': self.confidence,
        }


@dataclass
class HeatmapEntry:
    """Запис для теплової карти"""
    symbol: str
    k_4h: float = 0.0
    k_8h: float = 0.0
    k_24h: float = 0.0
    k_max: float = 0.0
    
    price_change_4h: float = 0.0
    price_change_24h: float = 0.0
    volatility_range_4h: float = 0.0
    oi_change_4h: float = 0.0
    oi_change_24h: float = 0.0
    
    funding_rate: float = 0.0
    funding_bias: str = "NEUTRAL"
    
    current_price: float = 0.0
    volume_24h: float = 0.0
    
    phase: str = "NONE"
    direction: str = "UNKNOWN"
    confidence: int = 0
    
    in_watchlist: bool = False
    consecutive_signals: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'k_4h': round(self.k_4h, 2),
            'k_8h': round(self.k_8h, 2),
            'k_24h': round(self.k_24h, 2),
            'k_max': round(self.k_max, 2),
            'price_change_4h': round(self.price_change_4h, 2),
            'price_change_24h': round(self.price_change_24h, 2),
            'volatility_range_4h': round(self.volatility_range_4h, 2),
            'oi_change_4h': round(self.oi_change_4h, 2),
            'oi_change_24h': round(self.oi_change_24h, 2),
            'funding_rate': round(self.funding_rate * 100, 4),  # В %
            'funding_bias': self.funding_bias,
            'current_price': self.current_price,
            'volume_24h': self.volume_24h,
            'phase': self.phase,
            'direction': self.direction,
            'confidence': self.confidence,
            'in_watchlist': self.in_watchlist,
            'consecutive_signals': self.consecutive_signals,
        }


# ============================================================================
#                              ANALYZER CLASS
# ============================================================================

class SqueezeAnalyzer:
    """
    Аналізатор для виявлення squeeze ситуацій.
    
    Основна логіка:
    1. Завантажує snapshots з БД
    2. Рахує K = ΔOI / ΔPrice
    3. Генерує сигнали якщо K > threshold
    4. Оновлює watchlist
    """
    
    def __init__(self, db_session_factory, config: Dict = None):
        self.db_session_factory = db_session_factory
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Статистика
        self.stats = {
            'analyses_run': 0,
            'signals_generated': 0,
            'last_analysis_time': None,
        }
        
        # Background analysis
        self._analyzing = False
        self._analyze_thread: Optional[threading.Thread] = None
        self._analyze_interval = 300  # 5 хвилин
        
        logger.info("🧠 SqueezeAnalyzer initialized")
    
    def update_config(self, config: Dict):
        """Оновлює конфігурацію"""
        self.config.update(config)
        logger.info("🔧 Analyzer config updated")
    
    def _get(self, key: str, default: Any = None) -> Any:
        """Отримує значення з конфігу"""
        return self.config.get(key, default)
    
    def analyze_symbol(
        self,
        symbol: str,
        lookback_hours: int = 4
    ) -> Optional[AnalysisResult]:
        """
        Аналізує одну монету за вказаний період.
        
        Args:
            symbol: Торгова пара
            lookback_hours: Період аналізу в годинах
            
        Returns:
            AnalysisResult або None якщо недостатньо даних
        """
        # Models вже імпортовані глобально
        
        session = self.db_session_factory()
        
        try:
            # Отримуємо snapshots за період
            snapshots = get_snapshots_for_period(session, symbol, lookback_hours)
            
            if len(snapshots) < 2:
                logger.debug(f"Not enough snapshots for {symbol}: {len(snapshots)}")
                return None
            
            # Перший і останній snapshot
            first = snapshots[0]
            last = snapshots[-1]
            
            # === NET CHANGE (Classic) ===
            # Порівнюємо першу і останню точку
            price_change = 0.0
            if first.mark_price and first.mark_price > 0:
                price_change = ((last.mark_price or last.last_price or 0) - first.mark_price) / first.mark_price * 100
            
            # === VOLATILITY RANGE (Strict) ===
            # Знаходимо min/max за весь період
            prices = [s.mark_price or s.last_price or 0 for s in snapshots if (s.mark_price or s.last_price)]
            
            volatility_range_pct = 0.0
            if prices:
                min_price = min(prices)
                max_price = max(prices)
                if min_price > 0:
                    volatility_range_pct = (max_price - min_price) / min_price * 100
            
            # OI change
            oi_change = 0.0
            if first.open_interest and first.open_interest > 0:
                oi_change = ((last.open_interest or 0) - first.open_interest) / first.open_interest * 100
            
            # K coefficient
            # Запобігаємо діленню на 0 або дуже маленьке число
            price_abs = max(abs(price_change), 0.1)
            k_coefficient = oi_change / price_abs if price_abs > 0 else 0
            
            # Funding bias
            funding_rate = last.funding_rate or 0
            funding_extreme_pos = self._get('sd_funding_extreme_positive', 0.0003)
            funding_extreme_neg = self._get('sd_funding_extreme_negative', -0.0003)
            
            if funding_rate >= funding_extreme_pos:
                funding_bias = "SHORT"  # Long Squeeze potential
            elif funding_rate <= funding_extreme_neg:
                funding_bias = "LONG"   # Short Squeeze potential
            else:
                funding_bias = "NEUTRAL"
            
            # === ФІЛЬТРИ ===
            price_threshold = self._get('sd_price_change_threshold', 2.0)
            volatility_threshold = self._get('sd_volatility_threshold', 4.0)
            oi_threshold = self._get('sd_oi_change_threshold', 5.0)
            k_threshold = self._get('sd_k_coefficient_threshold', 3.0)
            analysis_method = self._get('sd_analysis_method', 'combined')
            
            # Розраховуємо обидва фільтри
            is_flat = abs(price_change) < price_threshold           # Net Change фільтр
            is_low_volatility = volatility_range_pct < volatility_threshold  # Range фільтр
            is_accumulating = oi_change >= oi_threshold
            
            # === ВИЗНАЧЕННЯ СИГНАЛУ ЗА МЕТОДОМ ===
            if analysis_method == 'net_change':
                # Classic: тільки Net Change
                is_squeeze = is_flat
            elif analysis_method == 'volatility_range':
                # Strict: тільки Range
                is_squeeze = is_low_volatility
            else:  # combined (default)
                # Recommended: обидва фільтри
                is_squeeze = is_flat and is_low_volatility
            
            has_signal = is_squeeze and is_accumulating and k_coefficient >= k_threshold
            
            # Визначаємо напрямок
            direction = "UNKNOWN"
            if has_signal:
                if funding_bias == "LONG":
                    direction = "LONG"
                elif funding_bias == "SHORT":
                    direction = "SHORT"
                else:
                    # Якщо funding нейтральний, дивимось на OI trend
                    direction = "LONG" if oi_change > 0 else "SHORT"
            
            # Confidence score
            confidence = self._get('sd_base_confidence', 50)
            if has_signal:
                # +5 за кожен K понад threshold
                k_bonus = int((k_coefficient - k_threshold) * self._get('sd_k_confidence_multiplier', 5))
                confidence += min(k_bonus, 30)  # Max +30
                
                # +15 якщо funding extreme
                if funding_bias in ["LONG", "SHORT"]:
                    confidence += self._get('sd_funding_confidence_bonus', 15)
                
                # +10 якщо Combined (обидва фільтри спрацювали)
                if analysis_method == 'combined' and is_flat and is_low_volatility:
                    confidence += 10
                
                confidence = min(confidence, 100)
            
            # Signal type
            signal_type = None
            if has_signal:
                signal_type = SignalType.ACCUMULATION_START.value
            
            result = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price_change_pct=price_change,
                volatility_range_pct=volatility_range_pct,
                oi_change_pct=oi_change,
                k_coefficient=k_coefficient,
                funding_rate=funding_rate,
                funding_bias=funding_bias,
                current_price=last.mark_price or last.last_price or 0,
                current_oi=last.open_interest or 0,
                lookback_hours=lookback_hours,
                is_flat=is_flat,
                is_low_volatility=is_low_volatility,
                is_accumulating=is_accumulating,
                has_signal=has_signal,
                signal_type=signal_type,
                direction=direction,
                confidence=confidence,
                raw_data={
                    'first_timestamp': first.timestamp.isoformat() if first.timestamp else None,
                    'last_timestamp': last.timestamp.isoformat() if last.timestamp else None,
                    'snapshots_count': len(snapshots),
                    'first_oi': first.open_interest,
                    'last_oi': last.open_interest,
                    'first_price': first.mark_price,
                    'last_price': last.mark_price,
                    'min_price': min_price if prices else None,
                    'max_price': max_price if prices else None,
                    'analysis_method': analysis_method,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Analyze error for {symbol}: {e}")
            return None
        finally:
            session.close()
    
    def analyze_all_symbols(
        self,
        symbols: List[str],
        lookback_hours: int = 4
    ) -> List[AnalysisResult]:
        """
        Аналізує всі символи.
        
        Returns:
            List[AnalysisResult] відсортований за K coefficient
        """
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol, lookback_hours)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Analysis error for {symbol}: {e}")
        
        # Сортуємо за K (найвищі першими)
        results.sort(key=lambda x: x.k_coefficient, reverse=True)
        
        self.stats['analyses_run'] += 1
        self.stats['last_analysis_time'] = datetime.utcnow()
        
        return results
    
    def generate_heatmap(self, symbols: List[str]) -> List[HeatmapEntry]:
        """
        Генерує теплову карту для всіх символів.
        Аналізує за 4h, 8h, 24h періоди.
        
        Returns:
            List[HeatmapEntry] відсортований за k_max
        """
        # Models вже імпортовані глобально
        
        session = self.db_session_factory()
        heatmap = []
        
        try:
            # Отримуємо watchlist для перевірки статусу
            watchlist_items = session.query(SqueezeWatchlist).all()
            watchlist_dict = {w.symbol: w for w in watchlist_items}
            
            for symbol in symbols:
                try:
                    entry = HeatmapEntry(symbol=symbol)
                    
                    # Аналіз за різні періоди
                    for hours in [4, 8, 24]:
                        result = self.analyze_symbol(symbol, hours)
                        
                        if result:
                            if hours == 4:
                                entry.k_4h = result.k_coefficient
                                entry.price_change_4h = result.price_change_pct
                                entry.volatility_range_4h = result.volatility_range_pct
                                entry.oi_change_4h = result.oi_change_pct
                                entry.current_price = result.current_price
                                entry.funding_rate = result.funding_rate
                                entry.funding_bias = result.funding_bias
                                entry.direction = result.direction
                                entry.confidence = result.confidence
                            elif hours == 8:
                                entry.k_8h = result.k_coefficient
                            elif hours == 24:
                                entry.k_24h = result.k_coefficient
                                entry.price_change_24h = result.price_change_pct
                                entry.oi_change_24h = result.oi_change_pct
                    
                    # Max K
                    entry.k_max = max(entry.k_4h, entry.k_8h, entry.k_24h)
                    
                    # Watchlist info
                    if symbol in watchlist_dict:
                        wl = watchlist_dict[symbol]
                        entry.in_watchlist = True
                        entry.phase = wl.phase
                        entry.consecutive_signals = wl.consecutive_signals
                        entry.direction = wl.direction or entry.direction
                    
                    heatmap.append(entry)
                    
                except Exception as e:
                    logger.debug(f"Heatmap error for {symbol}: {e}")
            
            # Сортуємо за k_max
            heatmap.sort(key=lambda x: x.k_max, reverse=True)
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Generate heatmap error: {e}")
            return []
        finally:
            session.close()
    
    def update_watchlist(
        self,
        analysis_results: List[AnalysisResult]
    ) -> Dict[str, Any]:
        """
        Оновлює watchlist на основі результатів аналізу.
        
        Logic:
        1. Додає нові монети з сигналами
        2. Оновлює consecutive_signals для існуючих
        3. Підвищує фазу якщо достатньо сигналів
        4. Видаляє/expires старі записи
        
        Returns:
            Dict зі статистикою оновлень
        """
        # Models вже імпортовані глобально
        
        session = self.db_session_factory()
        stats = {
            'added': 0,
            'updated': 0,
            'phase_upgraded': 0,
            'expired': 0,
            'triggered': 0,
        }
        
        try:
            # Отримуємо всі записи watchlist
            existing = {w.symbol: w for w in session.query(SqueezeWatchlist).all()}
            
            symbols_with_signals = set()
            
            for result in analysis_results:
                if result.has_signal:
                    symbols_with_signals.add(result.symbol)
                    
                    if result.symbol in existing:
                        # Оновлюємо існуючий
                        wl = existing[result.symbol]
                        wl.consecutive_signals += 1
                        wl.total_signals += 1
                        wl.current_price = result.current_price
                        wl.current_oi = result.current_oi
                        wl.current_k = result.k_coefficient
                        wl.last_update = datetime.utcnow()
                        
                        # Оновлюємо накопичену статистику
                        if wl.entry_oi and wl.entry_oi > 0:
                            wl.total_oi_change_pct = ((wl.current_oi or 0) - wl.entry_oi) / wl.entry_oi * 100
                        if wl.entry_price and wl.entry_price > 0:
                            wl.total_price_change_pct = ((wl.current_price or 0) - wl.entry_price) / wl.entry_price * 100
                        
                        # Direction
                        wl.direction = result.direction
                        wl.direction_confidence = result.confidence
                        wl.funding_bias = result.funding_bias
                        
                        # Фаза
                        min_for_accumulating = self._get('sd_min_consecutive_signals', 2)
                        min_for_ready = self._get('sd_ready_consecutive_signals', 4)
                        
                        old_phase = wl.phase
                        
                        if wl.consecutive_signals >= min_for_ready:
                            wl.phase = WatchlistPhase.SQUEEZE_READY.value
                        elif wl.consecutive_signals >= min_for_accumulating:
                            wl.phase = WatchlistPhase.ACCUMULATING.value
                        
                        if wl.phase != old_phase:
                            stats['phase_upgraded'] += 1
                            logger.info(f"📈 {result.symbol}: {old_phase} → {wl.phase}")
                            
                            # === TELEGRAM ALERT ===
                            try:
                                from .routes import get_detector_manager
                                mgr = get_detector_manager()
                                
                                if mgr and mgr.config.get('sd_telegram_alerts') and hasattr(mgr, 'bot_instance') and mgr.bot_instance:
                                    emoji = "👀"
                                    if wl.phase == "ACCUMULATING": emoji = "🔋"
                                    if wl.phase == "SQUEEZE_READY": emoji = "🔥"
                                    if wl.phase == "TRIGGERED": emoji = "🚀"
                                    
                                    msg = (
                                        f"{emoji} <b>SQUEEZE ALERT: {result.symbol}</b>\n"
                                        f"Phase: {old_phase} ➡️ <b>{wl.phase}</b>\n"
                                        f"K-Coeff: <code>{result.k_coefficient:.2f}</code>\n"
                                        f"OI Δ: <code>{result.oi_change_pct:+.2f}%</code>\n"
                                        f"Price Δ: <code>{result.price_change_pct:+.2f}%</code>\n"
                                        f"Direction: {result.direction}"
                                    )
                                    mgr.bot_instance.send_message(msg)
                                    logger.debug(f"📱 Telegram sent for {result.symbol}")
                            except Exception as te:
                                logger.debug(f"Telegram send skip: {te}")
                        
                        stats['updated'] += 1
                        
                    else:
                        # Додаємо новий
                        wl = SqueezeWatchlist(
                            symbol=result.symbol,
                            phase=WatchlistPhase.WATCHING.value,
                            consecutive_signals=1,
                            total_signals=1,
                            entry_price=result.current_price,
                            entry_oi=result.current_oi,
                            current_price=result.current_price,
                            current_oi=result.current_oi,
                            current_k=result.k_coefficient,
                            direction=result.direction,
                            direction_confidence=result.confidence,
                            funding_bias=result.funding_bias,
                        )
                        session.add(wl)
                        existing[result.symbol] = wl
                        stats['added'] += 1
                        
                        logger.info(f"🆕 Added {result.symbol} to watchlist (K={result.k_coefficient:.2f})")
                    
                    # Записуємо сигнал
                    signal = SqueezeSignal(
                        symbol=result.symbol,
                        signal_type=result.signal_type,
                        direction=result.direction,
                        k_coefficient=result.k_coefficient,
                        price_change_pct=result.price_change_pct,
                        oi_change_pct=result.oi_change_pct,
                        funding_rate=result.funding_rate,
                        funding_bias=result.funding_bias,
                        price_at_signal=result.current_price,
                        oi_at_signal=result.current_oi,
                        lookback_hours=result.lookback_hours,
                        confidence=result.confidence,
                        raw_data=json.dumps(result.raw_data),
                    )
                    session.add(signal)
                    self.stats['signals_generated'] += 1
            
            # Обробляємо монети БЕЗ сигналів
            for symbol, wl in existing.items():
                if symbol not in symbols_with_signals:
                    # Скидаємо consecutive
                    if wl.consecutive_signals > 0:
                        wl.consecutive_signals = 0
                        wl.last_update = datetime.utcnow()
                    
                    # Перевіряємо breakout
                    # Знаходимо результат аналізу для цієї монети
                    result_for_symbol = next(
                        (r for r in analysis_results if r.symbol == symbol),
                        None
                    )
                    
                    if result_for_symbol:
                        breakout_threshold = self._get('sd_breakout_threshold', 3.0)
                        
                        # Breakout detection
                        if abs(result_for_symbol.price_change_pct) >= breakout_threshold:
                            old_phase = wl.phase
                            
                            if result_for_symbol.price_change_pct > 0:
                                wl.phase = WatchlistPhase.TRIGGERED.value
                                wl.result_status = "BREAKOUT_UP"
                            else:
                                wl.phase = WatchlistPhase.TRIGGERED.value
                                wl.result_status = "BREAKOUT_DOWN"
                            
                            # P&L
                            if wl.entry_price and wl.entry_price > 0:
                                if wl.direction == "LONG":
                                    wl.result_pnl_pct = result_for_symbol.price_change_pct
                                else:
                                    wl.result_pnl_pct = -result_for_symbol.price_change_pct
                            
                            stats['triggered'] += 1
                            logger.info(f"🚀 {symbol}: TRIGGERED! {wl.result_status} ({result_for_symbol.price_change_pct:.2f}%)")
                    
                    # Timeout check
                    timeout_hours = self._get('sd_watchlist_timeout_hours', 48)
                    if wl.added_at:
                        age_hours = (datetime.utcnow() - wl.added_at).total_seconds() / 3600
                        if age_hours > timeout_hours and wl.phase not in [
                            WatchlistPhase.TRIGGERED.value,
                            WatchlistPhase.EXPIRED.value
                        ]:
                            wl.phase = WatchlistPhase.EXPIRED.value
                            stats['expired'] += 1
                            logger.info(f"⏰ {symbol}: EXPIRED (no breakout after {timeout_hours}h)")
            
            session.commit()
            
            return stats
            
        except Exception as e:
            logger.error(f"Update watchlist error: {e}")
            session.rollback()
            return stats
        finally:
            session.close()
    
    def run_full_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Запускає повний цикл аналізу.
        
        1. Аналізує всі символи за 4h
        2. Оновлює watchlist
        3. Повертає результати
        """
        logger.info(f"🔍 Running full analysis for {len(symbols)} symbols...")
        
        # Аналізуємо
        results = self.analyze_all_symbols(symbols, lookback_hours=4)
        
        # Оновлюємо watchlist
        watchlist_stats = self.update_watchlist(results)
        
        # Фільтруємо сигнали
        signals = [r for r in results if r.has_signal]
        
        logger.info(f"✅ Analysis complete: {len(signals)} signals, "
                   f"added={watchlist_stats['added']}, "
                   f"triggered={watchlist_stats['triggered']}")
        
        return {
            'total_analyzed': len(results),
            'signals_found': len(signals),
            'signals': [s.to_dict() for s in signals],
            'top_k': [r.to_dict() for r in results[:10]],  # Top 10 за K
            'watchlist_stats': watchlist_stats,
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    def start_periodic_analysis(self, symbols: List[str], interval: int = 300):
        """
        Запускає періодичний аналіз.
        
        Args:
            symbols: Список символів
            interval: Інтервал в секундах
        """
        if self._analyzing:
            logger.warning("Analysis already running")
            return
        
        self._analyzing = True
        self._analyze_interval = interval
        self._analyze_symbols = symbols
        self._analyze_thread = threading.Thread(target=self._analyze_loop, daemon=True)
        self._analyze_thread.start()
        
        logger.info(f"🔄 Started periodic analysis every {interval}s")
    
    def stop_periodic_analysis(self):
        """Зупиняє періодичний аналіз"""
        self._analyzing = False
        
        if self._analyze_thread and self._analyze_thread.is_alive():
            self._analyze_thread.join(timeout=10)
        
        logger.info("⏹️ Stopped periodic analysis")
    
    def _analyze_loop(self):
        """Цикл періодичного аналізу"""
        while self._analyzing:
            try:
                self.run_full_analysis(self._analyze_symbols)
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
            
            # Чекаємо з можливістю переривання
            for _ in range(self._analyze_interval):
                if not self._analyzing:
                    break
                time.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Повертає статистику"""
        return {
            **self.stats,
            'analyzing_active': self._analyzing,
            'analyze_interval': self._analyze_interval,
        }
    
    def get_watchlist(self) -> List[Dict]:
        """Повертає поточний watchlist"""
        # Models вже імпортовані глобально
        
        session = self.db_session_factory()
        
        try:
            items = session.query(SqueezeWatchlist).filter(
                SqueezeWatchlist.phase.notin_([
                    WatchlistPhase.EXPIRED.value,
                ])
            ).order_by(SqueezeWatchlist.current_k.desc()).all()
            
            return [w.to_dict() for w in items]
            
        except Exception as e:
            logger.error(f"Get watchlist error: {e}")
            return []
        finally:
            session.close()
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Повертає останні сигнали"""
        # Models вже імпортовані глобально
        
        session = self.db_session_factory()
        
        try:
            signals = session.query(SqueezeSignal).order_by(
                SqueezeSignal.created_at.desc()
            ).limit(limit).all()
            
            return [s.to_dict() for s in signals]
            
        except Exception as e:
            logger.error(f"Get signals error: {e}")
            return []
        finally:
            session.close()
    
    def clear_watchlist(self):
        """Очищає watchlist"""
        # Models вже імпортовані глобально
        
        session = self.db_session_factory()
        
        try:
            session.query(SqueezeWatchlist).delete()
            session.commit()
            logger.info("🗑️ Watchlist cleared")
        except Exception as e:
            logger.error(f"Clear watchlist error: {e}")
            session.rollback()
        finally:
            session.close()
