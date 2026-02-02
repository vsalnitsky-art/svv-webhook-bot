"""
Entry Manager v1.0 - Smart Money Entry System

Керує станами входу після детекції CHoCH:
- STALKING: CHoCH detected, чекаємо відкат до Order Block
- ENTRY_FOUND: Ціна торкнулась OB, готовий до входу
- POSITION: Позиція відкрита

Логіка "Мисливця":
1. CHoCH detected → Стан STALKING (полюємо на відкат)
2. Price touches OB → Стан ENTRY_FOUND (сигнал до входу)
3. Entry executed → Стан POSITION

Автор: SVV Bot Team
Версія: 1.0 (2026-02-02)
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from detection.smc_analyzer import (
    SMCAnalysisResult, StructureSignal, MarketBias, PriceZone, OrderBlock
)


class EntryState(Enum):
    """Стани входу в угоду"""
    WATCHING = "WATCHING"       # Спостерігаємо, немає сигналу
    BUILDING = "BUILDING"       # Squeeze росте
    READY = "READY"             # CHoCH detected, готовий до полювання
    STALKING = "STALKING"       # Чекаємо відкат до OB (полюємо)
    ENTRY_FOUND = "ENTRY_FOUND" # Ціна в зоні OB - час входити!
    POSITION = "POSITION"       # Позиція відкрита
    INVALIDATED = "INVALIDATED" # Сигнал скасовано (пробито OB/SL)


class StopLossMode(Enum):
    """Режим стоп-лосс"""
    CONSERVATIVE = "CONSERVATIVE"  # За Swing Low (надійніше)
    AGGRESSIVE = "AGGRESSIVE"      # За межу OB (вищий R/R)


@dataclass
class EntrySetup:
    """Налаштування входу в угоду"""
    symbol: str
    direction: str  # LONG / SHORT
    state: EntryState
    
    # CHoCH інформація
    choch_detected: bool = False
    choch_signal: StructureSignal = StructureSignal.NONE
    choch_time: Optional[datetime] = None
    
    # Order Block для входу
    target_ob: Optional[OrderBlock] = None
    
    # Ціни
    entry_price: float = 0.0       # Точка входу (верх OB або медіана)
    stop_loss: float = 0.0         # Стоп під OB
    take_profit: float = 0.0       # Найближчий Swing High/Low
    
    # Метрики
    risk_reward: float = 0.0       # R/R ratio
    confidence: float = 0.0        # 0-100%
    smc_score: float = 0.0
    
    # Зона
    price_zone: PriceZone = PriceZone.EQUILIBRIUM
    zone_level: float = 0.5
    
    # HTF Bias
    htf_bias: str = "NEUTRAL"      # 4H trend
    htf_aligned: bool = False      # Чи співпадає з напрямком
    
    # Час
    created_at: datetime = None
    stalking_since: Optional[datetime] = None
    max_stalking_hours: int = 24   # Макс час очікування відкату
    
    # Причини
    reasons: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def is_expired(self) -> bool:
        """Перевіряє чи не застарів сигнал"""
        if self.stalking_since is None:
            return False
        elapsed = datetime.now() - self.stalking_since
        return elapsed.total_seconds() > self.max_stalking_hours * 3600
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'state': self.state.value,
            'choch_detected': self.choch_detected,
            'choch_signal': self.choch_signal.value,
            'entry_price': round(self.entry_price, 6),
            'stop_loss': round(self.stop_loss, 6),
            'take_profit': round(self.take_profit, 6),
            'risk_reward': round(self.risk_reward, 2),
            'confidence': round(self.confidence, 1),
            'smc_score': round(self.smc_score, 3),
            'price_zone': self.price_zone.value,
            'htf_bias': self.htf_bias,
            'htf_aligned': self.htf_aligned,
            'reasons': self.reasons,
            'is_expired': self.is_expired(),
        }


class EntryManager:
    """
    Entry Manager - Керує логікою входу після CHoCH
    
    Реалізує стратегію "Мисливець":
    - Не входимо одразу на CHoCH (занадто високо)
    - Чекаємо відкат до Order Block
    - Входимо з коротким стопом та високим R/R
    """
    
    # Stop Loss buffer (% нижче OB)
    SL_BUFFER_PCT = 0.2
    
    # Мінімальний R/R для входу
    MIN_RISK_REWARD = 2.0
    
    # Макс час очікування відкату
    MAX_STALKING_HOURS = 24
    
    def __init__(self, sl_mode: StopLossMode = StopLossMode.AGGRESSIVE):
        self.sl_mode = sl_mode
        self._active_setups: Dict[str, EntrySetup] = {}  # symbol -> setup
    
    def process_signal(self,
                       symbol: str,
                       current_price: float,
                       smc_result: SMCAnalysisResult,
                       htf_bias: str = "NEUTRAL",
                       swing_high: float = None,
                       swing_low: float = None) -> EntrySetup:
        """
        Головний метод обробки сигналу
        
        Args:
            symbol: Символ монети
            current_price: Поточна ціна
            smc_result: Результат SMC аналізу
            htf_bias: Напрямок на 4H (BULLISH/BEARISH/NEUTRAL)
            swing_high: Найближчий Swing High (для TP)
            swing_low: Найближчий Swing Low (для SL conservative)
        
        Returns:
            EntrySetup з поточним станом та параметрами
        """
        # Перевіряємо чи є активний setup для цього символу
        existing = self._active_setups.get(symbol)
        
        # ============================================
        # BULLISH CHoCH DETECTION
        # ============================================
        if smc_result.structure_signal == StructureSignal.BULLISH_CHOCH:
            return self._handle_bullish_choch(
                symbol, current_price, smc_result, htf_bias, 
                swing_high, swing_low, existing
            )
        
        # ============================================
        # BEARISH CHoCH DETECTION  
        # ============================================
        elif smc_result.structure_signal == StructureSignal.BEARISH_CHOCH:
            return self._handle_bearish_choch(
                symbol, current_price, smc_result, htf_bias,
                swing_high, swing_low, existing
            )
        
        # ============================================
        # BOS (підтвердження тренду, менш пріоритетний)
        # ============================================
        elif smc_result.structure_signal in [StructureSignal.BULLISH_BOS, StructureSignal.BEARISH_BOS]:
            # BOS - продовжуємо trend, але не так агресивно
            if existing and existing.state == EntryState.STALKING:
                return self._check_pullback(symbol, current_price, smc_result, existing)
            # Можна додати логіку для BOS entry
        
        # ============================================
        # NO SIGNAL - перевіряємо існуючі setups
        # ============================================
        if existing:
            if existing.state == EntryState.STALKING:
                return self._check_pullback(symbol, current_price, smc_result, existing)
            elif existing.is_expired():
                self._invalidate_setup(symbol, "Expired after 24h")
                return self._watching_state(symbol)
        
        return self._watching_state(symbol)
    
    def _handle_bullish_choch(self,
                              symbol: str,
                              current_price: float,
                              smc_result: SMCAnalysisResult,
                              htf_bias: str,
                              swing_high: float,
                              swing_low: float,
                              existing: Optional[EntrySetup]) -> EntrySetup:
        """Обробка Bullish CHoCH"""
        
        # Знаходимо найближчий Bullish OB для входу
        target_ob = smc_result.nearest_bullish_ob
        
        # Перевіряємо HTF alignment
        htf_aligned = htf_bias in ["BULLISH", "NEUTRAL"]
        
        # ============================================
        # ВИПАДОК 1: Ціна вже в Discount Zone + біля OB
        # → Миттєвий вхід!
        # ============================================
        if (smc_result.price_zone == PriceZone.DISCOUNT and 
            smc_result.price_at_bullish_ob and target_ob):
            
            setup = self._create_long_setup(
                symbol, current_price, target_ob, smc_result,
                htf_bias, htf_aligned, swing_high, swing_low
            )
            setup.state = EntryState.ENTRY_FOUND
            setup.reasons.append("🎯 Perfect Entry: CHoCH + Discount + At OB")
            
            self._active_setups[symbol] = setup
            return setup
        
        # ============================================
        # ВИПАДОК 2: CHoCH є, але ціна занадто високо
        # → Переходимо в режим STALKING
        # ============================================
        if target_ob:
            setup = self._create_long_setup(
                symbol, current_price, target_ob, smc_result,
                htf_bias, htf_aligned, swing_high, swing_low
            )
            setup.state = EntryState.STALKING
            setup.stalking_since = datetime.now()
            setup.reasons.append(f"🐆 Stalking: Waiting pullback to OB ({target_ob.high:.4f})")
            
            self._active_setups[symbol] = setup
            return setup
        
        # ============================================
        # ВИПАДОК 3: CHoCH є, але немає OB
        # → READY, але без точної точки входу
        # ============================================
        setup = EntrySetup(
            symbol=symbol,
            direction="LONG",
            state=EntryState.READY,
            choch_detected=True,
            choch_signal=StructureSignal.BULLISH_CHOCH,
            choch_time=datetime.now(),
            smc_score=smc_result.smc_score,
            price_zone=smc_result.price_zone,
            zone_level=smc_result.zone_level,
            htf_bias=htf_bias,
            htf_aligned=htf_aligned,
            confidence=70 if htf_aligned else 50,
        )
        setup.reasons.append("⚠️ CHoCH detected but no clear OB for entry")
        
        self._active_setups[symbol] = setup
        return setup
    
    def _handle_bearish_choch(self,
                              symbol: str,
                              current_price: float,
                              smc_result: SMCAnalysisResult,
                              htf_bias: str,
                              swing_high: float,
                              swing_low: float,
                              existing: Optional[EntrySetup]) -> EntrySetup:
        """Обробка Bearish CHoCH"""
        
        target_ob = smc_result.nearest_bearish_ob
        htf_aligned = htf_bias in ["BEARISH", "NEUTRAL"]
        
        # Миттєвий вхід якщо в Premium + біля OB
        if (smc_result.price_zone == PriceZone.PREMIUM and
            smc_result.price_at_bearish_ob and target_ob):
            
            setup = self._create_short_setup(
                symbol, current_price, target_ob, smc_result,
                htf_bias, htf_aligned, swing_high, swing_low
            )
            setup.state = EntryState.ENTRY_FOUND
            setup.reasons.append("🎯 Perfect Entry: CHoCH + Premium + At OB")
            
            self._active_setups[symbol] = setup
            return setup
        
        # STALKING mode
        if target_ob:
            setup = self._create_short_setup(
                symbol, current_price, target_ob, smc_result,
                htf_bias, htf_aligned, swing_high, swing_low
            )
            setup.state = EntryState.STALKING
            setup.stalking_since = datetime.now()
            setup.reasons.append(f"🐆 Stalking: Waiting pullback to OB ({target_ob.low:.4f})")
            
            self._active_setups[symbol] = setup
            return setup
        
        # Без OB
        setup = EntrySetup(
            symbol=symbol,
            direction="SHORT",
            state=EntryState.READY,
            choch_detected=True,
            choch_signal=StructureSignal.BEARISH_CHOCH,
            choch_time=datetime.now(),
            smc_score=smc_result.smc_score,
            price_zone=smc_result.price_zone,
            htf_bias=htf_bias,
            htf_aligned=htf_aligned,
            confidence=70 if htf_aligned else 50,
        )
        setup.reasons.append("⚠️ CHoCH detected but no clear OB for entry")
        
        self._active_setups[symbol] = setup
        return setup
    
    def _check_pullback(self,
                        symbol: str,
                        current_price: float,
                        smc_result: SMCAnalysisResult,
                        setup: EntrySetup) -> EntrySetup:
        """
        Перевіряє чи відбувся відкат до OB
        """
        if setup.is_expired():
            self._invalidate_setup(symbol, "Expired")
            return self._watching_state(symbol)
        
        # LONG setup - чекаємо падіння до OB
        if setup.direction == "LONG" and setup.target_ob:
            ob = setup.target_ob
            
            # Ціна торкнулась OB!
            if current_price <= ob.high:
                setup.state = EntryState.ENTRY_FOUND
                setup.entry_price = current_price
                setup.reasons.append(f"⚡ Pullback complete! Price at OB ({current_price:.4f})")
                return setup
            
            # Ціна пробила OB вниз - invalidate
            if current_price < ob.low * (1 - self.SL_BUFFER_PCT / 100):
                self._invalidate_setup(symbol, f"Price broke below OB ({current_price:.4f})")
                return self._watching_state(symbol)
        
        # SHORT setup - чекаємо зростання до OB
        elif setup.direction == "SHORT" and setup.target_ob:
            ob = setup.target_ob
            
            if current_price >= ob.low:
                setup.state = EntryState.ENTRY_FOUND
                setup.entry_price = current_price
                setup.reasons.append(f"⚡ Pullback complete! Price at OB ({current_price:.4f})")
                return setup
            
            if current_price > ob.high * (1 + self.SL_BUFFER_PCT / 100):
                self._invalidate_setup(symbol, f"Price broke above OB ({current_price:.4f})")
                return self._watching_state(symbol)
        
        return setup
    
    def _create_long_setup(self,
                           symbol: str,
                           current_price: float,
                           ob: OrderBlock,
                           smc_result: SMCAnalysisResult,
                           htf_bias: str,
                           htf_aligned: bool,
                           swing_high: float,
                           swing_low: float) -> EntrySetup:
        """Створює LONG setup з розрахунком Entry/SL/TP"""
        
        # Entry: верхня межа OB або медіана
        entry_price = ob.high  # Або ob.mid_price для консервативнішого входу
        
        # Stop Loss
        if self.sl_mode == StopLossMode.AGGRESSIVE:
            stop_loss = ob.low * (1 - self.SL_BUFFER_PCT / 100)
        else:  # CONSERVATIVE
            stop_loss = swing_low * (1 - self.SL_BUFFER_PCT / 100) if swing_low else ob.low * 0.99
        
        # Take Profit: найближчий Swing High
        take_profit = swing_high if swing_high else entry_price * 1.05  # Default 5%
        
        # Risk/Reward
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward = reward / risk if risk > 0 else 0
        
        # Confidence
        confidence = 50
        if htf_aligned:
            confidence += 15
        if smc_result.price_zone == PriceZone.DISCOUNT:
            confidence += 15
        if smc_result.price_at_bullish_ob:
            confidence += 10
        if risk_reward >= self.MIN_RISK_REWARD:
            confidence += 10
        
        return EntrySetup(
            symbol=symbol,
            direction="LONG",
            state=EntryState.READY,
            choch_detected=True,
            choch_signal=StructureSignal.BULLISH_CHOCH,
            choch_time=datetime.now(),
            target_ob=ob,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=min(95, confidence),
            smc_score=smc_result.smc_score,
            price_zone=smc_result.price_zone,
            zone_level=smc_result.zone_level,
            htf_bias=htf_bias,
            htf_aligned=htf_aligned,
        )
    
    def _create_short_setup(self,
                            symbol: str,
                            current_price: float,
                            ob: OrderBlock,
                            smc_result: SMCAnalysisResult,
                            htf_bias: str,
                            htf_aligned: bool,
                            swing_high: float,
                            swing_low: float) -> EntrySetup:
        """Створює SHORT setup з розрахунком Entry/SL/TP"""
        
        entry_price = ob.low
        
        if self.sl_mode == StopLossMode.AGGRESSIVE:
            stop_loss = ob.high * (1 + self.SL_BUFFER_PCT / 100)
        else:
            stop_loss = swing_high * (1 + self.SL_BUFFER_PCT / 100) if swing_high else ob.high * 1.01
        
        take_profit = swing_low if swing_low else entry_price * 0.95
        
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        risk_reward = reward / risk if risk > 0 else 0
        
        confidence = 50
        if htf_aligned:
            confidence += 15
        if smc_result.price_zone == PriceZone.PREMIUM:
            confidence += 15
        if smc_result.price_at_bearish_ob:
            confidence += 10
        if risk_reward >= self.MIN_RISK_REWARD:
            confidence += 10
        
        return EntrySetup(
            symbol=symbol,
            direction="SHORT",
            state=EntryState.READY,
            choch_detected=True,
            choch_signal=StructureSignal.BEARISH_CHOCH,
            choch_time=datetime.now(),
            target_ob=ob,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=min(95, confidence),
            smc_score=smc_result.smc_score,
            price_zone=smc_result.price_zone,
            zone_level=smc_result.zone_level,
            htf_bias=htf_bias,
            htf_aligned=htf_aligned,
        )
    
    def _watching_state(self, symbol: str) -> EntrySetup:
        """Повертає WATCHING стан"""
        return EntrySetup(
            symbol=symbol,
            direction="NEUTRAL",
            state=EntryState.WATCHING,
        )
    
    def _invalidate_setup(self, symbol: str, reason: str):
        """Скасовує активний setup"""
        if symbol in self._active_setups:
            setup = self._active_setups[symbol]
            setup.state = EntryState.INVALIDATED
            setup.reasons.append(f"❌ {reason}")
            del self._active_setups[symbol]
            print(f"[ENTRY] {symbol}: Setup invalidated - {reason}")
    
    def get_active_setups(self) -> Dict[str, EntrySetup]:
        """Повертає всі активні setups"""
        return self._active_setups.copy()
    
    def get_stalking_symbols(self) -> List[str]:
        """Повертає символи в режимі STALKING"""
        return [
            symbol for symbol, setup in self._active_setups.items()
            if setup.state == EntryState.STALKING
        ]
    
    def mark_position_opened(self, symbol: str):
        """Позначає що позиція відкрита"""
        if symbol in self._active_setups:
            self._active_setups[symbol].state = EntryState.POSITION


# Factory
_manager = None

def get_entry_manager(sl_mode: StopLossMode = StopLossMode.AGGRESSIVE) -> EntryManager:
    """Get Entry Manager instance (singleton)"""
    global _manager
    if _manager is None:
        _manager = EntryManager(sl_mode)
    return _manager
