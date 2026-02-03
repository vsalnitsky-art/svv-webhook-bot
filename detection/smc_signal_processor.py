"""
SMC Signal Processor v1.0 - Обробник сигналів стратегії "Пробудження Сплячого"

Повний цикл стратегії:
1. WATCHING: Sleeper знайдено (BB squeeze + низькі об'єми)
2. BUILDING: Squeeze зростає
3. READY: CHoCH detected - починаємо полювання!
4. STALKING: Чекаємо відкат до Order Block
5. ENTRY_FOUND: Ціна торкнулась OB - час входити!
6. POSITION: Позиція відкрита

Логіка "Мисливця":
- НЕ купуємо на CHoCH одразу (занадто дорого, низький R/R)
- Чекаємо поки ціна відкотиться до Order Block
- Входимо з коротким стопом під OB (R/R = 3-6)

Автор: SVV Bot Team
Версія: 1.0 (2026-02-02)
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from config.bot_settings import SleeperState
from storage import get_db
from core.market_data import get_fetcher
from detection.smc_analyzer import (
    SMCAnalyzer, SMCAnalysisResult, get_smc_analyzer,
    StructureSignal, MarketBias, PriceZone, OrderBlock
)
from detection.direction_engine_v8 import get_direction_engine_v8, BiasDirection
from detection.entry_manager import (
    EntryManager, EntrySetup, EntryState, StopLossMode, get_entry_manager
)
from alerts.telegram_notifier import get_notifier
from trading.risk_calculator import RiskCalculator


@dataclass
class SMCSignalResult:
    """Результат обробки SMC сигналу"""
    symbol: str
    state: str              # WATCHING/READY/STALKING/ENTRY_FOUND
    direction: str          # LONG/SHORT/NEUTRAL
    action: str             # WAIT/STALK/EXECUTE/NONE
    
    # SMC дані
    smc_signal: str = "NONE"
    market_bias: str = "NEUTRAL"
    price_zone: str = "EQUILIBRIUM"
    zone_level: float = 0.5
    at_ob: bool = False
    
    # HTF дані
    htf_bias: str = "NEUTRAL"
    htf_aligned: bool = False
    
    # Entry дані
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 0.0
    
    # Метрики
    confidence: float = 0.0
    smc_score: float = 0.0
    
    # Причини
    reasons: List[str] = field(default_factory=list)
    comment: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'state': self.state,
            'direction': self.direction,
            'action': self.action,
            'smc_signal': self.smc_signal,
            'market_bias': self.market_bias,
            'price_zone': self.price_zone,
            'zone_level': round(self.zone_level, 3),
            'at_ob': self.at_ob,
            'htf_bias': self.htf_bias,
            'htf_aligned': self.htf_aligned,
            'entry_price': round(self.entry_price, 6),
            'stop_loss': round(self.stop_loss, 6),
            'take_profit': round(self.take_profit, 6),
            'risk_reward': round(self.risk_reward, 2),
            'confidence': round(self.confidence, 1),
            'smc_score': round(self.smc_score, 3),
            'reasons': self.reasons,
            'comment': self.comment,
        }


class SMCSignalProcessor:
    """
    SMC Signal Processor - Повна інтеграція стратегії "Пробудження Сплячого"
    
    Workflow:
    1. Отримуємо READY sleepers (CHoCH вже виявлено)
    2. Для кожного перевіряємо: чи потрібно чекати відкату?
    3. Якщо ціна ще високо → STALKING (полюємо)
    4. Якщо ціна в OB → ENTRY_FOUND (час діяти!)
    """
    
    # Мінімальний R/R для входу
    MIN_RISK_REWARD = 2.0
    
    # Макс час очікування відкату (годин)
    MAX_STALKING_HOURS = 24
    
    # Мін впевненість для сигналу
    MIN_CONFIDENCE = 65
    
    def __init__(self, sl_mode: StopLossMode = StopLossMode.AGGRESSIVE):
        self.db = get_db()
        self.fetcher = get_fetcher()
        self.smc_analyzer = get_smc_analyzer()
        self.direction_engine = get_direction_engine_v8()
        self.entry_manager = get_entry_manager(sl_mode)
        self.notifier = get_notifier()
        
        # Активні "полювання" (STALKING)
        self._stalking_symbols: Dict[str, datetime] = {}  # symbol -> stalking_start_time
    
    def process_ready_sleepers(self) -> List[SMCSignalResult]:
        """
        Головний метод: обробляє всі READY та STALKING sleepers
        
        Returns:
            Список результатів з action = EXECUTE для тих, хто готовий до входу
        """
        results = []
        
        # 1. Отримуємо READY sleepers (CHoCH виявлено)
        ready_sleepers = self.db.get_sleepers(state='READY')
        
        # 2. Отримуємо STALKING sleepers (чекаємо відкат)
        stalking_sleepers = self.db.get_sleepers(state='STALKING')
        
        # 3. Обробляємо кожен
        all_sleepers = ready_sleepers + stalking_sleepers
        
        # v8.2.4: Логування для діагностики
        if all_sleepers:
            with_direction = [s for s in all_sleepers if s.get('direction') not in ['NEUTRAL', 'WAIT', None, '']]
            print(f"[SMC] Ready: {len(ready_sleepers)}, Stalking: {len(stalking_sleepers)}, With direction: {len(with_direction)}")
        
        for sleeper in all_sleepers:
            try:
                result = self._process_single_sleeper(sleeper)
                if result:
                    results.append(result)
                    
                    # Надсилаємо alert якщо потрібно
                    self._send_alert_if_needed(result, sleeper)
                    
            except Exception as e:
                print(f"[SMC] Error processing {sleeper.get('symbol')}: {e}")
        
        return results
    
    def _process_single_sleeper(self, sleeper: Dict) -> Optional[SMCSignalResult]:
        """
        Обробляє один sleeper за логікою "Мисливця"
        """
        symbol = sleeper.get('symbol')
        current_state = sleeper.get('state', 'WATCHING')
        direction = sleeper.get('direction', 'NEUTRAL')
        
        # v8.2.4: WAIT та NEUTRAL не обробляються - потрібен чіткий напрямок
        if direction in ['NEUTRAL', 'WAIT', None, '']:
            return None
        
        # Отримуємо свіжі дані
        klines_4h = self.fetcher.get_klines(symbol, '4h', 100)
        klines_1h = self.fetcher.get_klines(symbol, '1h', 100)
        
        if not klines_4h or not klines_1h or len(klines_1h) < 50:
            return None
        
        # Поточна ціна (klines is List[Dict] with keys: open, high, low, close, volume, etc.)
        last_candle = klines_1h[-1]
        if isinstance(last_candle, dict):
            current_price = float(last_candle.get('close', 0))
        else:
            # Fallback for list format [timestamp, open, high, low, close, volume]
            current_price = float(last_candle[4])
        
        # SMC аналіз на 1H з HTF bias від 4H
        smc_result = self.smc_analyzer.analyze(klines_1h, htf_klines=klines_4h)
        
        if not smc_result:
            return None
        
        # Визначаємо HTF bias
        htf_bias = "NEUTRAL"
        if smc_result.market_bias == MarketBias.BULLISH:
            htf_bias = "BULLISH"
        elif smc_result.market_bias == MarketBias.BEARISH:
            htf_bias = "BEARISH"
        
        htf_aligned = (
            (direction == "LONG" and htf_bias in ["BULLISH", "NEUTRAL"]) or
            (direction == "SHORT" and htf_bias in ["BEARISH", "NEUTRAL"])
        )
        
        # Отримуємо swing points для TP/SL
        swing_high = smc_result.last_hh if smc_result.last_hh else None
        swing_low = smc_result.last_ll if smc_result.last_ll else None
        
        # ============================================
        # ЛОГІКА СТАНІВ
        # ============================================
        
        # LONG сигнал
        if direction == "LONG":
            return self._process_long_signal(
                symbol, current_price, smc_result, htf_bias, htf_aligned,
                swing_high, swing_low, current_state, sleeper
            )
        
        # SHORT сигнал
        elif direction == "SHORT":
            return self._process_short_signal(
                symbol, current_price, smc_result, htf_bias, htf_aligned,
                swing_high, swing_low, current_state, sleeper
            )
        
        return None
    
    def _process_long_signal(self,
                             symbol: str,
                             current_price: float,
                             smc_result: SMCAnalysisResult,
                             htf_bias: str,
                             htf_aligned: bool,
                             swing_high: float,
                             swing_low: float,
                             current_state: str,
                             sleeper: Dict) -> SMCSignalResult:
        """
        Обробка LONG сигналу за логікою "Мисливця"
        """
        result = SMCSignalResult(
            symbol=symbol,
            direction="LONG",
            state=current_state,
            action="NONE",
            smc_signal=smc_result.structure_signal.value,
            market_bias=smc_result.market_bias.value,
            price_zone=smc_result.price_zone.value,
            zone_level=smc_result.zone_level,
            at_ob=smc_result.price_at_bullish_ob,
            htf_bias=htf_bias,
            htf_aligned=htf_aligned,
            smc_score=smc_result.smc_score,
        )
        
        # Знаходимо target OB
        target_ob = smc_result.nearest_bullish_ob
        
        # ============================================
        # ВИПАДОК 1: CHoCH є, ціна вже в Discount Zone + біля OB
        # → ENTRY_FOUND! Миттєвий вхід!
        # ============================================
        if (smc_result.price_zone == PriceZone.DISCOUNT and 
            smc_result.price_at_bullish_ob and target_ob):
            
            entry, sl, tp, rr = self._calculate_long_levels(
                current_price, target_ob, swing_high, swing_low
            )
            
            confidence = self._calculate_confidence(
                htf_aligned, smc_result, rr, target_ob is not None
            )
            
            result.state = "ENTRY_FOUND"
            result.action = "EXECUTE"
            result.entry_price = entry
            result.stop_loss = sl
            result.take_profit = tp
            result.risk_reward = rr
            result.confidence = confidence
            result.comment = "🎯 Perfect Entry: CHoCH + Discount + At OB"
            result.reasons = [
                f"✅ Bullish {smc_result.structure_signal.value}",
                f"✅ Discount Zone ({smc_result.zone_level:.2f})",
                f"✅ Price at Bullish OB",
                f"✅ R/R = {rr:.1f}",
            ]
            
            # Оновлюємо стан в БД
            self._update_sleeper_state(symbol, "ENTRY_FOUND", result)
            
            return result
        
        # ============================================
        # ВИПАДОК 2: CHoCH є, але ціна занадто високо
        # → STALKING (полюємо на відкат)
        # ============================================
        if target_ob and smc_result.structure_signal in [
            StructureSignal.BULLISH_CHOCH, StructureSignal.BULLISH_BOS
        ]:
            # Перевіряємо чи не занадто довго полюємо
            if symbol in self._stalking_symbols:
                elapsed = datetime.now() - self._stalking_symbols[symbol]
                if elapsed.total_seconds() > self.MAX_STALKING_HOURS * 3600:
                    # Timeout - скасовуємо
                    self._invalidate_stalking(symbol, "Timeout after 24h")
                    result.state = "WATCHING"
                    result.action = "NONE"
                    result.comment = "⏰ Stalking timeout - signal expired"
                    return result
            else:
                # Починаємо полювання
                self._stalking_symbols[symbol] = datetime.now()
            
            entry, sl, tp, rr = self._calculate_long_levels(
                target_ob.high, target_ob, swing_high, swing_low
            )
            
            result.state = "STALKING"
            result.action = "WAIT_PULLBACK"
            result.entry_price = target_ob.high  # Цільова ціна входу
            result.stop_loss = sl
            result.take_profit = tp
            result.risk_reward = rr
            result.confidence = self._calculate_confidence(htf_aligned, smc_result, rr, True)
            result.comment = f"🐆 Stalking: Waiting pullback to OB ({target_ob.high:.4f})"
            result.reasons = [
                f"✅ {smc_result.structure_signal.value} detected",
                f"⏳ Waiting for price to reach OB",
                f"🎯 Target Entry: {target_ob.high:.6f}",
            ]
            
            # Оновлюємо стан в БД
            self._update_sleeper_state(symbol, "STALKING", result)
            
            return result
        
        # ============================================
        # ВИПАДОК 3: Немає чіткого сигналу
        # → Залишаємо в READY
        # ============================================
        result.state = current_state
        result.action = "NONE"
        result.comment = "👀 Watching for CHoCH signal"
        
        return result
    
    def _process_short_signal(self,
                              symbol: str,
                              current_price: float,
                              smc_result: SMCAnalysisResult,
                              htf_bias: str,
                              htf_aligned: bool,
                              swing_high: float,
                              swing_low: float,
                              current_state: str,
                              sleeper: Dict) -> SMCSignalResult:
        """
        Обробка SHORT сигналу за логікою "Мисливця"
        """
        result = SMCSignalResult(
            symbol=symbol,
            direction="SHORT",
            state=current_state,
            action="NONE",
            smc_signal=smc_result.structure_signal.value,
            market_bias=smc_result.market_bias.value,
            price_zone=smc_result.price_zone.value,
            zone_level=smc_result.zone_level,
            at_ob=smc_result.price_at_bearish_ob,
            htf_bias=htf_bias,
            htf_aligned=htf_aligned,
            smc_score=smc_result.smc_score,
        )
        
        target_ob = smc_result.nearest_bearish_ob
        
        # ENTRY_FOUND: Premium + At OB
        if (smc_result.price_zone == PriceZone.PREMIUM and
            smc_result.price_at_bearish_ob and target_ob):
            
            entry, sl, tp, rr = self._calculate_short_levels(
                current_price, target_ob, swing_high, swing_low
            )
            
            confidence = self._calculate_confidence(htf_aligned, smc_result, rr, True)
            
            result.state = "ENTRY_FOUND"
            result.action = "EXECUTE"
            result.entry_price = entry
            result.stop_loss = sl
            result.take_profit = tp
            result.risk_reward = rr
            result.confidence = confidence
            result.comment = "🎯 Perfect Entry: CHoCH + Premium + At OB"
            result.reasons = [
                f"✅ Bearish {smc_result.structure_signal.value}",
                f"✅ Premium Zone ({smc_result.zone_level:.2f})",
                f"✅ Price at Bearish OB",
                f"✅ R/R = {rr:.1f}",
            ]
            
            self._update_sleeper_state(symbol, "ENTRY_FOUND", result)
            return result
        
        # STALKING
        if target_ob and smc_result.structure_signal in [
            StructureSignal.BEARISH_CHOCH, StructureSignal.BEARISH_BOS
        ]:
            if symbol in self._stalking_symbols:
                elapsed = datetime.now() - self._stalking_symbols[symbol]
                if elapsed.total_seconds() > self.MAX_STALKING_HOURS * 3600:
                    self._invalidate_stalking(symbol, "Timeout")
                    result.state = "WATCHING"
                    result.action = "NONE"
                    return result
            else:
                self._stalking_symbols[symbol] = datetime.now()
            
            entry, sl, tp, rr = self._calculate_short_levels(
                target_ob.low, target_ob, swing_high, swing_low
            )
            
            result.state = "STALKING"
            result.action = "WAIT_PULLBACK"
            result.entry_price = target_ob.low
            result.stop_loss = sl
            result.take_profit = tp
            result.risk_reward = rr
            result.confidence = self._calculate_confidence(htf_aligned, smc_result, rr, True)
            result.comment = f"🐆 Stalking: Waiting pullback to OB ({target_ob.low:.4f})"
            
            self._update_sleeper_state(symbol, "STALKING", result)
            return result
        
        result.state = current_state
        result.action = "NONE"
        return result
    
    def _calculate_long_levels(self,
                               entry_price: float,
                               ob: OrderBlock,
                               swing_high: float,
                               swing_low: float) -> Tuple[float, float, float, float]:
        """Розраховує Entry/SL/TP для LONG"""
        
        entry = entry_price
        
        # Stop Loss: під OB з буфером
        sl = ob.low * 0.998  # 0.2% buffer
        
        # Take Profit: найближчий Swing High або +5%
        if swing_high and swing_high > entry:
            tp = swing_high
        else:
            tp = entry * 1.05
        
        # R/R
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        
        return entry, sl, tp, rr
    
    def _calculate_short_levels(self,
                                entry_price: float,
                                ob: OrderBlock,
                                swing_high: float,
                                swing_low: float) -> Tuple[float, float, float, float]:
        """Розраховує Entry/SL/TP для SHORT"""
        
        entry = entry_price
        
        # Stop Loss: над OB з буфером
        sl = ob.high * 1.002
        
        # Take Profit: найближчий Swing Low або -5%
        if swing_low and swing_low < entry:
            tp = swing_low
        else:
            tp = entry * 0.95
        
        risk = sl - entry
        reward = entry - tp
        rr = reward / risk if risk > 0 else 0
        
        return entry, sl, tp, rr
    
    def _calculate_confidence(self,
                              htf_aligned: bool,
                              smc_result: SMCAnalysisResult,
                              rr: float,
                              has_ob: bool) -> float:
        """Розраховує confidence score"""
        
        confidence = 50
        
        # HTF alignment
        if htf_aligned:
            confidence += 15
        
        # CHoCH vs BOS
        if smc_result.structure_signal in [
            StructureSignal.BULLISH_CHOCH, StructureSignal.BEARISH_CHOCH
        ]:
            confidence += 15
        elif smc_result.structure_signal in [
            StructureSignal.BULLISH_BOS, StructureSignal.BEARISH_BOS
        ]:
            confidence += 10
        
        # Zone
        if smc_result.price_zone in [PriceZone.DISCOUNT, PriceZone.PREMIUM]:
            confidence += 10
        
        # OB
        if has_ob:
            confidence += 5
        
        # R/R
        if rr >= 3:
            confidence += 10
        elif rr >= 2:
            confidence += 5
        
        return min(95, confidence)
    
    def _update_sleeper_state(self, symbol: str, new_state: str, result: SMCSignalResult):
        """Оновлює стан sleeper в БД"""
        try:
            update_data = {
                'state': new_state,
                'smc_signal': result.smc_signal,
                'price_zone': result.price_zone,
                'entry_price': result.entry_price,
                'stop_loss': result.stop_loss,
                'take_profit': result.take_profit,
                'risk_reward': result.risk_reward,
            }
            self.db.update_sleeper(symbol, **update_data)
            print(f"[SMC] {symbol}: State -> {new_state}")
        except Exception as e:
            print(f"[SMC] Error updating {symbol}: {e}")
    
    def _invalidate_stalking(self, symbol: str, reason: str):
        """Скасовує полювання"""
        if symbol in self._stalking_symbols:
            del self._stalking_symbols[symbol]
        print(f"[SMC] {symbol}: Stalking invalidated - {reason}")
    
    def _send_alert_if_needed(self, result: SMCSignalResult, sleeper: Dict):
        """Надсилає Telegram alert якщо потрібно + розрахунок позиції"""
        
        old_state = sleeper.get('state', 'WATCHING')
        new_state = result.state
        
        # Alert при переході в STALKING
        if new_state == "STALKING" and old_state != "STALKING":
            self.notifier.send_stalking_alert(
                symbol=result.symbol,
                direction=result.direction,
                target_price=result.entry_price,
                ob_range=f"{result.stop_loss:.6f} - {result.entry_price:.6f}"
            )
        
        # Alert при ENTRY_FOUND
        elif new_state == "ENTRY_FOUND":
            # v8.2: Розраховуємо розмір позиції
            position_data = None
            try:
                risk_calc = RiskCalculator()
                
                # OB boundaries (entry is at OB edge, SL is beyond OB)
                if result.direction == "LONG":
                    ob_high = result.entry_price
                    ob_low = result.stop_loss * 1.002  # Remove buffer to get OB low
                else:
                    ob_high = result.stop_loss * 0.998
                    ob_low = result.entry_price
                
                position_data = risk_calc.calculate_ob_position(
                    symbol=result.symbol,
                    direction=result.direction,
                    entry_price=result.entry_price,
                    ob_high=ob_high,
                    ob_low=ob_low,
                    swing_target=result.take_profit
                )
                
                if position_data.get('success'):
                    print(f"[SMC] {result.symbol}: Position calculated - "
                          f"Size: ${position_data['position_value']:.0f}, "
                          f"Risk: ${position_data['risk_amount']:.0f}, "
                          f"R/R: {position_data['rr_ratio']:.1f}")
                          
            except Exception as e:
                print(f"[SMC] {result.symbol}: Position calc error: {e}")
            
            # Send alert with position data
            self.notifier.send_entry_alert(
                symbol=result.symbol,
                direction=result.direction,
                entry=result.entry_price,
                sl=result.stop_loss,
                tp=result.take_profit,
                rr=result.risk_reward,
                position_data=position_data
            )
            
            # Також надсилаємо повний SMC сигнал
            self.notifier.send_smc_signal(result.to_dict())
    
    def get_stalking_count(self) -> int:
        """Повертає кількість активних полювань"""
        return len(self._stalking_symbols)
    
    def get_stalking_symbols(self) -> List[str]:
        """Повертає список символів в режимі STALKING"""
        return list(self._stalking_symbols.keys())


# Singleton
_processor = None

def get_smc_processor(sl_mode: StopLossMode = StopLossMode.AGGRESSIVE) -> SMCSignalProcessor:
    """Get SMC Signal Processor instance"""
    global _processor
    if _processor is None:
        _processor = SMCSignalProcessor(sl_mode)
    return _processor
