"""
Quasimodo (QM) Pattern Detector v1.0

Детектор паттерна Квазимодо для крипто-ринків.
Працює на будь-якому таймфреймі, оптимізований для 5M.

Bearish QM (SELL):
      C (Head)
     / \\
    /   \\     E (Right Shoulder < A)
   /     \\   /
  A       \\ /
   \\     D     <- D < B (ключова умова!)
    B

  C > A > E, D < B -> SELL

Bullish QM (BUY):
    B     D     <- D > B (ключова умова!)
   /     / \\
  A     /   \\
   \\   /     E (Right Shoulder > A)
    \\ /
     C (Head)

  C < A < E, D > B -> BUY
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QMPattern:
    """Знайдений паттерн Квазимодо"""
    direction: str          # 'BUY' або 'SELL'
    
    # 5 точок паттерна
    price_A: float
    price_B: float
    price_C: float          # Голова
    price_D: float
    price_E: float          # Праве плече
    
    # Індекси свічок
    idx_A: int
    idx_B: int
    idx_C: int
    idx_D: int
    idx_E: int
    
    # Торгові рівні
    entry: float            # Рекомендований вхід
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    
    # Метрики
    strength: float         # 0-100
    confidence: float       # 0-100
    risk_reward: float
    risk_percent: float
    
    def to_dict(self) -> Dict:
        return {
            'direction': self.direction,
            'points': {
                'A': self.price_A, 'B': self.price_B, 'C': self.price_C,
                'D': self.price_D, 'E': self.price_E,
            },
            'indices': {
                'A': self.idx_A, 'B': self.idx_B, 'C': self.idx_C,
                'D': self.idx_D, 'E': self.idx_E,
            },
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'tp1': self.take_profit_1,
            'tp2': self.take_profit_2,
            'strength': round(self.strength, 1),
            'confidence': round(self.confidence, 1),
            'risk_reward': round(self.risk_reward, 2),
            'risk_percent': round(self.risk_percent, 2),
        }


class QMDetector:
    """
    Детектор паттерна Квазимодо.
    
    Не залежить від talib — використовує чистий numpy.
    """
    
    def __init__(
        self,
        min_swing_bars: int = 5,
        atr_period: int = 14,
        min_swing_atr: float = 1.2,
        min_pattern_bars: int = 25,
        lookback_bars: int = 150,
        min_db_diff_pct: float = 0.5,
        sl_buffer_pct: float = 0.2,
        min_confidence: float = 70.0,
        min_rr_ratio: float = 1.5,
        max_risk_pct: float = 2.0,
    ):
        self.min_swing_bars = min_swing_bars
        self.atr_period = atr_period
        self.min_swing_atr = min_swing_atr
        self.min_pattern_bars = min_pattern_bars
        self.lookback_bars = lookback_bars
        self.min_db_diff_pct = min_db_diff_pct
        self.sl_buffer_pct = sl_buffer_pct
        self.min_confidence = min_confidence
        self.min_rr_ratio = min_rr_ratio
        self.max_risk_pct = max_risk_pct
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def detect(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        direction_hint: Optional[str] = None
    ) -> Optional[QMPattern]:
        """
        Головний метод детекції паттерна QM.
        
        Args:
            highs, lows, closes: OHLC масиви (мінімум lookback_bars)
            direction_hint: 'BUY' або 'SELL' — якщо задано, шукає тільки в цьому напрямку
                           (визначається HTF зоною)
        
        Returns:
            QMPattern або None
        """
        n = len(closes)
        if n < self.lookback_bars:
            return None
        
        # Використовуємо тільки останні lookback_bars
        h = highs[-self.lookback_bars:]
        l = lows[-self.lookback_bars:]
        c = closes[-self.lookback_bars:]
        
        atr = self._atr(h, l, c)
        
        swing_highs = self._find_swing_highs(h, l, atr)
        swing_lows = self._find_swing_lows(h, l, atr)
        
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None
        
        # Шукаємо патерни
        result = None
        
        if direction_hint is None or direction_hint == 'SELL':
            result = self._find_bearish_qm(swing_highs, swing_lows, c, atr)
        
        if result is None and (direction_hint is None or direction_hint == 'BUY'):
            result = self._find_bullish_qm(swing_highs, swing_lows, c, atr)
        
        return result
    
    # ========================================
    # SWING DETECTION
    # ========================================
    
    def _find_swing_highs(self, highs: np.ndarray, lows: np.ndarray, atr: np.ndarray) -> List[Dict]:
        """Знайти свінгові максимуми"""
        result = []
        n = len(highs)
        bars = self.min_swing_bars
        
        for i in range(bars, n - bars):
            is_high = True
            for j in range(1, bars + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            
            if not is_high:
                continue
            
            # ATR фільтр
            lo = min(lows[max(0, i - 3):i + 4])
            height = highs[i] - lo
            atr_val = atr[i] if not np.isnan(atr[i]) else 1.0
            
            if atr_val > 0 and height >= atr_val * self.min_swing_atr:
                result.append({
                    'idx': i, 'price': float(highs[i]),
                    'height': height, 'atr_ratio': height / atr_val
                })
        
        return result
    
    def _find_swing_lows(self, highs: np.ndarray, lows: np.ndarray, atr: np.ndarray) -> List[Dict]:
        """Знайти свінгові мінімуми"""
        result = []
        n = len(lows)
        bars = self.min_swing_bars
        
        for i in range(bars, n - bars):
            is_low = True
            for j in range(1, bars + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            
            if not is_low:
                continue
            
            hi = max(highs[max(0, i - 3):i + 4])
            depth = hi - lows[i]
            atr_val = atr[i] if not np.isnan(atr[i]) else 1.0
            
            if atr_val > 0 and depth >= atr_val * self.min_swing_atr:
                result.append({
                    'idx': i, 'price': float(lows[i]),
                    'depth': depth, 'atr_ratio': depth / atr_val
                })
        
        return result
    
    # ========================================
    # BEARISH QM (SELL)
    # ========================================
    
    def _find_bearish_qm(
        self, swing_highs: List[Dict], swing_lows: List[Dict],
        closes: np.ndarray, atr: np.ndarray
    ) -> Optional[QMPattern]:
        """
        Пошук ведмежого QM: C > A > E, D < B → SELL
        Шукаємо з кінця (найсвіжіші патерни першими).
        """
        n_h = len(swing_highs)
        
        for i in range(n_h - 1, 1, -1):
            E = swing_highs[i]
            
            # E має бути серед останніх свічок (свіжий патерн)
            if E['idx'] < len(closes) - 20:
                continue
            
            for j in range(i - 1, 0, -1):
                C = swing_highs[j]
                
                for k in range(j - 1, -1, -1):
                    A = swing_highs[k]
                    
                    # 1. Базова структура: C > A > E
                    if not (C['price'] > A['price'] and A['price'] > E['price']):
                        continue
                    
                    # 2. Часові обмеження
                    total_bars = E['idx'] - A['idx']
                    if total_bars < self.min_pattern_bars or total_bars > self.lookback_bars * 0.8:
                        continue
                    
                    # 3. Знаходимо B (мінімум між A і C)
                    B = self._find_lowest_between(swing_lows, A['idx'], C['idx'])
                    if B is None:
                        continue
                    
                    # 4. Знаходимо D (мінімум між C і E)
                    D = self._find_lowest_between(swing_lows, C['idx'], E['idx'])
                    if D is None:
                        continue
                    
                    # 5. КРИТИЧНА УМОВА: D < B
                    if D['price'] >= B['price']:
                        continue
                    
                    # 6. Різниця D-B мінімум min_db_diff_pct%
                    db_diff = abs(D['price'] - B['price']) / B['price'] * 100
                    if db_diff < self.min_db_diff_pct:
                        continue
                    
                    # 7. Симетрія часу A→C та C→E
                    time_ac = C['idx'] - A['idx']
                    time_ce = E['idx'] - C['idx']
                    if time_ce > time_ac * 3 or time_ce < time_ac * 0.3:
                        continue
                    
                    # Патерн знайдено! Розраховуємо торгові рівні
                    return self._build_bearish_pattern(A, B, C, D, E, closes, atr)
        
        return None
    
    # ========================================
    # BULLISH QM (BUY)
    # ========================================
    
    def _find_bullish_qm(
        self, swing_highs: List[Dict], swing_lows: List[Dict],
        closes: np.ndarray, atr: np.ndarray
    ) -> Optional[QMPattern]:
        """
        Пошук бичого QM: C < A < E, D > B → BUY
        (Дзеркальна версія ведмежого)
        """
        n_l = len(swing_lows)
        
        for i in range(n_l - 1, 1, -1):
            E = swing_lows[i]
            
            if E['idx'] < len(closes) - 20:
                continue
            
            for j in range(i - 1, 0, -1):
                C = swing_lows[j]
                
                for k in range(j - 1, -1, -1):
                    A = swing_lows[k]
                    
                    # 1. Базова структура: C < A < E (голова нижче, праве плече вище)
                    if not (C['price'] < A['price'] and A['price'] < E['price']):
                        continue
                    
                    # 2. Часові обмеження
                    total_bars = E['idx'] - A['idx']
                    if total_bars < self.min_pattern_bars or total_bars > self.lookback_bars * 0.8:
                        continue
                    
                    # 3. B (максимум між A і C)
                    B = self._find_highest_between(swing_highs, A['idx'], C['idx'])
                    if B is None:
                        continue
                    
                    # 4. D (максимум між C і E)
                    D = self._find_highest_between(swing_highs, C['idx'], E['idx'])
                    if D is None:
                        continue
                    
                    # 5. КРИТИЧНА УМОВА: D > B
                    if D['price'] <= B['price']:
                        continue
                    
                    # 6. Мінімальна різниця D-B
                    db_diff = abs(D['price'] - B['price']) / B['price'] * 100
                    if db_diff < self.min_db_diff_pct:
                        continue
                    
                    # 7. Симетрія часу
                    time_ac = C['idx'] - A['idx']
                    time_ce = E['idx'] - C['idx']
                    if time_ce > time_ac * 3 or time_ce < time_ac * 0.3:
                        continue
                    
                    return self._build_bullish_pattern(A, B, C, D, E, closes, atr)
        
        return None
    
    # ========================================
    # PATTERN BUILDING
    # ========================================
    
    def _build_bearish_pattern(self, A, B, C, D, E, closes, atr) -> Optional[QMPattern]:
        """Побудувати ведмежий QM з торговими рівнями"""
        current_price = float(closes[-1])
        
        # Entry: біля рівня E (праве плече) або Fibo 61.8% від C→D
        fibo_618 = C['price'] - (C['price'] - D['price']) * 0.618
        entry = min(E['price'], fibo_618)
        
        # Stop Loss: вище C (голови)
        sl = C['price'] * (1 + self.sl_buffer_pct / 100)
        
        # Take Profit 1: рівень B
        tp1 = B['price']
        
        # Take Profit 2: розширення 127.2% від C→D
        move = C['price'] - D['price']
        tp2 = E['price'] - move * 1.272
        
        # Метрики
        risk = abs(entry - sl)
        reward = abs(entry - tp1)
        
        if risk == 0:
            return None
        
        rr = reward / risk
        risk_pct = risk / entry * 100
        
        # Фільтри
        if rr < self.min_rr_ratio or risk_pct > self.max_risk_pct:
            return None
        
        strength = self._calc_strength(A, B, C, D, E, 'SELL')
        confidence = self._calc_confidence(A, B, C, D, E, current_price, 'SELL')
        
        if confidence < self.min_confidence:
            return None
        
        return QMPattern(
            direction='SELL',
            price_A=A['price'], price_B=B['price'], price_C=C['price'],
            price_D=D['price'], price_E=E['price'],
            idx_A=A['idx'], idx_B=B['idx'], idx_C=C['idx'],
            idx_D=D['idx'], idx_E=E['idx'],
            entry=round(entry, 8), stop_loss=round(sl, 8),
            take_profit_1=round(tp1, 8), take_profit_2=round(tp2, 8),
            strength=strength, confidence=confidence,
            risk_reward=rr, risk_percent=risk_pct,
        )
    
    def _build_bullish_pattern(self, A, B, C, D, E, closes, atr) -> Optional[QMPattern]:
        """Побудувати бичий QM з торговими рівнями"""
        current_price = float(closes[-1])
        
        # Entry: біля рівня E або Fibo 61.8% від C→D
        fibo_618 = C['price'] + (D['price'] - C['price']) * 0.618
        entry = max(E['price'], fibo_618)
        
        # Stop Loss: нижче C (голови)
        sl = C['price'] * (1 - self.sl_buffer_pct / 100)
        
        # TP1: рівень B
        tp1 = B['price']
        
        # TP2: розширення 127.2%
        move = D['price'] - C['price']
        tp2 = E['price'] + move * 1.272
        
        risk = abs(entry - sl)
        reward = abs(tp1 - entry)
        
        if risk == 0:
            return None
        
        rr = reward / risk
        risk_pct = risk / entry * 100
        
        if rr < self.min_rr_ratio or risk_pct > self.max_risk_pct:
            return None
        
        strength = self._calc_strength(A, B, C, D, E, 'BUY')
        confidence = self._calc_confidence(A, B, C, D, E, current_price, 'BUY')
        
        if confidence < self.min_confidence:
            return None
        
        return QMPattern(
            direction='BUY',
            price_A=A['price'], price_B=B['price'], price_C=C['price'],
            price_D=D['price'], price_E=E['price'],
            idx_A=A['idx'], idx_B=B['idx'], idx_C=C['idx'],
            idx_D=D['idx'], idx_E=E['idx'],
            entry=round(entry, 8), stop_loss=round(sl, 8),
            take_profit_1=round(tp1, 8), take_profit_2=round(tp2, 8),
            strength=strength, confidence=confidence,
            risk_reward=rr, risk_percent=risk_pct,
        )
    
    # ========================================
    # HELPERS
    # ========================================
    
    def _find_lowest_between(self, swing_lows: List[Dict], start: int, end: int) -> Optional[Dict]:
        valid = [s for s in swing_lows if start < s['idx'] < end]
        return min(valid, key=lambda x: x['price']) if valid else None
    
    def _find_highest_between(self, swing_highs: List[Dict], start: int, end: int) -> Optional[Dict]:
        valid = [s for s in swing_highs if start < s['idx'] < end]
        return max(valid, key=lambda x: x['price']) if valid else None
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """ATR без talib"""
        n = len(closes)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        
        atr = np.full(n, np.nan)
        if n >= self.atr_period:
            atr[self.atr_period - 1] = np.mean(tr[:self.atr_period])
            alpha = 1.0 / self.atr_period
            for i in range(self.atr_period, n):
                atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
        
        return atr
    
    def _calc_strength(self, A, B, C, D, E, direction: str) -> float:
        """Сила паттерна 0-100"""
        strength = 50.0
        
        # Висота голови відносно плечей
        if direction == 'SELL':
            head_h = C['price'] - A['price']
            shoulder_diff = A['price'] - E['price']
        else:
            head_h = A['price'] - C['price']
            shoulder_diff = E['price'] - A['price']
        
        if head_h > 0:
            ratio = shoulder_diff / head_h
            if 0.3 <= ratio <= 0.7:
                strength += 20
            elif 0.2 <= ratio <= 0.8:
                strength += 10
        
        # Глибина D відносно B
        db_diff = abs(D['price'] - B['price']) / B['price'] * 100
        strength += min(db_diff * 5, 20)
        
        # Часова симетрія
        time_ac = C['idx'] - A['idx']
        time_ce = E['idx'] - C['idx']
        if time_ac > 0:
            sym = time_ce / time_ac
            if 0.7 <= sym <= 1.5:
                strength += 10
        
        return min(strength, 100.0)
    
    def _calc_confidence(self, A, B, C, D, E, price: float, direction: str) -> float:
        """Впевненість 0-100"""
        conf = 60.0
        
        # Позиція ціни відносно патерна
        if direction == 'SELL':
            if E['price'] >= price >= D['price']:
                conf += 15
            if price <= E['price']:
                conf += 5
        else:
            if E['price'] <= price <= D['price']:
                conf += 15
            if price >= E['price']:
                conf += 5
        
        # Різниця D-B
        db_diff = abs(D['price'] - B['price']) / B['price'] * 100
        if db_diff >= 0.8:
            conf += min(db_diff * 2, 15)
        
        # ATR ratio
        if hasattr(C, 'atr_ratio') and C.get('atr_ratio', 0) >= 1.5:
            conf += 5
        
        return min(conf, 100.0)
