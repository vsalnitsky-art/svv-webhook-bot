"""
Market Scanner - Сканирование рынка для поиска новых возможностей
"""

import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import db_manager, MarketCandidate, ScanHistory

logger = logging.getLogger(__name__)


class MarketScanner:
    """
    Сканирование рынка для поиска потенциальных сделок
    """
    
    def __init__(self, scanner):
        self.scanner = scanner
        self.bot = scanner.bot
        self.config = scanner.config
        self.indicator = scanner.indicator
        
        # Кэш данных рынка
        self.market_cache = {}
        self.cache_timestamp = None
        
        # Результаты последнего сканирования
        self.last_scan_results = []
        self.last_scan_id = 0
        self.last_scan_time = None
        self.scanning = False
        
        # Статистика
        self.total_scans = 0
        self.total_candidates_found = 0
        
        logger.info("✅ MarketScanner initialized")
    
    def start(self):
        """Запуск сканирования в отдельном потоке"""
        scanner_params = self.config.get_scanner_params()
        
        if not scanner_params['enabled']:
            logger.info("⏸️ Market scanner disabled in config")
            return
        
        threading.Thread(target=self._scan_loop, daemon=True).start()
        logger.info("🚀 MarketScanner started")
    
    def _scan_loop(self):
        """Основной цикл сканирования"""
        scanner_params = self.config.get_scanner_params()
        scan_interval = scanner_params['scan_interval']
        
        # Первое сканирование через 10 секунд после старта
        time.sleep(10)
        
        while True:
            try:
                if not self.scanning:
                    self.scan_market()
                time.sleep(scan_interval)
            except Exception as e:
                logger.error(f"❌ Scan loop error: {e}", exc_info=True)
                time.sleep(60)  # Подождать минуту при ошибке
    
    def scan_market(self) -> List[Dict]:
        """
        Основная функция сканирования рынка
        """
        if self.scanning:
            logger.warning("⚠️ Scan already in progress, skipping...")
            return []
        
        self.scanning = True
        scan_start = time.time()
        self.last_scan_id += 1
        scan_id = self.last_scan_id
        
        logger.info(f"🔍 Starting market scan #{scan_id}")
        
        try:
            # 1. Получить все тикеры
            all_tickers = self._get_all_tickers()
            total_coins = len(all_tickers)
            logger.info(f"   📊 Total coins: {total_coins}")
            
            if not all_tickers:
                logger.warning("⚠️ No tickers received from exchange")
                return []
            
            # 2. Фильтрация монет
            filtered_coins = self._filter_coins(all_tickers)
            filtered_count = len(filtered_coins)
            logger.info(f"   ✅ Filtered to: {filtered_count} coins")
            
            # 3. Применить индикатор и найти кандидатов
            candidates = self._process_candidates(filtered_coins)
            candidates_count = len(candidates)
            logger.info(f"   🎯 Candidates found: {candidates_count}")
            
            # 4. Рассчитать рейтинг и отсортировать
            if candidates:
                ranked_candidates = self._rank_candidates(candidates)
                
                # 5. Взять топ-N
                scanner_params = self.config.get_scanner_params()
                top_n = scanner_params['top_candidates_count']
                top_candidates = ranked_candidates[:top_n]
                
                # 6. Сохранить результаты
                self._save_scan_results(scan_id, top_candidates)
                self.last_scan_results = top_candidates
                
                logger.info(f"   🏆 Top {len(top_candidates)} candidates saved")
            else:
                self.last_scan_results = []
            
            # 7. Сохранить историю сканирования
            scan_duration = time.time() - scan_start
            self._save_scan_history(
                scan_id, total_coins, filtered_count, 
                candidates_count, scan_duration
            )
            
            self.last_scan_time = datetime.now()
            self.total_scans += 1
            self.total_candidates_found += candidates_count
            
            logger.info(f"✅ Scan #{scan_id} completed in {scan_duration:.1f}s")
            
            return self.last_scan_results
            
        except Exception as e:
            logger.error(f"❌ Market scan error: {e}", exc_info=True)
            return []
        finally:
            self.scanning = False
    
    def _get_all_tickers(self) -> List[Dict]:
        """
        Получить все тикеры с биржи
        """
        try:
            # Проверить кэш
            scanner_params = self.config.get_scanner_params()
            if scanner_params['use_cache'] and self.cache_timestamp:
                cache_age = (datetime.now() - self.cache_timestamp).total_seconds()
                if cache_age < scanner_params['cache_ttl']:
                    logger.info(f"   💾 Using cached tickers (age: {cache_age:.0f}s)")
                    return self.market_cache.get('tickers', [])
            
            # Получить свежие данные
            response = self.bot.session.get_tickers(category="linear")
            
            if response['retCode'] != 0:
                logger.error(f"❌ API error: {response.get('retMsg')}")
                return []
            
            tickers = response['result']['list']
            
            # Сохранить в кэш
            if scanner_params['use_cache']:
                self.market_cache['tickers'] = tickers
                self.cache_timestamp = datetime.now()
            
            return tickers
            
        except Exception as e:
            logger.error(f"❌ Error getting tickers: {e}")
            return []
    
    def _filter_coins(self, tickers: List[Dict]) -> List[Dict]:
        """
        Фильтрация монет по критериям
        """
        scanner_params = self.config.get_scanner_params()
        risk_params = self.config.get_risk_params()
        
        filtered = []
        
        # Статистика фільтрації
        stats = {
            'total': len(tickers),
            'after_blacklist': 0,
            'after_usdt_filter': 0,
            'after_volume_filter': 0,
            'after_price_change_filter': 0,
            'after_price_valid_filter': 0,
            'final': 0
        }
        
        logger.info(f"📋 ФІЛЬТРАЦІЯ МОНЕТ:")
        logger.info(f"   └─ Крок 0: Всього монет з біржі: {stats['total']}")
        
        for ticker in tickers:
            try:
                symbol = ticker['symbol']
                
                # 1. Проверка blacklist
                if symbol in risk_params['blacklist_symbols']:
                    continue
                
                stats['after_blacklist'] += 1
                
                # 2. Только USDT пары
                if not symbol.endswith('USDT'):
                    continue
                
                stats['after_usdt_filter'] += 1
                
                # 3. Минимальный объём 24h
                volume_24h = float(ticker.get('turnover24h', 0))
                if volume_24h < scanner_params['min_volume_24h']:
                    continue
                
                stats['after_volume_filter'] += 1
                
                # 4. Минимальное изменение цены (волатильность)
                price_change = abs(float(ticker.get('price24hPcnt', 0))) * 100
                if price_change < scanner_params['min_price_change_24h']:
                    continue
                
                stats['after_price_change_filter'] += 1
                
                # 5. Проверка spread (bid-ask)
                # Упрощённая проверка через lastPrice
                last_price = float(ticker.get('lastPrice', 0))
                if last_price <= 0:
                    continue
                
                stats['after_price_valid_filter'] += 1
                
                # Добавить в отфильтрованные
                filtered.append({
                    'symbol': symbol,
                    'last_price': last_price,
                    'volume_24h': volume_24h,
                    'price_change_24h': float(ticker.get('price24hPcnt', 0)) * 100,
                    'high_24h': float(ticker.get('highPrice24h', last_price)),
                    'low_24h': float(ticker.get('lowPrice24h', last_price)),
                })
                
            except Exception as e:
                logger.error(f"❌ Error filtering {ticker.get('symbol')}: {e}")
                continue
        
        stats['final'] = len(filtered)
        
        # Детальний звіт фільтрації
        logger.info(f"   ├─ Крок 1: Виключити blacklist: {stats['after_blacklist']} монет")
        logger.info(f"   ├─ Крок 2: Тільки USDT пари: {stats['after_usdt_filter']} монет")
        logger.info(f"   ├─ Крок 3: Об'єм > ${scanner_params['min_volume_24h']:,.0f}: {stats['after_volume_filter']} монет")
        logger.info(f"   │          (відсіяно {stats['after_usdt_filter'] - stats['after_volume_filter']} через низький об'єм)")
        logger.info(f"   ├─ Крок 4: Зміна ціни > {scanner_params['min_price_change_24h']}%: {stats['after_price_change_filter']} монет")
        logger.info(f"   │          (відсіяно {stats['after_volume_filter'] - stats['after_price_change_filter']} через низьку волатильність)")
        logger.info(f"   ├─ Крок 5: Валідна ціна: {stats['after_price_valid_filter']} монет")
        logger.info(f"   └─ 🎯 ФІНАЛЬНИЙ РЕЗУЛЬТАТ: {stats['final']} монет пройшли базові фільтри")
        
        if stats['final'] == 0:
            logger.warning("⚠️ УВАГА: Жодна монета не пройшла базові фільтри!")
            logger.warning(f"   💡 Спробуй зменшити:")
            logger.warning(f"      - min_volume_24h (зараз: ${scanner_params['min_volume_24h']:,.0f})")
            logger.warning(f"      - min_price_change_24h (зараз: {scanner_params['min_price_change_24h']}%)")
        
        return filtered
    
    def _process_candidates(self, filtered_coins: List[Dict]) -> List[Dict]:
        """
        Обработка монет через индикатор (параллельно)
        """
        scanner_params = self.config.get_scanner_params()
        batch_size = scanner_params['batch_size']
        use_parallel = scanner_params['parallel_processing']
        
        candidates = []
        
        if use_parallel and len(filtered_coins) > batch_size:
            # Параллельная обработка батчами
            candidates = self._process_parallel(filtered_coins, batch_size)
        else:
            # Последовательная обработка
            candidates = self._process_sequential(filtered_coins)
        
        return candidates
    
    def _process_sequential(self, coins: List[Dict]) -> List[Dict]:
        """Последовательная обработка монет"""
        candidates = []
        
        logger.info(f"🔬 АНАЛІЗ RSI/MFI ({len(coins)} монет):")
        start_time = time.time()
        
        for i, coin in enumerate(coins):
            # Логування прогресу кожні 5 монет
            if i % 5 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(coins) - i) * avg_time
                logger.info(f"   Прогрес: {i}/{len(coins)} ({i*100//len(coins)}%) | Залишилось: ~{remaining:.0f}s | Знайдено: {len(candidates)}")
            
            candidate = self._analyze_coin(coin)
            if candidate:
                candidates.append(candidate)
        
        total_time = time.time() - start_time
        logger.info(f"   ✅ Завершено аналіз за {total_time:.1f}s: знайдено {len(candidates)} кандидатів")
        
        if len(candidates) == 0 and len(coins) > 0:
            logger.warning("⚠️ УВАГА: Жодного кандидата не знайдено після RSI/MFI аналізу!")
            logger.warning(f"   💡 Можливі причини:")
            logger.warning(f"      - RSI всіх монет поза зонами входу")
            logger.warning(f"      - Занадто жорсткі вимоги до сили сигналу")
            logger.warning(f"      - Невірний таймфрейм")
        
        return candidates
    
    def _process_parallel(self, coins: List[Dict], batch_size: int) -> List[Dict]:
        """Параллельная обработка монет батчами"""
        candidates = []
        
        # Разбить на батчи
        batches = [coins[i:i+batch_size] for i in range(0, len(coins), batch_size)]
        
        logger.info(f"   📦 Processing {len(batches)} batches in parallel...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Отправить батчи на обработку
            future_to_batch = {
                executor.submit(self._process_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Собрать результаты
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_candidates = future.result()
                    candidates.extend(batch_candidates)
                    logger.info(f"   ✅ Batch {batch_num+1}/{len(batches)} completed: {len(batch_candidates)} candidates")
                except Exception as e:
                    logger.error(f"   ❌ Batch {batch_num} error: {e}")
        
        return candidates
    
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Обработка одного батча"""
        candidates = []
        
        for coin in batch:
            candidate = self._analyze_coin(coin)
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _analyze_coin(self, coin: Dict) -> Optional[Dict]:
        """
        Анализ одной монеты через индикатор
        """
        try:
            symbol = coin['symbol']
            
            # Получить исторические данные
            df = self._get_coin_historical_data(symbol)
            
            if df is None or len(df) < 50:
                logger.debug(f"      ⏭️  {symbol}: Недостатньо історичних даних ({len(df) if df is not None else 0} свічок)")
                return None
            
            # Применить индикатор
            result_df, alerts = self.indicator.process_data(df, symbol)
            
            # Получить последние сигналы
            signals = self.indicator.get_latest_signals(result_df)
            
            # Проверить наличие сигналов
            has_buy = signals.get('buy_signal', False)
            has_sell = signals.get('sell_signal', False)
            has_strong_buy = signals.get('strong_bullish_signal', False)
            has_strong_sell = signals.get('strong_bearish_signal', False)
            
            rsi_value = signals['rsi_value']
            
            # Определить направление и силу
            direction = None
            signal_strength = 'none'
            
            if has_strong_buy:
                direction = 'Long'
                signal_strength = 'Strong'
            elif has_strong_sell:
                direction = 'Short'
                signal_strength = 'Strong'
            elif has_buy:
                direction = 'Long'
                signal_strength = 'Regular'
            elif has_sell:
                direction = 'Short'
                signal_strength = 'Regular'
            
            # Если нет сигналов - пропустить
            if direction is None:
                logger.debug(f"      ⏭️  {symbol}: Немає сигналів (RSI: {rsi_value:.1f})")
                return None
            
            # Фильтр по силе сигнала
            scanner_params = self.config.get_scanner_params()
            min_strength = scanner_params['min_signal_strength']
            
            if min_strength == 'strong' and signal_strength != 'Strong':
                logger.debug(f"      ⏭️  {symbol}: Відсіяно через силу сигналу ({signal_strength}, потрібно: strong)")
                return None
            
            # Создать кандидата
            candidate = {
                'symbol': symbol,
                'direction': direction,
                'signal_strength': signal_strength,
                'price': coin['last_price'],
                'volume_24h': coin['volume_24h'],
                'change_24h': coin['price_change_24h'],
                'rsi': signals['rsi_value'],
                'mfi': signals.get('mfi_value', 0),
                'mfi_trend': signals.get('mfi_trend', 'Нейтральний'),
                'last_signal': signals.get('last_signal', 'Немає'),
                'bullish_cloud': signals.get('bullish_cloud', False),
                'bearish_cloud': signals.get('bearish_cloud', False),
            }
            
            logger.info(f"      ✅ {symbol}: {direction} {signal_strength} (RSI: {rsi_value:.1f})")
            
            return candidate
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {coin.get('symbol')}: {e}")
            return None
    
    def _get_coin_historical_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Получить исторические данные для монеты
        Использует таймфрейм из конфигурации (по умолчанию 4h)
        """
        try:
            # Получить таймфрейм из конфигурации
            timeframe = self.config.get_timeframe()
            
            response = self.bot.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timeframe,  # ✅ Использует timeframe из конфига
                limit=limit
            )
            
            if response['retCode'] != 0:
                return None
            
            klines = response['result']['list']
            
            if not klines:
                return None
            
            # Преобразовать в DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            return None
    
    def _rank_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Рассчитать рейтинг и отсортировать кандидатов
        """
        for candidate in candidates:
            # Рейтинговая формула
            rating = 0
            reason_parts = []
            
            # 1. Сила сигнала (главный фактор)
            if candidate['signal_strength'] == 'Strong':
                rating += 30
                reason_parts.append("Strong signal")
            else:
                rating += 10
                reason_parts.append("Regular signal")
            
            # 2. RSI в зоне (перепроданность для Long, перекупленность для Short)
            rsi = candidate['rsi']
            if candidate['direction'] == 'Long':
                if rsi < 30:
                    rating += 20
                    reason_parts.append("Oversold RSI")
                elif rsi < 40:
                    rating += 10
            else:  # Short
                if rsi > 70:
                    rating += 20
                    reason_parts.append("Overbought RSI")
                elif rsi > 60:
                    rating += 10
            
            # 3. MFI Cloud подтверждение
            if candidate['direction'] == 'Long' and candidate['bullish_cloud']:
                rating += 15
                reason_parts.append("Bullish MFI cloud")
            elif candidate['direction'] == 'Short' and candidate['bearish_cloud']:
                rating += 15
                reason_parts.append("Bearish MFI cloud")
            
            # 4. Объём торгов (ликвидность)
            volume_billions = candidate['volume_24h'] / 1_000_000_000
            if volume_billions > 1:
                rating += 15
                reason_parts.append("High volume")
            elif volume_billions > 0.5:
                rating += 10
                reason_parts.append("Good volume")
            elif volume_billions > 0.1:
                rating += 5
            
            # 5. Волатильность (движение цены)
            volatility = abs(candidate['change_24h'])
            if volatility > 5:
                rating += 10
                reason_parts.append("High volatility")
            elif volatility > 3:
                rating += 5
            
            candidate['rating'] = min(rating, 100)  # Максимум 100
            candidate['reason'] = ' + '.join(reason_parts)
        
        # Сортировать по рейтингу
        sorted_candidates = sorted(candidates, key=lambda x: x['rating'], reverse=True)
        
        return sorted_candidates
    
    def _save_scan_results(self, scan_id: int, candidates: List[Dict]):
        """Сохранить кандидатов в БД"""
        try:
            session = db_manager.get_session()
            
            for candidate in candidates:
                market_candidate = MarketCandidate(
                    scan_id=scan_id,
                    timestamp=datetime.now(),
                    symbol=candidate['symbol'],
                    direction=candidate['direction'],
                    signal_strength=candidate['signal_strength'],
                    rsi=candidate['rsi'],
                    mfi=candidate['mfi'],
                    mfi_trend=candidate['mfi_trend'],
                    price=candidate['price'],
                    volume_24h=candidate['volume_24h'],
                    change_24h=candidate['change_24h'],
                    rating=candidate['rating'],
                    reason=candidate['reason']
                )
                
                session.add(market_candidate)
            
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving scan results: {e}")
    
    def _save_scan_history(self, scan_id: int, total: int, filtered: int, 
                          candidates: int, duration: float):
        """Сохранить историю сканирования в БД"""
        try:
            session = db_manager.get_session()
            
            top_candidate = None
            top_rating = 0
            
            if self.last_scan_results:
                top_candidate = self.last_scan_results[0]['symbol']
                top_rating = self.last_scan_results[0]['rating']
            
            scan_history = ScanHistory(
                timestamp=datetime.now(),
                total_coins=total,
                filtered_coins=filtered,
                candidates_found=candidates,
                scan_duration=duration,
                top_candidate=top_candidate,
                top_rating=top_rating
            )
            
            session.add(scan_history)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving scan history: {e}")
    
    def get_latest_candidates(self, limit: int = 10) -> List[Dict]:
        """Получить последние кандидаты"""
        return self.last_scan_results[:limit]
    
    def get_stats(self) -> Dict:
        """Получить статистику сканирования"""
        return {
            'total_scans': self.total_scans,
            'total_candidates_found': self.total_candidates_found,
            'last_scan_time': self.last_scan_time,
            'last_scan_results_count': len(self.last_scan_results),
            'scanning': self.scanning,
        }
