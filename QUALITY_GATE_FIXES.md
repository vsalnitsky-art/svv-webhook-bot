# Trade Quality Gate Fixes - Implementation Summary

## Issues Fixed

### 1. **Health Score & Entry Score Weights Not Persisting** ✅
**Problem:** Weight slider values were being saved to the database but not loaded back when the page refreshed, causing settings to revert to defaults.

**Root Cause:** In `templates/smart_money.html` line 5169-5171, the `loadTMSettings()` function explicitly skipped rendering weight sliders with the comment "Weight sliders are hidden in the simplified UI — collectors gracefully fall back to defaults when DOM nodes aren't rendered, so we skip the render calls entirely."

**Fix:** Added calls to `renderHealthWeights(s.health_score_weights || {})` and `renderEntryWeights(s.entry_score_weights || {})` in the `loadTMSettings()` function to populate sliders with saved values from the server.

**Files Modified:**
- `templates/smart_money.html` (lines 5166-5173)

---

### 2. **Blocked Trades Tracking & Display** ✅
**Problem:** No mechanism to track or display trades that were blocked/rejected by Quality Gate filters or directional gates.

**Implementation:** Created a complete blocked trades tracking system with database storage, API endpoints, and UI display.

#### Database Layer

**New Table:** `BlockedTrade` in `storage/db_models.py`
- Fields: `id`, `is_paper`, `symbol`, `side`, `entry_price`, `blocked_at`, `blocked_reason`, `health_score`, `entry_score`, `snapshot` (JSON)
- Stores full quality metrics snapshot for later analysis
- Indexed on `symbol` and `blocked_at` for fast queries

**New DB Operations:** Added to `storage/db_operations.py`
- `record_blocked_trade()` - Store a blocked trade
- `get_blocked_trades()` - Retrieve blocked trades with filters (limit, is_paper, symbol)
- `get_blocked_trades_stats()` - Get counts (total, real, paper)
- `clear_blocked_trades()` - Delete blocked trades by scope

**Files Modified:**
- `storage/db_models.py` - Added `BlockedTrade` model (lines 457-483)
- `storage/db_operations.py` - Added 4 new functions + imports (lines 5, 15, 214-345)

#### Backend API

**New Endpoints:** Added to `web/flask_app.py`
- `GET /api/blocked-trades/list` - Fetch blocked trades with query params (limit, is_paper, symbol)
- `GET /api/blocked-trades/stats` - Get summary statistics
- `POST /api/blocked-trades/clear` - Clear blocked trades (scope: all/real/paper)

**Files Modified:**
- `web/flask_app.py` (lines 2609-2652)

#### Trade Manager Integration

**Recording Logic:** Added to `detection/trade_manager.py`
- New method `_record_blocked_trade()` to capture rejection details
- Called when trades are blocked by side gates (LONG/SHORT disabled)
- Captures: symbol, side, entry_price, blocked_reason, health/entry scores, full snapshot, is_paper flag

**Files Modified:**
- `detection/trade_manager.py`
  - Added `_record_blocked_trade()` method (lines 3678-3716)
  - Updated `_open_position()` to record blocked real trades (lines 1926-1932)
  - Updated `_open_shadow()` to record blocked paper trades (lines 2480-2487)

#### User Interface

**New Section:** Collapsed accordion in Trade Manager panel (`templates/smart_money.html`)
- Header shows total blocked trades count (real + paper breakdown)
- Table columns: Time, Symbol, Side, Entry Price, Blocked Reason, Health Score, Entry Score, Type (Real/Paper), Info button
- Info button shows full quality snapshot in alert dialog
- Clear button to delete all blocked trades
- Collapsed by default to avoid UI clutter

**JavaScript Functions:** Added to `templates/smart_money.html`
- `loadBlockedTrades()` - Fetch and display blocked trades
- `clearBlockedTrades()` - Clear all blocked trades with confirmation
- `showBlockedTradeInfo(tradeId)` - Display full snapshot details
- Integrated into `refreshAll()` to auto-refresh every 60 seconds

**Files Modified:**
- `templates/smart_money.html`
  - Added UI section (lines 2130-2156)
  - Added JavaScript functions (lines 6358-6439)
  - Updated `refreshAll()` to include `loadBlockedTrades()` (line 7013)

---

## Database Migration

The new `BlockedTrade` table will be created automatically on next app startup via `Base.metadata.create_all()` in `storage/db_models.py:init_db()`. No manual migration required.

---

## Testing Checklist

### Weight Persistence
- [ ] Open Smart Money page → Trade Manager settings
- [ ] Adjust Health Score weight sliders
- [ ] Adjust Entry Score weight sliders  
- [ ] Refresh the page
- [ ] Verify sliders retain the custom values (not reverting to defaults)

### Blocked Trades Tracking
- [ ] Disable LONG or SHORT side gate
- [ ] Trigger a signal for that side (e.g., manual signal or wait for SMC scanner)
- [ ] Verify console shows: `[TM] 🚫 REAL LONG entries disabled — BTCUSDT not opened`
- [ ] Open "🚫 Blocked Trades (Quality Gate)" section
- [ ] Verify the blocked trade appears in the table with reason "LONG_side_gate_disabled"
- [ ] Click "ℹ️ Details" button to view snapshot
- [ ] Test with paper trading (test_mode) to verify `is_paper` flag
- [ ] Click "🗑️ Clear" to delete blocked trades
- [ ] Verify table shows "No blocked trades yet"

### API Endpoints
```bash
# Get blocked trades list
curl http://localhost:5000/api/blocked-trades/list?limit=10

# Get stats
curl http://localhost:5000/api/blocked-trades/stats

# Clear all (requires POST)
curl -X POST http://localhost:5000/api/blocked-trades/clear \
  -H "Content-Type: application/json" \
  -d '{"scope":"all"}'
```

---

## Future Enhancements

1. **Expand Blocked Reasons:**
   - Currently only tracks side gate blocks
   - TODO: Track Health Score threshold blocks
   - TODO: Track Entry Score threshold blocks
   - TODO: Track position limit blocks
   - TODO: Track duplicate signal blocks

2. **Quality Snapshot Enrichment:**
   - Currently snapshot is empty for side gate blocks
   - TODO: Capture full Decision Center verdict at block time
   - TODO: Include HTF alignment, Forecast, CTR state

3. **Advanced Filtering:**
   - Add UI filters: by symbol, by reason, by date range
   - Export blocked trades to CSV for offline analysis

4. **Analytics Dashboard:**
   - Show most frequently blocked symbols
   - Show distribution of blocked reasons
   - Compare blocked vs accepted trade characteristics

---

## Notes

- Weight sliders remain `display:none` in HTML but are now properly populated on load
- Blocked trades are stored permanently until manually cleared (like Trade Archive)
- The feature is fully backwards compatible - old installations will auto-migrate on startup
- All blocked trade recording is defensive (try/except) to never break actual trading logic
