# Trade Quality Gate Fixes - Implementation Summary

## Issues Fixed

### 1. **Health Score & Entry Score Weights Not Persisting** ✅
**Problem:** Weight slider values were saved to the database but not loaded back
on page refresh, so settings reverted to defaults.

**Root Cause:** In `templates/smart_money.html`, `loadTMSettings()` explicitly
skipped rendering the weight sliders ("...so we skip the render calls entirely").
The saved values never made it back into the DOM, so the next save read defaults.

**Fix:** Added `renderHealthWeights(s.health_score_weights || {})` and
`renderEntryWeights(s.entry_score_weights || {})` to `loadTMSettings()` so the
sliders are repopulated with the values returned by the server.

**Files Modified:**
- `templates/smart_money.html` — `loadTMSettings()`

---

### 2. **Blocked Trades Table — signals rejected by the Trade Quality Gate** ✅

**What "Trade Quality Gate (data-driven filter)" actually is:** the SMC scanner's
signal filters in the Smart Money panel:
- **HTF Bias Filter** — blocks signals against the higher-timeframe bias
- **OB Filter** — blocks signals that disagree with the last valid Order Block
- **PD Zone Filter** — blocks LONG in Premium / SHORT in Discount (price position)
- **Forecast Filter** — blocks signals that disagree with the 1H/4H forecast

When a CHoCH/BOS structure event fires but is rejected by one of these filters, it
never reaches the Trade Manager — no position is opened. Previously these
rejections were only printed to the log and then lost. Now each one is captured
so the user can see **which filter blocked each opportunity and why**.

> Note: this is NOT the LONG/SHORT master toggle. Those toggles are a manual
> on/off switch, not a data-driven quality filter, so they are not recorded here.

#### Where recording happens (`detection/smc_scanner.py`)

- `_record_blocked_signal(symbol, side, entry_price, mode, filter_name, details)`
  — central helper that writes one row with a full snapshot. Best-effort; never
  breaks scanning.
- `_record_htf_block(symbol, event, mode)` — wired into all three HTF-block sites
  in `_process_alerts` (`choch`, `choch_or_bos` CHoCH leg, `choch_or_bos` BOS leg).
- OB / PD Zone / Forecast blocks recorded inline in `_send_alert`, capturing the
  relevant characteristics for each:
  - **OB Filter:** OB timeframe, OB bias, detail
  - **PD Zone:** timeframe, position %, long_max/short_min thresholds, detail
  - **Forecast:** which TFs are on, combine mode, raw 1H/4H forecasts, detail
  - **HTF Bias:** timeframe, method, bias, detail

`entry_price` (the structural break level) is now computed at the top of
`_send_alert` so a blocked signal still records the would-be entry.

#### Database (`storage/db_models.py`, `storage/db_operations.py`)

- New table `BlockedTrade` (`sob_blocked_trades`): `symbol`, `side`,
  `entry_price`, `blocked_at`, `blocked_reason` (filter name), `snapshot` (JSON
  with all characteristics), plus `health_score`/`entry_score` columns reserved
  for future scoring-based gates.
- New ops: `record_blocked_trade()`, `get_blocked_trades()`,
  `get_blocked_trades_stats()`, `clear_blocked_trades()`.
- Table auto-creates on startup via `Base.metadata.create_all()` — no migration.

#### Backend API (`web/flask_app.py`)

- `GET  /api/blocked-trades/list?limit=&is_paper=&symbol=`
- `GET  /api/blocked-trades/stats`
- `POST /api/blocked-trades/clear` (body `{"scope":"all"|"real"|"paper"}`)

#### UI (`templates/smart_money.html`)

- New **collapsed accordion** "🚫 Blocked Trades (Quality Gate)" under the Trade
  Archive panel, collapsed by default (`<details>`).
- Columns: Time · Symbol · Side · Entry Price · **Blocked By** (filter, colour-
  coded) · **Reason / Characteristics** (human-readable detail) · Info.
- Header shows total count + a per-filter breakdown
  (e.g. `HTF Bias Filter: 4 · PD Zone Filter: 2`).
- "ℹ️ Details" shows the full filter snapshot (all characteristics) for a row.
- "🗑️ Clear" wipes the list. Auto-refreshes every 60s with the rest of the panel.

JS functions added: `loadBlockedTrades()`, `clearBlockedTrades()`,
`showBlockedTradeInfo()`; `loadBlockedTrades()` added to `refreshAll()`.

---

## Verification

DB layer tested end-to-end on SQLite (record → stats → list ordering → symbol
filter → clear), all passing. Production runs Postgres; the table auto-creates on
boot. Python syntax of all five modified modules validated.

---

## Future Enhancements

- Record PD Zone / OB / Forecast snapshots also include raw forecast confidence.
- Add UI filters (by symbol / by filter / date range) and CSV export.
- When a scoring-based gate (Entry Score threshold) is added, populate the
  reserved `health_score` / `entry_score` columns.
