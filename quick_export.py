#!/usr/bin/env python3
"""Quick export without full DB init - just query the archive table."""

import sys
import json
import os
from sqlalchemy import create_engine, text

# Get DATABASE_URL from environment
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("❌ DATABASE_URL not set")
    sys.exit(1)

print(f"🔍 Connecting to: {db_url.split('@')[1].split('/')[0]}...")

try:
    engine = create_engine(db_url, pool_pre_ping=True, connect_args={'connect_timeout': 10})

    with engine.connect() as conn:
        # Simple query to get all trades
        print("📦 Querying trade archive...")
        result = conn.execute(text("""
            SELECT
                id, is_paper, symbol, side, entry_price, exit_price,
                qty, pnl_pct, pnl_usd, reason, reason_detail, opened_by,
                opened_at, closed_at, duration_secs, entry_snapshot
            FROM svv_trade_archive
            ORDER BY closed_at DESC
        """))

        trades = []
        for row in result:
            trade = {
                'id': row[0],
                'is_paper': row[1],
                'symbol': row[2],
                'side': row[3],
                'entry_price': row[4],
                'exit_price': row[5],
                'qty': row[6],
                'pnl_pct': row[7],
                'pnl_usd': row[8],
                'reason': row[9],
                'reason_detail': row[10],
                'opened_by': row[11],
                'opened_at': row[12],
                'closed_at': row[13],
                'duration_secs': row[14],
                'entry_snapshot': json.loads(row[15]) if row[15] else None,
            }
            trades.append(trade)

        print(f"✓ Loaded {len(trades)} trades")

        # Stats
        real = [t for t in trades if not t['is_paper']]
        paper = [t for t in trades if t['is_paper']]
        wins = [t for t in trades if t['pnl_pct'] > 0]

        avg_pnl = sum(t['pnl_pct'] for t in trades) / len(trades) if trades else 0

        stats = {
            'total': len(trades),
            'real': len(real),
            'paper': len(paper),
            'wins': len(wins),
            'losses': len(trades) - len(wins),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'avg_pnl_pct': avg_pnl,
        }

        # Export
        data = {
            'metadata': {
                'exported_at': __import__('time').time(),
                'source': 'svv-webhook-bot',
            },
            'stats': stats,
            'trades': trades,
        }

        output = 'trade_archive_full.json'
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        size = os.path.getsize(output)
        print(f"\n✓ Saved to: {output}")
        print(f"  Size: {size / 1024:.1f} KB")
        print(f"\n📊 Stats:")
        print(f"  Total: {stats['total']} (Real: {stats['real']}, Paper: {stats['paper']})")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Avg PnL: {stats['avg_pnl_pct']:+.2f}%")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
