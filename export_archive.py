#!/usr/bin/env python3
"""
Standalone script to export Trade Archive data from database.

Usage:
    python3 export_archive.py [output_file.json]

Exports all trade archive data (real + paper) with full entry snapshots
for analysis, backtesting, and ML training.
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from storage.db_operations import DBOperations

    def export_archive(output_path='trade_archive_export.json'):
        """Export all trade archive data to JSON."""

        print("🔍 Connecting to database...")
        db = DBOperations()

        print("📦 Fetching trade archive data...")
        # Get ALL trades (high limit to ensure we get everything)
        all_trades = db.get_trade_archive(limit=50000)

        if not all_trades:
            print("⚠️  No trades found in archive!")
            return

        print(f"✓ Loaded {len(all_trades)} trades")

        # Split by type
        real_trades = [t for t in all_trades if not t.get('is_paper', False)]
        paper_trades = [t for t in all_trades if t.get('is_paper', False)]

        print(f"  - Real: {len(real_trades)}")
        print(f"  - Paper: {len(paper_trades)}")

        # Calculate quick stats
        wins = [t for t in all_trades if t.get('pnl_pct', 0) > 0]
        losses = [t for t in all_trades if t.get('pnl_pct', 0) <= 0]

        total_pnl_pct = sum(t.get('pnl_pct', 0) for t in all_trades)
        total_pnl_usd = sum(t.get('pnl_usd', 0) for t in all_trades)
        avg_pnl_pct = total_pnl_pct / len(all_trades) if all_trades else 0
        avg_duration = sum(t.get('duration_secs', 0) for t in all_trades) / len(all_trades) if all_trades else 0

        # Count trades with entry snapshots
        trades_with_snapshot = [t for t in all_trades if t.get('entry_snapshot')]

        stats = {
            'total_trades': len(all_trades),
            'real_trades': len(real_trades),
            'paper_trades': len(paper_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(all_trades) * 100) if all_trades else 0,
            'total_pnl_pct': round(total_pnl_pct, 2),
            'total_pnl_usd': round(total_pnl_usd, 2),
            'avg_pnl_pct': round(avg_pnl_pct, 2),
            'avg_duration_mins': round(avg_duration / 60, 1),
            'best_trade_pct': max((t.get('pnl_pct', 0) for t in all_trades), default=0),
            'worst_trade_pct': min((t.get('pnl_pct', 0) for t in all_trades), default=0),
            'trades_with_entry_snapshot': len(trades_with_snapshot),
        }

        # Prepare export data
        export_data = {
            'metadata': {
                'exported_at': __import__('time').time(),
                'source': 'svv-webhook-bot trade archive',
                'description': 'Complete trade history with entry snapshots for backtesting',
            },
            'stats': stats,
            'trades': all_trades,
        }

        # Save to JSON
        print(f"\n💾 Saving to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        file_size = os.path.getsize(output_path)
        print(f"✓ Saved successfully!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size / 1024:.1f} KB")

        # Print detailed stats
        print(f"\n📊 Archive Statistics:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"    ├─ Real: {stats['real_trades']}")
        print(f"    └─ Paper: {stats['paper_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L)")
        print(f"  Avg PnL: {stats['avg_pnl_pct']:+.2f}%")
        print(f"  Total PnL: {stats['total_pnl_pct']:+.2f}% (${stats['total_pnl_usd']:+.2f})")
        print(f"  Best Trade: {stats['best_trade_pct']:+.2f}%")
        print(f"  Worst Trade: {stats['worst_trade_pct']:+.2f}%")
        print(f"  Avg Duration: {stats['avg_duration_mins']:.1f} min")
        print(f"  Entry Snapshots: {stats['trades_with_entry_snapshot']}/{stats['total_trades']}")

        # Top symbols
        from collections import Counter
        symbols = Counter(t.get('symbol', 'UNKNOWN') for t in all_trades)
        print(f"\n🔝 Top 5 Symbols:")
        for sym, count in symbols.most_common(5):
            sym_trades = [t for t in all_trades if t.get('symbol') == sym]
            sym_pnl = sum(t.get('pnl_pct', 0) for t in sym_trades) / len(sym_trades)
            print(f"    {sym}: {count} trades, avg {sym_pnl:+.2f}%")

        # Close reasons
        reasons = Counter(t.get('reason', 'unknown') for t in all_trades)
        print(f"\n🚪 Top Close Reasons:")
        for reason, count in reasons.most_common(5):
            print(f"    {reason}: {count} ({count/len(all_trades)*100:.1f}%)")

        print(f"\n✅ Export complete! Data ready for analysis.")
        return output_path

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you're running this script from the project root:")
    print("  cd /home/user/svv-webhook-bot")
    print("  python3 export_archive.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'trade_archive_export.json'
    export_archive(output_file)
