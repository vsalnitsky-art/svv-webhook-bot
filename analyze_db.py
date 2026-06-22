#!/usr/bin/env python3
"""Analyze PostgreSQL database size and identify foreign tables."""

import os
import sys

# This bot's table prefixes
BOT_PREFIXES = ['sob_', 'volumized_radar_']
BOT_EXACT_NAMES = ['volumized_radar_metadata', 'volumized_radar_stats', 'volumized_radar_snapshots']

def belongs_to_this_bot(table_name):
    """Check if table belongs to this bot."""
    if table_name in BOT_EXACT_NAMES:
        return True
    for prefix in BOT_PREFIXES:
        if table_name.startswith(prefix):
            return True
    return False

# Database connection
DB_URL = os.environ.get('DATABASE_URL', '')
if not DB_URL:
    print("❌ DATABASE_URL not set")
    sys.exit(1)

try:
    from sqlalchemy import create_engine, text
    engine = create_engine(DB_URL, pool_pre_ping=True)

    print(f"🔍 Analyzing database: {DB_URL.split('@')[1].split('/')[0]}")
    print("=" * 70)

    with engine.connect() as conn:
        # Get all tables with sizes
        result = conn.execute(text("""
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes,
                (SELECT COUNT(*) FROM information_schema.columns
                 WHERE table_schema = schemaname AND table_name = tablename) AS columns
            FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """))

        tables = list(result)

        if not tables:
            print("No tables found")
            sys.exit(0)

        # Categorize tables
        bot_tables = []
        foreign_tables = []

        for row in tables:
            schema, table, size, size_bytes, cols = row
            full_name = f"{schema}.{table}" if schema != 'public' else table

            table_info = {
                'name': full_name,
                'table_only': table,
                'size': size,
                'size_bytes': size_bytes,
                'columns': cols
            }

            if belongs_to_this_bot(table):
                bot_tables.append(table_info)
            else:
                foreign_tables.append(table_info)

        # Print bot tables
        print("\n✅ THIS BOT'S TABLES (sob_*, volumized_radar_*):")
        print("=" * 70)
        print(f"{'TABLE':<40} {'SIZE':<12} {'COLUMNS':<10}")
        print("-" * 70)

        bot_total = 0
        for tbl in bot_tables:
            print(f"{tbl['name']:<40} {tbl['size']:<12} {tbl['columns']:<10}")
            bot_total += tbl['size_bytes']

        print("-" * 70)
        print(f"{'BOT TABLES TOTAL':<40} {bot_total / (1024**2):.2f} MB")

        # Print foreign tables
        if foreign_tables:
            print("\n⚠️  FOREIGN TABLES (NOT this bot!):")
            print("=" * 70)
            print(f"{'TABLE':<40} {'SIZE':<12} {'COLUMNS':<10}")
            print("-" * 70)

            foreign_total = 0
            for tbl in foreign_tables:
                print(f"{tbl['name']:<40} {tbl['size']:<12} {tbl['columns']:<10}")
                foreign_total += tbl['size_bytes']

            print("-" * 70)
            print(f"{'FOREIGN TABLES TOTAL':<40} {foreign_total / (1024**2):.2f} MB")
        else:
            print("\n✅ No foreign tables found!")

        total_bytes = bot_total + (foreign_total if foreign_tables else 0)
        print("\n" + "=" * 70)
        print(f"GRAND TOTAL: {total_bytes / (1024**2):.2f} MB")
        print(f"  - This bot: {bot_total / (1024**2):.2f} MB ({bot_total/total_bytes*100:.1f}%)")
        if foreign_tables:
            print(f"  - Foreign:  {foreign_total / (1024**2):.2f} MB ({foreign_total/total_bytes*100:.1f}%)")
        print("=" * 70)

        # Row counts for largest tables
        all_tables = bot_tables + foreign_tables
        all_tables.sort(key=lambda x: x['size_bytes'], reverse=True)

        print("\n" + "=" * 70)
        print("ROW COUNTS (TOP 15 TABLES BY SIZE):")
        print("=" * 70)
        print(f"{'TABLE':<40} {'ROWS':>15} {'SIZE':<12} {'TYPE':<10}")
        print("-" * 70)

        for tbl in all_tables[:15]:
            try:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {tbl['name']}"))
                count = count_result.scalar()
                tbl_type = '✅ BOT' if belongs_to_this_bot(tbl['table_only']) else '⚠️ FOREIGN'
                print(f"{tbl['name']:<40} {count:>15,} {tbl['size']:<12} {tbl_type}")
            except Exception as e:
                tbl_type = '✅ BOT' if belongs_to_this_bot(tbl['table_only']) else '⚠️ FOREIGN'
                print(f"{tbl['name']:<40} {'ERROR':>15} {tbl['size']:<12} {tbl_type}")

        # Recommendations
        print("\n" + "=" * 70)
        print("🧹 CLEANUP RECOMMENDATIONS:")
        print("=" * 70)

        # 1. Foreign tables (PRIORITY!)
        if foreign_tables:
            print("\n🚨 PRIORITY: FOREIGN TABLES (not used by this bot)")
            print("-" * 70)
            foreign_total_mb = sum(t['size_bytes'] for t in foreign_tables) / (1024**2)
            print(f"Total foreign data: {foreign_total_mb:.2f} MB")
            print("\n⚠️  SAFE TO DELETE (these don't belong to svv-webhook-bot):")

            for tbl in foreign_tables:
                if tbl['size_bytes'] > 1024*1024:  # > 1 MB
                    print(f"\n   DROP TABLE IF EXISTS {tbl['name']} CASCADE;")
                    print(f"   -- Saves {tbl['size']}")

            print("\n💡 To delete ALL foreign tables at once:")
            print("   -- Copy-paste this into psql:")
            for tbl in foreign_tables:
                print(f"   DROP TABLE IF EXISTS {tbl['name']} CASCADE;")

        # 2. Bot tables cleanup
        print("\n\n✅ THIS BOT'S TABLES - Cleanup recommendations:")
        print("-" * 70)

        for tbl in bot_tables:
            name = tbl['name']
            table_only = tbl['table_only']
            size_mb = tbl['size_bytes'] / (1024**2)

            # Event logs
            if 'event_log' in table_only.lower() and size_mb > 50:
                print(f"\n⚠️  {name} ({tbl['size']})")
                print(f"   Event logs grow indefinitely. Keep last 30 days:")
                print(f"   DELETE FROM {name} WHERE timestamp < NOW() - INTERVAL '30 days';")

            # Trade archive
            if 'trade_archive' in table_only.lower() and size_mb > 10:
                try:
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {name}"))
                    count = count_result.scalar()
                    print(f"\n📦 {name} ({tbl['size']}, {count:,} rows)")
                    print(f"   ML training data. If not needed:")
                    print(f"   DELETE FROM {name};")
                except:
                    pass

            # Klines/OB data
            if any(x in table_only.lower() for x in ['kline', 'order_block', 'ob_snapshot']) and size_mb > 20:
                print(f"\n📊 {name} ({tbl['size']})")
                print(f"   Cache data. Keep last 7 days:")
                print(f"   DELETE FROM {name} WHERE timestamp < NOW() - INTERVAL '7 days';")

        print("\n" + "=" * 70)
        print("✅ Analysis complete!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
