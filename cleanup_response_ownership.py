"""Administrative helper for expiring old response-ownership records.

This script is intentionally manual. It does not schedule itself and does not
delete interaction or feedback data.
"""
import argparse
import sqlite3

from database import create_connection, create_tables, cleanup_old_response_ownership

DB_PATH = "data/chatbot.db"


def main():
    parser = argparse.ArgumentParser(
        description="Remove response-ownership records older than the feedback window."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Retention window in days (default: 30).",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="Path to the SQLite database (default: data/chatbot.db).",
    )
    args = parser.parse_args()

    create_tables(args.db)
    conn = create_connection(args.db)
    try:
        deleted = cleanup_old_response_ownership(conn, days=args.days)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Deleted {deleted} response-ownership records older than {args.days} days.")


if __name__ == "__main__":
    main()