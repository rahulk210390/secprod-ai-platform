"""Print freshly generated values for the Langfuse self-hosted secrets.

Copy the output into .env. Nothing is written to disk by this script.
"""

from __future__ import annotations

import secrets


def main() -> None:
    print(f"LANGFUSE_SALT={secrets.token_hex(16)}")
    print(f"LANGFUSE_ENCRYPTION_KEY={secrets.token_hex(32)}")
    print(f"LANGFUSE_NEXTAUTH_SECRET={secrets.token_hex(32)}")
    print(f"LANGFUSE_POSTGRES_PASSWORD={secrets.token_hex(12)}")
    print(f"CLICKHOUSE_PASSWORD={secrets.token_hex(12)}")
    print(f"MINIO_ROOT_PASSWORD={secrets.token_hex(12)}")
    print(f"REDIS_PASSWORD={secrets.token_hex(12)}")
    print(f"SECAI_DB_PASSWORD={secrets.token_hex(12)}")


if __name__ == "__main__":
    main()
