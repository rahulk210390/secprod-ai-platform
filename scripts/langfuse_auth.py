"""Print LANGFUSE_AUTH, the base64 Basic-auth value the OTel Collector needs.

The keys are read from the environment / .env via the normal settings tree.
"""

from __future__ import annotations

import sys

from secai.config import get_settings


def main() -> int:
    langfuse = get_settings().langfuse
    if not langfuse.is_configured:
        print(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set in .env",
            file=sys.stderr,
        )
        return 1
    print(f"LANGFUSE_AUTH={langfuse.basic_auth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
