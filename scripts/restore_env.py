"""Rebuild .env from the values the running stack was started with.

The local .env holds secrets that exist nowhere else: the Langfuse key pair and,
critically, ENCRYPTION_KEY — change that and Langfuse can no longer decrypt the
data already in its database. If .env is lost or reset while the containers are
still up, every value can be recovered from their environment.

    uv run python scripts/restore_env.py            # rewrite .env
    uv run python scripts/restore_env.py --check    # report drift, write nothing

This only works while the containers exist. Once they are removed the secrets
are gone for good, so keep a copy of .env somewhere safe.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = "secai-langfuse-web"
APP_DB = "secai-postgres"


def container_env(container: str) -> dict[str, str]:
    """The environment a running container was started with."""
    result = subprocess.run(
        ["docker", "inspect", container, "--format", "{{json .Config.Env}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    entries: list[str] = json.loads(result.stdout)
    return dict(entry.split("=", 1) for entry in entries if "=" in entry)


def recover() -> dict[str, str]:
    """Every .env value that must match the live stack."""
    web = container_env(WEB)
    app_db = container_env(APP_DB)

    public_key = web["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"]
    secret_key = web["LANGFUSE_INIT_PROJECT_SECRET_KEY"]
    langfuse_db_password = re.sub(r"^.*://langfuse:([^@]+)@.*$", r"\1", web["DATABASE_URL"])

    return {
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_SECRET_KEY": secret_key,
        "LANGFUSE_AUTH": base64.b64encode(f"{public_key}:{secret_key}".encode()).decode(),
        "LANGFUSE_SALT": web["SALT"],
        "LANGFUSE_ENCRYPTION_KEY": web["ENCRYPTION_KEY"],
        "LANGFUSE_NEXTAUTH_SECRET": web["NEXTAUTH_SECRET"],
        "LANGFUSE_POSTGRES_PASSWORD": langfuse_db_password,
        "CLICKHOUSE_PASSWORD": web["CLICKHOUSE_PASSWORD"],
        "MINIO_ROOT_PASSWORD": web["LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY"],
        "REDIS_PASSWORD": web["REDIS_AUTH"],
        "SECAI_DB_PASSWORD": app_db["POSTGRES_PASSWORD"],
        # Only meaningful on a fresh database, but recovering them keeps .env
        # honest about which account actually exists.
        "LANGFUSE_INIT_USER_EMAIL": web["LANGFUSE_INIT_USER_EMAIL"],
        "LANGFUSE_INIT_USER_PASSWORD": web["LANGFUSE_INIT_USER_PASSWORD"],
    }


def read_env(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report which values drifted from the running stack; write nothing",
    )
    args = parser.parse_args()

    env_path = REPO_ROOT / ".env"
    template = REPO_ROOT / ".env.example"

    try:
        recovered = recover()
    except (subprocess.CalledProcessError, KeyError, FileNotFoundError) as exc:
        print(f"cannot read the running containers: {exc}", file=sys.stderr)
        print("Start the stack first (make up-core), or restore .env from backup.", file=sys.stderr)
        return 1

    current = read_env(env_path)
    drifted = [k for k, v in recovered.items() if current.get(k) != v]

    if args.check:
        if drifted:
            print("drifted from the running stack:")
            for key in drifted:
                print(f"  {key}")
            return 1
        print(".env matches the running stack")
        return 0

    if not drifted:
        print(".env already matches the running stack; nothing to do")
        return 0

    text = template.read_text(encoding="utf-8")
    for key, value in recovered.items():
        pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(f"{key}={value}", text, count=1)
        else:
            text += f"\n{key}={value}\n"

    # Preserve any non-secret local overrides already present (e.g. VLLM_MODEL).
    for key, value in current.items():
        if key in recovered:
            continue
        pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(f"{key}={value}", text, count=1)

    env_path.write_text(text, encoding="utf-8")
    print(f"rewrote .env from the running stack ({len(drifted)} value(s) restored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
