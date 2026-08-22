#!/usr/bin/env python3
"""Verify the tracked-file and concrete-source claims in EXPORT_SOURCES_V2.

The public export intentionally omits private corpora, models and local paths.
Pass the private workspace root only when verifying concrete source hashes:

    python scripts/verify_export_sources_v2.py --authoritative-root /path/to/workspace
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


MANIFEST_PATH = "provenance/EXPORT_SOURCES_V2.json"
ABSOLUTE_PATH = re.compile(r"(?:^|[\"'])/(?:home|tmp|mnt)/|[A-Za-z]:[\\/]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_relative(path: str) -> bool:
    pure = PurePosixPath(path)
    return not pure.is_absolute() and ".." not in pure.parts


def walk_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-root", type=Path,
                        help="private workspace root; enables concrete source verification")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    data = json.loads((repo / MANIFEST_PATH).read_text(encoding="utf-8"))

    if data.get("schema") != "hex-agent-public-export-sources-v2":
        raise ValueError("unexpected manifest schema")
    if data.get("version") != 2:
        raise ValueError("unexpected manifest version")
    for text in walk_strings(data):
        if ABSOLUTE_PATH.search(text):
            raise ValueError("manifest contains a forbidden absolute local path")

    declared: set[str] = set()
    for entry in data["entries"]:
        for item in entry.get("files", []):
            public_path = item["public_path"]
            if not valid_relative(public_path):
                raise ValueError(f"invalid public path: {public_path}")
            if public_path in declared:
                raise ValueError(f"duplicate public path: {public_path}")
            declared.add(public_path)
            actual = repo / public_path
            if not actual.is_file():
                raise FileNotFoundError(f"missing public file: {public_path}")
            if sha256(actual) != item["public_sha256"]:
                raise ValueError(f"public hash mismatch: {public_path}")

            source_path = item.get("authoritative_path")
            source_hash = item.get("authoritative_sha256")
            if (source_path is None) != (source_hash is None):
                raise ValueError(f"incomplete authoritative hash pair: {public_path}")
            if source_path is not None:
                if not valid_relative(source_path):
                    raise ValueError(f"invalid authoritative path: {source_path}")
                if args.authoritative_root is not None:
                    source = args.authoritative_root.resolve() / source_path
                    if not source.is_file():
                        raise FileNotFoundError(f"missing authoritative file: {source_path}")
                    if sha256(source) != source_hash:
                        raise ValueError(f"authoritative hash mismatch: {source_path}")

        for evidence in entry.get("source_evidence", []):
            path = evidence.get("authoritative_path")
            expected = evidence.get("authoritative_sha256")
            if path is None or expected is None or not valid_relative(path):
                raise ValueError(f"invalid source evidence in {entry['id']}")
            if args.authoritative_root is not None:
                source = args.authoritative_root.resolve() / path
                if not source.is_file() or sha256(source) != expected:
                    raise ValueError(f"source evidence mismatch: {path}")

    tracked = set(subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True).splitlines())
    exempt = set(data["coverage"]["self_referential_exemptions"])
    if tracked - exempt != declared:
        missing = sorted((tracked - exempt) - declared)
        extra = sorted(declared - tracked)
        raise ValueError(f"tracked-file coverage mismatch; missing={missing}, extra={extra}")

    print(f"verified {len(declared)} declared public files; {len(tracked)} tracked files")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
