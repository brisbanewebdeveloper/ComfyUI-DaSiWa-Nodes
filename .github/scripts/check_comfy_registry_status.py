#!/usr/bin/env python3
"""Report the Comfy Registry status for one published node version."""

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

API_BASE_URL = "https://api.comfy.org/nodes"
REGISTRY_BASE_URL = "https://registry.comfy.org/nodes"
ACTIVE_STATUS = "NodeVersionStatusActive"


def write_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")


def response_detail(payload: object) -> str:
    if not isinstance(payload, dict):
        return "The Registry API returned a non-object response."

    details = []
    for field in ("status_reason", "comfy_node_extract_status"):
        value = payload.get(field)
        if value:
            details.append(f"{field}: {value}")
    return "; ".join(details) or "The Registry API did not provide a reason."


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {Path(sys.argv[0]).name} NODE_ID VERSION", file=sys.stderr)
        return 2

    node_id, version = sys.argv[1:]
    api_url = f"{API_BASE_URL}/{quote(node_id, safe='')}/versions/{quote(version, safe='')}"
    registry_url = f"{REGISTRY_BASE_URL}/{quote(node_id, safe='')}/versions/{quote(version, safe='')}"

    try:
        with urlopen(api_url, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:1000]
        message = f"Registry API returned HTTP {error.code}: {detail or error.reason}"
        print(message, file=sys.stderr)
        write_summary([
            "## Comfy Registry status",
            "",
            f"- Node: `{node_id}`",
            f"- Version: `{version}`",
            f"- Registry URL: {registry_url}",
            f"- Error: {message}",
        ])
        return 1
    except (URLError, TimeoutError) as error:
        message = f"Could not reach the Registry API: {error}"
        print(message, file=sys.stderr)
        write_summary([
            "## Comfy Registry status",
            "",
            f"- Node: `{node_id}`",
            f"- Version: `{version}`",
            f"- Registry URL: {registry_url}",
            f"- Error: {message}",
        ])
        return 1
    except json.JSONDecodeError as error:
        message = f"Registry API returned invalid JSON: {error}"
        print(message, file=sys.stderr)
        write_summary([
            "## Comfy Registry status",
            "",
            f"- Node: `{node_id}`",
            f"- Version: `{version}`",
            f"- Registry URL: {registry_url}",
            f"- Error: {message}",
        ])
        return 1

    status = payload.get("status") if isinstance(payload, dict) else None
    detail = response_detail(payload)
    print(f"Comfy Registry status for {node_id} {version}: {status or 'missing'}")
    print(f"Details: {detail}")
    print(f"Registry URL: {registry_url}")
    write_summary([
        "## Comfy Registry status",
        "",
        f"- Node: `{node_id}`",
        f"- Version: `{version}`",
        f"- Status: `{status or 'missing'}`",
        f"- Details: {detail}",
        f"- Registry URL: {registry_url}",
    ])

    if status == ACTIVE_STATUS:
        return 0

    print(
        f"::error title=Comfy Registry status::{node_id} {version} is "
        f"{status or 'missing'}. {detail}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
