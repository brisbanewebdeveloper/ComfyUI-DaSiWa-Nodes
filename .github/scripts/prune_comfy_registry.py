"""Keep the newly published Comfy Registry version and four deprecated predecessors."""

import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.comfy.org"
ACTIVE = "NodeVersionStatusActive"
DELETED = "NodeVersionStatusDeleted"


def request(method, path, token=None, payload=None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Registry {method} {path}: HTTP {error.code}") from error


def list_versions(node):
    versions = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"nodeId": node, "page": page, "pageSize": 100})
        result = request("GET", f"/versions?{query}")
        if not isinstance(result, dict) or not isinstance(result.get("versions"), list):
            raise RuntimeError("Unexpected registry version listing")
        versions.extend(result["versions"])
        if page >= result["totalPages"]:
            if len(versions) != result["total"]:
                raise RuntimeError("Incomplete registry version listing; refusing to prune")
            return versions
        page += 1


def prune(node, publisher, current, token):
    versions = []
    for attempt in range(20):
        versions = [v for v in list_versions(node) if v["status"] != DELETED]
        matches = [v for v in versions if v["version"] == current]
        if len(matches) == 1 and matches[0]["status"] == ACTIVE:
            break
        if len(matches) == 1 and matches[0]["status"] == "NodeVersionStatusPending" and attempt < 19:
            time.sleep(15)
            continue
        raise RuntimeError(f"Published version {current} is not active yet; refusing to prune")
    versions.sort(key=lambda v: (v["createdAt"], v["version"]), reverse=True)
    if versions[0]["version"] != current:
        raise RuntimeError("Published version is not the newest; refusing to prune")
    if len({v["id"] for v in versions}) != len(versions):
        raise RuntimeError("Duplicate version IDs; refusing to prune")
    prefix = f"/publishers/{urllib.parse.quote(publisher, safe='')}/nodes/{urllib.parse.quote(node, safe='')}/versions/"
    for version in versions[1:5]:
        if version["deprecated"]:
            continue
        path = prefix + urllib.parse.quote(version["id"], safe="")
        request("PUT", path, token, {"deprecated": True})
        print(f"Deprecated {version['version']}")
    for version in versions[5:]:
        path = prefix + urllib.parse.quote(version["id"], safe="")
        request("DELETE", path, token)
        print(f"Unpublished {version['version']}")
    remaining = [v for v in list_versions(node) if v["status"] != DELETED]
    expected = {v["id"] for v in versions[:5]}
    if {v["id"] for v in remaining} != expected or any(v["version"] != current and not v["deprecated"] for v in remaining):
        raise RuntimeError("Registry cleanup verification failed")
    print(f"Verified {current} and {len(remaining) - 1} deprecated predecessors remain")


if __name__ == "__main__":
    with open("pyproject.toml", "rb") as file:
        config = tomllib.load(file)
    token = os.environ.get("COMFY_REGISTRY_TOKEN")
    if not token:
        sys.exit("COMFY_REGISTRY_TOKEN is required")
    prune(config["project"]["name"], config["tool"]["comfy"]["PublisherId"], config["project"]["version"], token)
