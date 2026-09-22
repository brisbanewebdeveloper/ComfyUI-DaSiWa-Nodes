import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "audit_comfy_registry_status.py"


def _module():
    spec = importlib.util.spec_from_file_location("registry_status_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_versions_redacts_status_reason_and_counts_findings():
    module = _module()
    findings = [
        {"issue_type": "python_command_injection_risk"},
        {"issue_type": "python_command_injection_risk"},
        {"issue_type": "python_network_operations"},
    ]
    payload = {
        "versions": [
            {
                "version": "0.4.37",
                "status": "NodeVersionStatusBanned",
                "status_reason": json.dumps({
                    "message": "policy-v0.4: rce-remote-code — internal scan detail must not be printed",
                    "statusHistory": [{"message": json.dumps(findings)}],
                }),
            }
        ]
    }

    report = module.summarize_versions(payload)

    assert report == {
        "node_id": "ComfyUI-DaSiWa-Nodes",
        "versions": [{
            "version": "0.4.37",
            "status": "NodeVersionStatusBanned",
            "policy": "policy-v0.4: rce-remote-code",
            "finding_counts": {
                "python_command_injection_risk": 2,
                "python_network_operations": 1,
            },
        }],
    }
    assert "status_reason" not in json.dumps(report)


def test_summarize_versions_rejects_malformed_payload():
    module = _module()

    try:
        module.summarize_versions({"versions": "not-a-list"})
    except ValueError as error:
        assert str(error) == "Registry response has no versions list."
    else:
        raise AssertionError("Expected malformed Registry payload to be rejected.")


def test_require_active_version_accepts_only_the_requested_active_release():
    module = _module()
    report = {
        "versions": [
            {"version": "0.4.37", "status": "NodeVersionStatusBanned"},
            {"version": "0.4.38", "status": "NodeVersionStatusActive"},
        ]
    }

    assert module.require_active_version(report, "0.4.38") is None

    try:
        module.require_active_version(report, "0.4.37")
    except ValueError as error:
        assert str(error) == "Registry version 0.4.37 is NodeVersionStatusBanned, not active."
    else:
        raise AssertionError("Expected inactive release to be rejected.")
