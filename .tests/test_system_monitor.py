import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault("psutil", types.ModuleType("psutil"))

MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_system_monitor.py"
spec = importlib.util.spec_from_file_location("nodes_system_monitor", MODULE_PATH)
assert spec is not None and spec.loader is not None
system_monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(system_monitor)


def test_backend_telemetry_is_disabled_without_opt_in(monkeypatch):
    monkeypatch.delenv("DASWA_SYSTEM_MONITOR", raising=False)

    assert system_monitor._monitor_enabled() is False


def test_backend_telemetry_accepts_explicit_enable_values(monkeypatch):
    for value in ("1", "true", "yes", "on", "enable", "enabled"):
        monkeypatch.setenv("DASWA_SYSTEM_MONITOR", value)
        assert system_monitor._monitor_enabled() is True


def test_backend_telemetry_rejects_other_values(monkeypatch):
    for value in ("0", "false", "no", "off", "disable", "disabled", "unexpected"):
        monkeypatch.setenv("DASWA_SYSTEM_MONITOR", value)
        assert system_monitor._monitor_enabled() is False


def test_nvidia_parser_keeps_each_gpu_id_and_metrics():
    output = "0, GPU-a, RTX A, 55, 1024, 8192, 62\n1, GPU-b, RTX B, 21, 2048, 16384, 51\n"

    gpus = system_monitor._nvidia_gpus(lambda _: output)

    assert [gpu["id"] for gpu in gpus] == ["NVIDIA:0", "NVIDIA:1"]
    assert gpus[1]["vendor"] == "NVIDIA"
    assert gpus[0]["memory_used"] == 1024 * 1024 * 1024
    assert gpus[1]["memory_percent"] == 12.5


def test_amd_parser_keeps_vendor_and_device_identifier():
    output = '{"card0": {"Card series": "AMD Radeon", "GPU use (%)": "42", "Used Memory (MiB)": "1024", "Total Memory (MiB)": "8192", "Temperature (Sensor edge) (C)": "61"}}'

    gpu = system_monitor._amd_gpus(lambda _: output)[0]

    assert gpu["id"] == "AMD:card0"
    assert gpu["vendor"] == "AMD"
    assert gpu["utilization"] == 42
    assert gpu["memory_percent"] == 12.5


def test_intel_discovery_uses_drm_card_id(tmp_path):
    card = tmp_path / "card3" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    (card / "uevent").write_text("DRIVER=i915\n")

    gpu = system_monitor._intel_gpus(tmp_path)[0]

    assert gpu["id"] == "Intel:3"
    assert gpu["name"] == "Intel i915"


def test_windows_parser_detects_intel_and_amd_adapters_without_vendor_tools():
    output = '[{"Name":"Intel(R) Arc","PNPDeviceID":"PCI\\\\VEN_8086&DEV_7D55","VideoProcessor":"Intel Arc","AdapterRAM":8589934592},{"Name":"AMD Radeon","PNPDeviceID":"PCI\\\\VEN_1002&DEV_744C","VideoProcessor":"AMD Radeon","AdapterRAM":17179869184}]'

    gpus = system_monitor._windows_gpus(lambda _: output)

    assert [gpu["vendor"] for gpu in gpus] == ["Intel", "AMD"]
    assert gpus[0]["id"].startswith("Intel:PCI\\VEN_8086")
    assert gpus[1]["memory_total"] == 17179869184


def test_snapshot_reports_partition_capacity_and_io_rates(monkeypatch):
    monitor = system_monitor.DaSiWaSystemMonitor()
    partition = types.SimpleNamespace(device="/dev/nvme0n1p2", mountpoint="/", fstype="ext4")
    usage = types.SimpleNamespace(used=25 * 1024**3, total=100 * 1024**3, percent=25.0)
    counters = [
        {"nvme0n1": types.SimpleNamespace(read_bytes=100, write_bytes=200)},
        {"nvme0n1": types.SimpleNamespace(read_bytes=100 + 4 * 1024**2, write_bytes=200 + 2 * 1024**2)},
    ]
    monotonic = iter((10.0, 12.0))

    monkeypatch.setattr(system_monitor.psutil, "virtual_memory", lambda: types.SimpleNamespace(used=1, total=2, percent=50.0), raising=False)
    monkeypatch.setattr(system_monitor.psutil, "swap_memory", lambda: types.SimpleNamespace(used=0, total=1, percent=0.0), raising=False)
    monkeypatch.setattr(system_monitor.psutil, "cpu_percent", lambda interval=None: 10.0, raising=False)
    monkeypatch.setattr(system_monitor.psutil, "cpu_count", lambda logical=True: 8, raising=False)
    monkeypatch.setattr(system_monitor.psutil, "disk_partitions", lambda all=False: [partition], raising=False)
    monkeypatch.setattr(system_monitor.psutil, "disk_usage", lambda path: usage, raising=False)
    monkeypatch.setattr(system_monitor.psutil, "disk_io_counters", lambda perdisk=False: counters.pop(0), raising=False)
    monkeypatch.setattr(system_monitor, "time", types.SimpleNamespace(monotonic=lambda: next(monotonic)), raising=False)
    monkeypatch.setattr(monitor, "gpu_info", lambda: [])

    monitor.snapshot()
    disk = monitor.snapshot()["disks"][0]

    assert disk == {
        "path": "/", "device": "/dev/nvme0n1p2", "fstype": "ext4",
        "used": 25 * 1024**3, "total": 100 * 1024**3, "percent": 25.0,
        "read_mb_s": 2.0, "write_mb_s": 1.0,
    }


def test_snapshot_keeps_only_system_and_comfyui_filesystems(monkeypatch):
    monitor = system_monitor.DaSiWaSystemMonitor()
    partitions = [
        types.SimpleNamespace(device="/dev/nvme0n1p1", mountpoint="/", fstype="ext4"),
        types.SimpleNamespace(device="/dev/sda1", mountpoint="/mnt/comfy", fstype="ext4"),
        types.SimpleNamespace(device="/dev/sdb1", mountpoint="/mnt/comfy/models", fstype="ext4"),
    ]

    monkeypatch.setattr(system_monitor.psutil, "disk_partitions", lambda all=False: partitions, raising=False)
    monkeypatch.setattr(monitor, "_comfyui_path", lambda: "/mnt/comfy/custom_nodes/DaSiWa")

    assert [partition.mountpoint for partition in monitor._monitored_partitions()] == ["/", "/mnt/comfy"]


def test_snapshot_deduplicates_comfyui_bind_mount_on_system_disk(monkeypatch):
    monitor = system_monitor.DaSiWaSystemMonitor()
    partitions = [
        types.SimpleNamespace(device="/dev/nvme0n1p1", mountpoint="/", fstype="ext4"),
        types.SimpleNamespace(device="/dev/nvme0n1p1", mountpoint="/home", fstype="ext4"),
    ]

    monkeypatch.setattr(system_monitor.psutil, "disk_partitions", lambda all=False: partitions, raising=False)
    monkeypatch.setattr(monitor, "_comfyui_path", lambda: "/home/comfyui/custom_nodes/DaSiWa")

    assert [partition.mountpoint for partition in monitor._monitored_partitions()] == ["/"]
