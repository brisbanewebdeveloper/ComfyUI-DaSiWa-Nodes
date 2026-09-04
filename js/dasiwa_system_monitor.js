import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT_NAME = "dasiwa.system_monitor";
const ROOT_ID = "dasiwa-system-monitor";
const PANEL_ID = "dasiwa-system-monitor-panel";
const SETTINGS_KEY = "dasiwa.system_monitor.settings";
const HISTORY_LENGTH = 60;
const DEFAULT_SETTINGS = {
    enabled: true, mode: "lite", placement: "top", dockSide: "top", orientation: "horizontal",
    widgets: {}, toolbarIndex: null, x: 24, y: 60,
};

let settings = loadSettings();
let latestSnapshot = null;
let history = [];

function loadSettings() {
    try {
        const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY));
        return {
            enabled: saved?.enabled !== false,
            mode: saved?.mode === "full" ? "full" : "lite",
            placement: saved?.placement === "floating" ? "floating" : "top",
            dockSide: ["top", "left", "right"].includes(saved?.dockSide) ? saved.dockSide : "top",
            orientation: saved?.orientation === "vertical" ? "vertical" : "horizontal",
            widgets: saved?.widgets && typeof saved.widgets === "object" ? saved.widgets : {},
            toolbarIndex: Number.isInteger(saved?.toolbarIndex) ? saved.toolbarIndex : null,
            x: Number.isFinite(saved?.x) ? saved.x : DEFAULT_SETTINGS.x,
            y: Number.isFinite(saved?.y) ? saved.y : DEFAULT_SETTINGS.y,
        };
    } catch {
        return { ...DEFAULT_SETTINGS };
    }
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function applyOrientation(root) {
    root.classList.toggle("is-vertical", settings.orientation === "vertical");
    root.classList.toggle("is-horizontal", settings.orientation !== "vertical");
}

function hideSideDocks() {
    ["left", "right"].forEach((side) => {
        const dock = document.getElementById(`dasiwa-monitor-dock-${side}`);
        if (dock) dock.hidden = true;
    });
}

function placePanel(root, side = settings.dockSide) {
    settings.dockSide = side;
    settings.orientation = side === "top" ? "horizontal" : "vertical";
    applyOrientation(root);
    root.classList.toggle("is-dock-top", side === "top");
    root.classList.toggle("is-dock-left", side === "left");
    root.classList.toggle("is-dock-right", side === "right");
    root.classList.remove("is-floating");
    root.style.left = "";
    root.style.top = "";
    root.style.transform = "";
    hideSideDocks();
    if (side === "left" || side === "right") {
        const dock = document.getElementById(`dasiwa-monitor-dock-${side}`);
        if (!dock) return false;
        dock.hidden = false;
        dock.appendChild(root);
        return true;
    }
    const legacyTopbar = document.querySelector('[data-testid="legacy-topbar-container"] > .flex');
    if (legacyTopbar) {
        const children = [...legacyTopbar.children];
        const before = Number.isInteger(settings.toolbarIndex) ? children[settings.toolbarIndex] : children[0];
        legacyTopbar.insertBefore(root, before ?? null);
        return true;
    }
    const extensionsButton = document.querySelector('button[aria-label="Extensions"]');
    if (!extensionsButton?.parentElement) return false;
    extensionsButton.before(root);
    return true;
}

function saveToolbarPosition(root) {
    settings.toolbarIndex = [...root.parentElement.children].indexOf(root);
    saveSettings();
}

function dockPanel(root, pointerX = 0) {
    settings.dockSide = "top";
    if (!placePanel(root, "top")) return false;
    const toolbar = root.parentElement;
    const before = [...toolbar.children].find((child) => child !== root && pointerX < child.getBoundingClientRect().left + child.offsetWidth / 2);
    if (before) toolbar.insertBefore(root, before);
    else toolbar.appendChild(root);
    settings.placement = "top";
    saveToolbarPosition(root);
    return true;
}

function floatPanel(root, x, y) {
    document.body.appendChild(root);
    root.classList.remove("is-dock-top", "is-dock-left", "is-dock-right");
    root.classList.add("is-floating");
    [...root.querySelector(`#${PANEL_ID}`)?.children ?? []].forEach((metric) => metric.hidden = false);
    settings.placement = "floating";
    settings.x = Math.max(8, Math.min(x, window.innerWidth - root.offsetWidth - 8));
    settings.y = Math.max(48, Math.min(y, window.innerHeight - root.offsetHeight - 8));
    root.style.transform = "";
    root.style.left = `${Math.round(settings.x)}px`;
    root.style.top = `${Math.round(settings.y)}px`;
}

function dockTarget(side) {
    return document.getElementById(`dasiwa-monitor-dock-target-${side}`);
}

function dockSideAtPoint(x, y) {
    if (y <= 48) return "top";
    if (x <= 64) return "left";
    if (x >= window.innerWidth - 64) return "right";
    return null;
}

function showDockTargets(activeSide = null) {
    ["top", "left", "right"].forEach((side) => {
        const target = dockTarget(side);
        target.hidden = false;
        target.classList.toggle("is-active", side === activeSide);
    });
}

function hideDockTargets() {
    ["top", "left", "right"].forEach((side) => {
        const target = dockTarget(side);
        target.hidden = true;
        target.classList.remove("is-active");
    });
}

function enablePanelDrag(root) {
    const handle = root.querySelector(".dasiwa-monitor-drag-handle");
    handle.title = "Drag to float; drop on Dock to top to return it to the toolbar";
    handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        const bounds = root.getBoundingClientRect();
        const offsetX = event.clientX - bounds.left;
        const offsetY = event.clientY - bounds.top;
        let dragging = false;
        const move = (moveEvent) => {
            if (!dragging && Math.hypot(moveEvent.clientX - event.clientX, moveEvent.clientY - event.clientY) < 4) return;
            if (!dragging) {
                dragging = true;
                floatPanel(root, moveEvent.clientX - offsetX, moveEvent.clientY - offsetY);
                showDockTargets(dockSideAtPoint(moveEvent.clientX, moveEvent.clientY));
            } else {
                floatPanel(root, moveEvent.clientX - offsetX, moveEvent.clientY - offsetY);
                showDockTargets(dockSideAtPoint(moveEvent.clientX, moveEvent.clientY));
            }
        };
        const end = (endEvent) => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
            if (dragging) {
                if (endEvent.type === "pointercancel") {
                    saveSettings();
                    hideDockTargets();
                    return;
                }
                const side = dockSideAtPoint(endEvent.clientX, endEvent.clientY);
                if (side === "top") dockPanel(root, endEvent.clientX);
                else if (side) {
                    settings.dockSide = side;
                    settings.placement = "top";
                    placePanel(root, side);
                    saveSettings();
                }
                else saveSettings();
                hideDockTargets();
            }
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    });
}

function bytes(value) {
    if (value === null || value === undefined) return "n/a";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let index = 0;
    while (value >= 1024 && index < units.length - 1) {
        value /= 1024;
        index++;
    }
    return `${value.toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function mbPerSec(value) {
    if (value === null || value === undefined) return "n/a";
    const rounded = Math.round(value * 10) / 10;
    if (rounded >= 1000) return `${(rounded / 1000).toFixed(1)} GB/s`;
    return `${rounded.toFixed(1).padStart(4, "\u00A0")} MB/s`;
}

function percent(value) {
    return value === null || value === undefined ? "n/a" : `${String(Math.round(value)).padStart(3, "\u00A0")}%`;
}

function celsius(value) {
    return value === null || value === undefined ? "n/a" : `${String(Math.round(value)).padStart(3, "\u00A0")}°`;
}

function meterFill(value) {
    return Math.max(0, Math.min(100, Number(value) || 0));
}

function metric(kind, label, rawValue, value, detail = "") {
    return `<div class="dasiwa-monitor-metric ${kind}" style="--fill:${meterFill(rawValue)}%" title="${detail}"><span>${label}</span><strong>${value}</strong></div>`;
}

function isWidgetVisible(id) {
    return settings.widgets[id] !== false;
}

function allSnapshotMetrics(snapshot) {
    return [
        { id: "cpu", kind: "cpu", label: "CPU", value: snapshot.cpu_percent, text: percent(snapshot.cpu_percent), detail: `${snapshot.cpu_count ?? "?"} threads` },
        { id: "ram", kind: "ram", label: "RAM", value: snapshot.ram.percent, text: percent(snapshot.ram.percent), detail: `${bytes(snapshot.ram.used)} / ${bytes(snapshot.ram.total)}` },
        { id: "swap", kind: "swap", label: "SWAP", value: snapshot.swap.percent, text: percent(snapshot.swap.percent), detail: `${bytes(snapshot.swap.used)} / ${bytes(snapshot.swap.total)}` },
        ...(snapshot.disks ?? []).flatMap((disk) => [
            { id: `disk:${disk.path}`, kind: "disk", label: `DISK Space ${disk.path}`, value: disk.percent, text: percent(disk.percent), detail: `${disk.device}: ${bytes(disk.used)} / ${bytes(disk.total)}` },
            { id: `disk-rd:${disk.path}`, kind: "disk-rd", label: `DISK Read ${disk.path}`, value: Math.min(disk.read_mb_s, 100), text: mbPerSec(disk.read_mb_s), detail: `${disk.path}: disk read throughput` },
            { id: `disk-wr:${disk.path}`, kind: "disk-wr", label: `DISK Write ${disk.path}`, value: Math.min(disk.write_mb_s, 100), text: mbPerSec(disk.write_mb_s), detail: `${disk.path}: disk write throughput` },
        ]),
        ...snapshot.gpus.flatMap((gpu) => [
            { id: `gpu-util:${gpu.id}`, kind: "gpu-util", label: `GPU${gpu.index} Util`, value: gpu.utilization, text: percent(gpu.utilization), detail: `${gpu.id} — ${gpu.name}; GPU utilization` },
            { id: `gpu-vram:${gpu.id}`, kind: "gpu-vram", label: `GPU${gpu.index} VRAM`, value: gpu.memory_percent, text: percent(gpu.memory_percent), detail: `${gpu.id} — ${gpu.name}; VRAM ${bytes(gpu.memory_used)} / ${bytes(gpu.memory_total)}` },
            { id: `gpu-temp:${gpu.id}`, kind: "gpu-temp", label: `GPU${gpu.index} Temp`, value: gpu.temperature, text: celsius(gpu.temperature), detail: `${gpu.id} — ${gpu.name}; GPU temperature` },
        ]),
    ];
}

function snapshotMetrics(snapshot) {
    return allSnapshotMetrics(snapshot).filter(({ id }) => isWidgetVisible(id));
}

function historyPoint(snapshot) {
    return Object.fromEntries(snapshotMetrics(snapshot).map(({ label, value }) => [label, meterFill(value)]));
}

function sparkline(label) {
    const values = history.map((point) => point[label]).filter((value) => Number.isFinite(value));
    if (!values.length) return "<div class=\"dasiwa-monitor-empty-graph\">Collecting history…</div>";
    const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1) * 100).toFixed(1)},${(100 - value).toFixed(1)}`).join(" ");
    return `<svg class="dasiwa-monitor-graph" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="${label} history"><polyline points="${points}" /></svg>`;
}

function renderLite(panel, snapshot) {
    panel.className = "dasiwa-monitor-display is-lite";
    panel.innerHTML = snapshotMetrics(snapshot).map(({ kind, label, value, text, detail }) => metric(kind, label, value, text, detail)).join("");
    requestAnimationFrame(() => fitPanel(panel));
}

function renderFull(panel, snapshot) {
    panel.className = "dasiwa-monitor-display is-full";
    panel.innerHTML = `
        <div class="dasiwa-monitor-full-header"><strong>System Monitor</strong><span>last ${history.length}s</span></div>
        <div class="dasiwa-monitor-full-grid">
            ${snapshotMetrics(snapshot).map(({ kind, label, value, text, detail }) => `
                <section class="dasiwa-monitor-full-metric ${kind}" title="${detail}">
                    <div><span>${label}</span><strong>${text}</strong></div>
                    <div class="dasiwa-monitor-graph-wrap">${sparkline(label)}</div>
                    <small>${detail}</small>
                    <i style="--fill:${meterFill(value)}%"></i>
                </section>`).join("")}
        </div>`;
    positionFullPanel(panel);
}

function positionFullPanel(panel) {
    panel.style.top = "calc(100% + 8px)";
    if (settings.dockSide === "left") {
        panel.style.left = "calc(100% + 8px)";
        panel.style.right = "auto";
    } else {
        panel.style.left = "auto";
        panel.style.right = "0";
    }
}

function render(snapshot = latestSnapshot) {
    const panel = document.getElementById(PANEL_ID);
    if (!panel || !snapshot) return;
    panel.hidden = !settings.enabled;
    if (!settings.enabled) return;
    if (settings.mode === "full") renderFull(panel, snapshot);
    else renderLite(panel, snapshot);
}

function fitPanel(panel) {
    const extensionsButton = document.querySelector('button[aria-label="Extensions"]');
    const metrics = [...panel.children];
    metrics.forEach((metric) => metric.hidden = false);
    if (settings.mode !== "lite" || settings.placement === "floating" || settings.dockSide !== "top") return;
    while (metrics.length && extensionsButton && panel.getBoundingClientRect().left < extensionsButton.getBoundingClientRect().right + 8) {
        metrics.pop().hidden = true;
    }
}

function closeMenu() {
    document.getElementById("dasiwa-monitor-settings-menu")?.remove();
}

function openMenu(button) {
    const existing = document.getElementById("dasiwa-monitor-settings-menu");
    if (existing) {
        existing.remove();
        return;
    }
    const root = button.closest(`#${ROOT_ID}`);
    const widgetControls = latestSnapshot ? allSnapshotMetrics(latestSnapshot).map(({ id, label }) =>
        `<label><input type="checkbox" data-widget-id="${id}" ${isWidgetVisible(id) ? "checked" : ""}> ${label}</label>`
    ).join("") : "<small>Waiting for telemetry…</small>";
    const menu = document.createElement("div");
    menu.id = "dasiwa-monitor-settings-menu";
    menu.setAttribute("role", "menu");
    menu.innerHTML = `
        <label><input type="checkbox" ${settings.enabled ? "checked" : ""}> Show system monitor</label>
        <div class="dasiwa-monitor-menu-label">Display mode</div>
        <label><input type="radio" name="dasiwa-monitor-mode" value="lite" ${settings.mode === "lite" ? "checked" : ""}> Lite <small>toolbar meters</small></label>
        <label><input type="radio" name="dasiwa-monitor-mode" value="full" ${settings.mode === "full" ? "checked" : ""}> Full <small>all metrics + 60s graphs</small></label>
        <div class="dasiwa-monitor-menu-label">Dock</div>
        <label><input type="radio" name="dasiwa-monitor-dock" value="top" ${settings.dockSide === "top" ? "checked" : ""}> Top toolbar</label>
        <label><input type="radio" name="dasiwa-monitor-dock" value="left" ${settings.dockSide === "left" ? "checked" : ""}> Left side</label>
        <label><input type="radio" name="dasiwa-monitor-dock" value="right" ${settings.dockSide === "right" ? "checked" : ""}> Right side</label>
        <div class="dasiwa-monitor-menu-note">Layout is automatic: horizontal at top; vertical at either side.</div>
        <div class="dasiwa-monitor-menu-label">Widgets</div>
        <div class="dasiwa-monitor-widgets-list">${widgetControls}</div>`;
    button.after(menu);
    menu.querySelector('input[type="checkbox"]').addEventListener("change", (event) => {
        settings.enabled = event.target.checked;
        saveSettings();
        render();
    });
    menu.querySelectorAll('input[name="dasiwa-monitor-mode"]').forEach((input) => input.addEventListener("change", (event) => {
        settings.mode = event.target.value;
        settings.enabled = true;
        saveSettings();
        render();
        closeMenu();
    }));
    menu.querySelectorAll('input[name="dasiwa-monitor-dock"]').forEach((input) => input.addEventListener("change", (event) => {
        settings.dockSide = event.target.value;
        settings.placement = "top";
        placePanel(root, settings.dockSide);
        saveSettings();
        render();
    }));

    menu.querySelectorAll('[data-widget-id]').forEach((input) => input.addEventListener("change", (event) => {
        settings.widgets[event.target.dataset.widgetId] = event.target.checked;
        saveSettings();
        render();
    }));
    requestAnimationFrame(() => document.addEventListener("pointerdown", (event) => {
        if (!menu.contains(event.target) && event.target !== button) closeMenu();
    }, { once: true }));
}

function addStyles() {
    const style = document.createElement("style");
    style.textContent = `
        #${ROOT_ID} { position: relative; display: flex; align-items: flex-start; gap: 3px; min-width: 0; margin-right: 6px; color: var(--input-text); font: 600 11px/1 var(--font-inter, sans-serif); } #${ROOT_ID}.is-floating { position: fixed; z-index: 1002; margin: 0; } #${ROOT_ID} .dasiwa-monitor-drag-handle { width: 10px; min-height: 28px; border: 1px solid var(--border-color); background: var(--comfy-input-bg); cursor: grab; touch-action: none; } #${ROOT_ID} .dasiwa-monitor-drag-handle::after { content: "⠿"; display: block; padding-top: 7px; color: var(--descrip-text, #aaa); font-size: 13px; text-align: center; } #${ROOT_ID} .dasiwa-monitor-drag-handle:active { cursor: grabbing; }
        #dasiwa-monitor-dock-left, #dasiwa-monitor-dock-right { position: fixed; z-index: 1002; top: 120px; bottom: 8px; display: flex; align-items: flex-start; pointer-events: none; } #dasiwa-monitor-dock-left { left: 72px; } #dasiwa-monitor-dock-right { right: 72px; } #dasiwa-monitor-dock-left > #${ROOT_ID}, #dasiwa-monitor-dock-right > #${ROOT_ID} { pointer-events: auto; margin: 0; } #${ROOT_ID}.is-vertical .dasiwa-monitor-display.is-lite { flex-direction: column; align-items: stretch; } #${ROOT_ID}.is-horizontal .dasiwa-monitor-display.is-lite { flex-direction: row; }
        .dasiwa-monitor-dock-target { position: fixed; z-index: 1005; box-sizing: border-box; border: 2px dashed #22d3ee; background: #0891b533; color: #cffafe; font: 700 12px/38px var(--font-inter, sans-serif); letter-spacing: .04em; text-align: center; pointer-events: none; } .dasiwa-monitor-dock-target.is-active { background: #0891b588; border-style: solid; } #dasiwa-monitor-dock-target-top { top: 0; left: 0; width: 100vw; height: 42px; } #dasiwa-monitor-dock-target-left, #dasiwa-monitor-dock-target-right { top: 48px; bottom: 0; width: 56px; writing-mode: vertical-rl; padding-top: 12px; } #dasiwa-monitor-dock-target-left { left: 0; } #dasiwa-monitor-dock-target-right { right: 0; }
        #${PANEL_ID}.is-lite { display: flex; align-items: center; gap: 3px; min-width: 0; }
        #${ROOT_ID} .dasiwa-monitor-metric { position: relative; box-sizing: border-box; display: grid; grid-template-columns: max-content max-content; gap: 4px; align-items: center; flex: none; min-width: 56px; height: 28px; overflow: hidden; padding: 0 3px; border: 1px solid color-mix(in srgb, var(--meter) 65%, var(--border-color)); background: var(--comfy-input-bg); white-space: nowrap; }
        #${ROOT_ID} .dasiwa-monitor-metric::before, #${ROOT_ID} .dasiwa-monitor-full-metric i { content: ""; position: absolute; inset: 0 auto 0 0; width: var(--fill); opacity: .38; background: var(--meter); transition: width .35s ease; }
        #${ROOT_ID} span, #${ROOT_ID} strong { position: relative; z-index: 1; font-variant-numeric: tabular-nums; }
        #${ROOT_ID} .dasiwa-monitor-metric span { color: var(--input-text); font-size: 9px; letter-spacing: .01em; }
        #${ROOT_ID} .dasiwa-monitor-metric strong { text-align: right; font-size: 11px; }
        #${ROOT_ID} .cpu { --meter: #38bdf8; } #${ROOT_ID} .ram { --meter: #a78bfa; } #${ROOT_ID} .swap { --meter: #f59e0b; } #${ROOT_ID} .disk { --meter: #fb7185; } #${ROOT_ID} .disk-rd { --meter: #34d399; } #${ROOT_ID} .disk-wr { --meter: #f472b6; }
        #${ROOT_ID} .gpu-util { --meter: #4ade80; } #${ROOT_ID} .gpu-vram { --meter: #22d3ee; } #${ROOT_ID} .gpu-temp { --meter: #fb923c; }
        #${ROOT_ID} [hidden] { display: none !important; }
        #${ROOT_ID} .dasiwa-monitor-settings { position: relative; z-index: 1003; width: 28px; height: 28px; padding: 0; border: 1px solid var(--border-color); color: var(--input-text); background: var(--comfy-input-bg); cursor: pointer; font-size: 15px; } #${ROOT_ID} .dasiwa-monitor-settings:hover, #${ROOT_ID} .dasiwa-monitor-settings:focus-visible { border-color: #22d3ee; color: #22d3ee; outline: none; }
        #dasiwa-monitor-settings-menu { position: absolute; z-index: 1004; top: 32px; right: 0; display: grid; gap: 8px; width: 220px; padding: 10px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 8px 24px #0008; font: 500 12px/1.25 var(--font-inter, sans-serif); } #${ROOT_ID}.is-dock-left #dasiwa-monitor-settings-menu { right: auto; left: 0; } #dasiwa-monitor-settings-menu label { display: grid; grid-template-columns: 16px auto; gap: 6px; align-items: start; cursor: pointer; } #dasiwa-monitor-settings-menu small { grid-column: 2; color: var(--descrip-text, #aaa); } #dasiwa-monitor-settings-menu .dasiwa-monitor-menu-label { padding-top: 4px; border-top: 1px solid var(--border-color); color: var(--descrip-text, #aaa); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; } #dasiwa-monitor-settings-menu .dasiwa-monitor-menu-note { color: var(--descrip-text, #aaa); font-size: 11px; line-height: 1.35; } #dasiwa-monitor-settings-menu .dasiwa-monitor-widgets-list { display: grid; gap: 3px; max-height: 180px; overflow-y: auto; }
        #${PANEL_ID}.is-full { position: absolute; z-index: 1000; width: min(720px, calc(100vw - 24px)); max-height: calc(100vh - 64px); overflow: auto; padding: 14px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 12px 32px #0009; } #${ROOT_ID} .dasiwa-monitor-full-header { display: flex; justify-content: space-between; margin-bottom: 12px; } #${ROOT_ID} .dasiwa-monitor-full-header strong { font-size: 14px; } #${ROOT_ID} .dasiwa-monitor-full-header span, #${ROOT_ID} .dasiwa-monitor-full-metric small { color: var(--descrip-text, #aaa); }
        #${ROOT_ID} .dasiwa-monitor-full-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; } #${ROOT_ID} .dasiwa-monitor-full-metric { position: relative; min-height: 104px; overflow: hidden; padding: 9px; border: 1px solid color-mix(in srgb, var(--meter) 65%, var(--border-color)); background: var(--comfy-input-bg); } #${ROOT_ID} .dasiwa-monitor-full-metric > div:first-child { display: flex; justify-content: space-between; } #${ROOT_ID} .dasiwa-monitor-full-metric strong { font-size: 16px; } #${ROOT_ID} .dasiwa-monitor-full-metric small { position: relative; z-index: 1; display: block; overflow: hidden; margin-top: 5px; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; } #${ROOT_ID} .dasiwa-monitor-graph-wrap { position: relative; z-index: 1; height: 46px; margin-top: 7px; border-bottom: 1px solid color-mix(in srgb, var(--meter) 25%, transparent); } #${ROOT_ID} .dasiwa-monitor-graph { width: 100%; height: 100%; overflow: visible; fill: none; stroke: var(--meter); stroke-width: 3; vector-effect: non-scaling-stroke; } #${ROOT_ID} .dasiwa-monitor-empty-graph { color: var(--descrip-text, #aaa); font-size: 10px; padding-top: 18px; text-align: center; }
        @media (max-width: 640px) { #${PANEL_ID}.is-full { width: calc(100vw - 12px); padding: 9px; } #${ROOT_ID} .dasiwa-monitor-full-grid { grid-template-columns: 1fr; } }
    `;
    document.head.appendChild(style);
}

app.registerExtension({
    name: "DaSiWa.SystemMonitor",
    async setup() {
        addStyles();
        const root = document.createElement("div");
        root.id = ROOT_ID;
        root.innerHTML = `<span class="dasiwa-monitor-drag-handle" aria-label="Drag system monitor"></span><div id="${PANEL_ID}" class="dasiwa-monitor-display is-lite">Loading…</div><button class="dasiwa-monitor-settings" type="button" title="DaSiWa node settings" aria-label="DaSiWa node settings">⚙</button>`;
        const settingsButton = root.querySelector("button");
        settingsButton.addEventListener("click", () => openMenu(settingsButton));
        document.body.insertAdjacentHTML("beforeend", `
            <div id="dasiwa-monitor-dock-left" hidden></div>
            <div id="dasiwa-monitor-dock-right" hidden></div>
            <div id="dasiwa-monitor-dock-target-top" class="dasiwa-monitor-dock-target" hidden>Drop to dock to top</div>
            <div id="dasiwa-monitor-dock-target-left" class="dasiwa-monitor-dock-target" hidden>Dock left</div>
            <div id="dasiwa-monitor-dock-target-right" class="dasiwa-monitor-dock-target" hidden>Dock right</div>`);
        applyOrientation(root);
        enablePanelDrag(root);
        if (placePanel(root)) {
            if (settings.placement === "floating") floatPanel(root, settings.x, settings.y);
        } else {
            document.body.appendChild(root);
            const observer = new MutationObserver(() => {
                if (placePanel(root)) {
                    if (settings.placement === "floating") floatPanel(root, settings.x, settings.y);
                    observer.disconnect();
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
        api.addEventListener(EVENT_NAME, (event) => {
            latestSnapshot = event.detail;
            history = [...history, historyPoint(latestSnapshot)].slice(-HISTORY_LENGTH);
            render();
        });
        new ResizeObserver(() => {
            const panel = document.getElementById(PANEL_ID);
            if (!panel) return;
            if (settings.mode === "full") positionFullPanel(panel);
            else fitPanel(panel);
        }).observe(document.documentElement);
        try {
            latestSnapshot = await api.fetchApi("/dasiwa/system-monitor").then((response) => response.json());
            history = [historyPoint(latestSnapshot)];
            render();
        } catch (error) {
            document.getElementById(PANEL_ID).textContent = "DaSiWa System Monitor is waiting for backend telemetry.";
            console.warn("DaSiWa System Monitor", error);
        }
    },
});