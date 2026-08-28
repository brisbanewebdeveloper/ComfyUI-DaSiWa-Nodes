"""Standalone seed control: the Director's seed panel as its own node.

Mirrors every feature of the MiniMax H3 Director's seed control:

- SEED field with the full unsigned 64-bit range (H3 seed space).
- Random / Fixed mode pills.
- "New" roll (keeps the selected mode), "Use Last" restore, and the
  "Last 10 seeds" history with copy actions — all rendered by the
  pack's JS (js/dasiwa_seed_control.js) with the Director's UX style.
- External seed socket: linking ``seed`` disables the local controls and
  passes the connected value through, exactly like the Director's
  external seed input.

State model (persisted in the hidden ``seed_control_state`` widget):
``{"mode": "random"|"fixed", "last_seed": str|None, "recent": [str, ...]}``
— the same shape the Director stores as ``seed_control`` inside its
``timeline_data`` widget.

Outputs:
- ``seed``  (INT)   — the effective seed.
- ``noise`` (NOISE) — a ``Noise_RandomNoise``-compatible object wrapping
  the same seed, so the value can be passed straight to any node that
  accepts a ``NOISE`` input (e.g. the LTX sampler's ``noise`` input), in
  addition to the raw integer.

Headless behaviour: when the mode is ``random`` and no local seed value is
set, the node rolls a fresh 64-bit seed on every queue (``IS_CHANGED``),
so the node also works without the DOM UI.
"""
import json
import secrets

MAX_SEED = 0xFFFFFFFFFFFFFFFF
DEFAULT_STATE = {"mode": "random", "last_seed": None, "recent": []}


def _normalize_state(raw):
    """Coerce any widget payload into the canonical seed-control state."""
    state = dict(DEFAULT_STATE)
    parsed = {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            parsed = {}
    elif isinstance(raw, dict):
        parsed = raw
    if isinstance(parsed, dict):
        state.update({key: value for key, value in parsed.items() if key in state})
    state["mode"] = "fixed" if state.get("mode") == "fixed" else "random"
    last_seed = state.get("last_seed")
    state["last_seed"] = str(last_seed) if str(last_seed or "").strip().isdigit() else None
    state["recent"] = [
        str(value) for value in (state.get("recent") or []) if str(value).strip().isdigit()
    ][:10]
    return state


def _roll_seed():
    """One random unsigned 64-bit seed, matching the JS two-Uint32 roll."""
    return secrets.randbits(64)


def _make_noise(seed):
    """Return a ``NOISE``-compatible object wrapping ``seed``.

    Prefers the real ``Noise_RandomNoise`` from ComfyUI's custom sampler
    nodes (the exact object ``RandomNoise.execute`` yields and the LTX
    samplers consume via ``.seed`` / ``.generate_noise``). Falls back to
    a duck-typed shim so the node still works where ``comfy_extras`` is
    not importable (headless / isolated tests).
    """
    try:
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

        return Noise_RandomNoise(int(seed))
    except Exception:
        class _NoiseShim:
            def __init__(self, seed):
                self.seed = int(seed)

            def generate_noise(self, input_latent):
                import comfy.sample

                latent = input_latent["samples"]
                batch = input_latent.get("batch_index")
                return comfy.sample.prepare_noise(latent, self.seed, batch)

        return _NoiseShim(int(seed))


class DaSiWa_SeedControl:
    """Standalone seed control with the MiniMax H3 Director's seed UX."""

    DESCRIPTION = (
        "Seed Control: the Director's seed panel as a standalone node — "
        "Random/Fixed mode, unsigned 64-bit seed, New roll, Use Last, last-10 "
        "history, an external seed override socket, and a NOISE output for "
        "downstream samplers."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Local seed value; hidden in the UI — the DOM panel owns it.
                "seed_value": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "step": 1,
                        "hidden": True,
                        "tooltip": "Local seed value. The panel edits this; leave the native widget hidden.",
                    },
                ),
                # Persisted seed-control state (mode / last_seed / recent),
                # same shape as the Director's seed_control block.
                "seed_control_state": (
                    "STRING",
                    {
                        "default": json.dumps(DEFAULT_STATE),
                        "multiline": False,
                        "hidden": True,
                        "tooltip": "Persisted seed-control state (JSON): mode, last_seed and recent seeds.",
                    },
                ),
            },
            "optional": {
                # External override socket (Director semantics): when linked,
                # the connected seed wins and the local controls are replaced
                # by the "External seed connected" note in the panel.
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "step": 1,
                        "forceInput": True,
                        "tooltip": "Optional external seed. When linked, the local controls are disabled and this value is passed through.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "NOISE")
    RETURN_NAMES = ("seed", "noise")
    FUNCTION = "execute"
    CATEGORY = "DaSiWa"

    @classmethod
    def IS_CHANGED(cls, seed_value, seed_control_state, seed=None):
        # A linked external seed drives re-execution through the link itself.
        if seed is not None:
            return False
        state = _normalize_state(seed_control_state)
        try:
            value = int(seed_value)
        except (TypeError, ValueError):
            value = 0
        # Random mode without a local value rolls a fresh seed on every
        # queue, mirroring the Director's before-queue seed preparation.
        if state["mode"] == "random" and value == 0:
            return secrets.token_hex(4)
        return False

    def execute(self, seed_value, seed_control_state, seed=None):
        state = _normalize_state(seed_control_state)
        external = seed is not None
        value = int(seed) if external else int(seed_value or 0)
        if value < 0 or value > MAX_SEED:
            raise ValueError(
                f"seed must be within 0..{MAX_SEED} (unsigned 64-bit)"
            )
        # Random mode with no local value: roll here too, so the node works
        # headless (API clients) exactly like it works through the panel.
        if not external and state["mode"] == "random" and value == 0:
            value = _roll_seed()
        return (value, _make_noise(value))


NODE_CLASS_MAPPINGS = {"DaSiWa_SeedControl": DaSiWa_SeedControl}
