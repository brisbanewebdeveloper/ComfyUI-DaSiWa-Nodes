"""Lightweight continuity-policy regression checks; no ComfyUI or torch required."""
import ast
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


def policy_namespace():
    source = ast.parse((ROOT / "nodes/h3_continuity/core.py").read_text())
    chosen = [node for node in source.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "DEFAULT_PROMPT" for target in node.targets
    ) or isinstance(node, ast.FunctionDef) and node.name == "compose_prompt"]
    namespace = {}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), "core.py", "exec"), namespace)
    return namespace


class ContinuityPolicyTests(unittest.TestCase):
    def setUp(self):
        self.compose = policy_namespace()["compose_prompt"]
        self.settings = {"version": 3, "overlap_frames": 22, "extension_frames": 119,
                         "continuation_prompt": "", "idea": ""}

    def test_auto_welds_seam_then_continues_naturally(self):
        prompt = self.compose(self.settings)
        self.assertIn("hidden overlapping context", prompt)
        self.assertIn("seam", prompt)
        self.assertIn("continue the established action, camera motion, and sound naturally", prompt.lower())
        self.assertNotIn("Next action:", prompt)

    def test_written_next_action_can_change_pace_and_camera_after_seam(self):
        prompt = self.compose({**self.settings, "continuation_prompt": "She sprints; camera pans to follow."})
        self.assertIn("Next action: She sprints; camera pans to follow.", prompt)
        self.assertIn("after the overlap", prompt.lower())
        self.assertIn("camera", prompt.lower())
        self.assertNotIn("Do not invent a pan", prompt)

    def test_legacy_v2_custom_prompt_remains_authoritative(self):
        prompt = self.compose({**self.settings, "version": 2, "continuation_prompt": "A deliberate dolly shot."})
        self.assertIn("A deliberate dolly shot.", prompt)
        self.assertNotIn("Continue the same uninterrupted shot naturally.", prompt)

    def test_idea_without_written_prompt_is_not_treated_as_auto(self):
        prompt = self.compose({**self.settings, "idea": "Accelerate after the seam."})
        self.assertIn("Next action: Accelerate after the seam.", prompt)
        self.assertNotIn("With no next action", prompt)

    def test_v3_normalization_preserves_auto_and_custom_prompts(self):
        script = """import {readFileSync} from 'node:fs';
const text = readFileSync(process.argv[1], 'utf8');
const m = await import('data:text/javascript,' + encodeURIComponent(text));
if ('CONTINUITY_LEGACY_PROMPT' in m) process.exit(1);
const auto = m.normalizeContinuity({version:3, continuation_prompt:'', source_id:'', operation:'continue'});
if (auto.continuation_prompt !== '' || auto.operation !== 'new') process.exit(2);
const custom = m.normalizeContinuity({version:3, source_id:'source', continuation_prompt:'Pan after the seam.', operation:'new', extension_frames:119});
if (custom.continuation_prompt !== 'Pan after the seam.' || custom.operation !== 'continue' || 'extension_frames' in custom) process.exit(3);
"""
        subprocess.run(["node", "--input-type=module", "-e", script, str(ROOT / "js/minimax_h3_continuity.js")], check=True)

    def test_forge_allows_requested_changes_without_restarting(self):
        source = ast.parse((ROOT / "nodes/h3_forge.py").read_text())
        assignment = next(node for node in source.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "CONTINUATION_SYSTEM" for target in node.targets))
        system = ast.literal_eval(assignment.value).lower()
        self.assertIn("overlap", system)
        self.assertIn("after the seam", system)
        self.assertIn("camera", system)
        self.assertIn("next action", system)
    def test_forge_without_new_idea_respects_existing_next_action(self):
        source = ast.parse((ROOT / "nodes/h3_forge.py").read_text())
        func = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == "generate_continuity_draft")
        func.decorator_list = []
        namespace = {"__import__": __import__, "re": re, "os": __import__("os"),
                     "_THINK": re.compile(r"<think>.*?</think>", re.S),
                     "CONTINUATION_SYSTEM": "test-system", "urlerror": __import__("urllib.error", fromlist=["HTTPError"]),
                     "load_bundle": lambda: {"default_creativity": "normal", "default_detail": 5,
                                             "creativity_presets": {"normal": {"rule": "natural"}},
                                             "detail_levels": {"5": {"rule": "100-200 words"}},
                                             "context_length": 4096, "max_output_chars": 7000},
                     "scale_detail_rule": lambda rule, duration: rule,
                     "_is_this_machine": lambda base: False, "_FORGE_LOADED": set()}
        backend = Mock(base="http://remote.invalid")
        backend.models.return_value = [{"id": "ollama:test"}]
        backend.can_see.return_value = False
        backend.chat.return_value = ("She sprints while the camera pans.", {})
        backend.unload.return_value = True
        namespace["backends"] = lambda settings: {"ollama": backend}
        exec(compile(ast.Module(body=[func], type_ignores=[]), "h3_forge.py", "exec"), namespace)
        namespace["generate_continuity_draft"]({"prompt": "old clip"}, "", "/unused", "ollama:test", {},
                                                current_prompt="She sprints; camera pans after the seam.")
        user = backend.chat.call_args.args[2]
        self.assertIn("She sprints; camera pans after the seam.", user)
        self.assertIn("Follow the current next-action draft", user)
        self.assertNotIn("New idea: Continue the current action naturally.", user)


if __name__ == "__main__":
    unittest.main()
