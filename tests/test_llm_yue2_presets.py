"""YuE2 field-specific presets preserve the single-STRING writer contract."""
import json

import pytest

from nodes import llm_backends
from nodes import llm_prompt_presets as prompts
from nodes import nodes_llm_simple as simple


TARGETS = [
    ("YuE2 Song Style", "promptforge_yue2_style", "Style", "English, warm female vocal, contemporary pop, 96 BPM, piano"),
    ("YuE2 Lyrics", "promptforge_yue2_lyrics", "Lyrics", "[Verse]\nEin Licht am Fenster\n\n[Chorus]\nWir gehen weiter"),
    ("YuE2 Instrumental Style", "promptforge_yue2_instrumental", "Style", "Instrumental, ambient, 72 BPM, solo piano, no vocals"),
]


@pytest.mark.parametrize("target,preset,label,result", TARGETS)
def test_writer_routes_yue2_through_analyze(monkeypatch, target, preset, label, result):
    sent = []
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted")
    def stream(url, payload, timeout, cancel, headers=None):
        sent.append(payload)
        raw = f"<think>hidden</think>\n===SEGMENT: {label}===\n{result}"
        yield "data: " + json.dumps({"choices": [{"delta": {"content": raw}}]})
        yield "data: [DONE]"
    monkeypatch.setattr(llm_backends, "_stream_lines", stream)
    monkeypatch.setattr(llm_backends.OpenAICompatible, "unload", lambda self, name: False)
    assert simple.WRITE_FOR[target] == preset
    assert preset in prompts._SYSTEM_PROMPT_PRESET_LABELS
    assert simple.DaSiWa_LLMPromptWriter().write(
        model="Server: test", write_for=target, idea="German song about hope", seed=0,
        keep_loaded=False, style="Watercolor", detail=8, creativity=7) == (result,)
    assert sent[0]["messages"][0]["content"] == prompts.exported_system(preset)
    user = sent[0]["messages"][1]["content"]
    assert "Watercolor" not in user and "camera" not in user and "materials" not in user
    assert "Creativity 7 of 10" in user and "Detail 8 of 10" in user
    assert "musical" in user


def test_music_prefix_uses_a_line_break_for_lyrics():
    spec = prompts.preset_spec("promptforge_yue2_lyrics")
    assert simple.assemble("[Verse]\nNew lines", spec, "[Intro]\nKeep exactly") == "[Intro]\nKeep exactly\n[Verse]\nNew lines"


@pytest.mark.parametrize("target,preset,label,result", TARGETS)
def test_yue2_guides_keep_fields_and_runtime_separate(target, preset, label, result):
    guide = prompts.preset_spec(preset)
    assert guide["output"] == label
    assert "ABC" in guide["system"] and "duration" in guide["system"].lower()
    assert simple.style_line("Anime", guide) == ""
    assert prompts.prompt_response(preset, f"===SEGMENT: {label}===\n```text\n{result}\n```") == result
