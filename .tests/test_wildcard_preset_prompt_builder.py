import importlib.util
import math
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_wildcard_preset_prompt_builder.py"
spec = importlib.util.spec_from_file_location("nodes_wildcard_preset_prompt_builder", MODULE_PATH)
assert spec is not None and spec.loader is not None
wildcards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wildcards)


class TestWildcardResolution:
    def test_resolves_every_brace_group_and_preserves_bridge_text(self):
        resolved = wildcards.resolve_pattern(
            "{small nose|button nose}, detailed skin, {sharp focus|soft focus}",
            seed=7,
            subject_key="FACE/Face Details",
            reroll=0,
        )
        assert "{" not in resolved
        assert "}" not in resolved
        assert "detailed skin" in resolved
        assert len([part for part in resolved.split(", ") if part]) == 3

    def test_same_seed_subject_and_reroll_are_reproducible(self):
        args = ("{A|B|C}", 123, "EYES/Eye Shape", 4)
        assert wildcards.resolve_pattern(*args) == wildcards.resolve_pattern(*args)

    def test_reroll_changes_deterministic_selection(self):
        choices = {
            wildcards.resolve_pattern("{A|B|C|D|E|F|G}", 42, "EYES/Eye Shape", reroll)
            for reroll in range(12)
        }
        assert len(choices) > 1


class TestPromptAssembly:
    def test_library_accepts_a_compatible_user_customized_json(self, tmp_path, monkeypatch):
        custom_library = tmp_path / "custom_library.json"
        custom_library.write_text(
            '{"wildcard_library":{"categories":{"Custom":[{"subject":"Entry","booru_wildcards":["custom"]}]}}}',
            encoding="utf-8",
        )
        monkeypatch.setattr(wildcards, "DATA_PATH", custom_library)
        assert wildcards._load_library() == {"Custom": [{"subject": "Entry", "booru_wildcards": ["custom"]}]}

    def test_bundled_library_and_registration_contract(self):
        library = wildcards._load_library()
        assert list(library) == [
            "Positive Prompts",
            "Art Style",
            "Character Selection",
            "Species - General",
            "Species - Mythical",
            "Species - Botanical",
            "Species - Kemonomimi",
            "Species - Mammals",
            "Species - Aquatic",
            "Species - Reptiles, Amphibians & Birds",
            "Species - Insects & Arachnids",
            "Character - Eyes",
            "Character - Face",
            "Character - Hair",
            "Character - Body",
            "Wardrobe",
            "Weapons & Items",
            "Body Visibility",
            "Composition & Pose",
            "Scene",
            "Background",
            "Intimate Backgrounds",
            "NSFW",
            "Negative Prompts",
        ]
        assert "Positive Prompts" in library
        assert "Negative Prompts" in library
        assert "General Presets" not in library
        assert "Species - Mythical & Botanical" not in library
        assert {subject["subject"] for subject in library["Species - Botanical"]} == {
            "Mushroom", "Moss", "Venus Flytrap", "Flower", "Cactus",
        }
        character_selection = {subject["subject"]: subject for subject in library["Character Selection"]}
        assert character_selection["1 Girl"]["booru_presets"] == ["1girl"]
        assert character_selection["1 Girl"]["nl_presets"] == ["a girl"]
        assert character_selection["2 Boys"]["nl_presets"] == ["two boys"]
        assert character_selection["5 Men"]["nl_presets"] == ["five men"]
        assert character_selection["5 Women"]["booru_presets"] == ["5women"]
        assert character_selection["5 Women"]["nl_presets"] == ["five women"]
        assert character_selection["Solo"]["booru_presets"] == ["solo"]
        assert character_selection["Group"]["booru_presets"] == ["group"]
        assert character_selection["Crowd"]["booru_presets"] == ["crowd"]
        assert all(not subject.get("booru_wildcards") and not subject.get("nl_wildcards")
                   for subject in character_selection.values())
        assert "✅ Positive Prompt" not in library
        assert "❌ Negative Prompt" not in library
        assert all(
            subject.get("booru_wildcards") or subject.get("nl_wildcards")
            or subject.get("booru_presets") or subject.get("nl_presets")
            for subjects in library.values() for subject in subjects
        )
        package_source = (Path(__file__).parents[1] / "__init__.py").read_text(encoding="utf-8")
        assert "DaSiWa_WildcardPresetPromptBuilder" in package_source

    def test_node_defaults_to_a_200_token_budget(self):
        input_types = wildcards.DaSiWa_WildcardPresetPromptBuilder.INPUT_TYPES()
        controls = input_types["required"]
        assert controls["token_budget"][1]["default"] == 200
        assert controls["auto_reroll"][1]["default"] is False
        assert input_types["optional"]["positive_input"][1]["forceInput"] is True
        assert input_types["optional"]["negative_input"][1]["forceInput"] is True

    def test_connected_prompts_are_prepended_to_builder_output(self):
        positive, negative = wildcards.DaSiWa_WildcardPresetPromptBuilder().build(
            "Booru", 1, 200, "{}", 0, False,
            positive_input="base positive", negative_input="base negative",
        )
        assert positive.startswith("base positive")
        assert negative.startswith("base negative")

    def test_auto_reroll_bypasses_comfyui_output_cache(self):
        first = wildcards.DaSiWa_WildcardPresetPromptBuilder.IS_CHANGED(
            "Booru", 0, 200, "{}", 0, True
        )
        second = wildcards.DaSiWa_WildcardPresetPromptBuilder.IS_CHANGED(
            "Booru", 0, 200, "{}", 0, True
        )
        assert math.isnan(first)
        assert math.isnan(second)
        assert first is not second
        # ComfyUI uses tuples as cache keys.  A module-level math.nan singleton
        # compares equal here through tuple identity optimization, so every queue
        # needs its own NaN object to invalidate the selected-output cache.
        assert (first,) != (second,)

    def test_auto_reroll_uses_a_fresh_queue_time_choice(self):
        alternatives = "|".join(f"choice-{index}" for index in range(100))
        library = {
            "Positive Prompts": [{
                "subject": "Variation",
                "booru_wildcards": [f"{{{alternatives}}}"],
                "nl_wildcards": [],
            }],
        }
        selection = '{"Positive Prompts/Variation/wildcards":{"enabled":true,"weight":1}}'
        with patch.object(wildcards, "_load_library", return_value=library), \
             patch.object(wildcards.secrets, "randbits", side_effect=[101, 202]):
            node = wildcards.DaSiWa_WildcardPresetPromptBuilder()
            first, _ = node.build("Booru", 7, 200, selection, 0, True)
            second, _ = node.build("Booru", 7, 200, selection, 0, True)
        assert first != second

    def test_frontend_restores_picker_on_load_and_only_sizes_new_nodes_once(self):
        source = (Path(__file__).parents[1] / "js" / "wildcard_preset_prompt_builder.js").read_text(encoding="utf-8")
        assert "loadedGraphNode(node)" in source
        assert "afterConfigureGraph()" in source
        assert "for (const node of app.graph?._nodes || [])" in source
        assert "function isWildcardNode(node)" in source
        assert "let installWildcardPicker;" in source
        assert "installWildcardPicker = function" in source
        assert "installWildcardPicker?.call(node);" in source
        assert "dasiwaWildcardPickerInstalled" in source
        assert "installWildcardPicker.call(this, { fitInitialSize: true });" in source
        assert "if (fitInitialSize)" in source
        assert "if (this.dasiwaWildcardPickerRestored) return;" in source
        assert "node.dasiwaWildcardPickerRestored = true;" in source
        assert "Math.max(this.size?.[0] || 0, 620)" in source
        assert "overflow:auto" in source
        assert "domWidget.computeSize" not in source
        assert "const PICKER_HEIGHT = 490;" in source
        assert "getHeight: () => PICKER_HEIGHT" in source
        assert "(this.size?.[1] || 0) - 130" not in source
        assert 'controls.className = "dasiwa-wildcard-controls";' in source
        assert 'selections.className = "dasiwa-wildcard-selections";' in source
        assert 'previewSegment.className = "dasiwa-wildcard-preview-segment";' in source
        assert "controls.appendChild(toolbar);" in source
        assert "previewSegment.appendChild(output);" in source
        assert source.index('renderSection("presets", settings, selections);') < source.index('renderSection("wildcards", settings, selections);')

    def test_auto_reroll_checkbox_invokes_its_hidden_widget_callback(self):
        source = (Path(__file__).parents[1] / "js" / "wildcard_preset_prompt_builder.js").read_text(encoding="utf-8")
        assert "entry.callback?.(value);" in source
        assert "setWidgetValue(autoRerollWidget, autoRerollInput.checked);" in source

    def test_random_select_replaces_selection_with_one_to_ten_preset_or_wildcard_subjects(self):
        source = (Path(__file__).parents[1] / "js" / "wildcard_preset_prompt_builder.js").read_text(encoding="utf-8")
        assert 'randomSelect.textContent = "🎲 Random Select";' in source
        assert 'for (const entryType of ["presets", "wildcards"])' in source
        assert "const count = Math.min(candidates.length, 1 + randomIndex(10));" in source
        assert "crypto.getRandomValues(value);" in source

    def test_category_headers_show_a_right_aligned_selected_subject_count(self):
        source = (Path(__file__).parents[1] / "js" / "wildcard_preset_prompt_builder.js").read_text(encoding="utf-8")
        assert "details.dataset.hasSelection = String(selectedCount > 0);" in source
        assert 'selectedBadge.className = "dasiwa-wildcard-selected-count";' in source
        assert 'selectedBadge.textContent = `✓ ${selectedCount} selected`;' in source

    def test_uses_style_keys_applies_weight_and_trims_lowest_weight_subject_first(self):
        library = {
            "FIRST": [{"subject": "High", "booru_wildcards": ["high detail"], "nl_wildcards": ["very high detail"]}],
            "SECOND": [{"subject": "Low", "booru_wildcards": ["low priority"], "nl_wildcards": ["low priority natural"]}],
            "NEGATIVE PROMPTS": [{"subject": "Bad", "booru_wildcards": ["blurry"], "nl_wildcards": ["out of focus"]}],
        }
        state = {
            "FIRST/High/wildcards": {"enabled": True, "weight": 1.5},
            "SECOND/Low/wildcards": {"enabled": True, "weight": 0.5},
            "NEGATIVE PROMPTS/Bad/wildcards": {"enabled": True, "weight": 1.0},
        }
        positive, negative, positive_tokens, negative_tokens = wildcards.build_prompts(
            library, "Natural Language", 9, 0, 9, state
        )
        assert positive == "(very high detail:1.5)"
        assert negative == "out of focus"
        assert positive_tokens <= 9
        assert negative_tokens <= 3

    def test_joins_multiple_entries_in_subject_order(self):
        library = {"FACE": [{"subject": "Details", "booru_wildcards": ["nose", "lips", "skin"], "nl_wildcards": []}]}
        positive, negative, *_ = wildcards.build_prompts(
            library, "Booru", 1, 0, 75, {"FACE/Details/wildcards": {"enabled": True, "weight": 1.0}}
        )
        assert positive == "nose, lips, skin"
        assert negative == ""

    def test_negative_categories_are_detected_by_name(self):
        assert wildcards.is_negative_category("NEGATIVE PROMPTS")
        assert wildcards.is_negative_category("❌ Negative Prompt")
        assert not wildcards.is_negative_category("POSITIVE PROMPTS")
