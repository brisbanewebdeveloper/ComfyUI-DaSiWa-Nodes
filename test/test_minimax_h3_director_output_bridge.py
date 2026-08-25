"""Regression checks for Director Image Inpaint without hidden graph bridges."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
JS = ROOT / "js" / "minimax_h3_director_v2.js"


def source():
    return JS.read_text(encoding="utf-8")


def test_director_controls_have_single_owners_and_image_inpaint_can_start_empty():
    src = source()
    assert '"Image Inpaint"' in src
    assert 'mode() === "Image Inpaint" ? type === "image"' in src
    assert 'mode() === "I2VA" || mode() === "Image Inpaint"' in src
    assert 'output frame count 1' in src
    assert 'openStageSettings' in src
    assert 'save_workflow' in src
    # Model/prompt controls share panel one; save settings are never nested there.
    assert 'topRow.append(modesSide, actionsSide, docsButton)' in src
    assert 'promptRow.append(promptSide)' in src
    assert 'modeGroup.append(promptRow)' in src
    assert 'promptSettingsPanel' not in src
    assert 'Save Settings' not in src
    assert 'controlRow.append(modeGroup, saveNodePanel)' in src
    # The post-process, resolution and optimization panels render as one shared
    # info-feed row appended together inside the mode group (current v2 layout).
    assert 'modeGroup.append(resolutionPanel, postprocessPanel, optimizationPanel)' in src
    assert 'PP_STAGE_FIELDS' in src
    assert 'modelFolder' in src
    assert 'ppLoadModelList' in src
    assert 'ds-h3-pp-burger' in src
    # The legacy internal-execute input is intentionally invisible.
    assert 'const internalExecuteWidget = node.widgets?.find(w => w.name === "internal_execute")' in src
    assert 'internalExecuteWidget.hidden = true' in src
    # The save panel consumes the real saver UI image payload.
    assert 'message?.images?.[0]' in src
    assert 'ds-h3-save-preview' in src
    assert 'const outputKind = mode() === "Image Inpaint" ? "image" : "video"' in src
    assert 'IMAGE_SAVE_OPTION_KEYS = new Set(["filename_prefix", "file_format", "compression"])' in src
    assert 'VIDEO_SAVE_OPTION_KEYS = new Set(["filename_prefix", "codec", "container", "bit_depth", "quality", "pingpong", "crop_to_audio", "audio_codec", "audio_bitrate", "save_first_frame", "save_last_frame"])' in src
    assert 'durationWidget.callback?.call(durationWidget, 1)' in src
    assert 'old?.call(lengthWidget, value)' in src
    assert 'Image Inpaint requires exactly one image reference.", true); return;' not in src


def test_director_does_not_create_external_output_nodes():
    src = source()
    for forbidden in (
        '"ImageFromBatch"',
        '"CreateVideo"',
        '"SaveVideo"',
        '"DaSiWa_MetadataConfig"',
        '"DaSiWa_MetadataImageSaver"',
        'dasiwa_director_bridge_role',
        'ds-h3-output-column',
    ):
        assert forbidden not in src


def test_director_output_contract_remains_original():
    from nodes.nodes_minimax_h3_director import MiniMaxH3Director

    assert MiniMaxH3Director.RETURN_NAMES == (
        "guide", "duration", "positive_prompt", "width", "height", "model",
        "fl2va_requested", "ref2va_requested", "frame_rate",
    )
    assert MiniMaxH3Director.RETURN_TYPES[-1] == "FLOAT"
    assert "images" not in MiniMaxH3Director.INPUT_TYPES()["optional"]
