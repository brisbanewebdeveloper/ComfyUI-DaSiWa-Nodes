import importlib.util
from pathlib import Path
import pytest
SPEC = importlib.util.spec_from_file_location('segment_policy', Path(__file__).resolve().parents[2] / 'nodes/h3_continuity/segment_policy.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)

def test_delta_layout_uses_global_audio_boundaries():
    assert p.delta_layout(124, 119) == dict(segment_start_frame=124, segment_frames=119, frames=243, video_token_start=37, video_token_count=35, audio_token_start=207, audio_token_count=198)

@pytest.mark.parametrize('source,added', [(True,119),(124.0,119),(123,119),(4,119),(124,True),(124,118),(124,0),(124,374)])
def test_bad_boundaries(source, added):
    with pytest.raises(ValueError): p.delta_layout(source, added)

def test_global_rounding():
    a = p.delta_layout(124,119)
    b = p.delta_layout(243,119)
    assert a['audio_token_start'] + a['audio_token_count'] == b['audio_token_start']
    assert a['audio_token_count'] + b['audio_token_count'] == round(362*40/24)-round(124*40/24)
