"""Execute actual planner functions without importing GPU/model management."""
import ast
import math
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / 'nodes/h3_tiled_sampling.py').read_text())
namespace = {'math': math, 'FRAME_PER_TOKEN': (1, 4, 4, 4, 4)}
functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in
             ('plan_tiles', 'temporal_ranges', '_even_split', 'auto_local_tiles')]
exec(compile(ast.Module(body=functions, type_ignores=[]), '<real planner>', 'exec'), namespace)
plan_tiles = namespace['plan_tiles']

@pytest.mark.parametrize('reserve', [0, 5, 10, 25, 95])
def test_reserve_formula_and_diagnostic(reserve):
    # One token/canvas avoids geometry changes; inspect exact residual rows.
    memory = 8 * 1024**3
    weights = 128 * 1024**2
    reserved = 16 * 1024**2
    plan = plan_tiles((1,24,1,2,2),32,32,memory,model_bytes=weights,
                      reserved_bytes=reserved,workspace_reserve_pct=reserve)
    buffers = 24 * 1 * 2 * 2 * 20
    expected = int(((memory-reserved-weights)*(1-reserve/100)-buffers)/(96*1024))
    assert plan['target_rows'] == expected
    assert f'{reserve:g}% headroom' in plan['explanation']

@pytest.mark.parametrize('reserve', [-5, 105, float('nan'), float('inf')])
@pytest.mark.parametrize('enforce', [True,False])
def test_invalid_reserve_rejected_even_when_bypassed(reserve,enforce):
    with pytest.raises(ValueError,match='reserve'):
        plan_tiles((1,24,1,2,2),32,32,1024**3,
                   workspace_reserve_pct=reserve,enforce_budget=enforce)

def test_full_reserve_rejects_and_bypass_retains_splits():
    kwargs=dict(workspace_reserve_pct=100,spatial_tiling=True,temporal_chunking=True)
    with pytest.raises(MemoryError,match='100%'):
        plan_tiles((1,24,35,32,32),2048,1536,12*1024**3,**kwargs)
    plan=plan_tiles((1,24,35,32,32),2048,1536,12*1024**3,enforce_budget=False,**kwargs)
    assert plan['chunk_tokens'] <= 30
    assert plan['tile_width'] < 2048
    assert 'budget enforcement=off' in plan['explanation']

def test_zero_reserve_is_not_budget_bypass():
    with pytest.raises(MemoryError):
        plan_tiles((1,24,35,32,32),2048,1536,1,workspace_reserve_pct=0)

def test_smaller_reserve_accepts_actual_borderline_plan():
    kwargs=dict(model_bytes=128*1024**2,reserved_bytes=16*1024**2,
        text_tokens=128,ref_tokens=100,audio_tokens=20,keyframe_tokens=1,
        spatial_tiling=False,temporal_chunking=False)
    with pytest.raises(MemoryError):
        plan_tiles((1,24,5,32,32),1024,1024,800*1024**2,workspace_reserve_pct=25,**kwargs)
    plan=plan_tiles((1,24,5,32,32),1024,1024,800*1024**2,workspace_reserve_pct=5,**kwargs)
    assert (plan['tile_width'],plan['tile_height'],plan['chunk_tokens']) == (1024,1024,5)
