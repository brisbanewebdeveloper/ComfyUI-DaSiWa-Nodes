"""Run the real CPU segment contracts against an explicit checkout and Core."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--comfy',type=Path,required=True)
parser.add_argument('--case',choices=['stage_delta', 'range_reads', 'all'],default='all')
args = parser.parse_args()
if not (args.comfy/'comfy').is_dir():
    parser.error('--comfy must name a ComfyUI checkout')
env = {**os.environ,'H3_COMFY':str(args.comfy.resolve()),'OMP_NUM_THREADS':'2'}
command = [sys.executable,'-m','pytest','-q'] + [str(args.repo.resolve()/'.github/scripts'/f) for f in ['test_h3_segment_backend.py']]
if {'stage_delta': 'stage_delta_writes', 'range_reads': 'range_reads', 'all': 'not guide and not upscale'}[args.case]: command += ['-k',{'stage_delta': 'stage_delta_writes', 'range_reads': 'range_reads', 'all': 'not guide and not upscale'}[args.case]]
result = subprocess.run(command,cwd=args.repo,env=env)
if result.returncode == 0:
    print(json.dumps({'case':args.case,'status':'passed','device':'cpu','learned_checkpoints':False}))
sys.exit(result.returncode)
