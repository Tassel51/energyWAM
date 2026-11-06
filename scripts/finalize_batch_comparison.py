"""Supervisor-owned finite final validation/audit chain, safe to rerun completed steps."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def run(script,*args):subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
run('confirm_batch_checkpoints.py')
name='batch4_tiny16_samples1600_seed42'
state=json.loads((R/'search_status_23.json').read_text()).get(name,{})
if state.get('phase')!='complete':raise RuntimeError('Batch4 did not complete; inspect failure before selecting a batch')
proof=R/f'{name}_resume_update_verification/result.json'
if not proof.exists() or not json.loads(proof.read_text()).get('passed'):run('verify_batch_resume_update.py',name,'--step','400','--wait-confirmation')
run('audit_batch_checkpoint.py',name,'--step','400')
run('report_batch_size_comparison.py')
run('summarize_batch_confirmation.py')
run('plot_batch_size_comparison.py')
print('Finite batch comparison validation and checkpoint verification complete; recommendation still requires review.',flush=True)
