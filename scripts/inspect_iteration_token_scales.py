"""CPU inspection of checkpoint token scales on declared nominal inputs."""
import json,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.config import load_config
from energywam.models.energy_core import EnergyWAMCore
cfg=load_config(ROOT/'configs/experiments/iteration_tinyfit_bf16_r2.yaml')
checkpoint=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104/tinyfit_bf16_r2/ckpt_latest.pt')
if not checkpoint.exists():checkpoint=checkpoint.with_name('model_reference.pt')
payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
state=payload['model']
class Backbone(nn.Module):
    hidden_size=2560
core=EnergyWAMCore(cfg,Backbone(),dict(action_dim=12,state_dim=32,future_dim=384,
    future_patches=196,num_cameras=1,action_horizon=16,num_future_frames=3,dino_dim=384))
core.load_state_dict({k[len('core.'):]:v for k,v in state.items() if k.startswith('core.') and not k.startswith('core.backbone.')},strict=True)
torch.manual_seed(14)
a=torch.randn(1,16,12);z=torch.randn(1,3,1,196,384);q=torch.cat([torch.randn(1,16).sin(),torch.randn(1,16).cos()],-1)
cur=torch.randn(1,1,196,384)
def rms(t):return float(t.detach().float().square().mean().sqrt())
with torch.no_grad():
    result=dict(scope='Actual BF16 control checkpoint40; NOMINAL unit-Gaussian action/future/current features and sincos-like state, NOT empirical data-distribution statistics',
        action_content_rms=rms(core.action_projector(a)),future_content_rms=rms(core.future_projector(z)),
        current_content_rms=rms(core.current_dino_projector(cur)),state_content_rms=rms(core.state_projector(q)),
        action_semantic_encoding_rms=rms(core.encodings.action(1,16,torch.device('cpu'),torch.float32)),
        future_semantic_encoding_rms=rms(core.encodings.future(1,3,1,196,torch.device('cpu'),torch.float32,offsets=torch.tensor([[5,11,16]]))),
        current_semantic_encoding_rms=rms(core.encodings.dino_current(1,1,196,torch.device('cpu'),torch.float32)),
        energy_token_rms=rms(core.energy_token),unused_energy_type_table_entry_rms=rms(core.encodings.type_of('energy')))
    emb=next(v for k,v in state.items() if k.startswith('backbone.') and k.endswith('embed_tokens.weight'))
    result['first_2048_text_embeddings_rms']=rms(emb[:2048])
out=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104/token_scale_inspection.json')
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
