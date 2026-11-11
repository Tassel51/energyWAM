"""GPU exact-update and resume check for original AdamW8bit with CPU state storage."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical-gpus',required=True);p.add_argument('--execution-gpu-policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from energywam.training.gpu_policy import authorized_visibility
    os.environ['CUDA_VISIBLE_DEVICES']=authorized_visibility(a.physical_gpus,a.execution_gpu_policy,visible=a.physical_gpus)
    import torch
    import bitsandbytes as bnb
    from energywam.training.optimizer_offload import OptimizerStateOffloader
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    arrays=[torch.randn(512,512) for _ in range(2)]
    ref=[torch.nn.Parameter(v.to('cuda:'+str(i))) for i,v in enumerate(arrays)]
    actual=[torch.nn.Parameter(p.detach().clone()) for p in ref]
    def make(params):
        opt=bnb.optim.AdamW8bit(params,lr=.001,weight_decay=0.)
        opt.check_overrides();opt.to_gpu();opt.initialized=True
        for gi,g in enumerate(opt.param_groups):
            for pi,parameter in enumerate(g['params']):opt.init_state(g,parameter,gi,pi)
        return opt
    baseline=make(ref);candidate=make(actual);off=OptimizerStateOffloader(candidate)
    steps=[];offloaded=off.offload()
    assert offloaded>0
    assert all(v.device.type=='cpu' for state in candidate.state.values() for v in state.values() if isinstance(v,torch.Tensor))
    def compare():
        assert all(torch.equal(x,y) for x,y in zip(ref,actual)), 'Parameter update differs'
        for x,y in zip(ref,actual):
            assert baseline.state[x].keys()==candidate.state[y].keys()
            for k,v in baseline.state[x].items():
                w=candidate.state[y][k]
                assert torch.equal(v.cpu(),w.cpu()) if isinstance(v,torch.Tensor) else v==w, k
    for step in range(4):
        for i,(x,y) in enumerate(zip(ref,actual)):
            grad=torch.randn(x.shape,generator=torch.Generator().manual_seed(100+step*2+i))
            x.grad=grad.to(x.device);y.grad=grad.to(y.device)
        baseline.step();off.restore();candidate.step();off.offload();compare()
        steps.append(dict(update=step+1,parameter_bitwise_equal=True,state_bitwise_equal=True))
        if step==1:
            path=a.output.parent/'optimizer_resume_probe.pt'
            torch.save(dict(parameters=[p.detach().cpu() for p in actual],optimizer=candidate.state_dict()),path)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            actual=[torch.nn.Parameter(v.to('cuda:'+str(i))) for i,v in enumerate(payload['parameters'])]
            candidate=make(actual);candidate.load_state_dict(payload['optimizer']);off=OptimizerStateOffloader(candidate);off.offload();compare()
    report=dict(passed=True,scope='Synthetic FP32 parameters on both authorized GPUs; exact AdamW8bit updates and all optimizer state across CPU offload; save at2/resume through4. Real4B capacity/resume tested separately.',initial_offloaded_bytes=offloaded,steps=steps,physical_gpus=a.physical_gpus)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
