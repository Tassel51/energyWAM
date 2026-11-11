"""One shared (optionally layer-sharded) model; exact BI microbatch accumulation."""
import time
import os
import torch
from .trainer import EnergyWAMTrainer
from .replicated import slice_batch,accumulate_log


class SharedMicrobatchTrainer(EnergyWAMTrainer):
    def train_step(self,batch):
        if self.cfg.stage.stage!='bi' or not self.cfg.stage.bi_both_directions_each_batch:
            raise ValueError('shared microbatch runner requires both BI directions')
        size=int(batch['action_target'].shape[0])
        micro_size=self.cfg.trainer.shared_microbatch_size
        if type(micro_size) is not int or micro_size<1:raise ValueError('positive shared_microbatch_size required')
        if micro_size>1 and not (self.cfg.trainer.fixed_candidate_initialization or os.environ.get('ENERGYWAM_SAMPLE_ALIGNED_RNG')=='1'):
            raise ValueError('native microbatch requires sample-aligned or fixed candidate seeds')
        if micro_size>1 and self.cfg.framework.energy_head_dropout!=0:
            raise ValueError('native microbatch equivalence requires zero energy-head dropout')
        if size<1 or not self.cfg.data.require_full_horizon:
            raise ValueError('shared microbatch runner requires a positive batch and full horizon')
        for key in ['action_mask','future_mask']:
            if key in batch and not torch.as_tensor(batch[key]).all():raise ValueError('padding invalidates equal microbatch weighting')
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        if self.device.type=='cuda':
            for dev in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(dev)
        start=time.time();forward=backward=0.;total=0.;direction_logs=[];head_action=None
        only=os.environ.get('ENERGYWAM_ONLY_DIRECTION')
        if only not in [None,'','action','future']:raise ValueError('invalid diagnostic direction')
        for role,direction in enumerate(['action','future']):
            if only and direction!=only:continue
            # Same direction-specific candidate RNG as the replicated baseline.
            torch.manual_seed(self.cfg.trainer.seed+role+self.global_step*100003)
            logs={}
            for index in range(0,size,micro_size):
                end=min(size,index+micro_size)
                if self.cfg.trainer.fixed_candidate_initialization:
                    from .candidate_seeds import fixed_candidate_seed
                    if 'episode_index' not in batch or 'frame_index' not in batch:
                        raise ValueError('fixed candidate initialization needs episode/frame metadata')
                    torch.manual_seed(fixed_candidate_seed(self.cfg.trainer.seed,role,
                        batch['episode_index'][index],batch['frame_index'][index]))
                elif os.environ.get('ENERGYWAM_SAMPLE_ALIGNED_RNG')=='1':
                    torch.manual_seed(self.cfg.trainer.seed+role+(self.global_step*size+index)*100003)
                micro=dict(slice_batch(batch,index,end),_bi_only=direction,_global_step=self.global_step)
                if micro_size>1:
                    if self.cfg.trainer.fixed_candidate_initialization:
                        micro['_candidate_seeds']=[fixed_candidate_seed(self.cfg.trainer.seed,role,
                            batch['episode_index'][i],batch['frame_index'][i]) for i in range(index,end)]
                    else:
                        micro['_candidate_seeds']=[self.cfg.trainer.seed+role+(self.global_step*size+i)*100003 for i in range(index,end)]
                begin=time.time();result=self.model(micro);forward+=time.time()-begin
                term=result['action_loss']/size if micro_size==1 else result['action_loss']*((end-index)/size)
                if not torch.isfinite(term):raise RuntimeError('nonfinite conditional loss')
                begin=time.time();term.backward();backward+=time.time()-begin
                total+=float(term.detach())
                for key,value in result.items():
                    if key!='action_loss' and isinstance(value,torch.Tensor) and value.ndim==0:
                        accumulate_log(logs,key,float(value.detach()),(end-index)/size)
                del result,term,micro
            direction_logs.append(logs)
            if self.cfg.trainer.directional_head_diagnostics and not only and direction=='action':
                head_action=[(p,None if p.grad is None else p.grad.detach().clone())
                             for p in self.model.core.energy_head.parameters()]
            if self.device.type=='cuda':
                for dev in range(torch.cuda.device_count()):torch.cuda.synchronize(dev)
                if os.environ.get('ENERGYWAM_TRIM_BETWEEN_DIRECTIONS','1')=='1' and hasattr(torch._C,'_host_emptyCache'):torch._C._host_emptyCache()
        combined={}
        for key in set().union(*(d.keys() for d in direction_logs)):
            values=[d[key] for d in direction_logs if key in d]
            for value in values:accumulate_log(combined,key,value,1/len(values))
        result={key:torch.tensor(value) for key,value in combined.items()}
        if head_action is not None:
            # FP32 head only; this does not describe all 4B shared parameters.
            a2=z2=dot=0.
            with torch.no_grad():
                for parameter,action in head_action:
                    action=torch.zeros_like(parameter) if action is None else action
                    total_gradient=torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
                    future=total_gradient-action
                    a2+=float(action.float().square().sum())
                    z2+=float(future.float().square().sum())
                    dot+=float((action.float()*future.float()).sum())
            norm_a,norm_z=a2**.5,z2**.5
            result['diag/head_action_gradient_norm']=torch.tensor(norm_a)
            result['diag/head_weighted_future_gradient_norm']=torch.tensor(norm_z)
            result['diag/head_direction_cosine']=torch.tensor(dot/max(norm_a*norm_z,1e-30))
            if norm_z>0 and self.cfg.loss.lambda_action>0 and self.cfg.loss.lambda_future>0:
                result['diag/head_unweighted_norm_ratio_action_future']=torch.tensor(
                    norm_a*self.cfg.loss.lambda_future/norm_z/max(self.cfg.loss.lambda_action,1e-30))
        result['action_loss']=torch.tensor(total)
        result['shared_model_instances']=torch.tensor(1.)
        return self._finish_step(batch,result,start,forward,backward)
