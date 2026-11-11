"""Read-only training observers: no extra backwards or optimizer updates."""
import json
import math
from pathlib import Path
import torch


def rms_rows(tensor):
    value=tensor.detach().float().reshape(tensor.shape[0],-1)
    return (value.norm(dim=1)/math.sqrt(value.shape[1])).cpu().tolist()


def grouped_norm(rows, key):
    values={}
    for row in rows:
        group=row['group'];values[group]=values.get(group,0.)+row[key]**2
    return {k:math.sqrt(v) for k,v in values.items()}


class ReplayObserver:
    def __init__(self, trainer, output):
        self.trainer=trainer;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.parameters=[(n,p) for n,p in trainer.model.named_parameters() if p.requires_grad]
        self.groups={id(p):g.get('group_name','main') for g in trainer.optimizer.param_groups for p in g['params']}
        self.context=None;self.last_context=None;self.outer=False;self.micro=0;self.grad_calls=[];self.handles=[]

    def emit(self, filename, row):
        with (self.output/filename).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    def identity(self):
        return dict(update=self.trainer.global_step+1)

    def grad_snapshot(self):
        with torch.no_grad():
            return [dict(name=n,group=self.groups[id(p)],norm=float(p.grad.detach().norm()),numel=p.numel())
                    for n,p in self.parameters if p.grad is not None]

    def install(self):
        import energywam.energy.solver as solver
        self.solver=solver;self.old_step=solver._step
        self.old_bi=self.trainer.model.stage_runner._run_bi
        self.old_backward=torch.Tensor.backward
        self.old_clip=torch.nn.utils.clip_grad_norm_
        self.old_opt=self.trainer.optimizer.step
        observer=self

        def run_bi(cond,batch,training):
            observer.micro+=1
            count=batch['action_target'].shape[0]
            def ids(key):
                x=batch.get(key)
                if x is None:return [None]*count
                return x.detach().cpu().tolist() if isinstance(x,torch.Tensor) else list(x)
            context=dict(**observer.identity(),micro=observer.micro,direction=batch.get('_bi_only'),
                         episode=ids('episode_index'),frame=ids('frame_index'),
                         candidate_seeds=batch.get('_candidate_seeds'),iteration=0,batch=batch)
            observer.context=context;observer.last_context=context
            with torch.no_grad():
                data=dict(**observer.identity(),micro=observer.micro,direction=context['direction'],episode=context['episode'],frame=context['frame'],fields={})
                for key in ['action_target','future_target','state','current_dino','action_mask','future_mask']:
                    t=batch.get(key)
                    if isinstance(t,torch.Tensor):
                        v=t.detach().float();data['fields'][key]=dict(shape=list(v.shape),rms=rms_rows(v),absmax=float(v.abs().max()),nonfinite=int((~torch.isfinite(v)).sum()))
                observer.emit('input_statistics.jsonl',data)
            try:return observer.old_bi(cond,batch,training)
            finally:observer.context=None

        def step(*args,**kwargs):
            ctx=observer.context
            if ctx is None:return observer.old_step(*args,**kwargs)
            iteration=ctx['iteration'];ctx['iteration']+=1
            direction=ctx['direction'];before=args[1] if direction=='action' else args[2]
            target=ctx['batch']['action_target' if direction=='action' else 'future_target']
            with torch.no_grad():mse_before=[v*v for v in rms_rows(before.detach()-target.detach())]
            result=observer.old_step(*args,**kwargs)
            after=result[0] if direction=='action' else result[1]
            state=kwargs.get('state')
            with torch.no_grad():
                mse_after=[v*v for v in rms_rows(after.detach()-target.detach())]
                update=rms_rows(after.detach()-before.detach())
                grads=(state.action_grad_norms if direction=='action' else state.future_grad_norms)[-1]
                gradient=(grads/math.sqrt(after[0].numel())).cpu().tolist()
                energy=state.energies[-1].cpu().tolist()
                for i in range(after.shape[0]):
                    observer.emit('window_inner_trace.jsonl',dict(update=ctx['update'],micro=ctx['micro'],direction=direction,
                        iteration=iteration,episode=ctx['episode'][i],frame=ctx['frame'][i],sample_in_micro=i,
                        candidate_seed=ctx['candidate_seeds'][i] if ctx['candidate_seeds'] else None,
                        mse_before=mse_before[i],mse_after=mse_after[i],candidate_update_rms=update[i],candidate_gradient_rms=gradient[i],energy=energy[i]))
            def adjoint(grad):
                if observer.outer:
                    observer.emit('candidate_adjoint.jsonl',dict(update=ctx['update'],micro=ctx['micro'],direction=direction,
                        iteration=iteration,episode=ctx['episode'],frame=ctx['frame'],gradient_rms=rms_rows(grad)))
                return None
            after.register_hook(adjoint)
            return result

        def param_hook(name,p):
            def hook(grad):
                if observer.outer:
                    observer.grad_calls.append(dict(name=name,group=observer.groups[id(p)],norm=float(grad.detach().norm())))
                return None
            return hook

        def backward(tensor,*args,**kwargs):
            observer.outer=True;observer.grad_calls=[]
            try:result=observer.old_backward(tensor,*args,**kwargs)
            finally:observer.outer=False
            ctx=observer.last_context
            observer.emit('micro_parameter_gradients.jsonl',dict(update=ctx['update'],micro=ctx['micro'],direction=ctx['direction'],
                episode=ctx['episode'],frame=ctx['frame'],incoming_gradient_calls=observer.grad_calls,
                accumulated_parameters=observer.grad_snapshot()))
            return result

        def clip(*args,**kwargs):
            before=observer.grad_snapshot();result=observer.old_clip(*args,**kwargs);after=observer.grad_snapshot()
            observer.emit('gradient_clipping.jsonl',dict(**observer.identity(),threshold=observer.trainer.cfg.trainer.max_grad_norm,
                before_groups=grouped_norm(before,'norm'),after_groups=grouped_norm(after,'norm'),
                before_total=math.sqrt(sum(x['norm']**2 for x in before)),after_total=math.sqrt(sum(x['norm']**2 for x in after)),before_parameters=before,after_parameters=after))
            return result

        def opt(*args,**kwargs):
            # Off-GPU snapshots avoid retaining another 4B GPU model.
            with torch.no_grad():previous=[p.detach().to('cpu',copy=True) for _,p in observer.parameters]
            result=observer.old_opt(*args,**kwargs)
            values=[]
            with torch.no_grad():
                for (name,p),old in zip(observer.parameters,previous):
                    new=p.detach().to('cpu',copy=True);flat_old=old.view(-1);flat_new=new.view(-1)
                    norm2=delta2=0.;max_delta=0.
                    for start in range(0,old.numel(),1048576):
                        o=flat_old[start:start+1048576].double();d=flat_new[start:start+1048576].double()-o
                        norm2+=float(o.square().sum());delta2+=float(d.square().sum());max_delta=max(max_delta,float(d.abs().max()))
                    norm=math.sqrt(norm2);delta=math.sqrt(delta2)
                    values.append(dict(name=name,group=observer.groups[id(p)],parameter_norm=norm,update_norm=delta,
                        relative_update=delta/max(norm,1e-30),max_abs_update=max_delta,numel=p.numel()))
                    del new
            observer.emit('parameter_updates.jsonl',dict(**observer.identity(),parameters=values,
                parameter_group_norms=grouped_norm(values,'parameter_norm'),update_group_norms=grouped_norm(values,'update_norm')))
            return result

        opt._wrapped_by_lr_sched=getattr(self.old_opt,'_wrapped_by_lr_sched',False)
        self.trainer.model.stage_runner._run_bi=run_bi;solver._step=step
        torch.Tensor.backward=backward;torch.nn.utils.clip_grad_norm_=clip;self.trainer.optimizer.step=opt
        self.handles=[p.register_hook(param_hook(n,p)) for n,p in self.parameters]
        return self

    def remove(self):
        self.trainer.model.stage_runner._run_bi=self.old_bi;self.solver._step=self.old_step
        torch.Tensor.backward=self.old_backward;torch.nn.utils.clip_grad_norm_=self.old_clip
        self.trainer.optimizer.step=self.old_opt
        for handle in self.handles:handle.remove()
