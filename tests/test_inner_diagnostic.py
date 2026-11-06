import torch
from energywam.energy.inner_diagnostic import trace_candidate
from energywam.energy.solver import refine


def test_trace_matches_production_for_both_conditionals():
    torch.manual_seed(42)
    a=torch.randn(1,3,2);z=torch.randn(1,2,4)
    astar=torch.randn_like(a);zstar=torch.randn_like(z)
    def energy(a,z):return (a.flatten(1).sum(1)+z.flatten(1).sum(1)).square()/10+a.square().flatten(1).mean(1)+z.square().flatten(1).mean(1)
    for mode in ['action_given_future','future_given_action']:
        action=mode=='action_given_future'
        init,target=(a,astar) if action else (z,zstar)
        fn=(lambda x:energy(x,zstar)) if action else (lambda x:energy(astar,x))
        rows,final=trace_candidate(fn,init,target,.05,4)
        ap,zp,_=refine(energy,action=a if action else astar,future=zstar if action else z,
                        mode=mode,num_steps=4,training=False,alpha_action=.05,alpha_future=.05)
        torch.testing.assert_close(final,ap if action else zp,rtol=0,atol=0)
        assert rows[0]['prediction_mse']==float((init-target).square().mean())
        assert rows[0]['applied_update_rms']==0
        assert len(rows)==5 and rows[-1]['k']==4
        assert rows[-1]['energy']<=rows[0]['energy']


def test_energy_decrease_is_not_assumed_to_improve_target_error():
    initial=torch.tensor([[1.]])
    rows,_=trace_candidate(lambda x:x.square().sum(1),initial,torch.tensor([[3.]]),.1,4)
    assert rows[-1]['energy']<rows[0]['energy']
    assert rows[-1]['prediction_mse']>rows[0]['prediction_mse']


def test_long_trace_preserves_short_prefix():
    torch.manual_seed(7)
    x=torch.randn(1,3,2);target=torch.randn_like(x)
    energy=lambda a:(a-target).square().flatten(1).sum(1)
    short,_=trace_candidate(energy,x,target,.05,4)
    long,_=trace_candidate(energy,x,target,.05,32)
    for a,b in zip(short,long):
        for key in a:
            if key!='refinement_seconds':assert a[key]==b[key]
    assert len(long)==33 and long[-1]['prediction_mse']<short[-1]['prediction_mse']
    assert long[-1]['refinement_seconds']>=long[4]['refinement_seconds']
