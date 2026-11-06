import torch
from energywam.training.offload import parameter_aware_cpu_offload

class RepeatedSave(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):ctx.save_for_backward(x,x);return x*x
 @staticmethod
 def backward(ctx,g):
  a,b=ctx.saved_tensors
  return g*(a+b)

def test_dedup_preserves_full_second_derivative_and_reuses_live_saved_tensor():
 model=torch.nn.Linear(1,1);results=[]
 for enabled in [False,True]:
  x=torch.tensor([.7,-.4],requires_grad=True)
  with parameter_aware_cpu_offload(model,deduplicate_cpu=enabled) as stats:
   y=RepeatedSave.apply(x).square().sum()
   first=torch.autograd.grad(y,x,create_graph=True)[0]
  second=torch.autograd.grad(first.sum(),x)[0]
  results.append((first.detach(),second,stats))
 torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
 torch.testing.assert_close(results[0][1],results[1][1],rtol=0,atol=0)
 torch.testing.assert_close(results[1][1],12*torch.tensor([.7,-.4]).square())
 assert results[1][2]['cpu_dedup_hits']>0
