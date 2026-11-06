"""Native batching must preserve per-window initialization and mean gradients."""
import copy,torch,pytest
from .real_fixtures import build_real_model,make_real_batch
from energywam.training.shared_microbatch import SharedMicrobatchTrainer


@pytest.mark.parametrize('batch_size',[2,3])
@pytest.mark.parametrize('prior',[False,True])
def test_real_qwen_native_microbatch_matches_sequential_and_remainder(tmp_path,monkeypatch,batch_size,prior):
 monkeypatch.setenv('ENERGYWAM_SAMPLE_ALIGNED_RNG','1')
 model,cfg,geo=build_real_model(with_real_dino=False,seed=431,**{
  'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,
  'backbone.dino_num_patches':4,'framework.dino_patch_pack_factor':2,
  'framework.iteration_conditioning':True,'framework.iteration_embedding_dim':16,
  'framework.iteration_mlp_hidden':16,'solver.learnable_step_sizes':True,
  'solver.alpha_action':.05,'solver.alpha_future':.1,
  'framework.quadratic_candidate_prior':prior})
 cfg.trainer.max_grad_norm=0;cfg.trainer.fp32_master_weights=False;cfg.trainer.output_dir=str(tmp_path/'sequential')
 batch=make_real_batch(model,cfg,geo,B=batch_size)
 batch['langs']=['open drawer','please open the cabinet now','move'][0:batch_size]
 native=copy.deepcopy(model);cfg2=copy.deepcopy(cfg);native.cfg=cfg2;native.stage_runner.cfg=cfg2;native.core.cfg=cfg2
 cfg2.trainer.shared_microbatch_size=2;cfg2.trainer.output_dir=str(tmp_path/'native')
 first=SharedMicrobatchTrainer(cfg,model,optimizer=torch.optim.SGD(model.parameters(),lr=0.))
 second=SharedMicrobatchTrainer(cfg2,native,optimizer=torch.optim.SGD(native.parameters(),lr=0.))
 a=first.train_step(batch);b=second.train_step(batch)
 assert a['loss']==pytest.approx(b['loss'],rel=1e-4,abs=1e-6)
 for (name,p),(other,q) in zip(model.named_parameters(),native.named_parameters()):
  assert name==other and (p.grad is None)==(q.grad is None)
  if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-3,atol=2e-6,msg=name)
