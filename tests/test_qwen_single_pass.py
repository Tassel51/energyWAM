"""Count actual Qwen forwards on CPU; random small Qwen, no pretrained 4B.

The processor and DINO are test fixtures. The vision tower, embedding table,
language transformer and decoder layers are the actual Transformers modules.
"""

from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch

import torch

from .harness import assert_nonzero_finite, assert_true
from .real_fixtures import build_real_model, make_real_batch


def _fixture(seed=301):
    model, cfg, geometry = build_real_model(
        with_real_dino=False, seed=seed,
        **{"backbone.dino_hidden_dim": 8, "backbone.dino_num_patches": 4},
    )
    model.eval()
    return model, cfg, geometry, make_real_batch(model, cfg, geometry, B=2)


@contextmanager
def _forwards(model):
    counts = Counter()
    language_inputs = []
    backbone = model.backbone
    mm = backbone._mm
    modules = {
        "generation_model": backbone.model,
        "multimodal_model": mm,
        "vision": mm.visual,
        "embedding": mm.get_input_embeddings(),
        "language": mm.language_model,
        "lm_head": backbone.model.lm_head,
    }
    modules.update({f"layer_{i}": layer for i, layer in enumerate(mm.language_model.layers)})
    handles = []

    def recorder(name):
        def hook(module, args, kwargs):
            counts[name] += 1
            if name == "language":
                language_inputs.append(kwargs["inputs_embeds"].detach().clone())
        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(recorder(name), with_kwargs=True))
        yield counts, language_inputs
    finally:
        for handle in handles:
            handle.remove()


def _assert_language_count(model, counts, expected):
    assert_true(counts["language"] == expected, f"language forwards: {counts}")
    for i in range(len(model.backbone._mm.language_model.layers)):
        assert_true(counts[f"layer_{i}"] == expected, f"decoder layer {i}: {counts}")
    for key in ("generation_model", "multimodal_model", "lm_head"):
        assert_true(counts[key] == 0, f"unexpected full VLM/vocab forward: {counts}")


def test_condition_is_embedding_only():
    model, cfg, geometry, batch = _fixture()
    with _forwards(model) as (counts, language_inputs):
        cond = model.backbone.encode_condition(batch)
    assert_true(counts["vision"] == 1 and counts["embedding"] == 1, str(counts))
    _assert_language_count(model, counts, 0)
    assert_true(not language_inputs, "condition encoder called the language model")
    assert_true(cond.deepstack_visual_embeds is not None, "native DeepStack features lost")
    native = model.backbone._build_native_inputs(batch["images"], batch["langs"], model.device)
    lookup = model.backbone._mm.get_input_embeddings()(native["input_ids"])
    nonvisual = cond.native_attention_mask.bool() & ~cond.visual_pos_masks
    torch.testing.assert_close(cond.native["inputs_embeds"][nonvisual], lookup[nonvisual], rtol=0, atol=0)


def test_one_language_pass_per_energy_and_reused_condition():
    model, cfg, geometry, batch = _fixture(seed=302)
    a = torch.randn(2, 8, 3, requires_grad=True)
    z = torch.randn(2, 3, 1, 4, 8, requires_grad=True)
    with _forwards(model) as (counts, language_inputs):
        cond = model.backbone.encode_condition(batch)
        before = cond.native["inputs_embeds"].detach().clone()
        e1 = model.core.energy(cond, z, a)
        _assert_language_count(model, counts, 1)
        expected = before.shape[1] + cond.group.embeds.shape[1] + 3 * 1 * 4 + 8 + 1
        assert_true(language_inputs[0].shape[1] == expected, "LLM did not receive the complete sequence")
        torch.testing.assert_close(language_inputs[0][:, :before.shape[1]], before, rtol=0, atol=0)
        e2 = model.core.energy(cond, z + 0.1, a - 0.1)
        _assert_language_count(model, counts, 2)
        assert_true(counts["vision"] == 1 and counts["embedding"] == 1, "condition was encoded again")
        assert_true(not torch.equal(language_inputs[0], language_inputs[1]), "updated candidates not passed to LLM")
        torch.testing.assert_close(cond.native["inputs_embeds"], before, rtol=0, atol=0)
    ga, gz = torch.autograd.grad(e1.sum(), (a, z), create_graph=True)
    assert_nonzero_finite(ga, "action gradient")
    assert_nonzero_finite(gz, "future gradient")
    model.zero_grad(set_to_none=True)
    # Differentiating the candidate gradients exercises double backward.
    (ga.square().sum() + gz.square().sum()).backward()
    grads = [p.grad for p in model.backbone._mm.language_model.parameters() if p.grad is not None]
    assert_true(bool(grads), "no language parameter gradient through candidate gradients")
    assert_true(all(bool(torch.isfinite(g).all()) for g in grads), "non-finite second-order gradient")
    assert_true(any(bool(g.abs().sum() > 0) for g in grads), "zero second-order language gradient")


def test_training_and_prediction_have_no_extra_language_pass():
    model, cfg, geometry, batch = _fixture(seed=303)
    for stage in ("bi", "joint"):
        cfg.stage.stage = stage
        with _forwards(model) as (counts, _), patch.object(model.core, "energy", wraps=model.core.energy) as energy:
            result = model(batch)
            assert_true(bool(torch.isfinite(result["action_loss"])), f"{stage}: non-finite loss")
            assert_true(energy.call_count > 0, f"{stage}: no energy evaluation")
            _assert_language_count(model, counts, energy.call_count)
            expected_encodes = 2 if stage == "bi" else 1
            assert_true(counts["vision"] == expected_encodes and counts["embedding"] == expected_encodes, f"{stage}: {counts}")
    slim = {key: batch[key] for key in ("images", "langs", "states")}
    with _forwards(model) as (counts, _), patch.object(model.core, "energy", wraps=model.core.energy) as energy:
        with torch.inference_mode():
            result = model.predict(slim, mode="joint")
        assert_true(result["normalized_actions"].shape == (2, 8, 3), "prediction shape")
        assert_true(energy.call_count > 0, "prediction: no energy evaluation")
        _assert_language_count(model, counts, energy.call_count)
        assert_true(counts["vision"] == 1 and counts["embedding"] == 1, f"prediction: {counts}")


TESTS = [
    ("condition_is_embedding_only", test_condition_is_embedding_only),
    ("one_language_pass_per_energy_and_reused_condition", test_one_language_pass_per_energy_and_reused_condition),
    ("training_and_prediction_have_no_extra_language_pass", test_training_and_prediction_have_no_extra_language_pass),
]
