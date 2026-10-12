"""Offline Gemma 4 logit parity against independent Hugging Face models.

Locally saved, seeded checkpoints exercise PLE/KV sharing, K=V attention and MoE
through the real boot path, without relying on mutable Hub fixtures.
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    PreTrainedTokenizerFast,
)
from transformers.cache_utils import DynamicCache

from transformer_lens.model_bridge import TransformerBridge

VARIANTS = ("ple_kv_shared", "dense_k_eq_v", "moe")
IDS = torch.tensor([[2, 3, 4, 5, 6]])


@pytest.fixture(scope="module", params=VARIANTS, ids=VARIANTS)
def model_pair(request, tmp_path_factory):
    variant = request.param
    text_config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=4,
        max_position_embeddings=64,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8 if variant == "ple_kv_shared" else 0,
        num_kv_shared_layers=2 if variant == "ple_kv_shared" else 0,
        use_double_wide_mlp=variant == "ple_kv_shared",
        attention_k_eq_v=variant != "ple_kv_shared",
        enable_moe_block=variant == "moe",
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=16,
        final_logit_softcapping=0.3,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        patch_size=2,
        position_embedding_size=32,
        pooling_kernel_size=1,
    )
    config = Gemma4Config(text_config=text_config, vision_config=vision_config, audio_config=None)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Gemma4ForConditionalGeneration(config).eval()
    with torch.no_grad():
        for index, layer in enumerate(model.model.language_model.layers):
            layer.layer_scalar.fill_(0.7 + index * 0.1)
    path = tmp_path_factory.mktemp(variant)
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3, "a": 4}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    # Separate loads prevent wrapping or hooks from contaminating the HF oracle.
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    bridge = TransformerBridge.boot_transformers(
        str(path), device="cpu", dtype=torch.float32
    ).eval()
    return hf, bridge


@pytest.fixture(scope="module")
def bridge(model_pair):
    return model_pair[1]


def assert_logit_parity(actual, expected):
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seq_len", [1, 5, 11])
@pytest.mark.parametrize("use_cache", [False, True])
def test_text_only_logit_parity_vs_hf(model_pair, seq_len, use_cache):
    hf, bridge = model_pair
    ids = torch.arange(2, 2 + seq_len).unsqueeze(0)
    with torch.no_grad():
        ref = hf(ids, use_cache=use_cache).logits
        out = bridge(ids, use_cache=use_cache)
    assert_logit_parity(out, ref)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padded_batch_logit_parity(model_pair, padding_side):
    hf, bridge = model_pair
    ids = torch.arange(2, 13).repeat(2, 1)
    mask = torch.ones_like(ids)
    padding = slice(0, 3) if padding_side == "left" else slice(-3, None)
    ids[0, padding] = 0
    mask[0, padding] = 0
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    kwargs = dict(attention_mask=mask, position_ids=positions, use_cache=False)
    with torch.no_grad():
        ref = hf(ids, **kwargs).logits
        out = bridge(ids, **kwargs)
    assert_logit_parity(out[mask.bool()], ref[mask.bool()])


def test_cached_decode_logit_parity(model_pair):
    hf, bridge = model_pair
    hf_cache = DynamicCache(config=hf.config.get_text_config())
    bridge_cache = DynamicCache(config=bridge.original_model.config.get_text_config())
    ids = torch.arange(2, 13).unsqueeze(0)
    with torch.no_grad():
        for tokens in (ids[:, :7], ids[:, 7:8], ids[:, 8:]):
            ref = hf(tokens, past_key_values=hf_cache, use_cache=True).logits
            out = bridge(tokens, past_key_values=bridge_cache, use_cache=True)
            assert_logit_parity(out, ref)


def test_architecture_features_are_exercised(model_pair):
    hf, bridge = model_pair
    config = hf.config.text_config
    layers = hf.model.language_model.layers
    assert {layer.self_attn.head_dim for layer in layers} == {8, 16}
    assert bridge.embed.original_component is not hf.model.language_model.embed_tokens
    assert bridge.original_model.lm_head.weight is bridge.embed.original_component.weight
    if config.num_kv_shared_layers:
        assert not hasattr(layers[2].self_attn, "k_proj")
        assert layers[2].mlp.intermediate_size == 2 * layers[0].mlp.intermediate_size
        assert hasattr(bridge.blocks[0], "per_layer_input_gate")
    if config.attention_k_eq_v:
        assert layers[1].self_attn.v_proj is None
    if config.enable_moe_block:
        assert hasattr(bridge.blocks[0], "router")
        assert hasattr(bridge.blocks[0], "experts")


def test_multimodal_effect_vision_pipeline_bridged(bridge):
    """is_multimodal drives the booted bridge to wire the HF vision pipeline as real
    submodules and to open the multimodal-input gate. Assert those effects, not the flag.

    Text dims still resolve from the nested text_config of the multimodal model.
    """
    assert bridge.cfg.n_layers == 4

    # EFFECT 1: the vision pipeline is bridged. All three fixtures are
    # Gemma4ForConditionalGeneration (vision_tower encoder + embed_vision projector),
    # so both map to real attached nn.Modules on the booted bridge.
    real = bridge.real_components
    assert real["vision_projector"][0] == "model.embed_vision"
    assert real["vision_encoder"][0] == "model.vision_tower"
    assert isinstance(bridge.vision_projector, torch.nn.Module)
    assert isinstance(bridge.vision_encoder, torch.nn.Module)

    # EFFECT 2: the is_multimodal gate is open. The "requires a multimodal model" guard
    # fires only when cfg.is_multimodal is False, so it must NOT fire here. The call may
    # succeed (processor present) or raise a different error (e.g. no processor) — both fine.
    try:
        bridge.prepare_multimodal_inputs("a")
    except ValueError as exc:
        assert "requires a multimodal model" not in str(exc)


def test_resid_hooks_fire_with_conventional_shape(bridge):
    """The residual stream is a single conventional (batch, seq, d_model) tensor."""
    captured = {}

    def cap(tensor, hook):
        captured[hook.name] = tensor.detach()
        return tensor

    names = [
        n
        for n in bridge.hook_dict
        if n.endswith("blocks.0.hook_resid_pre") or n.endswith("blocks.0.hook_resid_post")
    ]
    assert names, "no residual hooks registered"
    with torch.no_grad():
        bridge.run_with_hooks(IDS, fwd_hooks=[(n, cap) for n in names])

    assert captured, "residual hooks did not fire"
    for tensor in captured.values():
        assert tensor.shape == (IDS.shape[0], IDS.shape[1], bridge.cfg.d_model)


def test_run_with_cache_text_only(bridge):
    with torch.no_grad():
        expected = bridge(IDS)
        logits, cache = bridge.run_with_cache(IDS)
    assert_logit_parity(logits, expected)
    assert len(cache) > 0


def test_residual_intervention_matches_hf(model_pair):
    """A live bridge hook must edit the same computation as an HF layer hook."""
    hf, bridge = model_pair
    with torch.no_grad():
        baseline = hf(IDS).logits
        handle = hf.model.language_model.layers[0].register_forward_hook(
            lambda module, args, output: output * 0.5
        )
        try:
            expected = hf(IDS).logits
        finally:
            handle.remove()
        actual = bridge.run_with_hooks(
            IDS, fwd_hooks=[("blocks.0.hook_out", lambda tensor, hook: tensor * 0.5)]
        )
        restored = bridge(IDS)
    assert not torch.allclose(expected, baseline, atol=1e-6, rtol=1e-5)
    assert_logit_parity(actual, expected)
    assert_logit_parity(restored, baseline)
