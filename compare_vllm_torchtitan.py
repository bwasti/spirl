"""
Direct comparison between vLLM and TorchTitan forward passes.

This script runs the same input through both implementations and compares
outputs at every layer to find where they diverge.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

import torchtitan.experiments.compat
from torchtitan.models.qwen3.model.model import Qwen3Model
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from weights import vllm_to_torchtitan, torchtitan_to_vllm

# We'll enable batch_invariant_backward mode later, after loading vLLM
# vLLM needs to set up its own batch_invariant mode first
from batch_invariant_backward import enable_batch_invariant_backward_mode, disable_batch_invariant_backward_mode
from vllm.model_executor.layers.batch_invariant import disable_batch_invariant_mode


def load_torchtitan_model(model_path: str, checkpoint_path: str) -> Qwen3Model:
    """Load TorchTitan model."""
    print(f"\nLoading TorchTitan model...")

    # Load config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model_args = Qwen3ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
        qk_norm=True,
        depth_init=True,
        eos_id=getattr(hf_config, "eos_token_id", 151645),
    )

    model = Qwen3Model(model_args)
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16)
    model.eval()

    return model


def load_vllm_model(model_path: str, checkpoint_path: str):
    """Load vLLM model."""
    print(f"\nLoading vLLM model...")

    from vllm import LLM
    import tempfile
    import os
    import shutil
    from safetensors.torch import save_file

    # Create temp directory with vLLM weights
    temp_dir = tempfile.mkdtemp(prefix="vllm_compare_")

    # Copy config files
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "merges.txt", "vocab.json"]:
        src = os.path.join(model_path, file)
        if os.path.exists(src):
            shutil.copy2(src, temp_dir)

    # Convert and save weights
    titan_state = load_file(checkpoint_path)
    vllm_state = torchtitan_to_vllm(titan_state)
    save_file(vllm_state, os.path.join(temp_dir, "model.safetensors"))

    # Load with vLLM
    llm = LLM(
        model=temp_dir,
        trust_remote_code=True,
        max_model_len=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
    )

    return llm, temp_dir


@torch.no_grad()
def compare_forward_passes(
    titan_model: Qwen3Model,
    vllm_llm,
    tokenizer,
    prompt: str,
    max_tokens: int = 1,
    debug: bool = False,
):
    """Compare forward passes token by token."""

    print(f"\n{'='*80}")
    print(f"Comparing forward passes")
    print(f"{'='*80}")
    print(f"Prompt: '{prompt}'")

    device = next(titan_model.parameters()).device

    # Tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Input tokens: {input_ids}")
    print(f"Input length: {len(input_ids)}")

    # Setup debug tracker if requested
    titan_tracker = None
    if debug:
        from debug_model import instrument_qwen3_model
        titan_tracker = instrument_qwen3_model(titan_model)

    # TorchTitan forward pass
    print(f"\n--- TorchTitan Forward Pass ---")
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    titan_logits = titan_model(input_tensor)
    titan_logits_last = titan_logits[0, -1, :].float()
    titan_log_probs = F.log_softmax(titan_logits_last, dim=-1)
    titan_top_token = torch.argmax(titan_log_probs).item()
    titan_top_logprob = titan_log_probs[titan_top_token].item()

    print(f"Output shape: {titan_logits.shape}")
    print(f"Top token: {titan_top_token} ('{tokenizer.decode([titan_top_token])}')")
    print(f"Top logprob: {titan_top_logprob:.6f}")

    # vLLM forward pass
    print(f"\n--- vLLM Forward Pass ---")
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=1,
    )

    outputs = vllm_llm.generate([prompt], sampling_params)
    output = outputs[0].outputs[0]
    vllm_top_token = output.token_ids[0]
    vllm_logprobs = output.logprobs[0]
    vllm_top_logprob = list(vllm_logprobs.values())[0].logprob

    print(f"Top token: {vllm_top_token} ('{tokenizer.decode([vllm_top_token])}')")
    print(f"Top logprob: {vllm_top_logprob:.6f}")

    # Compare
    print(f"\n{'='*80}")
    print(f"Comparison Results")
    print(f"{'='*80}")

    tokens_match = (titan_top_token == vllm_top_token)
    logprob_diff = abs(titan_top_logprob - vllm_top_logprob)

    print(f"Tokens match: {tokens_match}")
    if tokens_match:
        print(f"  ✓ Both predict: {titan_top_token} ('{tokenizer.decode([titan_top_token])}')")
    else:
        print(f"  ✗ TorchTitan: {titan_top_token} ('{tokenizer.decode([titan_top_token])}')")
        print(f"  ✗ vLLM:       {vllm_top_token} ('{tokenizer.decode([vllm_top_token])}')")

    print(f"\nLogprob difference: {logprob_diff:.8f}")
    if logprob_diff < 1e-4:
        print(f"  ✓ Excellent match (< 0.0001)")
    elif logprob_diff < 1e-2:
        print(f"  ⚠ Close match (< 0.01)")
    else:
        print(f"  ✗ Large difference (>= 0.01)")

    # Compare full distributions (top-10 tokens)
    print(f"\n--- Top 10 Token Comparison ---")
    titan_top10 = torch.topk(titan_log_probs, 10)

    print(f"{'Rank':<6} {'TorchTitan':<40} {'vLLM':<40} {'Delta':<10}")
    print(f"{'-'*100}")

    for rank in range(10):
        titan_tok = titan_top10.indices[rank].item()
        titan_lp = titan_top10.values[rank].item()
        titan_str = tokenizer.decode([titan_tok])

        # Find this token in vLLM logprobs
        vllm_lp = None
        if titan_tok in [k for k, v in vllm_logprobs.items()]:
            vllm_lp = vllm_logprobs[titan_tok].logprob

        if vllm_lp is not None:
            delta = abs(titan_lp - vllm_lp)
            print(f"{rank+1:<6} {titan_tok:>6} ('{titan_str:>10s}'): {titan_lp:8.4f}    "
                  f"{titan_tok:>6} ('{titan_str:>10s}'): {vllm_lp:8.4f}    {delta:8.6f}")
        else:
            print(f"{rank+1:<6} {titan_tok:>6} ('{titan_str:>10s}'): {titan_lp:8.4f}    "
                  f"{'--':>6} {'(not in top)':>15s}  {'--':>10}")


def main():
    """Main comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare vLLM and TorchTitan forward passes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with layer-by-layer tracking")
    parser.add_argument("--prompt", type=str, help="Custom prompt to test")
    args = parser.parse_args()

    # Paths
    model_path = "./models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    checkpoint_path = "./converted/qwen3_torchtitan.safetensors"

    # Load vLLM first (it will set up its own batch_invariant mode)
    vllm_llm, temp_dir = load_vllm_model(model_path, checkpoint_path)

    # Now enable our batch_invariant_backward mode for TorchTitan
    print("\nEnabling batch_invariant_backward mode for TorchTitan...")
    disable_batch_invariant_mode()  # Disable vLLM's (no backward support)
    enable_batch_invariant_backward_mode()  # Enable ours (with backward support)

    # Load TorchTitan model
    titan_model = load_torchtitan_model(model_path, checkpoint_path)
    titan_model = titan_model.cuda()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Test prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "The capital of France is",
            "What is 7 times 8?",
            "Hello, my name is",
        ]

    for prompt in prompts:
        compare_forward_passes(titan_model, vllm_llm, tokenizer, prompt, debug=args.debug)
        print("\n")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    main()
