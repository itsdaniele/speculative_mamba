import argparse
import torch
from transformers import AutoTokenizer
from specmamba.models.mixer_seq_simple_speculative import MambaLMHeadModel
from specmamba.models.mixer_seq_simple import MambaLMHeadModel as MambaLMHeadModelBase


from decode_speculative import decode_speculative


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


MAMBA_MODELS = {
    "mamba_130m": "state-spaces/mamba-130m",
    "mamba_790m": "state-spaces/mamba-790m",
    "mamba_2_8b": "state-spaces/mamba-2.8b",
}


def adjust_llama_layers(model):
    """Adjusts the LLaMA model layers to align with the expected format."""
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.mha.Wqkv.weight.data = torch.cat(
            (
                layer.self_attn.q_proj.weight.data,
                layer.self_attn.k_proj.weight.data,
                layer.self_attn.v_proj.weight.data,
            ),
            dim=0,
        )
        layer.self_attn.mha.out_proj.weight.data = layer.self_attn.o_proj.weight.data


def load_target_model_and_tokenizer(model_name, dtype=None):
    """Loads the appropriate target model and tokenizer."""
    if model_name in MAMBA_MODELS.values():
        model = MambaLMHeadModel.from_pretrained(model_name).to(dtype=dtype).cuda()
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=True)
    else:
        raise ValueError(f"Unknown target model: {model_name}")

    return model, tokenizer


def load_draft_model_and_tokenizer(model_name, dtype=None):
    """Loads the appropriate draft model and adjusts layers if needed."""
    if model_name in MAMBA_MODELS.values():
        model = MambaLMHeadModelBase.from_pretrained(model_name).cuda()
        draft_type = "mamba"
    else:
        raise ValueError(f"Unknown draft model: {model_name}")

    return model, draft_type


def main(prompt, n_tokens_to_generate, K, model_target, model_draft, cg, dtype, args):
    print(
        f"Prompt: {prompt},\n"
        f"n_tokens_to_generate: {n_tokens_to_generate},\n"
        f"K: {K},\n"
        f"dtype: {dtype},\n"
        f"cg: {cg},\n"
        f"model_target: {model_target},\n"
        f"model_draft: {model_draft}"
    )
    model_target, tokenizer = load_target_model_and_tokenizer(model_target, dtype=dtype)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    print(f"# tokens in prompt: {input_ids.shape[1]}")

    model_draft, draft_type = load_draft_model_and_tokenizer(model_draft, dtype=dtype)

    print("Decoding...")
    outputs = decode_speculative(
        input_ids,
        model_target,
        model_draft,
        max_length=n_tokens_to_generate + input_ids.shape[1],
        draft_type=draft_type,
        speculative_lookahead=K,
        cg=cg,
        enable_timing=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    print(tokenizer.decode(outputs[0].sequences[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speculative and baseline decoding models.")
    parser.add_argument("--prompt", type=str, default="Italy is a country", help="The prompt to start the generation.")
    parser.add_argument("--n_tokens_to_generate", type=int, default=64, help="Number of tokens to generate.")
    parser.add_argument("--K", type=int, default=3, help="Speculative lookahead value.")
    parser.add_argument("--model_target", type=str, default=MAMBA_MODELS["mamba_2_8b"], help="Target model to use.")
    parser.add_argument("--model_draft", type=str, default=MAMBA_MODELS["mamba_130m"], help="Draft model to use.")
    parser.add_argument("--cg", action="store_true", help="Enable Cuda Graph.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for model tensors.")
    parser.add_argument("--top_k", type=int, default=50, help="Top k value for generation.")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top p value for generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature value for generation")

    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    main(
        prompt=args.prompt,
        n_tokens_to_generate=args.n_tokens_to_generate,
        K=args.K,
        model_target=args.model_target,
        model_draft=args.model_draft,
        cg=args.cg,
        dtype=dtype,
        args=args,
    )
