import torch
from torch import Tensor
import torch.nn.functional as F

import time
import gc

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

from einops import rearrange, repeat

from specmamba.utils.generation import (
    sample,
    modify_logits_for_top_k_filtering,
    modify_logits_for_top_p_filtering,
    decode,
)
from specmamba.models.mixer_seq_simple_speculative import MambaLMHeadModel
from specmamba.models.mixer_seq_simple import MambaLMHeadModel as MambaLMHeadModelBase

from transformers import AutoTokenizer
from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput,
)



@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    steps: int = 1
    recompute_steps: int = -1
    spec_token_idx: int = -1
    lengths_per_sample: Optional[Tensor] = None
    cache_seqlens: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size, reset_offset=True, reset_steps=False):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.recompute_steps = -1
        self.spec_token_idx = -1
        if reset_steps:
            self.steps = 1
        if reset_offset:
            self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
        if self.cache_seqlens is not None:
            self.cache_seqlens.zero_()


def sample_speculative(logits, logits_draft, tokens_draft, top_k=1, top_p=0.0, temperature=1.0):
    """Algorithm 1 from [1]
    [1] Fast Inference from Transformers via Speculative Decoding
    Yaniv Leviathan, Matan Kalman, Yossi Matias
    https://arxiv.org/abs/2211.17192

    Arguments:
        logits: Tensor of shape (batch_size, seqlen + 1, vocab_size)
        logits_draft: Tensor of shape (batch_size, seqlen, vocab_size)
        tokens_draft: Tensor of shape (batch_size, seqlen)
    Return:
        tokens: Tensor of shape (batch_size, seqlen + 1)
        num_generated_tokens: Tensor of shape (batch_size), with value in [1, seqlen + 1].
            For each sequence in the batch, the number of valid tokens that were sampled by
            speculative sampling.
    """
    batch, seqlen_p_1, vocab_size = logits.shape
    seqlen = seqlen_p_1 - 1
    assert logits_draft.shape == (batch, seqlen, vocab_size)
    assert tokens_draft.shape == (batch, seqlen)
    assert tokens_draft.dtype in [torch.int64, torch.int32]
    # TODO: if top_k = 1 we can simplify things and only work with indices
    if top_p > 0.0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
    # Clone so that when we modify for top_p we don't change the original logits
    logits = logits / temperature if temperature != 1.0 else logits.clone()
    logits_draft = logits_draft / temperature if temperature != 1.0 else logits_draft.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        modify_logits_for_top_k_filtering(logits, top_k)
        modify_logits_for_top_k_filtering(logits_draft, top_k)
    modify_logits_for_top_p_filtering(logits, top_p)
    modify_logits_for_top_p_filtering(logits_draft, top_p)
    probs = torch.softmax(logits, dim=-1)
    probs_draft = torch.softmax(logits_draft, dim=-1)

    # acceptance_rate = 1 - (probs[:, :-1] - probs_draft).abs().sum(dim=-1) / 2
    # print("Acceptance rate", acceptance_rate)
    gather = lambda probs, tokens: rearrange(  # noqa: E731
        probs.gather(dim=-1, index=rearrange(tokens, "... -> ... 1")), "... 1 -> ..."
    )
    (batch, seqlen)
    accepted = torch.rand(batch, seqlen, device=probs.device) * gather(probs_draft, tokens_draft) <= gather(
        probs[:, :-1], tokens_draft
    )

    accepted_all = accepted.all(dim=-1)
    # (batch,)
    first_rejected_idx = torch.where(accepted_all, seqlen, accepted.int().argmin(dim=-1))
    probs_diff = torch.clamp(probs[:, :-1] - probs_draft, min=0.0)
    # torch.multinomial can deal with unnormalized probabilities
    # probs_diff /= probs_diff.sum(dim=-1, keepdim=True)
    resample_probs = torch.cat([probs_diff, probs[:, -1:]], dim=1)
    resample_probs = rearrange(
        resample_probs.gather(dim=1, index=repeat(first_rejected_idx, "b -> b 1 d", d=vocab_size)),
        "b 1 d -> b d",
    )
    valid_mask = resample_probs.sum(dim=-1) > 0
    resample_probs[~valid_mask] = 1.0  # Assign a uniform distribution for invalid rows
    resample_probs = resample_probs / resample_probs.sum(dim=-1, keepdim=True)
    resample = torch.multinomial(resample_probs, num_samples=1).squeeze(dim=-1)  # (batch,)
    tokens = F.pad(tokens_draft, (0, 1))
    tokens[:, first_rejected_idx] = resample
    return tokens, first_rejected_idx + 1


def initialize_inference_params(
    model, model_draft, input_ids, max_length, draft_type, speculative_lookahead, cg, batch_size, seqlen_og
):
    if cg:
        if not hasattr(model_draft, "_decoding_cache"):
            model_draft._decoding_cache = None

        model_draft._decoding_cache = update_graph_cache(
            model_draft,
            model_type="draft",
            cache=model_draft._decoding_cache,
            batch_size=batch_size,
            seqlen_og=seqlen_og,
            max_seqlen=max_length,
            decoding_seqlens=(1, 2) if draft_type == "transformer" else (1,),
            recompute_steps_values=range(-1, speculative_lookahead + 1),
            cache_speculative_steps=None if draft_type == "transformer" else speculative_lookahead - 1,
            spec_token_idx_values=None if draft_type == "transformer" else range(-1, speculative_lookahead),
            draft_type=draft_type,
        )

        inference_params_draft = model_draft._decoding_cache.inference_params
        inference_params_draft.reset(max_length, batch_size, reset_steps=True)

        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None

        model._decoding_cache = update_graph_cache(
            model,
            model_type="target",
            cache=model._decoding_cache,
            batch_size=batch_size,
            seqlen_og=seqlen_og,
            max_seqlen=max_length,
            decoding_seqlens=range(1, speculative_lookahead + 2),
            recompute_steps_values=range(-1, speculative_lookahead + 1),
            cache_speculative_steps=speculative_lookahead,
        )

        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size, reset_steps=True)

    else:
        inference_params_draft = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        inference_params_draft.cache_seqlens = torch.full((batch_size,), 0, dtype=torch.int32, device=input_ids.device)
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        inference_params.cache_seqlens = torch.full((batch_size,), 0, dtype=torch.int32, device=input_ids.device)
        inf_cache = model.allocate_inference_cache(
            batch_size,
            max_length,
            cache_speculative_steps=speculative_lookahead,
        )
        inference_params.key_value_memory_dict = inf_cache

        inf_cache_draft = model_draft.allocate_inference_cache(
            batch_size,
            max_length,
            # this is only for hybrid or mamba draft models. It will be ignored for transformers.
            cache_speculative_steps=speculative_lookahead - 1,
        )
        inference_params_draft.key_value_memory_dict = inf_cache_draft

    return inference_params, inference_params_draft


@torch.inference_mode()
def decode_speculative(
    input_ids,
    model,
    model_draft,
    max_length,
    draft_type="mamba",
    speculative_lookahead=3,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    vocab_size=None,
    cg=False,
    enable_timing=False,
    debug=False,
):
    assert draft_type in ["mamba", "transformer", "hybrid"]
    batch_size, seqlen_og = input_ids.shape
    assert batch_size == 1, "Only batch size 1 is supported for now"
    assert eos_token_id is None, "Speculative decoding implementation doesn't support eos_token_id"

    inference_params, inference_params_draft = initialize_inference_params(
        model, model_draft, input_ids, max_length, draft_type, speculative_lookahead, cg, batch_size, seqlen_og
    )

    def get_logits(input_ids, inference_params, model, num_last_tokens=1, cg=False):
        # If seqlen_offset > 0, we are in the 1 or multistep decoding phase. Otherwise, we are prefilling.
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens,
            ).logits
        else:
            # NOTE: careful, CUDA graph is set to have num_last_tokens=input_ids.shape[1].
            # This might not be compatible the num_last_tokens used here.
            assert num_last_tokens <= input_ids.shape[1]

            logits = model._decoding_cache.run(
                input_ids,
                position_ids,
                inference_params.seqlen_offset,
                inference_params.cache_seqlens,
                recompute_steps=inference_params.recompute_steps,
                spec_token_idx=inference_params.spec_token_idx,
            )[:, -num_last_tokens:]
            # )
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens_transformer(input_ids, get_logits_fn, inference_params, sample_fn, num_tokens=1, **kwargs):
        """Sample `num_tokens` tokens from the model, given the previous logits.
        Also return the logits of the sampled tokens.
        Arguments:
            input_ids: (batch, seqlen)
        Return:
            tokens: (batch, num_tokens)
            scores: (batch, num_tokens), which contains @previous_logits and the logits of the next
                (num_tokens - 1) tokens. The logits of the last token isn't computed.
        """

        assert num_tokens >= 1

        sequences, scores = [input_ids], []
        for _ in range(num_tokens):
            scores.append(get_logits_fn(sequences[-1], inference_params)[:, -1])
            inference_params.recompute_steps = -1
            inference_params.seqlen_offset += sequences[-1].shape[1]
            sequences.append(sample_fn(scores[-1]).unsqueeze(1))
        else:
            return torch.cat(sequences[1:], dim=1), torch.stack(scores, dim=1)

    def sample_tokens_from_single_logit(logits, get_logits_fn, inference_params, sample_fn, num_tokens=1):
        assert num_tokens >= 1
        assert logits.shape[1] == 1
        sequences, scores = [sample_fn(logits.squeeze(1)).unsqueeze(1)], [logits[:, -1]]
        inference_params.spec_token_idx += 1
        inference_params.recompute_steps = -1

        for i in range(num_tokens - 1):
            inference_params.steps = sequences[-1].shape[1]

            scores.append(get_logits_fn(sequences[-1], inference_params)[:, -1])
            inference_params.cache_seqlens[:] += sequences[-1].shape[1]
            inference_params.spec_token_idx += 1

            inference_params.seqlen_offset += sequences[-1].shape[1]
            sequences.append(sample_fn(scores[-1]).unsqueeze(1))

        return torch.cat(sequences, dim=1), torch.stack(scores, dim=1)

    def sample_tokens(input_ids, get_logits_fn, inference_params, sample_fn, num_tokens=1):
        """Sample `num_tokens` tokens from the model, given the previous logits.
        Also return the logits of the sampled tokens.
        Arguments:
            input_ids: (batch, seqlen)
        Return:
            tokens: (batch, num_tokens)
            scores: (batch, num_tokens), which contains @previous_logits and the logits of the next
                (num_tokens - 1) tokens. The logits of the last token isn't computed.
        """

        assert num_tokens >= 1

        sequences, scores = [input_ids], []
        assert input_ids.shape[1] in (1, 2)

        for _ in range(num_tokens):
            for token_idx in range(sequences[-1].shape[1]):
                score = get_logits_fn(sequences[-1][:, token_idx].unsqueeze(1), inference_params)[:, -1]
                inference_params.recompute_steps = -1
                inference_params.cache_seqlens[:] += 1
            scores.append(score)
            inference_params.spec_token_idx += 1

            inference_params.seqlen_offset += sequences[-1].shape[1]
            sequences.append(sample_fn(scores[-1]).unsqueeze(1))

        return torch.cat(sequences[1:], dim=1), torch.stack(scores, dim=1)

    sampling_kwargs = dict(top_k=top_k, top_p=top_p, temperature=temperature)
    sample_fn = partial(sample, **sampling_kwargs)

    get_logits_main = partial(get_logits, model=model, cg=cg)
    get_logits_draft = partial(get_logits, model=model_draft, cg=cg)

    sample_tokens_main = partial(
        sample_tokens,
        get_logits_fn=get_logits_main,
        sample_fn=sample_fn,
        inference_params=inference_params,
    )

    if draft_type == "transformer":
        sample_tokens_draft = partial(
            sample_tokens_transformer,
            get_logits_fn=get_logits_draft,
            inference_params=inference_params_draft,
            sample_fn=sample_fn,
        )
    else:
        sample_tokens_draft = partial(
            sample_tokens,
            get_logits_fn=get_logits_draft,
            sample_fn=sample_fn,
            inference_params=inference_params_draft,
        )

        sample_tokens_from_single_logit_draft = partial(
            sample_tokens_from_single_logit,
            get_logits_fn=get_logits_draft,
            sample_fn=sample_fn,
            inference_params=inference_params_draft,
        )

    # if enable_timing:
    #     torch.cuda.synchronize()
    #     start = time.time()

    scores, sequences = [], [input_ids]
    num_main_model_calls = 0
    num_draft_tokens = 0
    num_accepted_tokens_history = []

    # -----------------------
    if seqlen_og >= max_length - 1:
        # Don't do speculative sampling, just sample 1 token from the model
        tokens, scores_new = sample_tokens_main(input_ids, num_tokens=1)
        sequences.append(tokens)
        scores.append(scores_new)

    else:
        # Sample from draft model, which produces @n_spec_tokens, and @model
        # will then use to produce between 1 and 1 + @n_spec_tokens tokens.
        # We want seqlen_og + 1 + @n_spec_tokens to be <= @max_length.
        n_spec_tokens = min(speculative_lookahead, max_length - seqlen_og - 1)

        if draft_type == "transformer":
            tokens_draft, scores_draft = sample_tokens_draft(input_ids, num_tokens=n_spec_tokens)
        else:
            last_logit_prefill = get_logits_draft(input_ids, inference_params_draft, num_last_tokens=1)
            inference_params_draft.seqlen_offset = seqlen_og
            inference_params_draft.spec_token_idx = -1
            conv_state_len_draft = model_draft.backbone.layers[0].mixer.d_conv + speculative_lookahead
            inference_params_draft.cache_seqlens[:] = (
                (seqlen_og) if (seqlen_og) <= conv_state_len_draft else conv_state_len_draft
            )
            # tokens_draft, scores_draft = sample_tokens_draft(None, num_tokens=n_spec_tokens, logits=last_logit_prefill)
            tokens_draft, scores_draft = sample_tokens_from_single_logit_draft(
                last_logit_prefill, num_tokens=n_spec_tokens
            )
        num_draft_tokens += n_spec_tokens

        if debug:
            scores_draft_ref = model_draft(
                torch.cat([input_ids, tokens_draft], dim=1),
                num_last_tokens=n_spec_tokens + 1,
            ).logits
            print((scores_draft - scores_draft_ref[:, :-1]).abs().max())
            # assert (scores_draft - scores_draft_ref[:, :-1]).abs().max() < 1.0

        # pre-fill prompt. We don't pass the last token in the prompt for prefilling, because we want to cache it
        # in order to recompute stuff in case draft tokens are rejected. Caching happens in the multistep decoding and
        # not in the prefill stage. I don't like this, but it's how it is for now.
        logits_prefill = get_logits_main(input_ids[:, :-1], inference_params, num_last_tokens=input_ids.shape[1] - 1)

        if debug:
            logits_ref = model(input_ids[:, :-1]).logits
            print((logits_prefill - logits_ref).abs().max())
            # assert (logits_prefill - logits_ref).abs().max() < 1.0

        # Now we do a multi-step decoding with the last token of the prompt + the draft tokens.
        inference_params.seqlen_offset = seqlen_og - 1
        conv_state_len = model.backbone.layers[0].mixer.d_conv - 1 + speculative_lookahead
        inference_params.cache_seqlens[:] = (seqlen_og - 1) if (seqlen_og - 1) <= conv_state_len else conv_state_len
        inference_params.steps = n_spec_tokens + 1

        if enable_timing:
            torch.cuda.synchronize()
            start = time.time()

        logits = get_logits_main(
            torch.cat(
                [
                    input_ids[
                        :,
                        -1:,
                    ],
                    tokens_draft,
                ],
                dim=1,
            ),
            inference_params,
            num_last_tokens=n_spec_tokens + 1,
        )

        num_main_model_calls += 1

        if debug:
            logits_ref = model(
                torch.cat([input_ids, tokens_draft], dim=1),
                num_last_tokens=n_spec_tokens + 1,
            ).logits
            print((logits - logits_ref).abs().max())
            # assert (logits - logits_ref).abs().max() < 1.0

        tokens, num_generated_tokens = sample_speculative(logits, scores_draft, tokens_draft, **sampling_kwargs)

        num_accepted_tokens_history.append(num_generated_tokens - 1)

        sequences.append(tokens[:1, : num_generated_tokens[0]])
        scores.append(logits[:1, : num_generated_tokens[0]])

        # Note that @model has not evaluated the last sampled token yet, so we'll need to pass
        # that in the next time we call @model.
        num_generated = num_generated_tokens[0].item()

        # -1 because we have not evaluated the last sampled token yet with the main model.
        inference_params.seqlen_offset = seqlen_og + num_generated - 1
        inference_params.cache_seqlens[:] = seqlen_og + num_generated - 1

        inference_params_draft.seqlen_offset = (
            inference_params.seqlen_offset - 1 if num_generated > 1 else inference_params.seqlen_offset
        )

        # decrease by number of tokens rejected
        num_token_rejected = n_spec_tokens - num_accepted_tokens_history[-1].item()
        inference_params_draft.cache_seqlens[:] -= num_token_rejected
        if n_spec_tokens - num_accepted_tokens_history[-1].item() > 0:
            inference_params_draft.cache_seqlens[:] += 1

        if debug:
            cur_ids = torch.cat([input_ids, sequences[-1]], dim=1)
            scores_ref = model(cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1).logits
            print((scores[-1] - scores_ref[:, :-1]).abs().max())
            # assert (scores[-1] - scores_ref[:, :-1]).abs().max() < 1.0

    while True:
        # seqlen_offset is total length generated - 1
        if inference_params.seqlen_offset >= max_length - 1:
            break

        if inference_params.seqlen_offset >= max_length - 2:
            # Don't do speculative sampling, just sample 1 token from the model
            inference_params.steps = 1
            inference_params.recompute_steps = num_accepted_tokens_history[-1].item()
            tokens, scores_new = sample_tokens_main(sequences[-1][:, -1:], num_tokens=1)
            sequences.append(tokens)
            scores.append(scores_new)
            break

        n_spec_tokens = min(speculative_lookahead, max_length - inference_params_draft.seqlen_offset - 2)

        # If the main model accepts all the draft tokens, plus it samples one new token,
        # then at the next iteration the draft model need to evaluate the logits of the last draft
        # token and the logits of the newly sampled token. So here we pass in the last 2 tokens
        # of sequences[-1].
        # The exception is when the main model rejects all the draft tokens, in which case we
        # will only have 1 token to pass in.

        inference_params_draft.recompute_steps = num_accepted_tokens_history[-1].item()
        inference_params_draft.spec_token_idx = -1

        if draft_type == "transformer":
            idx = -2
        else:
            idx = -1 if inference_params_draft.recompute_steps < n_spec_tokens else -2

        tokens_draft, scores_draft = sample_tokens_draft(sequences[-1][:, idx:], num_tokens=n_spec_tokens)
        num_draft_tokens += n_spec_tokens

        if debug:
            scores_draft_ref = model_draft(
                torch.cat([cur_ids, tokens_draft], dim=1),
                num_last_tokens=n_spec_tokens + 1,
            ).logits
            print("Draft model scores diff")
            print((scores_draft - scores_draft_ref[:, :-1]).abs().max())
            # assert (scores_draft - scores_draft_ref[:, :-1]).abs().max() < 1.0

        # We want to evaluate the last sampled tokens + all the draft tokens.
        inference_params.steps = n_spec_tokens + 1
        inference_params.recompute_steps = num_accepted_tokens_history[-1].item()

        if inference_params.recompute_steps == speculative_lookahead:
            inference_params.recompute_steps = -1

        logits = get_logits_main(
            torch.cat([sequences[-1][:, -1:], tokens_draft], dim=1),
            inference_params,
            num_last_tokens=n_spec_tokens + 1,
        )  # (batch, n_spec_tokens + 1, vocab_size)
        num_main_model_calls += 1

        if debug:
            logits_ref = model(
                torch.cat([cur_ids, tokens_draft], dim=1),
                num_last_tokens=n_spec_tokens + 1,
            ).logits
            print((logits - logits_ref).abs().max())
            # assert (logits - logits_ref).abs().max() < 1.0

        tokens, num_generated_tokens = sample_speculative(logits, scores_draft, tokens_draft, **sampling_kwargs)

        num_accepted_tokens_history.append(num_generated_tokens - 1)

        sequences.append(tokens[:1, : num_generated_tokens[0]])
        scores.append(logits[:1, : num_generated_tokens[0]])

        # We've evaluated 1 token from sequences[-1][:, -1:] above, plus
        # num_generated_tokens[0].item() - 1 tokens from the draft model.
        num_generated = num_generated_tokens[0].item()
        inference_params.seqlen_offset += num_generated
        inference_params.cache_seqlens[:] += num_generated

        inference_params_draft.seqlen_offset = (
            inference_params.seqlen_offset - 1 if num_generated > 1 else inference_params.seqlen_offset
        )
        inference_params_draft.cache_seqlens[:] -= n_spec_tokens - num_accepted_tokens_history[-1].item()
        if n_spec_tokens - num_accepted_tokens_history[-1].item() > 0:
            inference_params_draft.cache_seqlens[:] += 1

        if debug:
            cur_ids = torch.cat([cur_ids, sequences[-1]], dim=1)
            scores_ref = model(cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1).logits
            print((scores[-1] - scores_ref[:, :-1]).abs().max())
            # assert (scores[-1] - scores_ref[:, :-1]).abs().max() < 1.0

    if enable_timing:
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
        if debug:
            print(f"Number of calls to main model: {num_main_model_calls}")
        print(f"Acceptance rate: {torch.cat(num_accepted_tokens_history).sum().item() / num_draft_tokens * 100:.2f}%")
    sequences = torch.cat(sequences, dim=1)
    scores = torch.cat(scores, dim=1)

    if debug:
        scores_ref = model(sequences).logits
        print((scores - scores_ref[:, seqlen_og - 1 : -1]).abs().max())

    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput

    if enable_timing:
        return (
            output_cls(sequences=sequences, scores=scores),
            (time.time() - start) * 1000,
            num_accepted_tokens_history,
            num_draft_tokens,
        )
    return output_cls(sequences=sequences, scores=scores)


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(
    model,
    model_type,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    decoding_seqlens=(1,),
    recompute_steps_values=(0,),
    spec_token_idx_values=(-1,),
    dtype=None,
    n_warmups=2,
    cache_speculative_steps=0,
    draft_type="mamba",
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        assert hasattr(
            model, "allocate_inference_cache"
        ), "CUDA graph decoding requires that the model has a method allocate_inference_cache"
        inf_cache = model.allocate_inference_cache(
            batch_size,
            max_seqlen,
            dtype=dtype,
            cache_speculative_steps=cache_speculative_steps,
        )

        # a single InferenceParams object is shared for each model.
        # Across iterations, elements are updated in place.
        if model_type == "target" or draft_type != "transformer":
            spec_token_idx = spec_token_idx_values[0]
            recompute_steps = -1
        else:
            spec_token_idx = None
            recompute_steps = None

        lenghts_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache_seqlens = torch.full((batch_size,), 0, dtype=torch.int32, device=device)

        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_og,
            steps=decoding_seqlens[0],
            key_value_memory_dict=inf_cache,
            recompute_steps=recompute_steps,
            spec_token_idx=spec_token_idx,
            lengths_per_sample=lenghts_per_sample,
            cache_seqlens=cache_seqlens,
        )

        cache.mempool = torch.cuda.graphs.graph_pool_handle()

    for decoding_seqlen in decoding_seqlens:
        if model_type == "target" or (model_type == "draft" and draft_type != "transformer"):
            for recompute_steps in recompute_steps_values:
                for spec_token_idx in spec_token_idx_values:
                    if (batch_size, decoding_seqlen, recompute_steps, spec_token_idx) not in cache.callables:
                        cache.callables[batch_size, decoding_seqlen, recompute_steps, spec_token_idx] = capture_graph(
                            model,
                            cache.inference_params,
                            batch_size,
                            max_seqlen,
                            decoding_seqlen=decoding_seqlen,
                            recompute_steps=recompute_steps,
                            spec_token_idx=spec_token_idx,
                            mempool=cache.mempool,
                            n_warmups=n_warmups,
                        )
        else:
            if (batch_size, decoding_seqlen) not in cache.callables:
                cache.callables[batch_size, decoding_seqlen] = capture_graph(
                    model,
                    cache.inference_params,
                    batch_size,
                    max_seqlen,
                    decoding_seqlen=decoding_seqlen,
                    mempool=cache.mempool,
                    n_warmups=n_warmups,
                )

    def dispatch(input_ids, position_ids, seqlen, cache_seqlens, recompute_steps, spec_token_idx, draft_type="mamba"):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        # print(f"Dispatching: {batch_size=}, {decoding_seqlen=}, {recompute_steps=}, {spec_token_idx=}")

        if model_type == "target" or (model_type == "draft" and draft_type != "transformer"):
            return cache.callables[batch_size, decoding_seqlen, recompute_steps, spec_token_idx](
                input_ids, position_ids, seqlen, cache_seqlens
            )
        else:
            return cache.callables[batch_size, decoding_seqlen](input_ids, position_ids, seqlen, cache_seqlens)

    cache.run = partial(dispatch, draft_type=draft_type)
    cache.inference_params.seqlen_offset = 0  # TODO do we need this?
    cache.inference_params.spec_token_idx = -1
    cache.inference_params.recompute_steps = -1
    return cache


def capture_graph(
    model,
    inference_params,
    batch_size,
    max_seqlen,
    decoding_seqlen=1,
    recompute_steps=0,
    spec_token_idx=-1,
    mempool=None,
    n_warmups=2,
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)

    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset
    inference_params.cache_seqlens[:] = inference_params.seqlen_offset
    inference_params.steps = decoding_seqlen
    inference_params.recompute_steps = recompute_steps
    inference_params.spec_token_idx = spec_token_idx

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
        ).logits

    def run(new_input_ids, new_position_ids, seqlen, cache_seqlens):
        inference_params.lengths_per_sample[:] = seqlen
        inference_params.cache_seqlens[:] = cache_seqlens
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og
    return run
