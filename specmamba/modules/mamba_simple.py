# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from specmamba.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


try:
    from specmamba.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from specmamba.ops.triton.selective_state_update_multi import (
        selective_state_update_multi,
    )
except ImportError:
    selective_state_update_multi = None

try:
    from specmamba.ops.triton.selective_state_update_multi_cached_fused_draft import (
        selective_state_update_multi_cached_draft,
    )
except ImportError:
    selective_state_update_multi_cached_draft = None

try:
    from specmamba.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, **kwargs):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state, cache = None, None, None

        if inference_params is not None:
            conv_state, ssm_state, cache = self._get_states_from_cache(inference_params, batch)

            if inference_params.seqlen_offset > 0:
                if hasattr(inference_params, "recompute_steps"):
                    out, _, _ = self.step_cached_fused(hidden_states, conv_state, ssm_state, cache, inference_params)
                    # out, _, _ = self.step_cached(hidden_states, conv_state, ssm_state, cache, inference_params)
                else:
                    out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path and causal_conv1d_fn is not None and inference_params is None
        ):  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                # conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

                # cache["conv_state_ref"].copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
                if seqlen <= conv_state.shape[2]:
                    conv_state.copy_(F.pad(x, (0, conv_state.shape[2] - x.shape[-1])))
                else:
                    conv_state.copy_(F.pad(x, (conv_state.shape[2] - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step_cached(self, hidden_states, conv_state, ssm_state, cache, inference_params):
        dtype = hidden_states.dtype
        A = -torch.exp(self.A_log.float())

        # If no tokens were accepted in the previous iteration, we need to reset the states.
        if inference_params.recompute_steps == 0:
            conv_state.copy_(cache["conv_state_initial"])
            ssm_state.copy_(cache["ssm_state_initial"])

        # Recompute the steps for the tokens that were accepted in the previous iteration.
        if inference_params.recompute_steps > 0 and inference_params.recompute_steps < cache["x"].shape[1]:
            x = cache["x"][:, : inference_params.recompute_steps]
            z = cache["z"][:, : inference_params.recompute_steps]
            dt = cache["dt"][:, : inference_params.recompute_steps]
            B = cache["B"][:, : inference_params.recompute_steps]
            C = cache["C"][:, : inference_params.recompute_steps]
            x_conv = cache["x_conv"][:, : inference_params.recompute_steps]
            conv_state.copy_(cache["conv_state_initial"])
            ssm_state.copy_(cache["ssm_state_initial"])

            x = x.transpose(1, 2).contiguous()

            # conv_state_ = torch.cat((conv_state, x), dim=-1)

            conv_state.copy_(torch.roll(conv_state, shifts=-inference_params.recompute_steps, dims=-1))
            conv_state[:, :, -inference_params.recompute_steps :] = x[:, :, -self.d_conv :]

            # padding = self.d_conv - 1
            # if causal_conv1d_fn is None:
            #     x = self.conv1d(conv_state_)[:, :, -inference_params.recompute_steps - padding : -padding]
            #     x = self.act(x).to(dtype=hidden_states.dtype)
            # else:
            #     x = causal_conv1d_fn(
            #         x=conv_state_,
            #         weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            #         bias=self.conv1d.bias,
            #         activation=self.activation,
            #     )[..., -inference_params.recompute_steps :]

            # x = x.transpose(1, 2).contiguous()

            _ = selective_state_update_multi(
                ssm_state,
                x_conv,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
                steps=inference_params.recompute_steps,  # needs to be tensor on GPU.
            )

        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # If we are generating the first draft token of the batch, cache the initial states.
        if inference_params.spec_token_idx == 0:
            cache["conv_state_initial"].copy_(conv_state)
            cache["ssm_state_initial"].copy_(ssm_state)

        # if inference_params.spec_token_idx >= 0:
        #     cache["z"][:, inference_params.spec_token_idx, :].copy_(z)
        #     cache["x"][:, inference_params.spec_token_idx, :].copy_(x)

        # Conv step
        if causal_conv1d_update is None:
            assert False
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x_conv = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x_conv)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)

        if inference_params.spec_token_idx >= 0:
            cache["z"][:, inference_params.spec_token_idx, :].copy_(z)
            cache["x"][:, inference_params.spec_token_idx, :].copy_(x)
            cache["x_conv"][:, inference_params.spec_token_idx, :].copy_(x_conv)
            cache["dt"][:, inference_params.spec_token_idx, :].copy_(dt)
            cache["B"][:, inference_params.spec_token_idx, :].copy_(B)
            cache["C"][:, inference_params.spec_token_idx, :].copy_(C)

        # SSM step
        if selective_state_update is None:
            assert False
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x_conv,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def step_cached_fused(self, hidden_states, conv_state, ssm_state, cache, inference_params):
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        A = -torch.exp(self.A_log.float())

        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        x_conv = (
            causal_conv1d_update(
                x.unsqueeze(1).transpose(-1, -2),
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                cache_seqlens=inference_params.cache_seqlens,
            )
            .transpose(-1, -2)
            .squeeze(1)
        )

        # If no tokens were accepted in the previous iteration, we need to reset the states.
        # if inference_params.recompute_steps == 0:
        #     conv_state.copy_(cache["conv_state_initial"])

        # # Recompute the steps for the tokens that were accepted in the previous iteration.
        # # Uf we need to recompute everything, no need to recompute the conv_state
        # if inference_params.recompute_steps > 0 and inference_params.recompute_steps < cache["x"].shape[1]:
        #     # restore the conv_state
        #     x_cache = cache["x"][:, : inference_params.recompute_steps]
        #     cache["conv_state_ref"].copy_(cache["conv_state_initial"])
        #     x_cache = x_cache.transpose(1, 2).contiguous()
        #     cache["conv_state_ref"].copy_(
        #         torch.roll(cache["conv_state_ref"], shifts=-inference_params.recompute_steps, dims=-1)
        #     )
        #     cache["conv_state_ref"][:, :, -inference_params.recompute_steps :] = x_cache[:, :, -self.d_conv :]

        # # If we are generating the first draft token of the batch, cache the initial states.
        # if inference_params.spec_token_idx == 0:
        #     cache["conv_state_initial"].copy_(cache["conv_state_ref"])

        # x_conv_ref = causal_conv1d_update(
        #     x,
        #     cache["conv_state_ref"],
        #     rearrange(self.conv1d.weight, "d 1 w -> d w"),
        #     self.conv1d.bias,
        #     self.activation,
        # )

        # if not torch.allclose(x_conv, x_conv_ref, atol=1e-5) and self.layer_idx == 0:
        #     print("Conv mismatch")

        x_db = self.x_proj(x_conv)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)

        y = selective_state_update_multi_cached_draft(
            ssm_state,
            x_conv,
            x,
            dt,
            A,
            B,
            C,
            cache["ssm_state_initial"],
            cache["x_conv"],
            cache["x"],
            cache["dt"],
            cache["B"],
            D=self.D,
            z=z,
            # z_cache=cache["z"],
            dt_bias=self.dt_proj.bias,
            dt_softplus=True,
            initial_state=cache["ssm_state_initial"]
            if inference_params.recompute_steps == 0
            or (inference_params.recompute_steps > 0 and inference_params.recompute_steps < cache["x"].shape[1])
            else None,
            spec_token_idx=inference_params.spec_token_idx,
            return_step=inference_params.recompute_steps
            if inference_params.recompute_steps > 0 and inference_params.recompute_steps < cache["x"].shape[1]
            else 0,
        )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype

        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv + kwargs.get("cache_speculative_steps", 0) + 1,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )

        cache = dict()
        cache_speculative_steps = kwargs.get("cache_speculative_steps", 0)
        cache["dt"] = torch.zeros(
            batch_size,
            cache_speculative_steps + 1,
            self.d_inner,
            device=device,
            dtype=ssm_dtype,
        )

        cache["B"] = torch.zeros(
            batch_size,
            cache_speculative_steps + 1,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )

        cache["ssm_state_initial"] = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        cache["x"] = torch.zeros(
            batch_size,
            cache_speculative_steps + 1,
            self.d_model * 2,
            device=device,
            dtype=ssm_dtype,
        )
        cache["x_conv"] = torch.zeros_like(cache["x"])
        # cache["z"] = torch.zeros(
        #     batch_size,
        #     cache_speculative_steps + 1,
        #     self.d_model * 2,
        #     device=device,
        #     dtype=ssm_dtype,
        # )

        return conv_state, ssm_state, cache

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None

        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state, ssm_state, cache = self.allocate_inference_cache(batch_size, inference_params.max_seqlen)
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state, cache)
        else:
            conv_state, ssm_state, cache = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                raise NotImplementedError()
        return conv_state, ssm_state, cache


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        steps=1,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, steps=steps)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
