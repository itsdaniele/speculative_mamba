"""We want triton==2.1.0 or triton==2.2.0 or triton==2.3.0 for this"""

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat


@triton.heuristics({"HAS_RETURN_STATE": lambda args: args["return_step"] is not None})
@triton.heuristics({"HAS_INITIAL_STATE": lambda args: args["initial_state_ptr"] is not None})
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    return_state_ptr,
    x_ptr,
    # x_preconv_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    # C_cache_ptr,
    # z_cache_ptr,
    x_cache_ptr,
    # x_preconv_cache_ptr,
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_B_ratio,
    nheads_ngroups_C_ratio,
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_seq,
    stride_x_head,
    stride_x_dim,
    # stride_x_preconv_batch,
    # stride_x_preconv_seq,
    # stride_x_preconv_head,
    # stride_x_preconv_dim,
    stride_dt_batch,
    stride_dt_seq,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_seq,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_seq,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_seq,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_seq,
    stride_out_head,
    stride_out_dim,
    stride_dt_cache_batch,
    stride_dt_cache_seq,
    stride_dt_cache_head,
    stride_dt_cache_dim,
    stride_B_cache_batch,
    stride_B_cache_seq,
    stride_B_cache_group,
    stride_B_cache_dstate,
    # stride_C_cache_batch,
    # stride_C_cache_seq,
    # stride_C_cache_group,
    # stride_C_cache_dstate,
    # stride_z_cache_batch,
    # stride_z_cache_seq,
    # stride_z_cache_head,
    # stride_z_cache_dim,
    stride_x_cache_batch,
    stride_x_cache_seq,
    stride_x_cache_head,
    stride_x_cache_dim,
    # stride_x_preconv_cache_batch,
    # stride_x_preconv_cache_seq,
    # stride_x_preconv_cache_head,
    # stride_x_preconv_cache_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    steps: tl.constexpr,
    return_step: tl.constexpr,
    initial_state_ptr,
    HAS_RETURN_STATE: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)

    if HAS_INITIAL_STATE:
        initial_state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
        initial_state_ptrs = initial_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    if HAS_RETURN_STATE:
        return_state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
        return_state_ptrs = return_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head

    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    if not TIE_HDIM:
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    else:
        A = tl.load(A_ptr).to(tl.float32)

    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim

    if HAS_INITIAL_STATE:
        state = tl.load(initial_state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    else:
        state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)

    for seq_i in tl.static_range(return_step):
        x_cache_ptr_i = (
            x_cache_ptr + pid_b * stride_x_cache_batch + seq_i * stride_x_cache_seq + pid_h * stride_x_cache_head
        )
        dt_cache_ptr_i = (
            dt_cache_ptr + pid_b * stride_dt_cache_batch + seq_i * stride_dt_cache_seq + pid_h * stride_dt_cache_head
        )

        B_cache_ptr_i = (
            B_cache_ptr
            + pid_b * stride_B_cache_batch
            + seq_i * stride_B_cache_seq
            + (pid_h // nheads_ngroups_B_ratio) * stride_B_cache_group
        )

        # C_cache_ptr_i = (
        #     C_cache_ptr
        #     + pid_b * stride_C_cache_batch
        #     + seq_i * stride_C_cache_seq
        #     + (pid_h // nheads_ngroups_C_ratio) * stride_C_cache_group
        # )

        # if HAS_Z:
        #     z_cache_ptr_i = (
        #         z_cache_ptr + pid_b * stride_z_cache_batch + seq_i * stride_z_cache_seq + pid_h * stride_z_cache_head
        #     )

        x_cache_ptrs = x_cache_ptr_i + offs_m * stride_x_cache_dim
        dt_cache_ptrs = dt_cache_ptr_i + offs_m * stride_dt_cache_dim
        B_cache_ptrs = B_cache_ptr_i + offs_n * stride_B_cache_dstate
        # C_cache_ptrs = C_cache_ptr_i + offs_n * stride_C_cache_dstate
        # if HAS_Z:
        #     z_cache_ptrs = z_cache_ptr_i + offs_m * stride_z_cache_dim

        x = tl.load(x_cache_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_cache_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
            dA = tl.exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
            dA = tl.exp(A * dt)  # scalar, not a matrix

        B = tl.load(B_cache_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        # C = tl.load(C_cache_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        # if HAS_Z:
        #     z = tl.load(z_cache_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        if not TIE_HDIM:
            dB = B[None, :] * dt[:, None]
        else:
            dB = B * dt  # vector of size (dstate,)
        state = state * dA + dB * x[:, None]

    if HAS_RETURN_STATE:
        tl.store(return_state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

    for seq_i in tl.static_range(steps - return_step):
        x_ptr_i = x_ptr + pid_b * stride_x_batch + seq_i * stride_x_seq + pid_h * stride_x_head
        x_cache_ptr_i = (
            x_cache_ptr + pid_b * stride_x_cache_batch + seq_i * stride_x_cache_seq + pid_h * stride_x_cache_head
        )
        # x_preconv_ptr_i = (
        #     x_preconv_ptr
        #     + pid_b * stride_x_preconv_batch
        #     + seq_i * stride_x_preconv_seq
        #     + pid_h * stride_x_preconv_head
        # )
        # x_preconv_cache_ptr_i = (
        #     x_preconv_cache_ptr
        #     + pid_b * stride_x_preconv_cache_batch
        #     + seq_i * stride_x_preconv_cache_seq
        #     + pid_h * stride_x_preconv_cache_head
        # )
        dt_ptr_i = dt_ptr + pid_b * stride_dt_batch + seq_i * stride_dt_seq + pid_h * stride_dt_head
        dt_cache_ptr_i = (
            dt_cache_ptr + pid_b * stride_dt_cache_batch + seq_i * stride_dt_cache_seq + pid_h * stride_dt_cache_head
        )

        B_ptr_i = (
            B_ptr + pid_b * stride_B_batch + seq_i * stride_B_seq + (pid_h // nheads_ngroups_B_ratio) * stride_B_group
        )
        B_cache_ptr_i = (
            B_cache_ptr
            + pid_b * stride_B_cache_batch
            + seq_i * stride_B_cache_seq
            + (pid_h // nheads_ngroups_B_ratio) * stride_B_cache_group
        )

        C_ptr_i = (
            C_ptr + pid_b * stride_C_batch + seq_i * stride_C_seq + (pid_h // nheads_ngroups_C_ratio) * stride_C_group
        )

        # C_cache_ptr_i = (
        #     C_cache_ptr
        #     + pid_b * stride_C_cache_batch
        #     + seq_i * stride_C_cache_seq
        #     + (pid_h // nheads_ngroups_C_ratio) * stride_C_cache_group
        # )

        if HAS_Z:
            z_ptr_i = z_ptr + pid_b * stride_z_batch + seq_i * stride_z_seq + pid_h * stride_z_head
            # z_cache_ptr_i = (
            #     z_cache_ptr + pid_b * stride_z_cache_batch + seq_i * stride_z_cache_seq + pid_h * stride_z_cache_head
            # )

        out_ptr_i = out_ptr + pid_b * stride_out_batch + seq_i * stride_out_seq + pid_h * stride_out_head

        x_ptrs = x_ptr_i + offs_m * stride_x_dim
        x_cache_ptrs = x_cache_ptr_i + offs_m * stride_x_cache_dim
        # x_preconv_ptrs = x_preconv_ptr_i + offs_m * stride_x_preconv_dim
        # x_preconv_cached_ptrs = x_preconv_cache_ptr_i + offs_m * stride_x_preconv_cache_dim
        dt_ptrs = dt_ptr_i + offs_m * stride_dt_dim
        dt_cached_ptrs = dt_cache_ptr_i + offs_m * stride_dt_cache_dim
        B_ptrs = B_ptr_i + offs_n * stride_B_dstate
        B_cached_ptrs = B_cache_ptr_i + offs_n * stride_B_cache_dstate
        C_ptrs = C_ptr_i + offs_n * stride_C_dstate
        # C_cached_ptrs = C_cache_ptr_i + offs_n * stride_C_cache_dstate

        if HAS_Z:
            z_ptrs = z_ptr_i + offs_m * stride_z_dim
            # z_cached_ptrs = z_cache_ptr_i + offs_m * stride_z_cache_dim

        out_ptrs = out_ptr_i + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(x_cache_ptrs, x, mask=offs_m < dim)

        # x_preconv = tl.load(x_preconv_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        # tl.store(x_preconv_cached_ptrs, x_preconv, mask=offs_m < dim)

        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            tl.store(dt_cached_ptrs, dt, mask=offs_m < dim)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
            # A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            dA = tl.exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
            # A = tl.load(A_ptr).to(tl.float32)
            dA = tl.exp(A * dt)  # scalar, not a matrix

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        tl.store(B_cached_ptrs, B, mask=offs_n < dstate)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        # tl.store(C_cached_ptrs, C, mask=offs_n < dstate)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            # tl.store(z_cached_ptrs, z, mask=offs_m < dim)
        if not TIE_HDIM:
            dB = B[None, :] * dt[:, None]
        else:
            dB = B * dt  # vector of size (dstate,)
        state = state * dA + dB * x[:, None]
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    state_cache,
    x_cache,
    dt_cache,
    B_cache,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    steps=None,
    return_step=None,
    initial_state=None,
):
    """
    If initial_state is not None, it will be used as the initial state for the first step.
    The SSM is computed for return_step steps on the cached elements.
    The state_cache is updated with the post-recomputation state.
    The SSM is then computed for the remaining steps on the non-cached elements, and the cache is updated accordingly.

    Arguments:
        - state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        - x: (batch, seq, dim) or (batch, seq, nheads, dim)
        - dt: (batch, seq,  dim) or (batch, seq, nheads, dim)
        - A: (dim, dstate) or (nheads, dim, dstate)
        - B: (batch, seq, dstate) or (batch, seq, ngroups_B, dstate)
        - C: (batch, seq, dstate) or (batch, seq, ngroups_C, dstate)
        - state_cache: (batch, seq, dim, dstate) or (batch, seq, nheads, dim, dstate)
        - x_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        - x_preconv_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        - dt_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        - B_cache: (batch, cache_seqlen, ngroups_B, dstate)
        - C_cache: (batch, cache_seqlen, ngroups_C, dstate)
        - D: (dim,) or (nheads, dim)
        - z: (batch, seq, dim) or (batch, seq, nheads, dim)
        - z_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        - dt_bias: (dim,) or (nheads, dim)
        - dt_softplus: bool
        - steps: int
        - return_step: int
        - initial_state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
    Return:
        - out: (batch, seq, dim) or (batch, seq, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    seq = x.shape[1]
    assert x.shape == (batch, seq, nheads, dim)
    # assert x.shape == x_preconv.shape
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups_B = B.shape[2]
    ngroups_C = C.shape[2]
    assert nheads % ngroups_B == 0, "nheads must be divisible by ngroups"
    assert nheads % ngroups_C == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seq, ngroups_B, dstate)
    assert C.shape == (batch, seq, ngroups_C, dstate)

    assert state_cache.shape == state.shape
    cached_steps = B_cache.shape[1]
    assert x_cache.shape == (batch, cached_steps, nheads, dim)
    # assert dt_cache.shape == x_cache.shape
    # assert x_cache.shape == x_preconv_cache.shape
    assert B_cache.shape == (batch, cached_steps, ngroups_B, dstate)
    # assert C_cache.shape == (batch, cached_steps, ngroups_C, dstate)

    # if z_cache is not None:
    #     assert z_cache.shape == x_cache.shape

    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)

    if return_step is not None:
        assert return_step in range(0, steps)
        return_state = state_cache
    else:
        return_state = None

    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)  # noqa: E731
    z_strides = (z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0)
    # z_cache_strides = (
    #     (z_cache.stride(0), z_cache.stride(1), z_cache.stride(2), z_cache.stride(3)) if z is not None else (0, 0, 0, 0)
    # )
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = (
        (32, 4)
        if dstate <= 16
        else ((16, 4) if dstate <= 32 else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else ((4, 8)))))
    )
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0

    assert steps is not None
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            return_state,
            x,
            # x_preconv,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            dt_cache,
            B_cache,
            # C_cache,
            # z_cache,
            x_cache,
            # x_preconv_cache,
            batch,
            nheads,
            dim,
            dstate,
            nheads // ngroups_B,
            nheads // ngroups_C,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            # x_preconv.stride(0),
            # x_preconv.stride(1),
            # x_preconv.stride(2),
            # x_preconv.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            z_strides[3],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            dt_cache.stride(3),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            B_cache.stride(3),
            # C_cache.stride(0),
            # C_cache.stride(1),
            # C_cache.stride(2),
            # C_cache.stride(3),
            # z_cache_strides[0],
            # z_cache_strides[1],
            # z_cache_strides[2],
            # z_cache_strides[3],
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(2),
            x_cache.stride(3),
            # x_preconv_cache.stride(0),
            # x_preconv_cache.stride(1),
            # x_preconv_cache.stride(2),
            # x_preconv_cache.stride(3),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            steps,
            return_step,
            initial_state,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_state_update_ref(
    state,
    x,
    # x_preconv,
    dt,
    A,
    B,
    C,
    state_cache,
    x_cache,
    # x_preconv_cache,
    dt_cache,
    B_cache,
    # C_cache,
    D=None,
    z=None,
    # z_cache=None,
    dt_bias=None,
    dt_softplus=False,
    steps=None,
    return_step=None,
    initial_state=None,
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seq, dim) or (batch, seq, nheads, dim)
        x_preconv: (batch, seq, dim) or (batch, seq, nheads, dim)
        dt: (batch,seq,  dim) or (batch, seq, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seq, dstate) or (batch, seq, ngroups, dstate)
        C: (batch, seq, dstate) or (batch, seq,ngroups, dstate)
        state_cache: (batch, seq, dim, dstate) or (batch, seq, nheads, dim, dstate)
        x_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        x_preconv_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        dt_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        B_cache: (batch, cache_seqlen, ngroups, dstate)
        C_cache: (batch, cache_seqlen, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, seq, dim) or (batch,seq,nheads, dim)
        z_cache: (batch, cache_seqlen, dim) or (batch, cache_seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seq, dim) or (batch, seq, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    seq = x.shape[1]
    cached_steps = x_cache.shape[1]
    assert x.shape == (batch, seq, nheads, dim)
    assert x_cache.shape == (batch, cached_steps, nheads, dim)
    assert dt.shape == x.shape
    assert dt_cache.shape == x_cache.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups_B = B.shape[2]
    ngroups_C = C.shape[2]
    assert nheads % ngroups_B == 0, "nheads must be divisible by ngroups"
    assert nheads % ngroups_C == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seq, ngroups_B, dstate)
    assert C.shape == (batch, seq, ngroups_C, dstate)
    assert B_cache.shape == (batch, cached_steps, ngroups_B, dstate)
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
        # assert z_cache.shape == x_cache.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)

    if return_step is not None:
        assert return_step in range(0, steps)
        return_state = state_cache
    else:
        return_state = None

    out = torch.empty_like(x)
    assert steps is not None

    if initial_state is not None:
        state_tmp = copy.deepcopy(initial_state)
    else:
        state_tmp = copy.deepcopy(state)

    for seq_i in range(return_step):
        x_i = x_cache[:, seq_i]
        dt_i = dt_cache[:, seq_i]
        B_i = B_cache[:, seq_i]
        # C_i = C_cache[:, seq_i]

        if dt_bias is not None:
            dt_i += dt_bias

        dt_i = F.softplus(dt_i) if dt_softplus else dt_i
        dA = torch.exp(rearrange(dt_i, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
        B_i = repeat(B_i, "b g n -> b (g h) n", h=nheads // ngroups_B)  # (batch, nheads, dstate)
        # C_i = repeat(C_i, "b g n -> b (g h) n", h=nheads // ngroups_C)  # (batch, nheads, dstate)
        dB = rearrange(dt_i, "b h d -> b h d 1") * rearrange(B_i, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
        state_tmp = state_tmp * dA + dB * rearrange(x_i, "b h d -> b h d 1")  # (batch, dim, dstate

    if return_state is not None:
        return_state.copy_(state_tmp)

    x_cache[:, :seq] = x
    # x_preconv_cache[:, :seq] = x_preconv
    dt_cache[:, :seq] = dt
    B_cache[:, :seq] = B
    # C_cache[:, :seq] = C
    # if z is not None:
    #    z_cache[:, :seq] = z

    for seq_i in range(steps - return_step):
        x_i = x[:, seq_i]
        dt_i = dt[:, seq_i]
        B_i = B[:, seq_i]
        C_i = C[:, seq_i]
        z_i = z[:, seq_i] if z is not None else None

        if dt_bias is not None:
            dt_i += dt_bias

        dt_i = F.softplus(dt_i) if dt_softplus else dt_i
        dA = torch.exp(rearrange(dt_i, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
        B_i = repeat(B_i, "b g n -> b (g h) n", h=nheads // ngroups_B)  # (batch, nheads, dstate)
        C_i = repeat(C_i, "b g n -> b (g h) n", h=nheads // ngroups_C)  # (batch, nheads, dstate)
        dB = rearrange(dt_i, "b h d -> b h d 1") * rearrange(B_i, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
        state_tmp = state_tmp * dA + dB * rearrange(x_i, "b h d -> b h d 1")  # (batch, dim, dstate
        out_i = torch.einsum("bhdn,bhn->bhd", state_tmp.to(C.dtype), C_i)
        if D is not None:
            out_i += (x_i * D).to(out.dtype)
        out[:, seq_i, :, :] = (out_i if z is None else out_i * F.silu(z_i)).to(x.dtype)
        if not has_heads:
            out = out.squeeze(1)

    state.copy_(state_tmp)
    return out


if __name__ == "__main__":
    import copy

    batch = 1
    seq = 4
    cache_seqlen = 5
    nheads = 256
    dim = 1
    dstate = 16
    ngroups_C = 64
    ngroups_B = 32
    steps = 4
    return_step = 2
    dtype = torch.float32

    state = torch.randn(batch, nheads, dim, dstate, dtype=dtype, device="cuda") - 1.0
    state_ref = copy.deepcopy(state)

    x = torch.randn(batch, seq, nheads, dim, dtype=dtype, device="cuda")
    x_cache = torch.zeros(batch, cache_seqlen, nheads, dim, dtype=dtype, device="cuda")
    x_cache_ref = copy.deepcopy(x_cache)

    # x_preconv = torch.randn(batch, seq, nheads, dim, dtype=dtype, device="cuda")
    # x_preconv_cache = torch.zeros(batch, cache_seqlen, nheads, dim, dtype=dtype, device="cuda")
    # x_preconv_cache_ref = copy.deepcopy(x_preconv_cache)

    dt = torch.randn(batch, seq, nheads, dim, dtype=dtype, device="cuda")
    dt_ref = copy.deepcopy(dt)
    dt_cache = torch.zeros(batch, cache_seqlen, nheads, dim, dtype=dtype, device="cuda")
    dt_cache_ref = copy.deepcopy(dt_cache)

    B = torch.randn(batch, seq, ngroups_B, dstate, dtype=dtype, device="cuda")
    B_cache = torch.zeros(batch, cache_seqlen, (ngroups_B * dstate), dtype=dtype, device="cuda")
    B_cache = rearrange(B_cache, "b s (g d) -> b s g d", d=dstate)
    B_cache_ref = copy.deepcopy(B_cache)

    C = torch.randn(batch, seq, ngroups_C, dstate, dtype=dtype, device="cuda") - 4.0
    # C_cache = torch.zeros(batch, cache_seqlen, (ngroups_C * dstate), dtype=dtype, device="cuda")
    # C_cache = rearrange(C_cache, "b s (g d) -> b s g d", d=dstate)
    # C_cache_ref = copy.deepcopy(C_cache)

    state_cache = torch.randn_like(state).type(dtype) - 1.0
    state_cache_ref = copy.deepcopy(state_cache)

    initial_state = torch.randn(batch, nheads, dim, dstate, dtype=dtype, device="cuda") - 1.0
    initial_state_ref = copy.deepcopy(initial_state)

    # initial_state = None
    # initial_state_ref = None

    z = torch.randn(batch, seq, nheads, dim, dtype=dtype, device="cuda")
    # z_cache = torch.zeros(batch, cache_seqlen, nheads, dim, dtype=dtype, device="cuda")
    # z_cache_ref = copy.deepcopy(z_cache)

    D = torch.randn(nheads, dim, dtype=dtype, device="cuda") - 0.5
    dt_bias = torch.randn(nheads, dim, dtype=dtype, device="cuda") - 4.0
    A = torch.randn(nheads, dim, dstate, dtype=dtype, device="cuda") - 1.0
    output_ref = selective_state_update_ref(
        state_ref,
        x,
        # x_preconv,
        dt_ref,
        A,
        B,
        C,
        state_cache_ref,
        x_cache_ref,
        # x_preconv_cache_ref,
        dt_cache_ref,
        B_cache_ref,
        # C_cache_ref,
        D=D,
        z=z,
        # z_cache=z_cache_ref,
        dt_bias=dt_bias,
        dt_softplus=True,
        steps=steps + return_step,
        return_step=return_step,
        initial_state=initial_state_ref,
    )

    output = selective_state_update(
        state,
        x,
        # x_preconv,
        dt,
        A,
        B,
        C,
        state_cache,
        x_cache,
        # x_preconv_cache,
        dt_cache,
        B_cache,
        # C_cache,
        D=D,
        z=z,
        # z_cache=z_cache,
        dt_bias=dt_bias,
        dt_softplus=True,
        steps=steps + return_step,
        return_step=return_step,
        initial_state=initial_state,
    )

    print(torch.allclose(output, output_ref))
    print(torch.allclose(state_cache, state_cache_ref))
    print(torch.allclose(state, state_ref))
    print(torch.allclose(x_cache, x_cache_ref))
    # print(torch.allclose(x_preconv_cache, x_preconv_cache_ref))
    print(torch.allclose(dt_cache, dt_cache_ref))
    print(torch.allclose(B_cache, B_cache_ref))
    # print(torch.allclose(C_cache, C_cache_ref))
    # print(torch.allclose(z_cache, z_cache_ref))

    # print max diff for all of them
    print(torch.max(torch.abs(output - output_ref)))
    print(torch.max(torch.abs(state_cache - state_cache_ref)))
    print(torch.max(torch.abs(state - state_ref)))
    print(torch.max(torch.abs(x_cache - x_cache_ref)))
    # print(torch.max(torch.abs(x_preconv_cache - x_preconv_cache_ref)))
    print(torch.max(torch.abs(dt_cache - dt_cache_ref)))
    print(torch.max(torch.abs(B_cache - B_cache_ref)))
    # print(torch.max(torch.abs(C_cache - C_cache_ref)))
# print(torch.max(torch.abs(z_cache - z_cache_ref)))
