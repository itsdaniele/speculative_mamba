"""We want triton==2.1.0 for this"""

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange


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
    x_cache_ptr,
    batch,
    dim,
    dstate,
    # Strides
    stride_state_batch,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_seq,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_seq,
    stride_dt_dim,
    stride_dt_bias_dim,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_seq,
    stride_B_dstate,
    stride_C_batch,
    stride_C_seq,
    stride_C_dstate,
    stride_D_dim,
    stride_z_batch,
    stride_z_seq,
    stride_z_dim,
    stride_out_batch,
    stride_out_seq,
    stride_out_dim,
    stride_dt_cache_batch,
    stride_dt_cache_seq,
    stride_dt_cache_dim,
    stride_B_cache_batch,
    stride_B_cache_seq,
    stride_B_cache_dstate,
    stride_x_cache_batch,
    stride_x_cache_seq,
    stride_x_cache_dim,
    DT_SOFTPLUS: tl.constexpr,
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
    state_ptr += pid_b * stride_state_batch

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)

    if HAS_INITIAL_STATE:
        initial_state_ptr += pid_b * stride_state_batch
        initial_state_ptrs = initial_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    if HAS_RETURN_STATE:
        return_state_ptr += pid_b * stride_state_batch
        return_state_ptrs = return_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
        dt_bias = tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)

    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if HAS_INITIAL_STATE:
        state = tl.load(initial_state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    else:
        state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)

    A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    for seq_i in tl.static_range(return_step):
        x_cached_ptr_i = x_cache_ptr + pid_b * stride_x_cache_batch + seq_i * stride_x_cache_seq
        dt_cached_ptr_i = dt_cache_ptr + pid_b * stride_dt_cache_batch + seq_i * stride_dt_cache_seq
        B_cached_ptr_i = B_cache_ptr + pid_b * stride_B_cache_batch + seq_i * stride_B_cache_seq

        x_cached_ptrs = x_cached_ptr_i + offs_m * stride_x_cache_dim
        dt_cached_ptrs = dt_cached_ptr_i + offs_m * stride_dt_cache_dim
        B_cached_ptrs = B_cached_ptr_i + offs_n * stride_B_cache_dstate

        x = tl.load(x_cached_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dt = tl.load(dt_cached_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += dt_bias
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)

        dA = tl.exp(A * dt[:, None])
        B = tl.load(B_cached_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None]
        state = state * dA + dB * x[:, None]

    if HAS_RETURN_STATE:
        tl.store(return_state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

    for i in tl.static_range(steps - return_step):
        x_ptr_i = x_ptr + pid_b * stride_x_batch + i * stride_x_seq
        x_cache_ptr_i = x_cache_ptr + pid_b * stride_x_cache_batch + i * stride_x_cache_seq

        dt_ptr_i = dt_ptr + pid_b * stride_dt_batch + i * stride_dt_seq
        dt_cache_ptr_i = dt_cache_ptr + pid_b * stride_dt_cache_batch + i * stride_dt_cache_seq

        B_ptr_i = B_ptr + pid_b * stride_B_batch + i * stride_B_seq
        B_cache_ptr_i = B_cache_ptr + pid_b * stride_B_cache_batch + i * stride_B_cache_seq

        C_ptr_i = C_ptr + pid_b * stride_C_batch + i * stride_C_seq

        if HAS_Z:
            z_ptr_i = z_ptr + pid_b * stride_z_batch + i * stride_z_seq

        out_ptr_i = out_ptr + pid_b * stride_out_batch + i * stride_out_seq

        x_ptrs = x_ptr_i + offs_m * stride_x_dim
        x_cache_ptrs = x_cache_ptr_i + offs_m * stride_x_cache_dim

        dt_ptrs = dt_ptr_i + offs_m * stride_dt_dim
        dt_cache_ptrs = dt_cache_ptr_i + offs_m * stride_dt_cache_dim

        B_ptrs = B_ptr_i + offs_n * stride_B_dstate
        B_cache_ptrs = B_cache_ptr_i + offs_n * stride_B_cache_dstate

        C_ptrs = C_ptr_i + offs_n * stride_C_dstate

        if HAS_Z:
            z_ptrs = z_ptr_i + offs_m * stride_z_dim

        out_ptrs = out_ptr_i + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(x_cache_ptrs, x, mask=offs_m < dim)

        dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(dt_cache_ptrs, dt, mask=offs_m < dim)

        if HAS_DT_BIAS:
            dt += dt_bias
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        dA = tl.exp(A * dt[:, None])

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        tl.store(B_cache_ptrs, B, mask=offs_n < dstate)

        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)

        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None]
        state = state * dA + dB * x[:, None]

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)

        tl.store(out_ptrs, out, mask=offs_m < dim)

    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))


def selective_state_update_multi_cached(
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
    Argument:
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """

    batch, dim, dstate = state.shape
    seqlen = x.shape[1]
    assert x.shape == (batch, seqlen, dim)
    assert dt.shape == x.shape
    assert A.shape == (dim, dstate)
    assert B.shape == (batch, seqlen, dstate)
    assert C.shape == B.shape

    assert state_cache.shape == state.shape
    cached_steps = x_cache.shape[1]
    assert x_cache.shape == (batch, cached_steps, dim)
    assert dt_cache.shape == x_cache.shape
    assert B_cache.shape == (batch, cached_steps, dstate)

    if D is not None:
        assert D.shape == (dim,)
    if z is not None:
        assert z.shape == x.shape

    if dt_bias is not None:
        assert dt_bias.shape == (dim,)

    if return_step is not None:
        assert return_step in range(0, steps)
        return_state = state_cache
    else:
        return_state = None

    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch)  # noqa: E731
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)

    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = (
        (32, 4)
        if dstate <= 16
        else ((16, 4) if dstate <= 32 else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else ((4, 8)))))
    )

    assert steps is not None
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            return_state,
            x,
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
            x_cache,
            batch,
            dim,
            dstate,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(2),
            dt_softplus,
            BLOCK_SIZE_M,
            steps,
            return_step,
            initial_state,
            num_warps=num_warps,
        )
    return out


def selective_state_update_multi_cached_ref(
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
    Argument:
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = state.shape
    assert state_cache.shape == state.shape
    seqlen = x.shape[1]
    cached_steps = x_cache.shape[1]
    assert x.shape == (batch, seqlen, dim)
    assert x_cache.shape == (batch, cached_steps, dim)
    assert dt.shape == x.shape
    assert dt_cache.shape == x_cache.shape
    assert A.shape == (dim, dstate)
    assert B.shape == (batch, seqlen, dstate)
    assert B_cache.shape == (batch, cached_steps, dstate)
    assert C.shape == B.shape

    if D is not None:
        assert D.shape == (dim,)
    if z is not None:
        assert z.shape == x.shape

    if dt_bias is not None:
        assert dt_bias.shape == (dim,)

    if return_step is not None:
        return_state = state_cache
    else:
        return_state = None

    out = torch.empty_like(x)

    assert steps is not None

    if initial_state is not None:
        state_tmp = copy.deepcopy(initial_state)

    for seq_i in range(return_step):
        x_i_cached = x_cache[:, seq_i, :]
        dt_i_cached = dt_cache[:, seq_i, :]
        B_i_cached = B_cache[:, seq_i, :]
        if dt_bias is not None:
            dt_i_cached += dt_bias
        dt_i_cached = F.softplus(dt_i_cached) if dt_softplus else dt_i_cached
        dA_cached = torch.exp(rearrange(dt_i_cached, "b d -> b d 1") * A)
        dB_cached = rearrange(dt_i_cached, "b d -> b d 1") * rearrange(B_i_cached, "b n -> b 1 n")
        state_tmp = state_tmp * dA_cached + dB_cached * rearrange(x_i_cached, "b d -> b d 1")

    return_state.copy_(state_tmp)

    for seq_i in range(steps - return_step):
        x_i = x[:, seq_i, :]
        dt_i = dt[:, seq_i, :]
        B_i = B[:, seq_i, :]
        C_i = C[:, seq_i, :]

        x_cache[:, seq_i, :] = x_i
        dt_cache[:, seq_i, :] = copy.deepcopy(dt_i)
        B_cache[:, seq_i, :] = B_i
        if z is not None:
            z_i = z[:, seq_i, :]
        if dt_bias is not None:
            dt_i += dt_bias

        dt_i = F.softplus(dt_i) if dt_softplus else dt_i
        dA = torch.exp(rearrange(dt_i, "b d -> b d 1") * A)
        dB = rearrange(dt_i, "b d -> b d 1") * rearrange(B_i, "b n -> b 1 n")
        state_tmp = state_tmp * dA + dB * rearrange(x_i, "b d -> b d 1")
        out_i = torch.einsum("bdn,bn->bd", state_tmp.to(C.dtype), C_i)
        if D is not None:
            out_i += (x_i * D).to(out_i.dtype)
        out[:, seq_i, :] = out_i if z is None else out_i * F.silu(z_i)

    state.copy_(state_tmp)
    return out


if __name__ == "__main__":
    import copy

    torch.random.manual_seed(0)
    batch_size = 1
    seqlen = 2
    dim = 512
    itype = torch.float32
    device = torch.device("cuda")
    dstate = 16
    has_z = True
    cache_sequence = 4
    return_step = 2

    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    state_ref = copy.deepcopy(state)

    state_cache = torch.randn_like(state) - 1.0
    state_cache_ref = copy.deepcopy(state_cache)

    x = torch.randn(batch_size, seqlen, dim, device=device, dtype=itype)
    x_ref = copy.deepcopy(x)
    x_cache = torch.randn(batch_size, cache_sequence, dim, device=device, dtype=itype)
    x_cache_ref = copy.deepcopy(x_cache)

    dt = torch.randn(batch_size, seqlen, dim, device=device, dtype=itype)
    dt_ref = copy.deepcopy(dt)
    dt_cache = torch.randn(batch_size, cache_sequence, dim, device=device, dtype=itype)
    dt_cache_ref = copy.deepcopy(dt_cache)

    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    D = torch.randn(dim, device=device) - 0.5

    B = torch.randn(batch_size, seqlen, dstate, device=device)
    B_ref = copy.deepcopy(B)
    B_cache = torch.randn(batch_size, cache_sequence, dstate, device=device)
    B_cache_ref = copy.deepcopy(B_cache)

    C = torch.randn(batch_size, seqlen, dstate, device=device) - 4.0
    C_ref = copy.deepcopy(C)

    if has_z:
        z = torch.randn_like(x)
        z_ref = copy.deepcopy(z)
    else:
        z = None

    with torch.no_grad():
        out_ref = selective_state_update_multi_cached_ref(
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
            D,
            z,
            dt_bias,
            dt_softplus=True,
            steps=seqlen + return_step,
            return_step=return_step,
            initial_state=state_cache,
        )

    with torch.no_grad():
        out = selective_state_update_multi_cached(
            state_ref,
            x_ref,
            dt_ref,
            A,
            B_ref,
            C_ref,
            state_cache_ref,
            x_cache_ref,
            dt_cache_ref,
            B_cache_ref,
            D,
            z_ref,
            dt_bias,
            dt_softplus=True,
            steps=seqlen + return_step,
            return_step=return_step,
            initial_state=state_cache_ref,
        )

    print(torch.allclose(out, out_ref))
    print(torch.allclose(state_cache, state_cache_ref))
    print(torch.allclose(state, state_ref))
    print(torch.allclose(x_cache, x_cache_ref))
    print(torch.allclose(dt_cache, dt_cache_ref))
    print(torch.allclose(B_cache, B_cache_ref))

    # print max diff for all of them
    print(torch.max(torch.abs(out - out_ref)))
    print(torch.max(torch.abs(state_cache - state_cache_ref)))
    print(torch.max(torch.abs(state - state_ref)))
    print(torch.max(torch.abs(x_cache - x_cache_ref)))
    print(torch.max(torch.abs(dt_cache - dt_cache_ref)))
    print(torch.max(torch.abs(B_cache - B_cache_ref)))
