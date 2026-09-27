"""Two-pass CUDA reductions. Imported only by an explicit Triton request."""

import triton
import triton.language as tl


@triton.jit
def partial_reduce(Logits, Mask, Max, Index, Count, Bad,
                   V: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    token = block * BLOCK + tl.arange(0, BLOCK)
    inside = token < V
    scores = tl.load(Logits + token, inside, other=-float('inf'))
    words = tl.load(Mask + token // 32, inside, other=0).to(tl.uint32)
    allowed = inside & (((words >> (token % 32)) & 1) != 0)
    nan = scores != scores
    clean = tl.where(allowed & ~nan, scores, -float('inf'))
    maximum = tl.max(clean, 0)
    index = tl.min(tl.where(allowed & ~nan & (scores == maximum), token, V), 0)
    tl.store(Max + block, maximum)
    tl.store(Index + block, index)
    tl.store(Count + block, tl.sum(allowed.to(tl.int32), 0))
    tl.store(Bad + block, tl.max((inside & nan).to(tl.int32), 0))


@triton.jit
def finish_reduce(Max, Index, Count, Bad, TokenOut, CountOut, StatusOut,
                  V: tl.constexpr, N: tl.constexpr, REDUCE: tl.constexpr):
    block = tl.arange(0, REDUCE)
    valid = block < N
    maximums = tl.load(Max + block, valid, other=-float('inf'))
    indices = tl.load(Index + block, valid, other=V)
    counts = tl.load(Count + block, valid, other=0)
    bad = tl.max(tl.load(Bad + block, valid, other=0), 0)
    maximum = tl.max(maximums, 0)
    token = tl.min(tl.where(valid & (maximums == maximum), indices, V), 0)
    count = tl.sum(counts.to(tl.int64), 0)
    status = tl.where(bad != 0, 2, tl.where(count == 0, 1, 0))
    tl.store(TokenOut, token.to(tl.int64))
    tl.store(CountOut, count)
    tl.store(StatusOut, status)
