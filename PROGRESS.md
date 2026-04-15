# Progress Log: Tiny-LLM on Roblox/Luau

End-to-end LLM inference in pure Luau on a Roblox server. Goal: push a pretrained Llama-arch model to usable tok/s, then scale up to a larger model for actual capability.

## Session 1: Proof of concept

Built the working end-to-end pipeline, verified correctness bit-exact vs numpy, established baseline performance numbers.

### Model: `arnir0/Tiny-LLM`
- Llama arch, 13M params. Config: 1 layer, hidden=192, GQA 2q/1kv, head_dim=96, intermediate=1024, vocab=32000, max_ctx=1024, rope_theta=10000.
- **Untied** embeddings (separate `embed` and `lm_head`).
- Per-token compute: 13.69 MFLOP. lm_head (32000×192 matvec) is 90% of it.

### Wire format: `handover-files/README.md`
- Header 52 bytes: `TLLM` magic + u32 version + u32 dtype + 8×u32 config + f32 rope_theta + f32 rms_eps.
- Q8 layout per tensor: `int8[rows*cols]` followed by `f32[rows]` per-row scales.
- Norm tensors always f32. Tokenizer has its own format (`TLLT` magic + per-piece u16 len + utf8 + f32 score).
- Canonical tensor order: embed → per-layer{attn_norm, q, k, v, o, ffn_norm, gate, up, down} → final_norm → lm_head.

### Python pipeline
- `handover-files/tinyllm_numpy.py` — numpy reference forward (RMSNorm, halves-convention RoPE, GQA, SwiGLU). Ground-truth for round-trip verification.
- `handover-files/export_and_verify.py` — binary exporter + `LuauSim` class that reads the blob using only Luau-equivalent primitives; round-trip confirmed bit-exact for f32, top-10 overlap 10/10 and Spearman ρ=0.9999 for q8.
- `handover-files/run_real.py` — loads real safetensors, generates text with sentencepiece tokenizer.

### Binary blobs (hosted on GitHub raw)
- `handover-files/tinyllm_real_q8.bin` — 13.26 MB quantized weights.
- `handover-files/tokenizer.bin` — 400 KB SentencePiece converted to simple binary.
- URLs: `https://raw.githubusercontent.com/RussellBustamante/tinyllm_real_q8_v1/main/{tinyllm_real_q8.bin,tokenizer.bin}`.

### Luau smoke-test suite (command-bar scripts, each verifies one component)
- `handover-files/TinyLLM_kernels.luau` — standalone kernel module (matvec_f32, matvec_q8, rmsnorm, silu_mul, rope_inplace with precomputed cos/sin).
- `handover-files/loader_cmdbar.lua` — header parse + per-tensor slicing.
- `handover-files/smoke_test_cmdbar.lua` — matmul microbenchmark on the 32000×192 shape.
- `handover-files/forward_smoke_cmdbar.lua` — bit-exact prefill vs numpy golden (top-5 argmax `{338, 756, 471, 322, 393}`).
- `handover-files/decode_smoke_cmdbar.lua` — KV-cache single-token decode, 12/12 tokens match numpy greedy.
- `handover-files/tokenizer_smoke_cmdbar.lua` — 7/7 encode + 7/7 round-trip match Python sentencepiece.
- `handover-files/profile_sampler_cmdbar.lua` — phase profiler + heap-based sampler A/B (written in session 1, measured in session 2).
- `handover-files/full_demo_cmdbar.lua` — end-to-end demo (prompt → tokenize → prefill → sample-decode → detokenize).

### Measured Session-1 baselines
- Matmul kernel (W[32000,192] skinny): 14 ms / 0.88 GFLOPS unrolled x4, 25 ms naive.
- Model fetch: 0.5 s for 13 MB from GitHub raw. Parse: 4 ms.
- Prefill: 13.5 ms/tok (bit-exact match to numpy).
- Decode (KV cache): 13.1 ms/tok → 76 tok/s decode-only. 12/12 tokens match numpy greedy.
- Tokenizer: 7/7 encode, 7/7 roundtrip match Python sentencepiece.
- Full demo with sampling: **34 tok/s end-to-end** (sampler dominated).

### Critical platform discoveries
1. **Luau is op-bound, not bandwidth-bound.** Measured ~520M mul-adds/sec in `--!native` q8 kernel. Empirical formula: `tok/s ≈ 520M / param_count`. This **inverts** naive bandwidth reasoning — e.g. Bonsai-1.7B Q1 (small bytes) would be ~0.4 tok/s despite small file size because 1.7B is the bottleneck, not the bytes.
2. **`io.write` doesn't exist in Roblox Luau.** Only `print`. No streaming stdout.
3. **`[[ ... ]]` long strings break on embedded `]]`** (e.g. `table[key[i]]`). Use `[==[ ... ]==]` when source contains Luau code.
4. **Buffer hard cap: 1 GB per buffer.** Multiple buffers fine.
5. **HttpService body cap: 50 MB.** Rate limit 500 req/min. Bigger models need chunked fetch.
6. **Script timeout ~20 s without yielding.** `task.wait()` resets.
7. **Native codegen silently deopts** on type ambiguity. Check with `debug.dumpcodesize()`.
8. **Server has 2 parallel workers, client has 8.** HttpService is server-only.
9. **`buffer.readi8`/`readf32` have per-call overhead.** Unrolling inner loops x4 gave measurable wins.
10. **Luau RoPE is halves convention** (like Llama), not interleaved (like GPT-NeoX).
11. **Embedding is q8**: must multiply by per-row scale during lookup.

### Rejected alternatives (don't revisit)
- **LFM2.5-350M**: non-standard Liquid arch (LIV conv blocks), major re-implementation.
- **TinyStories-33M**: GPT-Neo arch (learned positional, LayerNorm, GELU, MHA, bias). Too many new kernels.
- **SmolLM2-360M**: ~1.5 tok/s current, ~4 optimized. Revisit after SmolLM2-135M is shipping.
- **SmolLM3-3B**: too big, also has NoPE (skip RoPE every 4th layer).
- **Bonsai-1.7B Q1**: ~0.4 tok/s on Luau regardless of quantization (op-bound, ternary helps bandwidth not ops).

---

## Session 2: Optimization — 33 → 171.7 tok/s (5.2× total)

Starting state: `handover-files/full_demo_cmdbar.lua` at 33 tok/s with `table.sort` sampler bottleneck. Goal: push toward 150+ tok/s on Tiny-LLM. Actual: **171.7 tok/s**.

### Phase A: Profile + Tier 1 fixes → 73.7 tok/s

Ran the pre-written `handover-files/profile_sampler_cmdbar.lua`. Findings:

| phase | ms/step | % |
|---|---|---|
| lm_head | 11.634 | **88.7%** |
| ffn | 1.134 | 8.6% |
| attn_proj | 0.143 | 1.1% |
| attn | 0.120 | 0.9% |
| o_res | 0.071 | 0.5% |
| rope | 0.008 | **0.1%** |
| embed | 0.003 | 0.0% |

- **Native codegen confirmed engaged** for all hot paths (`step` 57.89 KB, `matvec_q8` 4.56 KB, `apply_rope` 4.28 KB, `rmsnorm` 1.86 KB).
- **Heap-based top-K sampler**: 40× over `table.sort` (16.5 ms → 0.412 ms), 5/5 top-5 overlap with old sampler at seed 1337. Flat parallel `heap_ids`/`heap_vals` arrays at module scope, no per-call allocation.
- **Non-obvious finding that updated the handover's Tier 1 list**: the handover named "precompute RoPE cos/sin tables" as Tier 1. Profile showed RoPE is 0.1% of step time — precomputing would save nothing. Deprioritized (later re-added in Phase D as SmolLM2 infra).

Produced `full_demo_v2_cmdbar.lua`. Measured: **73.7 tok/s** end-to-end (projection was 70.1 — matched within noise). Prefill unchanged at 13.5 ms/tok.

### Phase B: Tier 2 matvec optimization → 157.5 tok/s

lm_head is 88.7% of step. Only matvec optimization matters.

**Vector SIMD experiment** (`simd_matvec_cmdbar.lua`): FAILED.
- Variants tested: A=scalar x4 baseline, A2=scalar x8, B=vec3 componentwise, C=vec3 via `vector.dot`, D=scalar with activation preloaded into `{number}` table.
- Results: A=11.71 ms, A2=8.29 ms (1.41×), B=26.53 ms (0.44×), C=17.61 ms (0.67×), D=15.04 ms (0.78×).
- **Conclusion**: Luau `vector.create(readi8(...), readi8(...), readi8(...))` in a hot loop is 2× slower than scalar. Native codegen doesn't fuse the i8→vector pack into useful SIMD. Handover's bitsplicer-based 2-3× SIMD projection doesn't translate to int8×f32 in Luau. **Kill this direction permanently.**
- Also: activation-backed `{number}` table (D) is slower than buffer reads. Buffer primitives stay.

**Unroll sweep** (`unroll_sweep_cmdbar.lua`, then `unroll_sweep_v2_cmdbar.lua` pushing further):
- x4: 11.60 ms (1.06 GFLOPS)
- x8: 8.37 ms (1.47 GFLOPS, 1.39×)
- x16: 6.69 ms (1.84 GFLOPS, 1.74×)
- x32: 5.69 ms (2.16 GFLOPS, 2.04×)
- x64: 5.29 ms (2.32 GFLOPS, 2.19×)
- **No compiler bailout through x64.** Diminishing returns past x32 but still positive.
- Both lm_head (cols=192) and FFN down (cols=1024) are divisible by 64 — single unified kernel, no tail.

Produced `full_demo_v3_cmdbar.lua` (x32) projected 132 tok/s; then `full_demo_v4_cmdbar.lua` (x64). Measured: **157.5 tok/s** end-to-end (projection was 138.8 — overperformed by 13%, because FFN/attn_proj scaled equally and non-matvec overhead compressed).

Prefill: 13.2 → 6.0 ms/tok. Handover's 150+ tok/s target hit.

### Phase C: Squeeze experiments (6 files) → mostly negative results

Ran 6 focused experiments in parallel. Files at top-level of repo.

**Matvec-kernel benches** (isolated lm_head shape):

1. `bench_split_acc_cmdbar.lua` — split-accumulator sweep.
   - x64 baseline: 5.20 ms
   - x64_split4 (4 accumulators × 16 terms): **4.67 ms, 1.11× — WINNER**
   - x64_split8 (8 accumulators × 8 terms): 4.93 ms, 1.06× (worse than split4 — register pressure)
   - **Finding**: the single 64-term `sum +=` chain in x64 was a serial dependency chain. 4 parallel accumulators expose ILP.

2. `bench_packed_reads_cmdbar.lua` — readu32 + bit-extract + sign-extend. CATASTROPHIC FAIL.
   - x64_packed: 56.23 ms (0.10×), x64_packed_split4: 54.69 ms (0.10×).
   - **Finding**: `buffer.readi8` is effectively free under native codegen; bit ops + sign-extension + inner Lua `for g=0,15 do` loop utterly dominate. The handover's "readi8 has meaningful per-call overhead" is correct but the overhead is < bit-op cost. **Kill this direction.**

3. `bench_row_pair_cmdbar.lua` — process multiple rows per iter sharing x-reads.
   - x64 baseline: 5.35 ms
   - 2row_x32: 5.86 ms, **0.91× (LOSS)** — 32 x-locals + 2 sum accumulators = register pressure.
   - 4row_x16: 4.90 ms, 1.09× — 16 x-locals + 4 accumulators fits better.
   - **Finding**: 16 x-locals is the sweet spot. More than that = register spill.

**End-to-end variants** (each = v4 + one change, all within ~1% noise of v4's 157.5):

4. `full_demo_v4_rope_precompute_cmdbar.lua` — precomputed cos/sin tables: 158.0 tok/s (+0.3%).
5. `full_demo_v4_bufferfill_cmdbar.lua` — `buffer.fill` for attn_out zeroing: 155.5 tok/s (-1.3%).
6. `full_demo_v4_attn_unroll_cmdbar.lua` — x16 unroll on qK dot: 156.2 tok/s (-0.8%).

All three within noise (±2% run-to-run). None is a measurable perf win or loss — all folded into v5 anyway as they're clean / future-infrastructure.

### Phase D: v5 + compound test → 171.7 tok/s (plateau)

Produced `full_demo_v5_cmdbar.lua` folding:
1. **x64_split4 matvec** (the 1.11× win).
2. **Precomputed RoPE tables** (neutral now, matters at 30 layers for SmolLM2).
3. **`buffer.fill`** for attn_out (cleaner).
4. **x16 unrolled qK dot** (neutral now, matters at long context).

Measured: **171.7 tok/s end-to-end** (1.09× over v4). Prefill: 6.0 → 5.4 ms/tok.

**Compound test** (`bench_combo_cmdbar.lua`): tested whether split4 + 4row_x16 stack.
- x64: 5.26 ms
- x64_split4: 4.73 ms (1.11×)
- 4row_x16: 4.87 ms (1.08×)
- 4row_x16_split2 (compound: 4 rows × 2 accumulators = 8 sums + 16 x-locals): 4.94 ms (1.07×)
- **Finding**: compound LOSES vs split4 alone. Three non-scalar variants cluster tightly (4.73–4.94 ms) — this is the native codegen's effective ceiling for q8 matvec on this VM. Register pressure from combining both optimizations outweighs their individual gains.

### Session-2 final state

**v5 is the shipping artifact**: `full_demo_v5_cmdbar.lua` at 171.7 tok/s end-to-end (5.2× over original 33 tok/s baseline).

Decode step breakdown at v5 (~5.8 ms/step):
- lm_head ≈ 88% (limited by native-codegen ceiling at ~2.6 GFLOPS)
- FFN ≈ 9%, attn ≈ 1%, other ≈ 2%, sampler ≈ 0.4 ms

### Remaining Tier 2/3 options (pocketed, not pursued)

- **Actor-shard lm_head**: server has 2 workers → theoretical 1.8× on the 88% phase. Complex plumbing (SharedTable for h/logits). Viable for Tiny-LLM only; doesn't transfer cleanly to SmolLM2 (where compute is spread across 30 layers, not concentrated in one matvec — sync overhead would compound).
- **Q4 kernel**: not needed for Tiny-LLM (Q8 fits HTTP cap). Reserved for SmolLM2 port.
- **Top-K vocab truncation**: chicken-and-egg without a proxy/draft.
- **Speculative decoding**: multi-week project (needs a trained draft model).

### Files produced this session (top-level of repo)

**Shipping artifacts**:
- `full_demo_v5_cmdbar.lua` — current production (171.7 tok/s).

**Versioned demos (each a measurement point)**:
- `full_demo_v2_cmdbar.lua` (heap sampler) → 73.7 tok/s
- `full_demo_v3_cmdbar.lua` (x32 matvec) — not measured end-to-end; v4 superseded
- `full_demo_v4_cmdbar.lua` (x64 matvec) → 157.5 tok/s
- `full_demo_v4_{rope_precompute,bufferfill,attn_unroll}_cmdbar.lua` — ablation studies, all neutral

**Matvec benchmarks**:
- `simd_matvec_cmdbar.lua` (Luau `vector` type — disproved as SIMD path)
- `unroll_sweep_cmdbar.lua` / `unroll_sweep_v2_cmdbar.lua` (x4→x64 sweep)
- `bench_split_acc_cmdbar.lua` (split4 winner)
- `bench_packed_reads_cmdbar.lua` (packed-byte reads — disproved)
- `bench_row_pair_cmdbar.lua` (4row_x16 win, 2row_x32 loss)
- `bench_combo_cmdbar.lua` (compound test — didn't stack)

### Key findings worth remembering

- **Plateau at ~2.6 GFLOPS on q8 matvec** under Luau `--!native`. Three different optimization directions all converged within 5% of each other at the top.
- **`buffer.readi8` is free in native code.** Packing reads via `readu32` + bit-extract is strictly worse.
- **Luau `vector` type doesn't fuse into SIMD** when weights are int8. Handover's 2-3× SIMD projection was wrong for this regime.
- **Unroll factor matters more than expected.** x4 → x32 gave 2.04× just from straight-line code scheduling.
- **16 x-locals is the register-pressure sweet spot** on this VM.
- **Projections consistently undershot by ~10%** — isolated matvec measurements translate to slightly better end-to-end than naive math predicts, likely because non-matvec phases also benefit from warmer cache / better scheduling.

---

## Session 3: SmolLM2-135M end-to-end — real coherent English at 9 tok/s

Ported the runtime to `HuggingFaceTB/SmolLM2-135M`. Same v5 kernels, new binary format version, tied embeddings, chunked HTTP fetch, and — the real novel work — a from-scratch GPT-2-style BPE tokenizer in pure Luau that matches HF output bit-exactly.

### Binary format v2

Added a u32 `flags` field after `dtype`. Bit 0 = tied embeddings (lm_head tensor is omitted in the blob, loader aliases `lm_head = embed`). Header grew from 52 → 56 bytes. Python exporter accepts `tied_embeddings=True` + `chunk_size=...` (splits the finished blob into N contiguous byte-range files). LuauSim handles both v1 and v2 transparently.

Files:
- `handover-files/export_and_verify.py` — v2 format + tied flag + 48 MB chunked writer. Round-trip verified on Tiny-LLM (f32 bit-exact, q8 argmax match) and on a synthetic tied toy model.
- `handover-files/tinyllm_numpy.py` — added `CFG_SMOLLM2` constant.

### SmolLM2 weights export

`handover-files/export_smollm2.py` downloads `model.safetensors` from HF, loads it with a manual bf16 decoder (`numpy reinterpret u16→u32<<16→f32`, no torch dependency), constructs the 30-layer weight dict with tied lm_head, exports Q8 in 48 MB chunks (three files: 48.00 / 48.00 / 39.44 MB, total 135.44 MB), runs a numpy reference forward on a fixed prompt, and verifies round-trip via LuauSim.

Round-trip quality: top-10 logit overlap 9/10, argmax match, against the numpy reference. Good enough that text generation is coherent.

Chunk size: 48 MB (decimal, not 48 MiB) leaves 2 MB headroom under the HttpService 50 MB cap.

### Luau side: config swap + chunked fetch + tied alias

`smollm2_demo_v1_cmdbar.lua` is a clone of `full_demo_v5_cmdbar.lua` with:
- v2 header parser (reads `flags` u32, `bit32.band(flags, 1)` for tied check).
- `fetch_chunked(urls)`: `HttpService:GetAsync` each URL, concat into one `buffer` via `buffer.writestring(blob, offset, body)`. Works at 48 MB scale; Lua strings are byte-safe (null bytes included).
- Tied-embed aliasing: `w.lm_head = w.embed; s.lm_head = s.embed` after parse.
- `runtime_max_ctx` param on `load_model` clamps header's `max_ctx`. Used at 1024 (47 MB KV cache across 30 layers) vs header's 8192 (378 MB).
- `task.wait()` every 4 tokens in the prefill/decode loops. Keeps us under the 20s timeout on long prefills.

**v5 kernels required zero changes.** All SmolLM2 matvec shapes have `cols % 64 == 0` (576 = 9×64, 1536 = 24×64) so the x64 unroll runs exact iterations with no tail.

### Stage-1 run (pre-tokenized prompt, first live boot)

Hardcoded Python-tokenized prompt IDs, skipped Luau tokenizer. First run produced real, coherent English text ("Once upon a time, there was a magical place called Techland..."). Measured 9.4 tok/s end-to-end, prefill 5.4 ms/tok, fetch 4.66 s for 135 MB, parse 37 ms. On projection exactly.

### Stage 2: BPE tokenizer in pure Luau

SmolLM2's tokenizer is HF `tokenizers` byte-level BPE, **not** SentencePiece Unigram (which Tiny-LLM's Luau encoder implemented). Different algorithm — not a re-export, a full rewrite.

**Tokenizer binary format (TLLB)**: magic 'TLLB' + u32 version + vocab_size + n_merges + bos_id + eos_id + reserved + 256 byte_encoder entries (u8 len + utf8) + vocab_size piece strings (u16 len + utf8) + n_merges ranked (lhs, rhs) pairs. Total ~1 MB.

**Luau BPE pipeline** (`smollm2_demo_v1_cmdbar.lua`, in-module):
1. `split_digits` — individual_digits=True: every ASCII digit becomes its own pre-token. Required for SmolLM2's `Digits` pre-tokenizer.
2. `bytelevel_regex` — ASCII-only implementation of the canonical GPT-2 pattern `'s|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+`. Python `re` alternation is **first-match-wins**, not longest-match — so `pretok_match_len` tries alternatives in order and returns the first hit's length. Trailing-ws-before-non-ws handled by the `\s+(?!\S)` branch emitting N-1 chars so the last space attaches to the next token.
3. `bpe` — byte-encode each byte via the 256-entry GPT-2 printable map, then iteratively merge adjacent symbols by lowest rank. Merge table keyed by `lhs .. "\0" .. rhs`; null byte is guaranteed safe because byte_encoder never emits byte 0x00 (smallest output is `!`, codepoints 0x20 and others map to U+0100..U+01FF).
4. Vocab lookup → token IDs.

**Decode**: concat pieces → byte-decode UTF-8 chars → raw bytes → Lua string.

**Verification**: `handover-files/verify_luau_bpe.py` is a Python reimplementation of the exact Luau algorithm operating on the same TLLB binary. 7/7 test prompts match HF output bit-exactly (including tricky cases: `"Hello, world! It's nice."` with contractions, `"I have 3 apples and 42 oranges."` with individual-digit splits).

In-Luau self-check runs the same 7 vectors at startup; aborts the demo if any mismatch.

### Stage-2 run (full pipeline, tokenizer live)

Fetch 4.08 s, tokenizer parse 26 ms, model parse 53 ms, self-check pass 7/7, prefill 103 ms/tok, decode **9.0 tok/s** with PROFILE_MODE on (~4% overhead from markers + os.clock accumulators). Real output:

> *"Once upon a time, there was a faraway land called Egypt. People in Egypt loved and respected these special people named Pharaohs. One day, a little boy named Timmy asked his family, 'Who is Pharaoh in Egypt?' His mother smiled and replied..."*

### Best measured end-to-end (Session 3 shipping state)

Reference numbers from the profiled 120-token run (`smollm2_demo_v1_cmdbar.lua`, `PROFILE_MODE=true`, prompt `"Once upon a time"`, seed random, temperature 0.8, top-K 40, RUNTIME_MAX_CTX 1024, server-side single actor):

| stage | time | rate |
|---|---|---|
| fetch 3 chunks (135.44 MB) | 3.29 s | 41 MB/s |
| tokenizer fetch (1.03 MB) | 0.09 s | — |
| parse model | 0.046 s | — |
| parse tokenizer | 0.025 s | — |
| tokenizer self-check (7 vectors) | instant | 7/7 OK |
| prefill (5 tokens) | 515.5 ms | 103 ms/tok |
| decode (120 tokens) | 13.26 s | **9.0 tok/s** |

Non-profile baseline (PROFILE_MODE off, same prompt, 60 tokens) measures **9.4 tok/s**. PROFILE_MODE overhead = ~4%.

Decode step (≈103 ms) phase breakdown (35 profiled steps):

| phase | ms/step | % step | MACs/layer × layers | effective GFLOPS |
|---|---|---|---|---|
| ffn          | 59.91 | 57.9% | 2.65M × 30 = 79.6M | 1.33 |
| lm_head      | 20.99 | 20.3% | 28.3M              | 1.35 |
| attn_proj    | 12.49 | 12.1% | 553K × 30 = 16.6M  | 1.33 |
| o_res        |  7.48 |  7.2% | 332K × 30 = 9.96M  | 1.33 |
| attn         |  2.45 |  2.4% | grows with pos     | — |
| rope         |  0.10 |  0.1% | — | — |
| embed        |  0.004 |  0.0% | — | — |
| **total**    | **103.44** | **100%** | — | — |

Representative generation (same seed, `PROFILE_MODE=true`, MAX_TOKENS=120):

> *"Once upon a time (well, before the pandemic), doctors used to take blood samples from kids to test for diseases like autism. But as time went by, more and more people started getting these tests. Some of them even started to complain, and eventually they were diagnosed with autism. But what does it mean to be diagnosed with autism? Well, it means that a person has a kind of brain problem that affects how they communicate with others..."*

### Profile measurements on v5 / SmolLM2

Per decode step (≈103 ms), summed across 35 profiled steps:

| phase | ms/step | % | MACs/layer × layers | effective GFLOPS |
|---|---|---|---|---|
| ffn | 59.9 | **57.9%** | 2.65M × 30 = 79.6M | 1.33 |
| lm_head | 21.0 | 20.3% | 28.3M | 1.35 |
| attn_proj | 12.5 | 12.1% | 553K × 30 = 16.6M | 1.33 |
| o_res | 7.5 | 7.2% | 332K × 30 = 9.96M | 1.33 |
| attn | 2.4 | 2.4% | grows w/ pos | — |
| rope | 0.1 | 0.1% | — | — |
| embed | 0.0 | 0.0% | — | — |

**Every matvec-heavy phase clocks ~1.33 G-MACs/s.** 100% compute-bound at the v5 kernel ceiling, uniform across phases. No hidden overhead, no mis-weighted bucket. Total MACs per token: ~134.5M → 134.5M ÷ 1.33G = 101 ms projected, measured 103 ms. The op-bound model from Session 1 (`tok/s ≈ 520M / param_count`) holds for SmolLM2 almost exactly: measured ≈ 520M ÷ 135M × adjustment ≈ 9.5 tok/s.

### Findings worth remembering

1. **Matvec plateau is a hard wall.** Session 2 established ~1.3 G-MACs/s with the v5 kernel on 32000×192. Session 3 confirms the same throughput on 1536×576, 576×1536, 49152×576, 576×576, 192×576. Shape-independent, truly the Luau native-codegen ceiling for q8 matvec on this VM.
2. **Every phase runs at the ceiling.** This means there's no single "slow phase" to unlock — further speed requires pushing the per-matvec rate itself or reducing total MACs.
3. **`buffer.writestring` with binary strings works at 48 MB scale.** Lua strings are byte-safe; null bytes pass through HttpService and into buffers without corruption.
4. **`safetensors.numpy` refuses bf16** (numpy has no native type). A 10-line manual decoder (`uint16 → uint32<<16 → f32`) does the job without torch.
5. **HF `tokenizers` BPE is not SentencePiece.** Two different algorithms. `tokenizer.model` → Unigram + scores; `tokenizer.json` → BPE + ranked merges. Don't try to reuse one encoder for the other.
6. **Python `re` alternation is first-match-wins.** Several BPE ports get this wrong by assuming longest-match; matters for the `\s+(?!\S)|\s+` tail of the GPT-2 regex.
7. **Null byte is a safe separator for byte-encoded BPE merge keys.** GPT-2's byte_encoder never emits byte 0x00 (byte 0 maps to U+0100 = UTF-8 0xC4 0x80; byte 32 maps to U+0120 'Ġ' = UTF-8 0xC4 0xA0; all outputs ≥ 0x21).

### Session-3 files

**Python**:
- `handover-files/export_and_verify.py` (v2 format + tied + chunking)
- `handover-files/tinyllm_numpy.py` (added CFG_SMOLLM2)
- `handover-files/export_smollm2.py` (download + convert + verify)
- `handover-files/export_tokenizer.py` (HF tokenizer.json → TLLB binary)
- `handover-files/tokenize_prompts.py` (stage-1 helper, obsolete after stage 2)
- `handover-files/decode_output.py` (stage-1 helper, obsolete after stage 2)
- `handover-files/verify_luau_bpe.py` (independent Python sim of the Luau BPE; 7/7 OK vs HF)

**Luau**:
- `smollm2_demo_v1_cmdbar.lua` — shipping SmolLM2 demo (953 lines). Includes: v2 header parser, chunked fetch, tied-embed loader, BPE tokenizer (load_tokenizer/encode/decode), v5 kernels unchanged, `step_profiled` with debug markers + `os.clock()` accumulators, self-check on startup.

**Hosted blobs** (at `https://raw.githubusercontent.com/RussellBustamante/tinyllm_real_q8_v1/main/`):
- `smollm2_135m_q8.chunk00.bin` (48.00 MB)
- `smollm2_135m_q8.chunk01.bin` (48.00 MB)
- `smollm2_135m_q8.chunk02.bin` (39.44 MB)
- `smollm2_tokenizer.bin` (1.03 MB)

### Session 3 shipping status

Project is at a shippable milestone. 135M-param Llama inference in pure Luau, end-to-end BPE, 9 tok/s streaming real coherent English — capability bar hit. All future optimization directions live in the "To explore" section below.

---

## To explore (Session 4+ candidates)

Comprehensive list of every non-dead-end optimization direction. Each entry: **expected win**, **effort**, **risk/unknown**, **validation step** (the cheap bench that would accept or kill it before investing in full integration). Organized by category and ranked within each.

Replace/move/delete entries from this list as Session 4 measures them. **Confirmed dead ends from sessions 1–3 are listed separately at the end — don't retry those.**

### Category A: Kernel-level fusions (low risk, quick to bench)

These aim to amortize function-call overhead and collapse redundant passes. All reuse the v5 matvec; no new kernel math. Each is a ~1-hour test.

| # | idea | expected win | effort | validation |
|---|---|---|---|---|
| A1 | **Fuse gate + up into one `matvec_q8_pair`** (stacked [3072, 576] weights, writes into two output buffers per row pass) | 2–4% end-to-end (saves ~30 function calls/step, may also warm `h`'s cache across both) | 1 h | time full decode 60 tok before vs after |
| A2 | **Fuse Q / K / V into one `matvec_q8_triple`** (stacked [960, 576]) | 1–3% end-to-end | 1 h | same |
| A3 | **`matvec_q8_residual` for Wo** (output projection accumulates directly into `x`, skipping the attn_h intermediate and residual loop) | 1–2% end-to-end | 0.5 h | same |
| A4 | **`matvec_q8_residual` for Wdown** (same idea for FFN down) | 1–2% end-to-end | 0.5 h | same |
| A5 | **`matvec_q8_silu_mul` for the silu(gate)·up step** — fold the element-wise silu-and-multiply into the `down` matvec's input-read so we don't write a scratch buffer | <1% (silu loop is ~1.5% of step already; mostly save the 1536-element write) | 1 h | same |
| A6 | **2-row × x16 matvec** (process two output rows per iteration sharing 16 x-locals). Session 2's bench_row_pair measured 4row_x16 at 1.09× and 2row_x32 at 0.91×; the 2row_x16 middle-point was never benched but is register-friendly. Worth one more sweep. | possibly 3–5% on isolated matvec | 1 h | `bench_row_pair_v2` on the 1536×576 shape |
| A7 | **Custom silu lookup table** (precomputed 16-bit quantized exp values) to replace `math.exp` in the 1536-element silu loop. | <1% (silu is 1.5% of step; halving it recovers 0.7%). | 1 h | silu-only micro-bench with 100k iterations |
| A8 | **Constant folding the RoPE cos/sin reads** into apply_rope inner loop (precomputed tables already exist; verify the reads aren't redundant). | <0.5% | 0.5 h | inspect v5 + profile delta |
| A9 | **Re-measure v5 baseline after every Luau runtime update** — free check that confirms nothing silently regressed. Automate as `bench_v5_regression.lua`. | 0 (defensive) | 0.5 h setup | run once per month |

Stacking A1+A2+A3+A4: plausibly 5–10% end-to-end. Low risk, small effort.

### Category B: Parallelism on Roblox actors (biggest potential, highest uncertainty)

Roblox has built-in parallel Luau via `Actor` instances and `SharedTable`. **Server has 2 workers, client has 8.** See user-facing explanation below the PROGRESS section for how this works in Studio.

The gating unknown for all B-items is SharedTable write/read latency for small (1–100 KB) payloads. Session 1's handover cited "0.003–0.01 ms" but unverified. Any B-item's ROI depends on this; bench B0 *first*.

| # | idea | expected win | effort | validation |
|---|---|---|---|---|
| B0 | **Baseline: measure SharedTable write/read cost** for 2 KB, 10 KB, 100 KB f32 payloads. Actor A writes, actor B reads, round-trip timed over 1000 iterations. | 0 (essential baseline) | 2 h | one-shot bench |
| B1 | **Tensor-parallel lm_head across 2 server actors** — split the 49152 rows in half, each actor does 24576 rows, merge via SharedTable write of result buffers. 1 sync point/step. | ~10% end-to-end (lm_head 20.3% → 10%) if B0 < 0.5 ms | 0.5 day | measure full decode after integration |
| B2 | **Tensor-parallel FFN matvecs** (gate/up/down split row-wise) across 2 server actors. 3 syncs × 30 layers = 90 syncs/step. | ~20–25% if per-sync cost < 0.05 ms; regression if > 0.5 ms | 1–2 days | end-to-end decode bench |
| B3 | **Tensor-parallel attn_proj** (Q/K/V split) across 2 server actors. Relatively small phase (12%) so reward limited. | ~5% if sync is cheap | 1 day | same |
| B4 | **Client-side 8-actor tensor-parallel** — server fetches + parses + RemoteEvent's weights to client, client runs inference with 8-way split on FFN+lm_head. Maximum theoretical win. | 2–3× if sync scales (8 syncs/matvec × 90 matvecs × ~0.01 ms = 7 ms overhead per step); potentially 27 tok/s. Could also be 0.8× if sync cost dominates. | 1 week | client-side prototype; measure |
| B5 | **Parallel heap top-K sampler** — split V=49152 across actors each computing local top-K, merge local heaps on main. | ~0.3% (sampler is already 0.4 ms) | 0.5 day | sampler-only bench |
| B6 | **Server-side layer-pipeline prefill** (worker A processes layer 0, hands off to worker B for layer 1 while A starts next token's layer 0). Only wins for prefill batching, not single-token decode. Useful if prefill is ever the bottleneck (long prompts). | 1.5–2× on prefill only | 2–3 days | prefill bench with 100-token prompt |

### Category C: Algorithmic / model-level (correctness trade)

These reduce total MACs by computing less, accepting a quality hit.

| # | idea | expected win | effort | validation |
|---|---|---|---|---|
| C1 | **Top-K vocab truncation via bigram LM**: precompute for each token-in-vocab the top-N likely follow-up tokens (n-gram table). At inference, only project lm_head for those N candidates + a safety tail. | ~12–18% end-to-end (lm_head 20.3% → ~3%) | 2 days | A/B story quality (human read) vs full-vocab baseline on 100 prompts |
| C2 | **Trigram / 3-gram LM** for the same truncation — more accurate predictions, larger table. | same, slightly better quality | 2 days | same |
| C3 | **Speculative decoding** (needs a tiny trained draft model + full-model validator). 1–2M param draft would run at ~100 tok/s in Luau; validator accepts batches of 4–8. | 2–4× end-to-end **if** the draft's acceptance rate is > 50% | 2–3 weeks (training + infra) | cross-model acceptance rate measured on 500 generations |
| C4 | **Draft from n-gram LM** (skip training a neural draft): use a static n-gram table as the draft. Lower acceptance rate but zero training cost. | 1.3–1.8× plausibly | 1 week | acceptance rate measurement |
| C5 | **Adaptive early-exit** — if the intermediate hidden state matches a "confident" cluster, skip remaining layers. Requires retraining with aux losses per layer. | 1.3–1.8× on easy tokens | 3–6 weeks (retraining) | calibration on held-out prompts |
| C6 | **Layer pruning** — drop every 3rd layer post-hoc. Requires fine-tune to recover quality. | ~33% fewer MACs → ~1.3× end-to-end | 1–2 weeks | perplexity vs baseline on held-out set |

### Category D: Model compression (most uncertain in Luau)

| # | idea | expected win | effort | validation |
|---|---|---|---|---|
| D1 | **Isolated Q4 matvec bench on 1536×576 shape**: first prove Q4 can beat Q8 in Luau at all. Hand-pack 2 weights per byte, unpack with `bit32.band`+shift, compare ms against v5 Q8. | gate-check for all Q4 work | 1 day | if Q4 > Q8 by >10%, pursue; otherwise abandon |
| D2 | **Pre-unpack Q4 row to f32 scratch** (not per-element) — Q4 row unpack once, matvec as f32×f32. Decouples unpack from hot loop. | uncertain (saves memory bandwidth which doesn't matter in Luau; adds scratch writes) | 1 day | isolated matvec bench |
| D3 | **Q8 KV cache** — K and V stored as int8 with per-row scales. Halves KV memory (47 MB → 24 MB at max_ctx 1024). Halves attn compute (attn is 2.4%). | ~1.2% speed + 2× memory headroom | 1 day | same output quality on N generations |
| D4 | **Group-quantized Q4 (AWQ-style)** — per-column groups of 64 with group scales. Better precision than naive Q4. Same performance concerns as D1. | gated by D1 result | 2 days | post-D1 |
| D5 | **Bitnet W1A8 / W2A8** — exists only for 1.7B+ models, too big for us. Keep watching for smaller trained checkpoints. | conditional | 0 (waiting) | monitor releases |
| D6 | **Factorized lm_head** (SVD-init + fine-tune to rank r=128): `logits = (h·U)·V`, reduces lm_head MACs 4.4× (28.3M → 6.4M). Can combine with factorized tied embed. | ~15% end-to-end | 1–2 weeks (fine-tune) | held-out perplexity + top-K overlap vs unfactored |
| D7 | **Factorized FFN down only** (retain full up/gate) — similar SVD+fine-tune, targeting the largest of the 3 FFN matvecs. | ~5–8% end-to-end | 1–2 weeks | same |

### Category E: Alternative architectures (speculative, substantial work)

Mostly not-better-here, listed to close the loop on "what else could we run".

| # | idea | expected tok/s (at 135M) | why consider | why not |
|---|---|---|---|---|
| E1 | **SmolLM2-135M-Instruct** | same 9 tok/s | *Quality*, not speed — prompt following, ChatML template. Same kernels. | Already works as a ~10-line Python export change. Worth pursuing for UX. |
| E2 | **SmolLM2-360M** (same arch, larger) | ~3.5 tok/s | better quality | too slow for interactive |
| E3 | **RWKV-4 World 169M** | ~8 tok/s | RNN, no KV cache → constant memory | new kernels needed (time-mix, channel-mix); quality lower than SmolLM2 |
| E4 | **Mamba-130M** (SSM) | ~7 tok/s | linear in sequence length | selective-scan kernel is op-heavy, weakens at short context |
| E5 | **Small MoE** (hypothetical) | — | active params < total | no trained <500M MoE available |
| E6 | **Deep-narrow Llama** (e.g. 60L × 384H, same 135M total) | ~9 tok/s | could match our matvec ceiling differently | smaller hidden → less unroll benefit; probably neutral |
| E7 | **Custom-trained deeper-narrower model** (20L × 768H or similar) | uncertain | could be tuned for MAC count | requires full training run |
| E8 | **Same model with sliding-window attn** | same | helps at long context only | attn is 2.4%, irrelevant at ctx=1024 |

### Category F: Hosting / loading / context

Not speed wins on the hot path but enable bigger use cases.

| # | idea | effect | effort |
|---|---|---|---|
| F1 | **Bump RUNTIME_MAX_CTX to 2048 / 4096 / 8192** — 94 / 189 / 378 MB KV cache. Enables longer generations / conversations. | memory / ctx trade | 5 min |
| F2 | **KV cache LRU eviction** for very long dialogues — recycle oldest tokens past a sliding window. | memory cap for arbitrary-length dialogues | 1 day |
| F3 | **zstd-compressed blob hosting** — maybe 2× smaller wire payload, requires a Luau zstd decompressor at load. | faster cold-load only | 2 days |
| F4 | **Prefetch next layer's weights while current layer is computing** (tensor-parallel style, same actor). Useless if weights already in RAM — they are. | 0 | — |
| F5 | **Checkpointing KV cache to disk** for sessions > process lifetime (via DataStore). | session persistence | 2 days |

### Category G: UX / application layer

| # | idea | effect | effort |
|---|---|---|---|
| G1 | **Streaming token output to player UI** (RemoteEvent each new token, client displays as it arrives) | feels faster at same tok/s | 1 day |
| G2 | **Chat loop** with ChatML formatting (+ E1) | real chatbot | 3 days |
| G3 | **Request queue** for multi-user scenarios (single LLM, multiple requesters) | multi-user viable | 1 week |
| G4 | **Per-player personality prompts** cached as pre-computed prefill state | near-zero-latency persona switching | 3 days |

### Confirmed dead ends (do not retry without Luau runtime changes)

From sessions 1–3, measured to regress or plateau:

- **`vector.create(readi8, readi8, readi8)` SIMD** — 2–3× slower than scalar. Luau native codegen doesn't fuse int8→vector into SIMD instructions.
- **Packed reads via `readu32` + bit-extract + sign-extend** — **10× slower** than scalar `readi8`. Bit ops dominate; `readi8` is effectively free under native codegen.
- **Unroll depth > x64** — diminishing returns (x32→x64 was only +8%), and exceeds the register budget past x64.
- **Tables `{number}` for activation `x` instead of `buffer`** — measurably slower than buffer reads.
- **2-row × x32 matvec tiling** — 0.91× (register spill from 32 x-locals + 2 accumulators).
- **Compound row-pair × split-accumulator (4row_x16 × split4)** — plateaued within 5% of any single optimization; register pressure cancels gains.
- **Native-code bailout from type ambiguity** — easy to regress into this silently; always verify with `debug.dumpcodesize()`.

### Compound opportunity sketches (ambitious combinations)

- **Conservative (1–2 days work, ~15–20% end-to-end)**: A1+A2+A3+A4 kernel fusions, then B0 bench, then B1 if B0 < 0.5 ms.
- **Aggressive (1–2 weeks, ~40–60% end-to-end)**: all of Category A + B2 + C1 top-K draft. Plausible landing: **13–15 tok/s**.
- **Research-scale (1–2 months, up to 3× end-to-end)**: D6 factorized lm_head fine-tune + B4 client 8-actor + C3 speculative decoding. Landing: **~27 tok/s**. Ambitious, not obvious it all compounds.

---

## Session 4: A1–A4 kernel fusions — 9.7 → 10.5 tok/s (+8%)

Starting state: `smollm2_demo_v1_cmdbar.lua` at 9.0–9.4 tok/s (v5 kernel, 1.33 G-MACs/s ceiling).

### What was done

Implemented and measured all four A-category kernel fusions in one bench pass (20 baseline steps vs 20 v2 steps on the same loaded model, 3-step warmup each):

| kernel | description | used for |
|---|---|---|
| `matvec_q8_pair` | loads each x[k] once, accumulates into two W matrices per row pass | A1: gate+up (rows=1536, cols=576); A2: K+V (rows=192, cols=576) |
| `matvec_q8_add` | same as matvec_q8 but final write is `out[r] += sum*scale` | A3: Wo adds directly into x (no attn_h buffer); A4: Wdown adds directly into x (no ffn_out buffer) |

Note: Q is kept separate from K+V (A2 does pair not triple) because Q has rows=H=576 while K+V have rows=KV=192 — they can't be paired into a single call.

### Benchmark result

```
baseline 103.63 ms/step (9.7 tok/s) | v2 95.33 ms/step (10.5 tok/s) | delta 8.0%
```

Within the projected 5–10% range from PROGRESS.md. No errors, coherent output confirmed.

### Shipping artifact

`smollm2_demo_v2_cmdbar.lua` — integrates all A1–A4 fusions into the full demo. Hosted at GitHub (same repo). Two intermediate buffers removed from `load_model`: `attn_h` (replaced by fused Wo residual) and `ffn_out` (replaced by fused Wdown residual).

### Phase breakdown (v2, updated)

All four fused kernels still run at the ~1.33 G-MACs/s native codegen ceiling — the fusion wins come from reducing function-call overhead and loop overhead (shared x-reads in pair, skipped residual loop in add), not from changing the compute-bound rate itself.

Expected phase redistribution vs v1:
- `attn_proj` reduced (K+V fusion saves one matvec call overhead)
- `o_res` reduced (Wo+residual fused, no separate H-length loop)
- `ffn` reduced (gate+up fusion + Wdown+residual fused)

### Dead ends confirmed this session

- **A2 as triple (Q+K+V)**: not possible — Q has rows=576, K/V have rows=192. Different shapes prevent fusing all three into one pass.

### Next-action summary (outgoing)

1. **B0 SharedTable baseline bench.** Gates entire parallelism tier. Must use real Actor+Script hierarchy with hard timeouts — never bare `task.desynchronize()`. (Safety: see feedback_mcp_safety.md)
2. **B1 tensor-parallel lm_head** if B0 < 0.5 ms. lm_head is still 20% of step.
3. **A5–A8** (silu fusion, 2row×x16 sweep, lookup-table silu, RoPE constant-folding) — <2% each, worth benching if time allows.
4. **E1 SmolLM2-Instruct + ChatML** — pure quality win, ~10-line Python change.

---

## Session 3 next-action summary (outgoing)

Concrete suggestions for whoever picks up:

1. **E1 — Swap to SmolLM2-Instruct + ChatML.** Pure quality win, ~10-line change. Start here.
2. **A1+A2+A3+A4 kernel fusions.** Afternoon of easy wins, 5–10% compound.
3. **B0 SharedTable baseline bench.** Gates the entire parallelism tier.
4. **B1 tensor-parallel lm_head** if B0 passes. First measurable parallel win.
5. After above: decide between Category C (algorithmic), D (quantization), or D6 (factorized retraining) based on appetite for complexity.
5. **Q4**: only if the unchanged-kernel ceiling actually matters for a specific use case.
