-- SmolLM2-135M end-to-end demo v2 (A1-A4 kernel fusions: +8% vs v1).
-- Kernels: v6 set — matvec_q8_pair (shared x-reads for gate+up, K+V),
--          matvec_q8_add (fused residual for Wo, Wdown).
-- Benchmark: 103.6 ms/step (9.7 tok/s) baseline → 95.3 ms/step (10.5 tok/s) v2.
--
-- Stage-2 features unchanged from v1:
--   * chunked Q8 weights fetch (3 × 48 MB, binary format v2 with flags field),
--   * tied-embed aliasing (lm_head = embed),
--   * HF GPT-2 BPE tokenizer in pure Luau (Digits + ByteLevel pre-tokenizer,
--     merge-rank BPE, byte-level decode), ~1 MB tokenizer blob,
--   * self-check against Python-produced reference token IDs,
--   * optional PROFILE_MODE runs an instrumented step with debug.profilebegin/end
--     for each phase (embed/attn_proj/rope/attn/o_res/ffn/lm_head/sampler).
--
-- Memory: k_cache + v_cache grow as max_ctx × kv_dim × 4 × layers × 2.
-- At max_ctx=1024: ~47 MB. At 8192: ~378 MB.

local BLOB_CHUNK_URLS = {
	"https://raw.githubusercontent.com/RussellBustamante/tinyllm_real_q8_v1/main/smollm2_135m_q8.chunk00.bin",
	"https://raw.githubusercontent.com/RussellBustamante/tinyllm_real_q8_v1/main/smollm2_135m_q8.chunk01.bin",
	"https://raw.githubusercontent.com/RussellBustamante/tinyllm_real_q8_v1/main/smollm2_135m_q8.chunk02.bin",
}
local TOKENIZER_URL =
	"https://raw.githubusercontent.com/RussellBustamante/tinyllm_real_q8_v1/main/smollm2_tokenizer.bin"

local PROMPT = "Once upon a time"
local MAX_TOKENS = 120
local TEMPERATURE = 0.8
local TOP_K = 40
local SEED = nil
local RUNTIME_MAX_CTX = 1024
local PROFILE_MODE = true -- set true to dump per-phase timings from Studio's microprofiler
local PROFILE_STEPS = 30 -- number of decode steps to profile when PROFILE_MODE is on
local PREPEND_BOS = true -- SmolLM2 was pretrained without explicit BOS; leaving this true is mostly benign

-- Tokenizer self-check vectors (from handover-files/export_tokenizer.py).
-- encode() must match these IDs bit-exactly or we abort before running the model.
local TEST_VECTORS = {
	{ text = "Once upon a time", ids = { 6403, 1980, 253, 655 } },
	{ text = "The quick brown fox", ids = { 504, 2365, 6354, 16438 } },
	{ text = "In the beginning", ids = { 788, 260, 3616 } },
	{ text = "The capital of France is", ids = { 504, 3575, 282, 4649, 314 } },
	{ text = "A recipe for chocolate cake:", ids = { 49, 11594, 327, 9678, 16253, 42 } },
	{ text = "I have 3 apples and 42 oranges.", ids = { 57, 457, 216, 35, 13855, 284, 216, 36, 34, 27068, 30 } },
	{ text = "Hello, world! It's nice.", ids = { 19556, 28, 905, 17, 657, 506, 9239, 30 } },
}

local ModuleScript = Instance.new("ModuleScript")
ModuleScript.Name = "SmolLM2"
ModuleScript.Source = [==[
--!strict
--!native
--!optimize 2

local M = {}
local F4 = 4
local DT_Q8 = 1

type Config = {
	dtype: number, hidden: number, layers: number, heads: number, kv_heads: number,
	head_dim: number, inter: number, vocab: number, max_ctx: number,
	rope_theta: number, rms_eps: number, kv_dim: number,
}

local FLAG_TIED_EMBED = 1

local function parse_model(blob: buffer)
	assert(buffer.readstring(blob, 0, 4) == "TLLM", "bad model magic")
	local version = buffer.readu32(blob, 4)
	assert(version == 1 or version == 2, "bad model version: "..tostring(version))
	local off = 8
	local dtype = buffer.readu32(blob, off); off += 4
	local flags = 0
	if version >= 2 then flags = buffer.readu32(blob, off); off += 4 end
	local tied = bit32.band(flags, FLAG_TIED_EMBED) ~= 0
	local cfg: Config = {
		dtype      = dtype,
		hidden     = buffer.readu32(blob, off + 0),
		layers     = buffer.readu32(blob, off + 4),
		heads      = buffer.readu32(blob, off + 8),
		kv_heads   = buffer.readu32(blob, off + 12),
		head_dim   = buffer.readu32(blob, off + 16),
		inter      = buffer.readu32(blob, off + 20),
		vocab      = buffer.readu32(blob, off + 24),
		max_ctx    = buffer.readu32(blob, off + 28),
		rope_theta = buffer.readf32(blob, off + 32),
		rms_eps    = buffer.readf32(blob, off + 36),
		kv_dim     = 0,
	}
	cfg.kv_dim = cfg.kv_heads * cfg.head_dim
	off += 40
	local w, s = {}, {}
	local function rf(n) local b = buffer.create(n*4); buffer.copy(b,0,blob,off,n*4); off+=n*4; return b end
	local function ri(n) local b = buffer.create(n);   buffer.copy(b,0,blob,off,n);   off+=n;   return b end
	local function mat(name, rows, cols)
		if cfg.dtype == DT_Q8 then w[name] = ri(rows*cols); s[name] = rf(rows)
		else w[name] = rf(rows*cols) end
	end
	local function vec(name) w[name] = rf(cfg.hidden) end
	mat("embed", cfg.vocab, cfg.hidden)
	for i = 0, cfg.layers - 1 do
		vec("L"..i..".attn_norm")
		mat("L"..i..".q", cfg.hidden, cfg.hidden)
		mat("L"..i..".k", cfg.kv_dim, cfg.hidden)
		mat("L"..i..".v", cfg.kv_dim, cfg.hidden)
		mat("L"..i..".o", cfg.hidden, cfg.hidden)
		vec("L"..i..".ffn_norm")
		mat("L"..i..".gate", cfg.inter, cfg.hidden)
		mat("L"..i..".up",   cfg.inter, cfg.hidden)
		mat("L"..i..".down", cfg.hidden, cfg.inter)
	end
	vec("final_norm")
	if tied then
		w["lm_head"] = w["embed"]
		s["lm_head"] = s["embed"]
	else
		mat("lm_head", cfg.vocab, cfg.hidden)
	end
	return cfg, w, s, tied
end

-- v5 kernel: x64-unrolled matvec_q8. Straight-line 64-term FMA per iter.
local function matvec_q8(W, sc, x, y, rows, cols)
	local ri, rf, wf = buffer.readi8, buffer.readf32, buffer.writef32
	for r = 0, rows - 1 do
		local ro = r * cols
		local s1, s2, s3, s4 = 0.0, 0.0, 0.0, 0.0
		local k = 0
		while k + 64 <= cols do
			s1 += ri(W, ro + k)      * rf(x, k * F4)
			    + ri(W, ro + k + 1)  * rf(x, (k+1) * F4)
			    + ri(W, ro + k + 2)  * rf(x, (k+2) * F4)
			    + ri(W, ro + k + 3)  * rf(x, (k+3) * F4)
			    + ri(W, ro + k + 4)  * rf(x, (k+4) * F4)
			    + ri(W, ro + k + 5)  * rf(x, (k+5) * F4)
			    + ri(W, ro + k + 6)  * rf(x, (k+6) * F4)
			    + ri(W, ro + k + 7)  * rf(x, (k+7) * F4)
			    + ri(W, ro + k + 8)  * rf(x, (k+8) * F4)
			    + ri(W, ro + k + 9)  * rf(x, (k+9) * F4)
			    + ri(W, ro + k + 10) * rf(x, (k+10) * F4)
			    + ri(W, ro + k + 11) * rf(x, (k+11) * F4)
			    + ri(W, ro + k + 12) * rf(x, (k+12) * F4)
			    + ri(W, ro + k + 13) * rf(x, (k+13) * F4)
			    + ri(W, ro + k + 14) * rf(x, (k+14) * F4)
			    + ri(W, ro + k + 15) * rf(x, (k+15) * F4)
			s2 += ri(W, ro + k + 16) * rf(x, (k+16) * F4)
			    + ri(W, ro + k + 17) * rf(x, (k+17) * F4)
			    + ri(W, ro + k + 18) * rf(x, (k+18) * F4)
			    + ri(W, ro + k + 19) * rf(x, (k+19) * F4)
			    + ri(W, ro + k + 20) * rf(x, (k+20) * F4)
			    + ri(W, ro + k + 21) * rf(x, (k+21) * F4)
			    + ri(W, ro + k + 22) * rf(x, (k+22) * F4)
			    + ri(W, ro + k + 23) * rf(x, (k+23) * F4)
			    + ri(W, ro + k + 24) * rf(x, (k+24) * F4)
			    + ri(W, ro + k + 25) * rf(x, (k+25) * F4)
			    + ri(W, ro + k + 26) * rf(x, (k+26) * F4)
			    + ri(W, ro + k + 27) * rf(x, (k+27) * F4)
			    + ri(W, ro + k + 28) * rf(x, (k+28) * F4)
			    + ri(W, ro + k + 29) * rf(x, (k+29) * F4)
			    + ri(W, ro + k + 30) * rf(x, (k+30) * F4)
			    + ri(W, ro + k + 31) * rf(x, (k+31) * F4)
			s3 += ri(W, ro + k + 32) * rf(x, (k+32) * F4)
			    + ri(W, ro + k + 33) * rf(x, (k+33) * F4)
			    + ri(W, ro + k + 34) * rf(x, (k+34) * F4)
			    + ri(W, ro + k + 35) * rf(x, (k+35) * F4)
			    + ri(W, ro + k + 36) * rf(x, (k+36) * F4)
			    + ri(W, ro + k + 37) * rf(x, (k+37) * F4)
			    + ri(W, ro + k + 38) * rf(x, (k+38) * F4)
			    + ri(W, ro + k + 39) * rf(x, (k+39) * F4)
			    + ri(W, ro + k + 40) * rf(x, (k+40) * F4)
			    + ri(W, ro + k + 41) * rf(x, (k+41) * F4)
			    + ri(W, ro + k + 42) * rf(x, (k+42) * F4)
			    + ri(W, ro + k + 43) * rf(x, (k+43) * F4)
			    + ri(W, ro + k + 44) * rf(x, (k+44) * F4)
			    + ri(W, ro + k + 45) * rf(x, (k+45) * F4)
			    + ri(W, ro + k + 46) * rf(x, (k+46) * F4)
			    + ri(W, ro + k + 47) * rf(x, (k+47) * F4)
			s4 += ri(W, ro + k + 48) * rf(x, (k+48) * F4)
			    + ri(W, ro + k + 49) * rf(x, (k+49) * F4)
			    + ri(W, ro + k + 50) * rf(x, (k+50) * F4)
			    + ri(W, ro + k + 51) * rf(x, (k+51) * F4)
			    + ri(W, ro + k + 52) * rf(x, (k+52) * F4)
			    + ri(W, ro + k + 53) * rf(x, (k+53) * F4)
			    + ri(W, ro + k + 54) * rf(x, (k+54) * F4)
			    + ri(W, ro + k + 55) * rf(x, (k+55) * F4)
			    + ri(W, ro + k + 56) * rf(x, (k+56) * F4)
			    + ri(W, ro + k + 57) * rf(x, (k+57) * F4)
			    + ri(W, ro + k + 58) * rf(x, (k+58) * F4)
			    + ri(W, ro + k + 59) * rf(x, (k+59) * F4)
			    + ri(W, ro + k + 60) * rf(x, (k+60) * F4)
			    + ri(W, ro + k + 61) * rf(x, (k+61) * F4)
			    + ri(W, ro + k + 62) * rf(x, (k+62) * F4)
			    + ri(W, ro + k + 63) * rf(x, (k+63) * F4)
			k += 64
		end
		local sum = s1 + s2 + s3 + s4
		while k < cols do sum += ri(W, ro + k) * rf(x, k * F4); k += 1 end
		wf(y, r * F4, sum * rf(sc, r * F4))
	end
end

-- A1+A2: paired matvec — loads each x[k] once, accumulates into both W1 and W2.
-- Used for gate+up (rows=1536, cols=576) and K+V (rows=192, cols=576).
local function matvec_q8_pair(W1, sc1, W2, sc2, x, y1, y2, rows, cols)
	local ri, rf, wf = buffer.readi8, buffer.readf32, buffer.writef32
	for r = 0, rows - 1 do
		local ro = r * cols
		local a1, a2, a3, a4 = 0.0, 0.0, 0.0, 0.0
		local b1, b2, b3, b4 = 0.0, 0.0, 0.0, 0.0
		local k = 0
		while k + 64 <= cols do
			local v0  = rf(x, k*F4);       local v1  = rf(x, (k+1)*F4)
			local v2  = rf(x, (k+2)*F4);   local v3  = rf(x, (k+3)*F4)
			local v4  = rf(x, (k+4)*F4);   local v5  = rf(x, (k+5)*F4)
			local v6  = rf(x, (k+6)*F4);   local v7  = rf(x, (k+7)*F4)
			local v8  = rf(x, (k+8)*F4);   local v9  = rf(x, (k+9)*F4)
			local v10 = rf(x, (k+10)*F4);  local v11 = rf(x, (k+11)*F4)
			local v12 = rf(x, (k+12)*F4);  local v13 = rf(x, (k+13)*F4)
			local v14 = rf(x, (k+14)*F4);  local v15 = rf(x, (k+15)*F4)
			a1 += ri(W1,ro+k)*v0 + ri(W1,ro+k+1)*v1 + ri(W1,ro+k+2)*v2 + ri(W1,ro+k+3)*v3
			    + ri(W1,ro+k+4)*v4 + ri(W1,ro+k+5)*v5 + ri(W1,ro+k+6)*v6 + ri(W1,ro+k+7)*v7
			    + ri(W1,ro+k+8)*v8 + ri(W1,ro+k+9)*v9 + ri(W1,ro+k+10)*v10 + ri(W1,ro+k+11)*v11
			    + ri(W1,ro+k+12)*v12 + ri(W1,ro+k+13)*v13 + ri(W1,ro+k+14)*v14 + ri(W1,ro+k+15)*v15
			b1 += ri(W2,ro+k)*v0 + ri(W2,ro+k+1)*v1 + ri(W2,ro+k+2)*v2 + ri(W2,ro+k+3)*v3
			    + ri(W2,ro+k+4)*v4 + ri(W2,ro+k+5)*v5 + ri(W2,ro+k+6)*v6 + ri(W2,ro+k+7)*v7
			    + ri(W2,ro+k+8)*v8 + ri(W2,ro+k+9)*v9 + ri(W2,ro+k+10)*v10 + ri(W2,ro+k+11)*v11
			    + ri(W2,ro+k+12)*v12 + ri(W2,ro+k+13)*v13 + ri(W2,ro+k+14)*v14 + ri(W2,ro+k+15)*v15
			v0  = rf(x,(k+16)*F4);  v1  = rf(x,(k+17)*F4); v2  = rf(x,(k+18)*F4); v3  = rf(x,(k+19)*F4)
			v4  = rf(x,(k+20)*F4);  v5  = rf(x,(k+21)*F4); v6  = rf(x,(k+22)*F4); v7  = rf(x,(k+23)*F4)
			v8  = rf(x,(k+24)*F4);  v9  = rf(x,(k+25)*F4); v10 = rf(x,(k+26)*F4); v11 = rf(x,(k+27)*F4)
			v12 = rf(x,(k+28)*F4);  v13 = rf(x,(k+29)*F4); v14 = rf(x,(k+30)*F4); v15 = rf(x,(k+31)*F4)
			a2 += ri(W1,ro+k+16)*v0 + ri(W1,ro+k+17)*v1 + ri(W1,ro+k+18)*v2 + ri(W1,ro+k+19)*v3
			    + ri(W1,ro+k+20)*v4 + ri(W1,ro+k+21)*v5 + ri(W1,ro+k+22)*v6 + ri(W1,ro+k+23)*v7
			    + ri(W1,ro+k+24)*v8 + ri(W1,ro+k+25)*v9 + ri(W1,ro+k+26)*v10 + ri(W1,ro+k+27)*v11
			    + ri(W1,ro+k+28)*v12 + ri(W1,ro+k+29)*v13 + ri(W1,ro+k+30)*v14 + ri(W1,ro+k+31)*v15
			b2 += ri(W2,ro+k+16)*v0 + ri(W2,ro+k+17)*v1 + ri(W2,ro+k+18)*v2 + ri(W2,ro+k+19)*v3
			    + ri(W2,ro+k+20)*v4 + ri(W2,ro+k+21)*v5 + ri(W2,ro+k+22)*v6 + ri(W2,ro+k+23)*v7
			    + ri(W2,ro+k+24)*v8 + ri(W2,ro+k+25)*v9 + ri(W2,ro+k+26)*v10 + ri(W2,ro+k+27)*v11
			    + ri(W2,ro+k+28)*v12 + ri(W2,ro+k+29)*v13 + ri(W2,ro+k+30)*v14 + ri(W2,ro+k+31)*v15
			v0  = rf(x,(k+32)*F4);  v1  = rf(x,(k+33)*F4); v2  = rf(x,(k+34)*F4); v3  = rf(x,(k+35)*F4)
			v4  = rf(x,(k+36)*F4);  v5  = rf(x,(k+37)*F4); v6  = rf(x,(k+38)*F4); v7  = rf(x,(k+39)*F4)
			v8  = rf(x,(k+40)*F4);  v9  = rf(x,(k+41)*F4); v10 = rf(x,(k+42)*F4); v11 = rf(x,(k+43)*F4)
			v12 = rf(x,(k+44)*F4);  v13 = rf(x,(k+45)*F4); v14 = rf(x,(k+46)*F4); v15 = rf(x,(k+47)*F4)
			a3 += ri(W1,ro+k+32)*v0 + ri(W1,ro+k+33)*v1 + ri(W1,ro+k+34)*v2 + ri(W1,ro+k+35)*v3
			    + ri(W1,ro+k+36)*v4 + ri(W1,ro+k+37)*v5 + ri(W1,ro+k+38)*v6 + ri(W1,ro+k+39)*v7
			    + ri(W1,ro+k+40)*v8 + ri(W1,ro+k+41)*v9 + ri(W1,ro+k+42)*v10 + ri(W1,ro+k+43)*v11
			    + ri(W1,ro+k+44)*v12 + ri(W1,ro+k+45)*v13 + ri(W1,ro+k+46)*v14 + ri(W1,ro+k+47)*v15
			b3 += ri(W2,ro+k+32)*v0 + ri(W2,ro+k+33)*v1 + ri(W2,ro+k+34)*v2 + ri(W2,ro+k+35)*v3
			    + ri(W2,ro+k+36)*v4 + ri(W2,ro+k+37)*v5 + ri(W2,ro+k+38)*v6 + ri(W2,ro+k+39)*v7
			    + ri(W2,ro+k+40)*v8 + ri(W2,ro+k+41)*v9 + ri(W2,ro+k+42)*v10 + ri(W2,ro+k+43)*v11
			    + ri(W2,ro+k+44)*v12 + ri(W2,ro+k+45)*v13 + ri(W2,ro+k+46)*v14 + ri(W2,ro+k+47)*v15
			v0  = rf(x,(k+48)*F4);  v1  = rf(x,(k+49)*F4); v2  = rf(x,(k+50)*F4); v3  = rf(x,(k+51)*F4)
			v4  = rf(x,(k+52)*F4);  v5  = rf(x,(k+53)*F4); v6  = rf(x,(k+54)*F4); v7  = rf(x,(k+55)*F4)
			v8  = rf(x,(k+56)*F4);  v9  = rf(x,(k+57)*F4); v10 = rf(x,(k+58)*F4); v11 = rf(x,(k+59)*F4)
			v12 = rf(x,(k+60)*F4);  v13 = rf(x,(k+61)*F4); v14 = rf(x,(k+62)*F4); v15 = rf(x,(k+63)*F4)
			a4 += ri(W1,ro+k+48)*v0 + ri(W1,ro+k+49)*v1 + ri(W1,ro+k+50)*v2 + ri(W1,ro+k+51)*v3
			    + ri(W1,ro+k+52)*v4 + ri(W1,ro+k+53)*v5 + ri(W1,ro+k+54)*v6 + ri(W1,ro+k+55)*v7
			    + ri(W1,ro+k+56)*v8 + ri(W1,ro+k+57)*v9 + ri(W1,ro+k+58)*v10 + ri(W1,ro+k+59)*v11
			    + ri(W1,ro+k+60)*v12 + ri(W1,ro+k+61)*v13 + ri(W1,ro+k+62)*v14 + ri(W1,ro+k+63)*v15
			b4 += ri(W2,ro+k+48)*v0 + ri(W2,ro+k+49)*v1 + ri(W2,ro+k+50)*v2 + ri(W2,ro+k+51)*v3
			    + ri(W2,ro+k+52)*v4 + ri(W2,ro+k+53)*v5 + ri(W2,ro+k+54)*v6 + ri(W2,ro+k+55)*v7
			    + ri(W2,ro+k+56)*v8 + ri(W2,ro+k+57)*v9 + ri(W2,ro+k+58)*v10 + ri(W2,ro+k+59)*v11
			    + ri(W2,ro+k+60)*v12 + ri(W2,ro+k+61)*v13 + ri(W2,ro+k+62)*v14 + ri(W2,ro+k+63)*v15
			k += 64
		end
		local sum1 = a1 + a2 + a3 + a4
		local sum2 = b1 + b2 + b3 + b4
		while k < cols do
			local xk = rf(x, k*F4)
			sum1 += ri(W1, ro+k) * xk
			sum2 += ri(W2, ro+k) * xk
			k += 1
		end
		wf(y1, r*F4, sum1 * rf(sc1, r*F4))
		wf(y2, r*F4, sum2 * rf(sc2, r*F4))
	end
end

-- A3+A4: fused residual matvec — adds W*x*scale directly into out (no separate buffer).
-- Used for Wo (adds into x, rows=576, cols=576) and Wdown (adds into x, rows=576, cols=1536).
local function matvec_q8_add(W, sc, x, out, rows, cols)
	local ri, rf, wf = buffer.readi8, buffer.readf32, buffer.writef32
	for r = 0, rows - 1 do
		local ro = r * cols
		local s1, s2, s3, s4 = 0.0, 0.0, 0.0, 0.0
		local k = 0
		while k + 64 <= cols do
			s1 += ri(W, ro + k)      * rf(x, k * F4)
			    + ri(W, ro + k + 1)  * rf(x, (k+1) * F4)
			    + ri(W, ro + k + 2)  * rf(x, (k+2) * F4)
			    + ri(W, ro + k + 3)  * rf(x, (k+3) * F4)
			    + ri(W, ro + k + 4)  * rf(x, (k+4) * F4)
			    + ri(W, ro + k + 5)  * rf(x, (k+5) * F4)
			    + ri(W, ro + k + 6)  * rf(x, (k+6) * F4)
			    + ri(W, ro + k + 7)  * rf(x, (k+7) * F4)
			    + ri(W, ro + k + 8)  * rf(x, (k+8) * F4)
			    + ri(W, ro + k + 9)  * rf(x, (k+9) * F4)
			    + ri(W, ro + k + 10) * rf(x, (k+10) * F4)
			    + ri(W, ro + k + 11) * rf(x, (k+11) * F4)
			    + ri(W, ro + k + 12) * rf(x, (k+12) * F4)
			    + ri(W, ro + k + 13) * rf(x, (k+13) * F4)
			    + ri(W, ro + k + 14) * rf(x, (k+14) * F4)
			    + ri(W, ro + k + 15) * rf(x, (k+15) * F4)
			s2 += ri(W, ro + k + 16) * rf(x, (k+16) * F4)
			    + ri(W, ro + k + 17) * rf(x, (k+17) * F4)
			    + ri(W, ro + k + 18) * rf(x, (k+18) * F4)
			    + ri(W, ro + k + 19) * rf(x, (k+19) * F4)
			    + ri(W, ro + k + 20) * rf(x, (k+20) * F4)
			    + ri(W, ro + k + 21) * rf(x, (k+21) * F4)
			    + ri(W, ro + k + 22) * rf(x, (k+22) * F4)
			    + ri(W, ro + k + 23) * rf(x, (k+23) * F4)
			    + ri(W, ro + k + 24) * rf(x, (k+24) * F4)
			    + ri(W, ro + k + 25) * rf(x, (k+25) * F4)
			    + ri(W, ro + k + 26) * rf(x, (k+26) * F4)
			    + ri(W, ro + k + 27) * rf(x, (k+27) * F4)
			    + ri(W, ro + k + 28) * rf(x, (k+28) * F4)
			    + ri(W, ro + k + 29) * rf(x, (k+29) * F4)
			    + ri(W, ro + k + 30) * rf(x, (k+30) * F4)
			    + ri(W, ro + k + 31) * rf(x, (k+31) * F4)
			s3 += ri(W, ro + k + 32) * rf(x, (k+32) * F4)
			    + ri(W, ro + k + 33) * rf(x, (k+33) * F4)
			    + ri(W, ro + k + 34) * rf(x, (k+34) * F4)
			    + ri(W, ro + k + 35) * rf(x, (k+35) * F4)
			    + ri(W, ro + k + 36) * rf(x, (k+36) * F4)
			    + ri(W, ro + k + 37) * rf(x, (k+37) * F4)
			    + ri(W, ro + k + 38) * rf(x, (k+38) * F4)
			    + ri(W, ro + k + 39) * rf(x, (k+39) * F4)
			    + ri(W, ro + k + 40) * rf(x, (k+40) * F4)
			    + ri(W, ro + k + 41) * rf(x, (k+41) * F4)
			    + ri(W, ro + k + 42) * rf(x, (k+42) * F4)
			    + ri(W, ro + k + 43) * rf(x, (k+43) * F4)
			    + ri(W, ro + k + 44) * rf(x, (k+44) * F4)
			    + ri(W, ro + k + 45) * rf(x, (k+45) * F4)
			    + ri(W, ro + k + 46) * rf(x, (k+46) * F4)
			    + ri(W, ro + k + 47) * rf(x, (k+47) * F4)
			s4 += ri(W, ro + k + 48) * rf(x, (k+48) * F4)
			    + ri(W, ro + k + 49) * rf(x, (k+49) * F4)
			    + ri(W, ro + k + 50) * rf(x, (k+50) * F4)
			    + ri(W, ro + k + 51) * rf(x, (k+51) * F4)
			    + ri(W, ro + k + 52) * rf(x, (k+52) * F4)
			    + ri(W, ro + k + 53) * rf(x, (k+53) * F4)
			    + ri(W, ro + k + 54) * rf(x, (k+54) * F4)
			    + ri(W, ro + k + 55) * rf(x, (k+55) * F4)
			    + ri(W, ro + k + 56) * rf(x, (k+56) * F4)
			    + ri(W, ro + k + 57) * rf(x, (k+57) * F4)
			    + ri(W, ro + k + 58) * rf(x, (k+58) * F4)
			    + ri(W, ro + k + 59) * rf(x, (k+59) * F4)
			    + ri(W, ro + k + 60) * rf(x, (k+60) * F4)
			    + ri(W, ro + k + 61) * rf(x, (k+61) * F4)
			    + ri(W, ro + k + 62) * rf(x, (k+62) * F4)
			    + ri(W, ro + k + 63) * rf(x, (k+63) * F4)
			k += 64
		end
		local sum = s1 + s2 + s3 + s4
		while k < cols do sum += ri(W, ro + k) * rf(x, k * F4); k += 1 end
		wf(out, r * F4, rf(out, r * F4) + sum * rf(sc, r * F4))
	end
end

local function rmsnorm(x, w, out, n, eps)
	local rf, wf = buffer.readf32, buffer.writef32
	local ss = 0.0
	for i = 0, n - 1 do local v = rf(x, i * F4); ss += v * v end
	local inv = 1.0 / math.sqrt(ss / n + eps)
	for i = 0, n - 1 do wf(out, i * F4, rf(x, i * F4) * inv * rf(w, i * F4)) end
end

local function apply_rope(x, cos_tbl, sin_tbl, pos, heads, hd)
	local rf, wf = buffer.readf32, buffer.writef32
	local half = hd // 2
	local row_off = pos * half * F4
	for h = 0, heads - 1 do
		local base = h * hd * F4
		for i = 0, half - 1 do
			local c = rf(cos_tbl, row_off + i * F4)
			local s = rf(sin_tbl, row_off + i * F4)
			local a = rf(x, base + i * F4); local b = rf(x, base + (i + half) * F4)
			wf(x, base + i * F4,          a * c - b * s)
			wf(x, base + (i + half) * F4, a * s + b * c)
		end
	end
end

local function build_rope_tables(max_ctx, hd, theta)
	local half = hd // 2
	local cos_tbl = buffer.create(max_ctx * half * F4)
	local sin_tbl = buffer.create(max_ctx * half * F4)
	for pos = 0, max_ctx - 1 do
		for i = 0, half - 1 do
			local freq = 1.0 / (theta ^ (2 * i / hd))
			local off = (pos * half + i) * F4
			buffer.writef32(cos_tbl, off, math.cos(pos * freq))
			buffer.writef32(sin_tbl, off, math.sin(pos * freq))
		end
	end
	return cos_tbl, sin_tbl
end

function M.load_model(blob: buffer, runtime_max_ctx: number?)
	local cfg, w, s, tied = parse_model(blob)
	-- Cap runtime context to limit KV cache memory (header may advertise huge ctx).
	if runtime_max_ctx and runtime_max_ctx > 0 and runtime_max_ctx < cfg.max_ctx then
		cfg.max_ctx = runtime_max_ctx
	end
	local H, KV, I, V = cfg.hidden, cfg.kv_dim, cfg.inter, cfg.vocab
	local st = {
		cfg = cfg, w = w, s = s, tied = tied, pos = 0,
		x = buffer.create(H*F4), h = buffer.create(H*F4), q = buffer.create(H*F4),
		k_tmp = buffer.create(KV*F4), v_tmp = buffer.create(KV*F4),
		attn_out = buffer.create(H*F4),
		gate = buffer.create(I*F4), up = buffer.create(I*F4),
		logits = buffer.create(V*F4),
		scores = buffer.create(cfg.max_ctx*F4),
		k_cache = table.create(cfg.layers), v_cache = table.create(cfg.layers),
	}
	for l = 0, cfg.layers - 1 do
		st.k_cache[l+1] = buffer.create(cfg.max_ctx*KV*F4)
		st.v_cache[l+1] = buffer.create(cfg.max_ctx*KV*F4)
	end
	st.cos_tbl, st.sin_tbl = build_rope_tables(cfg.max_ctx, cfg.head_dim, cfg.rope_theta)
	return st
end

function M.step(st, token_id: number): buffer
	local cfg = st.cfg; local w, s = st.w, st.s
	local H, HD, NH, KVH, KV, I, V = cfg.hidden, cfg.head_dim, cfg.heads, cfg.kv_heads, cfg.kv_dim, cfg.inter, cfg.vocab
	local eps = cfg.rms_eps
	local inv_sqrt_hd = 1.0 / math.sqrt(HD)
	local pos = st.pos
	local x, h, q = st.x, st.h, st.q
	local k_tmp, v_tmp = st.k_tmp, st.v_tmp
	local attn_out = st.attn_out
	local gate, up = st.gate, st.up
	local logits, scores = st.logits, st.scores

	local escale = buffer.readf32(s["embed"], token_id * F4)
	local eoff = token_id * H
	for i = 0, H - 1 do
		buffer.writef32(x, i * F4, buffer.readi8(w["embed"], eoff + i) * escale)
	end

	for layer = 0, cfg.layers - 1 do
		local Wq, Sq = w["L"..layer..".q"], s["L"..layer..".q"]
		local Wk, Sk = w["L"..layer..".k"], s["L"..layer..".k"]
		local Wv, Sv = w["L"..layer..".v"], s["L"..layer..".v"]
		local Wo, So = w["L"..layer..".o"], s["L"..layer..".o"]
		local Wg, Sg = w["L"..layer..".gate"], s["L"..layer..".gate"]
		local Wu, Su = w["L"..layer..".up"], s["L"..layer..".up"]
		local Wd, Sd = w["L"..layer..".down"], s["L"..layer..".down"]
		local Wan = w["L"..layer..".attn_norm"]
		local Wfn = w["L"..layer..".ffn_norm"]
		local k_cache = st.k_cache[layer + 1]
		local v_cache = st.v_cache[layer + 1]

		rmsnorm(x, Wan, h, H, eps)
		matvec_q8(Wq, Sq, h, q, H, H)                          -- Q (separate: rows=H≠KV)
		matvec_q8_pair(Wk, Sk, Wv, Sv, h, k_tmp, v_tmp, KV, H) -- A2: K+V fused
		apply_rope(q,     st.cos_tbl, st.sin_tbl, pos, NH,  HD)
		apply_rope(k_tmp, st.cos_tbl, st.sin_tbl, pos, KVH, HD)
		buffer.copy(k_cache, pos * KV * F4, k_tmp, 0, KV * F4)
		buffer.copy(v_cache, pos * KV * F4, v_tmp, 0, KV * F4)

		buffer.fill(attn_out, 0, 0, H * F4)
		for qh = 0, NH - 1 do
			local kvh = qh * KVH // NH
			local q_off = qh * HD * F4
			local max_s = -math.huge
			for tk = 0, pos do
				local k_off = (tk * KV + kvh * HD) * F4
				local dot = 0.0
				local i = 0
				while i + 16 <= HD do
					dot += buffer.readf32(q, q_off + i * F4)      * buffer.readf32(k_cache, k_off + i * F4)
					     + buffer.readf32(q, q_off + (i+1) * F4)  * buffer.readf32(k_cache, k_off + (i+1) * F4)
					     + buffer.readf32(q, q_off + (i+2) * F4)  * buffer.readf32(k_cache, k_off + (i+2) * F4)
					     + buffer.readf32(q, q_off + (i+3) * F4)  * buffer.readf32(k_cache, k_off + (i+3) * F4)
					     + buffer.readf32(q, q_off + (i+4) * F4)  * buffer.readf32(k_cache, k_off + (i+4) * F4)
					     + buffer.readf32(q, q_off + (i+5) * F4)  * buffer.readf32(k_cache, k_off + (i+5) * F4)
					     + buffer.readf32(q, q_off + (i+6) * F4)  * buffer.readf32(k_cache, k_off + (i+6) * F4)
					     + buffer.readf32(q, q_off + (i+7) * F4)  * buffer.readf32(k_cache, k_off + (i+7) * F4)
					     + buffer.readf32(q, q_off + (i+8) * F4)  * buffer.readf32(k_cache, k_off + (i+8) * F4)
					     + buffer.readf32(q, q_off + (i+9) * F4)  * buffer.readf32(k_cache, k_off + (i+9) * F4)
					     + buffer.readf32(q, q_off + (i+10) * F4) * buffer.readf32(k_cache, k_off + (i+10) * F4)
					     + buffer.readf32(q, q_off + (i+11) * F4) * buffer.readf32(k_cache, k_off + (i+11) * F4)
					     + buffer.readf32(q, q_off + (i+12) * F4) * buffer.readf32(k_cache, k_off + (i+12) * F4)
					     + buffer.readf32(q, q_off + (i+13) * F4) * buffer.readf32(k_cache, k_off + (i+13) * F4)
					     + buffer.readf32(q, q_off + (i+14) * F4) * buffer.readf32(k_cache, k_off + (i+14) * F4)
					     + buffer.readf32(q, q_off + (i+15) * F4) * buffer.readf32(k_cache, k_off + (i+15) * F4)
					i += 16
				end
				while i < HD do
					dot += buffer.readf32(q, q_off + i * F4) * buffer.readf32(k_cache, k_off + i * F4)
					i += 1
				end
				local sc_ = dot * inv_sqrt_hd
				buffer.writef32(scores, tk * F4, sc_)
				if sc_ > max_s then max_s = sc_ end
			end
			local denom = 0.0
			for tk = 0, pos do
				local e = math.exp(buffer.readf32(scores, tk * F4) - max_s)
				buffer.writef32(scores, tk * F4, e); denom += e
			end
			local inv_d = 1.0 / denom
			local out_off = qh * HD * F4
			for i = 0, HD - 1 do
				local acc = 0.0
				for tk = 0, pos do
					local v_off = (tk * KV + kvh * HD) * F4
					acc += buffer.readf32(scores, tk * F4) * inv_d * buffer.readf32(v_cache, v_off + i * F4)
				end
				buffer.writef32(attn_out, out_off + i * F4, acc)
			end
		end
		matvec_q8_add(Wo, So, attn_out, x, H, H) -- A3: Wo fused residual add into x
		rmsnorm(x, Wfn, h, H, eps)
		matvec_q8_pair(Wg, Sg, Wu, Su, h, gate, up, I, H) -- A1: gate+up fused
		for i = 0, I - 1 do
			local g = buffer.readf32(gate, i * F4)
			local silu = g / (1.0 + math.exp(-g))
			buffer.writef32(gate, i * F4, silu * buffer.readf32(up, i * F4))
		end
		matvec_q8_add(Wd, Sd, gate, x, H, I)     -- A4: Wdown fused residual add into x
	end

	rmsnorm(x, w["final_norm"], h, H, eps)
	matvec_q8(w["lm_head"], s["lm_head"], h, logits, V, H)
	st.pos = pos + 1
	return logits
end

-- Heap top-K sampler (from v2, unchanged).
local heap_ids: {number} = {}
local heap_vals: {number} = {}

local function heap_push(id: number, v: number, K: number, size: number): number
	if size < K then
		size += 1
		heap_ids[size] = id; heap_vals[size] = v
		local i = size
		while i > 1 do
			local p = i // 2
			if heap_vals[p] <= heap_vals[i] then break end
			heap_ids[i], heap_ids[p] = heap_ids[p], heap_ids[i]
			heap_vals[i], heap_vals[p] = heap_vals[p], heap_vals[i]
			i = p
		end
	elseif v > heap_vals[1] then
		heap_ids[1] = id; heap_vals[1] = v
		local i = 1
		while true do
			local l = i * 2; local r = l + 1; local smallest = i
			if l <= K and heap_vals[l] < heap_vals[smallest] then smallest = l end
			if r <= K and heap_vals[r] < heap_vals[smallest] then smallest = r end
			if smallest == i then break end
			heap_ids[i], heap_ids[smallest] = heap_ids[smallest], heap_ids[i]
			heap_vals[i], heap_vals[smallest] = heap_vals[smallest], heap_vals[i]
			i = smallest
		end
	end
	return size
end

function M.sample_topk(logits: buffer, V: number, temperature: number, top_k: number, rng): number
	local inv_temp = 1.0 / math.max(temperature, 1e-6)
	local K = math.min(top_k, V)
	local size = 0
	for i = 0, V - 1 do
		local v = buffer.readf32(logits, i * F4) * inv_temp
		size = heap_push(i, v, K, size)
	end
	local max_v = heap_vals[1]
	for i = 2, K do if heap_vals[i] > max_v then max_v = heap_vals[i] end end
	local total = 0.0
	for i = 1, K do
		local p = math.exp(heap_vals[i] - max_v)
		heap_vals[i] = p
		total += p
	end
	local r = rng:NextNumber() * total
	local acc = 0.0
	for i = 1, K do
		acc += heap_vals[i]
		if r <= acc then return heap_ids[i] end
	end
	return heap_ids[K]
end

-- Profile-instrumented step: same math as M.step with debug.profilebegin/end markers
-- and os.clock() accumulators for per-phase timing.
M.phase_ms = {embed=0, attn_proj=0, rope=0, attn=0, o_res=0, ffn=0, lm_head=0}
M.phase_count = 0

function M.reset_phase_ms()
	for k in pairs(M.phase_ms) do M.phase_ms[k] = 0 end
	M.phase_count = 0
end

function M.print_phase_ms()
	local total = 0
	for _, v in pairs(M.phase_ms) do total += v end
	local n = math.max(M.phase_count, 1)
	print(("=== phase timings (summed across %d steps, per-step avg) ==="):format(n))
	local order = {"embed","attn_proj","rope","attn","o_res","ffn","lm_head"}
	for _, k in ipairs(order) do
		local ms = M.phase_ms[k]
		print(("  %-10s %8.2f ms total  %7.3f ms/step  (%5.1f%%)"):format(
			k, ms, ms / n, total > 0 and (ms / total * 100) or 0))
	end
	print(("  %-10s %8.2f ms total  %7.3f ms/step"):format("TOTAL", total, total / n))
end

function M.step_profiled(st, token_id: number): buffer
	local cfg = st.cfg; local w, s = st.w, st.s
	local H, HD, NH, KVH, KV, I, V = cfg.hidden, cfg.head_dim, cfg.heads, cfg.kv_heads, cfg.kv_dim, cfg.inter, cfg.vocab
	local eps = cfg.rms_eps
	local inv_sqrt_hd = 1.0 / math.sqrt(HD)
	local pos = st.pos
	local x, h, q = st.x, st.h, st.q
	local k_tmp, v_tmp = st.k_tmp, st.v_tmp
	local attn_out = st.attn_out
	local gate, up = st.gate, st.up
	local logits, scores = st.logits, st.scores
	local phase, clock = M.phase_ms, os.clock

	local t = clock()
	debug.profilebegin("embed")
	local escale = buffer.readf32(s["embed"], token_id * F4)
	local eoff = token_id * H
	for i = 0, H - 1 do
		buffer.writef32(x, i * F4, buffer.readi8(w["embed"], eoff + i) * escale)
	end
	debug.profileend()
	phase.embed += (clock() - t) * 1000

	for layer = 0, cfg.layers - 1 do
		local Wq, Sq = w["L"..layer..".q"], s["L"..layer..".q"]
		local Wk, Sk = w["L"..layer..".k"], s["L"..layer..".k"]
		local Wv, Sv = w["L"..layer..".v"], s["L"..layer..".v"]
		local Wo, So = w["L"..layer..".o"], s["L"..layer..".o"]
		local Wg, Sg = w["L"..layer..".gate"], s["L"..layer..".gate"]
		local Wu, Su = w["L"..layer..".up"], s["L"..layer..".up"]
		local Wd, Sd = w["L"..layer..".down"], s["L"..layer..".down"]
		local Wan = w["L"..layer..".attn_norm"]
		local Wfn = w["L"..layer..".ffn_norm"]
		local k_cache = st.k_cache[layer + 1]
		local v_cache = st.v_cache[layer + 1]

		t = clock()
		debug.profilebegin("attn_proj")
		rmsnorm(x, Wan, h, H, eps)
		matvec_q8(Wq, Sq, h, q, H, H)
		matvec_q8_pair(Wk, Sk, Wv, Sv, h, k_tmp, v_tmp, KV, H) -- A2
		debug.profileend()
		phase.attn_proj += (clock() - t) * 1000

		t = clock()
		debug.profilebegin("rope")
		apply_rope(q,     st.cos_tbl, st.sin_tbl, pos, NH,  HD)
		apply_rope(k_tmp, st.cos_tbl, st.sin_tbl, pos, KVH, HD)
		buffer.copy(k_cache, pos * KV * F4, k_tmp, 0, KV * F4)
		buffer.copy(v_cache, pos * KV * F4, v_tmp, 0, KV * F4)
		debug.profileend()
		phase.rope += (clock() - t) * 1000

		t = clock()
		debug.profilebegin("attn")
		buffer.fill(attn_out, 0, 0, H * F4)
		for qh = 0, NH - 1 do
			local kvh = qh * KVH // NH
			local q_off = qh * HD * F4
			local max_s = -math.huge
			for tk = 0, pos do
				local k_off = (tk * KV + kvh * HD) * F4
				local dot = 0.0
				local i = 0
				while i + 16 <= HD do
					dot += buffer.readf32(q, q_off + i * F4)      * buffer.readf32(k_cache, k_off + i * F4)
					     + buffer.readf32(q, q_off + (i+1) * F4)  * buffer.readf32(k_cache, k_off + (i+1) * F4)
					     + buffer.readf32(q, q_off + (i+2) * F4)  * buffer.readf32(k_cache, k_off + (i+2) * F4)
					     + buffer.readf32(q, q_off + (i+3) * F4)  * buffer.readf32(k_cache, k_off + (i+3) * F4)
					     + buffer.readf32(q, q_off + (i+4) * F4)  * buffer.readf32(k_cache, k_off + (i+4) * F4)
					     + buffer.readf32(q, q_off + (i+5) * F4)  * buffer.readf32(k_cache, k_off + (i+5) * F4)
					     + buffer.readf32(q, q_off + (i+6) * F4)  * buffer.readf32(k_cache, k_off + (i+6) * F4)
					     + buffer.readf32(q, q_off + (i+7) * F4)  * buffer.readf32(k_cache, k_off + (i+7) * F4)
					     + buffer.readf32(q, q_off + (i+8) * F4)  * buffer.readf32(k_cache, k_off + (i+8) * F4)
					     + buffer.readf32(q, q_off + (i+9) * F4)  * buffer.readf32(k_cache, k_off + (i+9) * F4)
					     + buffer.readf32(q, q_off + (i+10) * F4) * buffer.readf32(k_cache, k_off + (i+10) * F4)
					     + buffer.readf32(q, q_off + (i+11) * F4) * buffer.readf32(k_cache, k_off + (i+11) * F4)
					     + buffer.readf32(q, q_off + (i+12) * F4) * buffer.readf32(k_cache, k_off + (i+12) * F4)
					     + buffer.readf32(q, q_off + (i+13) * F4) * buffer.readf32(k_cache, k_off + (i+13) * F4)
					     + buffer.readf32(q, q_off + (i+14) * F4) * buffer.readf32(k_cache, k_off + (i+14) * F4)
					     + buffer.readf32(q, q_off + (i+15) * F4) * buffer.readf32(k_cache, k_off + (i+15) * F4)
					i += 16
				end
				while i < HD do
					dot += buffer.readf32(q, q_off + i * F4) * buffer.readf32(k_cache, k_off + i * F4)
					i += 1
				end
				local sc_ = dot * inv_sqrt_hd
				buffer.writef32(scores, tk * F4, sc_)
				if sc_ > max_s then max_s = sc_ end
			end
			local denom = 0.0
			for tk = 0, pos do
				local e = math.exp(buffer.readf32(scores, tk * F4) - max_s)
				buffer.writef32(scores, tk * F4, e); denom += e
			end
			local inv_d = 1.0 / denom
			local out_off = qh * HD * F4
			for i = 0, HD - 1 do
				local acc = 0.0
				for tk = 0, pos do
					local v_off = (tk * KV + kvh * HD) * F4
					acc += buffer.readf32(scores, tk * F4) * inv_d * buffer.readf32(v_cache, v_off + i * F4)
				end
				buffer.writef32(attn_out, out_off + i * F4, acc)
			end
		end
		debug.profileend()
		phase.attn += (clock() - t) * 1000

		t = clock()
		debug.profilebegin("o_res")
		matvec_q8_add(Wo, So, attn_out, x, H, H) -- A3: fused Wo + residual
		debug.profileend()
		phase.o_res += (clock() - t) * 1000

		t = clock()
		debug.profilebegin("ffn")
		rmsnorm(x, Wfn, h, H, eps)
		matvec_q8_pair(Wg, Sg, Wu, Su, h, gate, up, I, H) -- A1: fused gate+up
		for i = 0, I - 1 do
			local g = buffer.readf32(gate, i * F4)
			local silu = g / (1.0 + math.exp(-g))
			buffer.writef32(gate, i * F4, silu * buffer.readf32(up, i * F4))
		end
		matvec_q8_add(Wd, Sd, gate, x, H, I)     -- A4: fused Wdown + residual
		debug.profileend()
		phase.ffn += (clock() - t) * 1000
	end

	t = clock()
	debug.profilebegin("lm_head")
	rmsnorm(x, w["final_norm"], h, H, eps)
	matvec_q8(w["lm_head"], s["lm_head"], h, logits, V, H)
	debug.profileend()
	phase.lm_head += (clock() - t) * 1000
	st.pos = pos + 1
	M.phase_count += 1
	return logits
end

-- ============================================================================
-- HF GPT-2-style BPE tokenizer (matches HuggingFaceTB/SmolLM2 tokenizer.json).
-- Wire format produced by handover-files/export_tokenizer.py (magic 'TLLB').
-- Pipeline: Digits(individual_digits) → ByteLevel(use_regex) → byte-encode → BPE.
-- ============================================================================

function M.load_tokenizer(body: string)
	local blob = buffer.fromstring(body)
	assert(buffer.readstring(blob, 0, 4) == "TLLB", "bad tokenizer magic")
	assert(buffer.readu32(blob, 4) == 1, "bad tokenizer version")
	local vocab_size = buffer.readu32(blob, 8)
	local n_merges   = buffer.readu32(blob, 12)
	local bos_id     = buffer.readu32(blob, 16)
	local eos_id     = buffer.readu32(blob, 20)
	local off        = 28
	-- byte_encoder: 256 entries (byte b -> 1-2 byte UTF-8 string)
	local byte_enc: {string} = table.create(256)
	local byte_dec: {[string]: number} = {}
	for b = 0, 255 do
		local l = buffer.readu8(blob, off); off += 1
		local s = buffer.readstring(blob, off, l); off += l
		byte_enc[b + 1] = s
		byte_dec[s]     = b
	end
	-- vocab: id -> piece string
	local piece_by_id: {string}          = table.create(vocab_size)
	local id_by_piece: {[string]: number} = {}
	for id = 0, vocab_size - 1 do
		local l = buffer.readu16(blob, off); off += 2
		local p = buffer.readstring(blob, off, l); off += l
		piece_by_id[id + 1] = p
		id_by_piece[p]      = id
	end
	-- merges: (lhs, rhs) pair -> rank (lower rank = earlier merge)
	local merges_rank: {[string]: number} = {}
	for rank = 0, n_merges - 1 do
		local ll = buffer.readu16(blob, off); off += 2
		local l  = buffer.readstring(blob, off, ll); off += ll
		local rl = buffer.readu16(blob, off); off += 2
		local r  = buffer.readstring(blob, off, rl); off += rl
		merges_rank[l .. "\0" .. r] = rank
	end
	return {
		vocab_size  = vocab_size,
		n_merges    = n_merges,
		bos_id      = bos_id,
		eos_id      = eos_id,
		byte_enc    = byte_enc,
		byte_dec    = byte_dec,
		piece_by_id = piece_by_id,
		id_by_piece = id_by_piece,
		merges_rank = merges_rank,
	}
end

-- GPT-2 pre-tokenizer stage 1: Digits(individual_digits=True).
-- Every ASCII digit becomes its own pre-token; non-digit runs stay intact.
local function split_digits(text: string): {string}
	local out = {}
	local buf = {}
	local n = #text
	for i = 1, n do
		local c = string.byte(text, i)
		if c >= 48 and c <= 57 then
			if #buf > 0 then out[#out + 1] = table.concat(buf); buf = {} end
			out[#out + 1] = string.char(c)
		else
			buf[#buf + 1] = string.char(c)
		end
	end
	if #buf > 0 then out[#out + 1] = table.concat(buf) end
	return out
end

-- GPT-2 pre-tokenizer stage 2: ByteLevel(use_regex=True).
-- ASCII-only implementation of the canonical regex:
--   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
-- Python `re` alternation is ordered (first-match-wins at each position), not
-- longest-match, so we try alternatives in order and return the first hit's length.
local CONTRACTIONS = { "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" }

local function is_letter(b: number): boolean
	return (b >= 65 and b <= 90) or (b >= 97 and b <= 122)
end
local function is_digit(b: number): boolean
	return b >= 48 and b <= 57
end
local function is_ws(b: number): boolean
	return b == 32 or b == 9 or b == 10 or b == 13
end

local function pretok_match_len(text: string, i: number, n: number): number
	-- 1. contractions
	if string.byte(text, i) == 39 then
		for _, suf in ipairs(CONTRACTIONS) do
			local sl = #suf
			if i + sl - 1 <= n and string.sub(text, i, i + sl - 1) == suf then
				return sl
			end
		end
	end
	-- 2. optional-space + letter run
	local j = i
	if string.byte(text, j) == 32 then j += 1 end
	if j <= n and is_letter(string.byte(text, j)) then
		local k = j + 1
		while k <= n and is_letter(string.byte(text, k)) do k += 1 end
		return k - i
	end
	-- 3. optional-space + digit run (mostly dead because Digits split runs first)
	j = i
	if string.byte(text, j) == 32 then j += 1 end
	if j <= n and is_digit(string.byte(text, j)) then
		local k = j + 1
		while k <= n and is_digit(string.byte(text, k)) do k += 1 end
		return k - i
	end
	-- 4. optional-space + non-space-non-letter-non-digit run (punctuation etc.)
	j = i
	if string.byte(text, j) == 32 then j += 1 end
	if j <= n then
		local b = string.byte(text, j)
		if not (is_letter(b) or is_digit(b) or is_ws(b)) then
			local k = j + 1
			while k <= n do
				local bb = string.byte(text, k)
				if is_letter(bb) or is_digit(bb) or is_ws(bb) then break end
				k += 1
			end
			return k - i
		end
	end
	-- 5. \s+(?!\S) — ws run that is followed by ws or EOF (i.e. a trailing-ws segment)
	if is_ws(string.byte(text, i)) then
		local k = i
		while k <= n and is_ws(string.byte(text, k)) do k += 1 end
		if k > n then
			return k - i  -- whole run is trailing, take all
		end
		-- followed by non-ws: \s+(?!\S) matches everything except the final space
		if k - 1 > i then return k - 1 - i end
		-- single-char ws followed by non-ws: fall through to rule 6
	end
	-- 6. \s+
	if is_ws(string.byte(text, i)) then
		local k = i + 1
		while k <= n and is_ws(string.byte(text, k)) do k += 1 end
		return k - i
	end
	-- Shouldn't reach here for valid ASCII input; advance by 1 to make progress.
	return 1
end

local function bytelevel_regex(text: string): {string}
	local out = {}
	local i, n = 1, #text
	while i <= n do
		local len = pretok_match_len(text, i, n)
		out[#out + 1] = string.sub(text, i, i + len - 1)
		i += len
	end
	return out
end

-- BPE merge: adjacent chars repeatedly merged by lowest rank until stable.
local function bpe(chars: {string}, merges_rank: {[string]: number}): {string}
	while #chars > 1 do
		local best_rank, best_i = math.huge, 0
		for i = 1, #chars - 1 do
			local r = merges_rank[chars[i] .. "\0" .. chars[i + 1]]
			if r and r < best_rank then
				best_rank = r
				best_i = i
			end
		end
		if best_i == 0 then break end
		chars[best_i] = chars[best_i] .. chars[best_i + 1]
		table.remove(chars, best_i + 1)
	end
	return chars
end

function M.encode(tok, text: string): {number}
	local ids = {}
	local byte_enc = tok.byte_enc
	local merges_rank = tok.merges_rank
	local id_by_piece = tok.id_by_piece
	for _, d_piece in ipairs(split_digits(text)) do
		for _, piece in ipairs(bytelevel_regex(d_piece)) do
			-- byte-encode each byte of the piece into a list of byte-encoder chars
			local n = #piece
			local chars = table.create(n)
			for i = 1, n do
				chars[i] = byte_enc[string.byte(piece, i) + 1]
			end
			chars = bpe(chars, merges_rank)
			for _, c in ipairs(chars) do
				local id = id_by_piece[c]
				if id == nil then
					error("BPE produced piece not in vocab: "..string.format("%q", c))
				end
				ids[#ids + 1] = id
			end
		end
	end
	return ids
end

function M.decode(tok, ids: {number}): string
	local parts = {}
	local piece_by_id = tok.piece_by_id
	for _, id in ipairs(ids) do
		if id >= 0 and id < tok.vocab_size then
			parts[#parts + 1] = piece_by_id[id + 1]
		end
	end
	local enc_str = table.concat(parts)
	local out = {}
	local i, n = 1, #enc_str
	local byte_dec = tok.byte_dec
	while i <= n do
		local b1 = string.byte(enc_str, i)
		local clen = 1
		if b1 >= 0xC0 and b1 < 0xE0 then clen = 2
		elseif b1 >= 0xE0 and b1 < 0xF0 then clen = 3
		elseif b1 >= 0xF0 then clen = 4 end
		local ch = string.sub(enc_str, i, i + clen - 1)
		local b = byte_dec[ch]
		if b then
			out[#out + 1] = string.char(b)
		else
			-- Special-token piece (ASCII, not a byte-encoded char) — pass through.
			out[#out + 1] = ch
		end
		i += clen
	end
	return table.concat(out)
end

return M
]==]
ModuleScript.Parent = game:GetService("ServerStorage")

local HttpService = game:GetService("HttpService")

-- Fetch N chunks, concat into one buffer via buffer.writestring.
-- Each chunk must be a contiguous byte slice of the full blob.
local function fetch_chunked(urls: { string }): buffer
	local bodies = table.create(#urls)
	local total = 0
	for i, url in ipairs(urls) do
		local t = os.clock()
		local body = HttpService:GetAsync(url)
		bodies[i] = body
		total += #body
		print(("  chunk %d/%d: %.2f MB in %.2f s"):format(i, #urls, #body / 1e6, os.clock() - t))
	end
	local blob = buffer.create(total)
	local off = 0
	for _, body in ipairs(bodies) do
		buffer.writestring(blob, off, body)
		off += #body
	end
	return blob
end

print("=== loading ===")
local t0 = os.clock()
local blob = fetch_chunked(BLOB_CHUNK_URLS)
print(("fetched %.2f MB total in %.2f s"):format(buffer.len(blob) / 1e6, os.clock() - t0))

local t_tok_fetch = os.clock()
local tok_body = HttpService:GetAsync(TOKENIZER_URL)
print(("tokenizer: %.2f KB in %.2f s"):format(#tok_body / 1024, os.clock() - t_tok_fetch))

local SmolLM = require(ModuleScript)
local t_parse = os.clock()
local model = SmolLM.load_model(blob, RUNTIME_MAX_CTX)
print(
	("parsed model in %.3f s  (layers=%d, vocab=%d, max_ctx=%d, tied=%s)"):format(
		os.clock() - t_parse,
		model.cfg.layers,
		model.cfg.vocab,
		model.cfg.max_ctx,
		tostring(model.tied)
	)
)
local t_tok_parse = os.clock()
local tok = SmolLM.load_tokenizer(tok_body)
print(
	("parsed tokenizer in %.3f s  (vocab=%d, merges=%d, bos=%d, eos=%d)"):format(
		os.clock() - t_tok_parse,
		tok.vocab_size,
		tok.n_merges,
		tok.bos_id,
		tok.eos_id
	)
)
print("")

-- Tokenizer self-check: Luau encode() must match Python reference bit-exactly.
do
	print("=== tokenizer self-check ===")
	local fails = 0
	for _, tv in ipairs(TEST_VECTORS) do
		local got = SmolLM.encode(tok, tv.text)
		local ok = (#got == #tv.ids)
		if ok then
			for i = 1, #got do
				if got[i] ~= tv.ids[i] then
					ok = false
					break
				end
			end
		end
		if ok then
			print(("  OK   %-40s -> %d tokens"):format(string.format("%q", tv.text), #got))
		else
			fails += 1
			local a, b = {}, {}
			for _, v in ipairs(tv.ids) do
				a[#a + 1] = tostring(v)
			end
			for _, v in ipairs(got) do
				b[#b + 1] = tostring(v)
			end
			print(("  FAIL %q"):format(tv.text))
			print("       expected: {" .. table.concat(a, ", ") .. "}")
			print("       got:      {" .. table.concat(b, ", ") .. "}")
		end
	end
	if fails > 0 then
		error(("tokenizer self-check FAILED on %d/%d vectors"):format(fails, #TEST_VECTORS))
	end
	print("")
end

local rng = Random.new(SEED or os.clock() * 1e6)
print(("=== prompt: %q ==="):format(PROMPT))
local prompt_ids = SmolLM.encode(tok, PROMPT)
print(("encoded into %d tokens: {%s}"):format(
	#prompt_ids,
	(function()
		local t = {}
		for _, x in ipairs(prompt_ids) do
			t[#t + 1] = tostring(x)
		end
		return table.concat(t, ", ")
	end)()
))

local step_fn = if PROFILE_MODE then SmolLM.step_profiled else SmolLM.step
if PROFILE_MODE then
	print(
		("(PROFILE_MODE on — first %d decode steps use step_profiled; phase totals printed after)"):format(
			PROFILE_STEPS
		)
	)
	SmolLM.reset_phase_ms()
end

-- Prefill
local t_prefill = os.clock()
local logits
if PREPEND_BOS then
	logits = step_fn(model, tok.bos_id)
end
for i, id in ipairs(prompt_ids) do
	logits = step_fn(model, id)
	if i % 4 == 0 then
		task.wait()
	end
end
local dt_prefill = os.clock() - t_prefill
local n_prefill = (if PREPEND_BOS then 1 else 0) + #prompt_ids
print(
	("prefill: %d tokens in %.1f ms  (%.1f ms/tok,  %.1f tok/s)"):format(
		n_prefill,
		dt_prefill * 1000,
		dt_prefill * 1000 / math.max(n_prefill, 1),
		n_prefill / math.max(dt_prefill, 1e-6)
	)
)

print("")
print("=== generating ===")
local generated_ids = {}
local t_gen = os.clock()
for step = 1, MAX_TOKENS do
	local use_profiled = PROFILE_MODE and step <= PROFILE_STEPS
	local next_id
	if use_profiled then
		debug.profilebegin("sampler")
		next_id = SmolLM.sample_topk(logits, model.cfg.vocab, TEMPERATURE, TOP_K, rng)
		debug.profileend()
	else
		next_id = SmolLM.sample_topk(logits, model.cfg.vocab, TEMPERATURE, TOP_K, rng)
	end
	if next_id == tok.eos_id then
		break
	end
	generated_ids[#generated_ids + 1] = next_id
	if use_profiled then
		logits = SmolLM.step_profiled(model, next_id)
	else
		logits = SmolLM.step(model, next_id)
	end
	if step % 4 == 0 then
		task.wait()
	end
end
local dt = os.clock() - t_gen
local n = #generated_ids
print(("generated %d tokens in %.2f s  (%.1f tok/s)"):format(n, dt, n / math.max(dt, 1e-6)))
print("")

if PROFILE_MODE then
	SmolLM.print_phase_ms()
	print("")
end

print("=== output ===")
print(PROMPT .. SmolLM.decode(tok, generated_ids))
print("")
print("=== generated token IDs ===")
do
	local t = {}
	for i = 1, #generated_ids do
		t[i] = tostring(generated_ids[i])
	end
	print("{" .. table.concat(t, ", ") .. "}")
end

ModuleScript:Destroy()
