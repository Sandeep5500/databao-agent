# Local Setup: GLM-4.7-Flash-AWQ + Spider 2.0 Benchmark

This guide covers running databao-agent on the Babel cluster using GLM-4.7-Flash-AWQ
served via vLLM for the Spider 2.0 benchmark.

---

## Architecture Overview

```
[login node]                        [compute node — A6000 GPU]
  fast_embed_server (port 11434)       vLLM server (port 8765)
  benchmark runner  ─────────────────────────────────────────→  OpenAI-compatible API
```

Three independent components must all be running for the benchmark to work:

1. **fast_embed_server** — serves `nomic-embed-text` embeddings for DCE vector search
   (`search_context` tool). Runs on the login node.
2. **vLLM server** — serves GLM-4.7-Flash-AWQ on a GPU compute node via SLURM.
3. **Benchmark runner** — reads `logs/vllm_endpoint.txt` to find the vLLM node,
   then runs questions through the databao agent.

---

## Quick Start

```bash
# 1. Start embed server (login node — must be running before benchmark)
cd /data/user_data/<user>/personal/capstone/databao-context-engine
nohup uv run python ../scripts/fast_embed_server.py --port 11434 \
  > /tmp/fast_embed_login.log 2>&1 &
curl -s http://localhost:11434/api/tags   # verify: returns nomic-embed-text model

# 2. Submit vLLM SLURM job (GLM-4.7-Flash-AWQ is the default)
cd /data/user_data/<user>/personal/capstone
sbatch scripts/serve_vllm.slurm

# Watch until "Application startup complete" (~8 minutes):
tail -f logs/vllm_<jobid>.err   # weight loading progress
tail -f logs/vllm_<jobid>.out   # routes + startup message

# 3. Verify vLLM is responding
cat logs/vllm_endpoint.txt                      # e.g. babel-s9-24:8765
curl -s http://$(cat logs/vllm_endpoint.txt | head -1 | awk '{print $1}')/v1/models

# 4. Run benchmark
cd databao-agent

# First 5 questions (quick sanity check)
PYTHONUNBUFFERED=1 uv run python -u ../scripts/spider2_benchmark.py \
  --instances local002,local003,local004,local007,local008 \
  2>&1 | tee ../logs/benchmark_5q_$(date +%Y%m%d_%H%M%S).log

# First 20 questions
PYTHONUNBUFFERED=1 uv run python -u ../scripts/spider2_benchmark.py \
  --limit 20 2>&1 | tee ../logs/benchmark_20q_$(date +%Y%m%d_%H%M%S).log

# Full 135-question benchmark (background, ~6 hours)
PYTHONUNBUFFERED=1 nohup uv run python -u ../scripts/spider2_benchmark.py \
  2>&1 | tee ../logs/benchmark_full_$(date +%Y%m%d_%H%M%S).log > /tmp/benchmark.log &
```

---

## Model: GLM-4.7-Flash-AWQ

```
HuggingFace ID:  QuantTrio/GLM-4.7-Flash-AWQ
Architecture:    GLM-4.7-Flash — 30B-A3B MoE (30B total, 3B active per token)
Quantization:    INT4 AWQ
VRAM:            ~17 GB (fits on single A6000 48GB)
Context window:  40960 tokens (capped to 28000 due to A6000 KV cache constraint)
Tool calling:    glm47 parser (vLLM >= 0.18.1)
Thinking:        Yes — generates <think>...</think> blocks, but does not interfere
                 with tool call parsing (tool calls still use finish_reason=tool_calls)
```

### Why max_model_len=28000 (not 40960)?

GLM at 40960 tokens requires ~36.7 GiB of KV cache. A6000 nodes only have ~26 GiB
available after model weights (~17 GiB) and CUDA overhead. Setting
`--max-model-len 28000` keeps KV cache requirements within budget.

### Switching to Qwen3-32B-AWQ (recommended fallback)

```bash
MODEL=Qwen/Qwen3-32B-AWQ sbatch scripts/serve_vllm.slurm
```

The SLURM script auto-detects the model name and adjusts the tool-call parser
(`glm47` for GLM, `hermes` for others). No other changes needed.

---

## SLURM Job: serve_vllm.slurm

**Location:** `scripts/serve_vllm.slurm`

**Key settings:**
```bash
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=babel-s9-28    # hardware bus error on this node — excluded
```

**Configurable env vars (override at submission):**
```bash
MODEL=QuantTrio/GLM-4.7-Flash-AWQ  # default
VLLM_PORT=8765                      # default
MAX_MODEL_LEN=28000                 # default — GLM KV cache limit on A6000
```

**Critical GLM-specific settings baked into the script:**

| Setting | Value | Why |
|---------|-------|-----|
| `VLLM_MLA_DISABLE=1` | env var before launch | Disables TRITON_MLA backend; forces FLASH_ATTN. TRITON_MLA causes bus errors on A6000 nodes with GLM's MLA attention. |
| `--quantization awq` | not `awq_marlin` | `awq_marlin` kernel causes bus errors with the `glm4_moe_lite` architecture in vLLM 0.18.1. |
| `--dtype float16` | not `auto` | AWQ quantization requires float16. vLLM defaults to bfloat16 on A6000, causing a validation error at startup. |
| `--enforce-eager` | flag | Disables CUDA graph capture, avoiding OOM during warmup on A6000. |
| `--gpu-memory-utilization 0.92` | not 0.95 | Leaves headroom for GLM's larger KV cache. |

**Why `$VENV_PYTHON` instead of `uv run python`:**

The script uses `.venv/bin/python` directly to launch vLLM. `uv run` syncs the
`uv.lock` before execution, which would revert `regex` from `2026.4.4` back to
`2025.9.18`. GLM requires `transformers` from git main (5.6.0.dev0), which
requires `regex>=2025.10.22`. Using `$VENV_PYTHON` skips the lockfile sync.

---

## Benchmark Script: spider2_benchmark.py

**Location:** `scripts/spider2_benchmark.py`

**Current configuration:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `max_tokens` | 1024 | GLM context is 28000. Reducing from 2048 frees 1024 tokens, preventing overflow on questions with large external knowledge docs. SQL rarely exceeds 500 tokens. |
| `recursion_limit` | 12 (AgentConfig) | Caps agent loops at 12 steps. Maps to 24 LangGraph nodes (2 per step). |
| `_graph_recursion_limit` | 24 (executor) | Must be set alongside `recursion_limit`. `base.py` uses `max(self._graph_recursion_limit, agent_config.recursion_limit)` — default is 50, overriding AgentConfig. |
| `min_retrievals` | 1 | Agent must call `search_context` at least once per question. |
| `_max_schema_summary_length` | 60,000 chars | ~15K tokens. Triggers 3-tier schema fallback for large DBs (E_commerce, Baseball) to prevent schema overflow. |
| External knowledge truncation | 20,000 chars | ~5K tokens. Prevents haversine_formula.md and similar long docs from overflowing 28K context. |

**Schema compression (3-tier fallback):**

When a database schema exceeds `_max_schema_summary_length`:
1. **Full schema** — all tables + columns (≤ 60K chars)
2. **Table names only** — no column details; agent uses `search_context` for specifics
3. **Schema overview** — high-level description only

This is critical for large databases like E_commerce (~100K chars) and Baseball (~112K chars).

---

## DCE Retrieval (search_context tool)

The `search_context` tool lets the agent query database schemas and column descriptions
via vector similarity search. It requires:

1. **fast_embed_server** running on `localhost:11434` — embeds the search query
2. **dce.duckdb** — pre-built vector index at `spider2-dce/output/dce.duckdb`

Per question, a temporary DCE project is created with only the relevant database's
YAML, so the agent sees exactly one schema at a time.

**Chunk types in the vector index:**

- `TABLE` chunk: embeds a description of the table (columns, PKs, FKs, row count)
- `COLUMN` chunk: embeds a description of a single column (type, description, stats)

The agent retrieves the most relevant chunks and reads the YAML to understand schema
details before writing SQL.

---

## Benchmark Results

| Run | Model | Questions | Score | Notes |
|-----|-------|-----------|-------|-------|
| Run 1 | Qwen3-8B | 5 | 1/5 (20%) | Initial test |
| Run 2 | GLM-4.7-Flash-AWQ | 5 | 1/5 (20%) | v1 prompts |
| Run 3 | GLM-4.7-Flash-AWQ | 5 | 1/5 (20%) | v2 prompts |
| Run 4 | Qwen3-32B-AWQ | 123 | 15/123 (12%) | First full run |
| Run 5 | GLM-4.7-Flash-AWQ | 20 | 3/20 (15%) | awq fix confirmed stable |

**Run 4 breakdown (Qwen3-32B-AWQ, 123 questions):**

| Status | Count |
|--------|-------|
| result_mismatch | 54 |
| no_sql | 29 |
| agent_error (context overflow) | 25 |
| pass | 15 |

Main failure areas: `bank_sales_trading` (0/15), `f1` (0/8), `sqlite-sakila` (0/7).
Best areas: `chinook` (2/3), `Baseball` (1/2), `IPL` (2/11).

---

## Troubleshooting

### vLLM Bus Error on First Inference

**Symptom:** Server starts ("Application startup complete") then crashes with
`Bus error (core dumped)` on the first request.

**Root cause (GLM-specific):** GLM-4.7-Flash uses Multi-head Latent Attention (MLA),
which vLLM routes to the `TRITON_MLA` backend by default. This backend causes hardware
bus errors on A6000 nodes.

**Fix:** Set `VLLM_MLA_DISABLE=1` before launching vLLM. This forces `FLASH_ATTN`
instead of `TRITON_MLA`. Already set in `serve_vllm.slurm`.

```bash
# Verify the fix is active — look for this line in vllm_<jobid>.err:
# "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', ...]"
grep "attention backend" logs/vllm_*.err | tail -3
```

**Note:** `VLLM_ATTENTION_BACKEND=FLASH_ATTN` was removed in vLLM 0.18.1. Use
`VLLM_MLA_DISABLE=1` instead.

**Secondary cause:** `awq_marlin` kernel incompatibility with `glm4_moe_lite`
architecture. Script uses `--quantization awq` (not `awq_marlin`) to avoid this.

### transformers ImportError: regex>=2025.10.22 required

**Symptom:**
```
ImportError: regex>=2025.10.22 is required for a normal functioning of this module
```

**Cause:** `uv.lock` pins `regex==2025.9.18`. GLM needs `transformers` from git main.

**Fix (already in slurm script):**
1. Install `transformers` from git main
2. Force-install `regex==2026.4.4 --no-deps`
3. Run vLLM via `$VENV_PYTHON` (not `uv run`) to skip lockfile sync

**Manual fix if needed:**
```bash
cd /data/user_data/<user>/personal/capstone/databao-agent
uv pip install --python .venv/bin/python "regex==2026.4.4" --no-deps
.venv/bin/python -c "import transformers; print(transformers.__version__)"
# Expected: 5.6.0.dev0
```

### Context Overflow (400 error)

**Symptom:**
```
Error code: 400 - maximum context length is 28000 tokens. However, you requested
1024 output tokens and your prompt contains at least XXXXX input tokens
```

**Cause:** Question uses a large external knowledge doc + large schema. Airlines
questions (local009, local010) use `haversine_formula.md`; the schema + doc + history
exceeds 27,000 input tokens.

**Fix options:**
- Reduce `max_tokens` further (currently 1024)
- Reduce `_max_schema_summary_length` below 60,000 chars
- Reduce external knowledge truncation below 20,000 chars

### Agent Loops (Recursion Limit)

**Symptom:** `Recursion limit of 50 reached` in agent trace.

**Root cause:** `AgentConfig.recursion_limit` alone is insufficient. `base.py` computes
the actual LangGraph limit as `max(self._graph_recursion_limit, agent_config.recursion_limit)`,
and `_graph_recursion_limit` defaults to 50.

**Fix (already in benchmark script):**
```python
agent_config = AgentConfig(recursion_limit=12, min_retrievals=1)
executor = LighthouseExecutor()
executor._graph_recursion_limit = 24  # must match recursion_limit × 2
```

### fast_embed_server Not Running

**Symptom:** `search_context` fails or the agent never retrieves schema context.

**Fix:**
```bash
# Check
curl -s http://localhost:11434/api/tags

# Restart
cd /data/user_data/<user>/personal/capstone/databao-context-engine
nohup uv run python ../scripts/fast_embed_server.py --port 11434 \
  > /tmp/fast_embed_login.log 2>&1 &
sleep 30
curl -s http://localhost:11434/api/tags
```

### Connection Error in Benchmark

**Symptom:** All questions show `AGENT ERROR: Connection error.`

```bash
# Check if vLLM job is still running
squeue -u $USER

# Check for crash
tail -20 logs/vllm_*.err

# Re-submit if crashed
sbatch scripts/serve_vllm.slurm

# Verify connectivity
curl -s http://$(cat logs/vllm_endpoint.txt | head -1 | awk '{print $1}')/v1/models
```

---

## File Reference

```
capstone/
├── scripts/
│   ├── serve_vllm.slurm          # SLURM: launches vLLM on A6000 GPU node
│   ├── spider2_benchmark.py      # Benchmark runner for Spider 2.0 local SQLite track
│   ├── fast_embed_server.py      # Embedding server for search_context tool
│   └── enrich_dce.py             # Builds DCE vector index (one-time setup)
├── logs/
│   ├── vllm_endpoint.txt         # Written by SLURM: "<node>:8765\nModel: ..."
│   ├── vllm_<jobid>.out          # vLLM stdout (routes, startup banner)
│   ├── vllm_<jobid>.err          # vLLM stderr (weights, errors, crashes)
│   └── benchmark_*.log           # Benchmark output with agent traces
├── results/
│   └── spider2_results_*.csv     # Per-question scores
├── databao-agent/                # Agent library (this repo)
│   ├── databao/agent/configs/agent.py           # AgentConfig (recursion_limit, min_retrievals)
│   ├── databao/agent/executors/lighthouse/      # LighthouseExecutor + system prompt
│   └── LOCAL_SETUP_README.md     # This file
├── databao-context-engine/       # DCE vector search library
├── spider2-dce/                  # DCE project for Spider 2.0
│   ├── dce.ini
│   ├── src/databases/            # Per-database source YAMLs
│   └── output/
│       ├── dce.duckdb            # 270MB vector index (pre-built)
│       └── databases/            # Enriched YAMLs with LLM descriptions
└── Spider2/spider2-lite/         # Spider 2.0 dataset
    ├── spider2-lite.jsonl        # 135 questions
    └── evaluation_suite/gold/    # Gold CSVs + condition_cols/ignore_order config
```
