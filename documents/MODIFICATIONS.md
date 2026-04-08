# Modifications to databao-agent and databao-context-engine

This document summarizes the changes made to the databao stack while running
the Spider 2.0 local benchmark. Changes are grouped by repository and by
purpose so it is clear *what* was added and *why*.

The companion guide [LOCAL_SETUP_README.md](LOCAL_SETUP_README.md) covers how
to actually run the modified stack on the Babel cluster.

---

## databao-agent

### 1. `min_retrievals` agent config

**File:** `databao/agent/configs/agent.py`

Added a `min_retrievals: int = 0` field to `AgentConfig`. When set, the system
prompt instructs the agent that it must call `search_context` at least N times
before submitting a result.

**Why:** During Spider 2.0 runs the agent often skipped `search_context`
entirely on questions where it should have looked up table/column metadata
first (e.g. `local004` jumping straight to SQL with the wrong customer ID
column). Forcing a minimum number of retrievals biases the agent toward
exploration before commitment.

**File:** `databao/agent/executors/lighthouse/executor.py`

`LighthouseExecutor.render_system_prompt(...)` now accepts and passes
`min_retrievals` through to the Jinja template.

**File:** `databao/agent/executors/lighthouse/system_prompt.jinja`

The system prompt now reads:

> Always use the search_context tool to find relevant context (table
> descriptions, column meanings, relationships) before writing SQL.
> You MUST call search_context at least {{ min_retrievals }} time(s)
> before submitting your result. Use different queries each time to
> explore relevant tables and columns.

Also tightened the SQL output rule:

> Always output exactly ONE DuckDB SELECT statement. Never use SET, PRAGMA,
> or SQLite-specific syntax. Combine multiple result sets with UNION ALL
> rather than separate statements.

---

## databao-context-engine

### 1. Increased OllamaConfig timeout (30s → 120s)

**File:** `src/databao_context_engine/llm/config.py`

Embedding requests for `nomic-embed-text:v1.5` running on CPU on the login
node sometimes exceeded the default 30-second timeout when the embed server
was warming up after a fresh restart. Bumped the default to 120s to absorb
cold starts without failing the first few enrichment calls.

### 2. Critique pass during table enrichment

**File:** `src/databao_context_engine/plugins/databases/context_enricher.py`

Added a second LLM pass that runs immediately after the per-column
enrichment loop in `_get_enriched_table`. The new helper
`_critique_table_columns(...)`:

1. Builds a single prompt containing every column in the table, its current
   draft description, and up to 5 distinct sample values pulled from
   `table.samples`.
2. Asks the model to rewrite *only* the columns whose draft descriptions do
   not clearly distinguish them from sibling columns, citing concrete sample
   values from both columns whenever the values differ.
3. Parses a YAML `rewrites` block, validates that every returned column name
   exists in the original table, drops invalid entries, and applies the
   surviving rewrites in place.

The pass is wrapped in a `try`/`except` so a critique failure (parser error,
unreachable LLM, etc.) leaves the original draft descriptions in place rather
than blocking the build.

**Prompt design highlights** (full text in
[context_enricher.py](../../databao-context-engine/src/databao_context_engine/plugins/databases/context_enricher.py)):

- "Mandatory sample citation rule" — if sibling columns have distinct sample
  values, the rewrite must quote a real value from each.
- "Symmetric treatment" — if one column in a sibling group is rewritten,
  every other column in that group must also be rewritten.
- "No invented relationships" — only mention cross-column relationships
  visible in column names, foreign keys, or sample values.

**Why:** The original per-column enrichment describes each column in
isolation, so semantically distinct columns with similar names end up with
nearly identical descriptions:

```yaml
- name: customer_id
  description: customer_id is a nullable TEXT column used to uniquely identify customer records.
- name: customer_unique_id
  description: customer_unique_id is a text column that contains unique identifiers for customers.
```

This is the root cause of result_mismatch failures like `local004`
(`customer_id` vs `customer_unique_id`) and `local008` (`name_first || name_last`
vs `name_given`). The agent has no signal to pick the right column.

After the critique pass, the same columns become:

```yaml
- name: customer_id
  description: customer_id is the order-scoped foreign key, e.g. "06b8999e..."; differs from customer_unique_id which is the persistent customer identity used in the customers table.
- name: customer_unique_id
  description: customer_unique_id is the persistent customer identity that survives across orders, e.g. "861eff4711a542e..."; use this instead of customer_id when computing per-customer aggregates.
```

**Validation runs (Qwen3-32B-AWQ on babel-n5-20):**

| Table | Cols | Rewrites | Time | Notes |
|---|---|---|---|---|
| `baseball.player` | 24 | 6 | ~14s | Caught `name_first/name_last/name_given` and `player_id/retro_id/bbref_id` triple. |
| `baseball.team` | 48 | 14 | ~47s | Caught 4-way `team_id*` IDs, `lg/div/wc/ws_win` group, `bb/bba`, `so/soa`, `double/dp`. |

Both runs returned `finish_reason=stop`, all rewrites validated cleanly with
zero hallucinated column names. Token budget: prompt ≤3.2K, response ≤1.3K.

The standalone experiment script that produced these numbers lives at
[scripts/experiment_critique_enrich.py](../../scripts/experiment_critique_enrich.py)
in the capstone repo.

---

## Companion changes outside the forked repos

Several supporting changes live in `scripts/` (not part of either fork) and
are documented here for completeness:

- **`scripts/serve_vllm.slurm`** — vLLM SLURM job. Rewritten to support both
  GLM-4.7-Flash-AWQ and Qwen3-32B-AWQ from a single script. Critical
  GLM-specific settings: `VLLM_MLA_DISABLE=1` (replaces removed
  `VLLM_ATTENTION_BACKEND` env var in vLLM ≥ 0.18.1), `--quantization awq`
  (not `awq_marlin`), `--dtype float16`, `--enforce-eager`,
  `--tool-call-parser glm47`, transformers from git main + force-installed
  `regex==2026.4.4`. See [LOCAL_SETUP_README.md](LOCAL_SETUP_README.md) for
  the full reasoning.

- **`scripts/spider2_benchmark.py`** — Spider 2.0 benchmark runner.
  Configures `LLMConfig`, `AgentConfig(recursion_limit=12, min_retrievals=1)`,
  forces `executor._graph_recursion_limit = 24` (LangGraph limit is
  `max(self._graph_recursion_limit, agent_config.recursion_limit)` and
  defaults to 50, so AgentConfig alone is ignored), sets
  `executor._max_schema_summary_length = 60_000` to compress huge schemas
  into the 3-tier fallback, truncates external knowledge docs to 20K chars,
  and disables Qwen3 thinking mode via
  `chat_template_kwargs={"enable_thinking": False}`.

- **`scripts/enrich_dce.py`** — Standalone DCE enrichment script that calls
  vLLM via a custom `VLLMDescriptionProvider` instead of the default Ollama
  provider. Uses `Qwen/Qwen3-32B-AWQ` for description generation. Now also
  exercises the new critique pass automatically because the critique runs
  inside the shared `_get_enriched_table` code path.

---

## Open work

- Re-run full enrichment on all 28 Spider 2.0 databases with the critique
  pass enabled, and re-build the DCE vector index from the new YAMLs.
- Re-run the 20-question and 135-question Spider 2.0 benchmarks against the
  re-enriched DCE index to measure the score delta.
- Investigate `bank_sales_trading` (0/15 in Run 4 — largest failure
  category) — likely candidates for sibling-disambiguation wins.
- Consider a follow-up critique-pass tweak to add explicit "prefer X over Y
  when..." preference language alongside the descriptive disambiguation.
