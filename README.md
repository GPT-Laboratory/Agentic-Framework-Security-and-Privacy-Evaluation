# Agentic Framework Benchmark Evaluation

This repository evaluates agentic frameworks on three benchmark suites:

- **ConfAIde** — for contextual privacy reasoning
- **DecodingTrust** — for trustworthiness (privacy benchmark only)
- **Agent Security Bench (ASB)** — for adversarial and security robustness in tool-using agents

The purpose of this project is to compare how different agentic frameworks behave under privacy-sensitive, trustworthiness-sensitive, and security-sensitive settings using a unified evaluation workflow.

---

## Benchmarks Used

### 1. ConfAIde

ConfAIde tests whether language models can reason about privacy in context, rather than only recognizing obviously sensitive information. It is organized into multiple tiers with increasing contextual difficulty.

**Tiers used:** `1`, `2a`, `2b`, `3`, `4`

**Metrics computed per tier:**

| Metric | Description |
|---|---|
| `privacy_safe_rate` | Rate of privacy-safe responses |
| `privacy_leak_rate` | Rate of responses leaking private info |
| `refusal_rate` | Rate of refusals |
| `utility_rate` | Rate of useful/helpful responses |

---

### 2. DecodingTrust

DecodingTrust is a broader trustworthiness benchmark. In this repository, **only the privacy benchmark** is used to compare how different agentic frameworks behave under privacy-sensitive prompts.

**Dataset variants used:** `context`, `one_shot`, `two_shot`, `five_shot`

---

### 3. Agent Security Bench (ASB)

ASB evaluates the security robustness of LLM-based agents under adversarial conditions.

**Scenarios used:** `clean`, `dpi`, `opi`, `mp`, `mixed`, `pot`

---

## Repository Structure

A typical run consists of:

1. Loading tasks or samples from one benchmark
2. Running a framework-specific app or agent wrapper
3. Collecting responses
4. Evaluating responses with benchmark-specific logic
5. Saving per-sample results as `.jsonl` and aggregate summaries as `.json`

**Output folders:**

```
outputs_confaide_<framework>/
outputs_decodingtrust_<framework>/
outputs_<framework>/
```

**Typical local directory layout:**

```
crewai_asb_eval/
├── ASB/
├── confAIde/
├── DecodingTrust/
├── src/
├── requirements.txt
└── ...
```

---

## Getting Started

### Step 1 — Clone this repository

```bash
git clone <your-repo-url>
cd crewai_asb_eval
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a framework package is missing, install it separately (see Step 6).

### Step 4 — Clone the benchmark repositories

The runners read benchmark files directly from local clones.

```bash
# Clone ConfAIde
git clone https://github.com/skywalker023/confAIde.git

# Clone DecodingTrust
git clone https://github.com/AI-secure/DecodingTrust.git

# Clone ASB
git clone https://github.com/agiresearch/ASB.git
```

After cloning, your project should look like this:

```
crewai_asb_eval/
├── ASB/
├── confAIde/
├── DecodingTrust/
├── src/
├── requirements.txt
└── ...
```

### Step 5 — Set your OpenAI API key

```bash
# Set the OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Optionally set the model
export OPENAI_MODEL_NAME="gpt-4o-mini"
```

Verify the key is loaded:

```bash
python - <<'PY'
import os
print("OPENAI_API_KEY is set" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY is NOT set")
print("Model:", os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
PY
```

### Step 6 — Install the framework you want to evaluate

Install only the framework you plan to use for that run.

```bash
# SuperAGI
pip install superagi

# AutoGen
pip install autogen-agentchat autogen-ext[openai]

# CrewAI
pip install crewai
```

---

## Running ConfAIde

ConfAIde evaluates contextual privacy reasoning across tiers: `1`, `2a`, `2b`, `3`, `4`.

### Setup

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL_NAME="gpt-4o-mini"
```

### Run a small test first

```bash
python -m src.run_confaide_superagi_eval --confaide-root ./confAIde --tier 1 --sample-limit 5 --model gpt-4o-mini
```

### Run a single tier

```bash
python -m src.run_confaide_superagi_eval --confaide-root ./confAIde --tier 1 --model gpt-4o-mini
```

### Run all tiers individually

```bash
python -m src.run_confaide_superagi_eval --confaide-root ./confAIde --tier 2a --model gpt-4o-mini
python -m src.run_confaide_superagi_eval --confaide-root ./confAIde --tier 2b --model gpt-4o-mini
python -m src.run_confaide_superagi_eval --confaide-root ./confAIde --tier 3  --model gpt-4o-mini
python -m src.run_confaide_superagi_eval --confaide-root ./confAIde --tier 4  --model gpt-4o-mini
```

### Run all tiers at once

```bash
python -m src.run_confaide_superagi_eval --confaide-root ./confAIde --tier all --model gpt-4o-mini
```

### Inspect outputs

Results are saved in `outputs_confaide_superagi/`:

```bash
ls outputs_confaide_superagi
cat outputs_confaide_superagi/tier_1_summary.json
```

---

## Running DecodingTrust (Privacy Benchmark)

Only the privacy benchmark from DecodingTrust is used here. Dataset variants: `context`, `one_shot`, `two_shot`, `five_shot`.

### Setup

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL_NAME="gpt-4o-mini"
```

### Run a small test first

```bash
python -m src.run_dt_privacy_superagi_eval --dt-root ./DecodingTrust --dataset one_shot --limit 5 --model gpt-4o-mini
```

### Run one dataset fully

```bash
python -m src.run_dt_privacy_superagi_eval --dt-root ./DecodingTrust --dataset one_shot --model gpt-4o-mini
```

### Run all datasets individually

```bash
python -m src.run_dt_privacy_superagi_eval --dt-root ./DecodingTrust --dataset context   --model gpt-4o-mini
python -m src.run_dt_privacy_superagi_eval --dt-root ./DecodingTrust --dataset one_shot  --model gpt-4o-mini
python -m src.run_dt_privacy_superagi_eval --dt-root ./DecodingTrust --dataset two_shot  --model gpt-4o-mini
python -m src.run_dt_privacy_superagi_eval --dt-root ./DecodingTrust --dataset five_shot --model gpt-4o-mini
```

### Run all datasets in a loop

```bash
for d in context one_shot two_shot five_shot; do
  python -m src.run_dt_privacy_superagi_eval --dt-root ./DecodingTrust --dataset "$d" --model gpt-4o-mini
done
```

### Inspect outputs

Results are saved in `outputs_decodingtrust_superagi/`:

```bash
ls outputs_decodingtrust_superagi
cat outputs_decodingtrust_superagi/decodingtrust_superagi_privacy_one_shot_summary.json
```

### Metrics

| Metric | Description |
|---|---|
| `safe_behavior_rate` | Rate of privacy-safe responses |
| `pii_leak_rate` | Rate of personally identifiable info leaks |
| `refusal_rate` | Rate of refusals |
| `generic_pii_detected_rate` | *(updated runners)* Generic PII detection rate |
| `dataset_value_leak_rate` | *(updated runners)* Dataset value leak rate |
| `name_like_leak_signal_rate` | *(updated runners)* Name-like leak signal rate |

---

## Running ASB

ASB evaluates security robustness of LLM agents. Scenarios: `clean`, `dpi`, `opi`, `mp`, `mixed`, `pot`, `all`.

### Setup

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL_NAME="gpt-4o-mini"
```

### Run a small test first

```bash
python -m src.run_superagi_asb_eval --asb-root ./ASB --scenario clean --task-limit 5 --model gpt-4o-mini
```

### Run one scenario fully

```bash
python -m src.run_superagi_asb_eval --asb-root ./ASB --scenario clean --model gpt-4o-mini
```

### Run all scenarios

```bash
python -m src.run_superagi_asb_eval --asb-root ./ASB --scenario all --model gpt-4o-mini
```

### Run with a specific agent name (if required by the ASB loader)

```bash
python -m src.run_superagi_asb_eval --asb-root ./ASB --agent-name superagi --scenario all --model gpt-4o-mini
```

### Inspect outputs

Results are saved in `outputs_superagi/`:

```bash
ls outputs_superagi
cat outputs_superagi/clean_summary.json
cat outputs_superagi/dpi_summary.json
```

ASB evaluates whether the agent resists malicious instructions, avoids unsafe tool use, and preserves safe task behavior under attack scenarios. The evaluator aggregates per-task records into scenario summaries for comparison across frameworks.
