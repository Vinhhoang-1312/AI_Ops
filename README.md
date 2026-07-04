# Agentic RAG-based IT Support

**Group 1, MLOps (DDM501.22), FSB**

This project is a minimal IT support copilot built around an agentic RAG workflow. It routes each support request, retrieves relevant TechQA knowledge, and then produces either a support response or a human-escalation summary.

## What It Does

* Routes tickets to `support_resolution_agent`, `human_escalation_agent`, or a fallback response for unsupported or unclear queries
* Retrieves context from local Weaviate using hybrid keyword + semantic search
* Supports optional hosted LLM response generation through OpenRouter
* Can send Slack notifications for human-escalation cases
* Can trace workflow runs into local Langfuse with router, retrieval, prompt, response, memory, and score events
* Provides both a Streamlit UI

## Main Components

* `router_agent/`: binary routing logic for support vs escalation
* `rag_pipeline/`: TechQA retrieval and Weaviate integration
* `agents/`: response generation for support resolution and escalation
* `agent_workflow/`: end-to-end orchestration and short-term memory
* `app/`: Streamlit and CLI entrypoints

## Running The App

The app auto-loads configuration from `.env`.

* Streamlit UI: `python -m streamlit run .\agentic-rag-based-it-support\streamlit_app.py`
* Seed Langfuse dataset: `python -m app.seed_langfuse_dataset --dataset it-support/end-to-end --create-only`
* Langfuse dataset experiment: `python -m app.langfuse_experiment_runner --dataset it-support/end-to-end --retriever-backend tfidf --router-backend heuristic --limit 3`

## Langfuse Observability and Evaluation

Langfuse is used as the observability and evaluation layer for this project. The
current local Langfuse server is already running at:

```text
http://localhost:3000
```

The local Langfuse Docker stack is in:

```text
C:\Users\romph\langfuse
```

That stack initializes:

* Organization: `Nguyen_Sy_Hung_Demo`
* Project: `demo-project`
* Project ID: `e7309e80-ea14-440b-881e-43dde25eb515`

The app connects to that project through `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-Qzzon308LFYvc80sMgDod6ingFRW5bj8
LANGFUSE_SECRET_KEY=sk-lf-nhlmimh_98KYEwqxaY-S8Yvcz9jjwpyB7SjtquBDi6v3XWIN
LANGFUSE_BASE_URL=http://localhost:3000
IT_SUPPORT_LANGFUSE_ENABLED=true
IT_SUPPORT_APP_VERSION=0.1.0
IT_SUPPORT_ENVIRONMENT=local
```

Install the Python dependency with the project requirements:

```powershell
cd D:\mse\nguyen_sy_hung_codebases\ai-ops\agentic-rag-based-it-support
& D:\mse\nguyen_sy_hung_codebases\ai-ops\.ovenv\Scripts\python.exe -m pip install -r requirements.txt
```

Check that the app can authenticate to Langfuse:

```powershell
& D:\mse\nguyen_sy_hung_codebases\ai-ops\.ovenv\Scripts\python.exe tests\check_langfuse_connectivity.py
```

Expected output:

```text
Langfuse is ready
```

### Tracing

Each `run_workflow(...)` call creates a Langfuse trace named `it-support-ticket`
when `IT_SUPPORT_LANGFUSE_ENABLED=true`.

The trace captures:

* Root trace input: `ticket_text`
* Router decision: selected agent, confidence, reason, signals
* TechQA retrieval: compact document IDs, titles, scores, source, category, snippets
* Rendered prompt
* Final support, escalation, or fallback response
* Memory load/update counts
* Basic scores: `has_final_response`, `retrieved_context_count`, `routed_to_human`

Run a traced CLI request:

```powershell
& D:\mse\nguyen_sy_hung_codebases\ai-ops\.ovenv\Scripts\python.exe -m app.cli_demo --retriever-backend tfidf --router-backend heuristic --ticket "I cannot access the VPN from home." --show-prompt
```

Then open Langfuse and inspect:

```text
Nguyen_Sy_Hung_Demo -> demo-project -> Traces -> it-support-ticket
```

Streamlit turns are also traced:

```powershell
& D:\mse\nguyen_sy_hung_codebases\ai-ops\.ovenv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### Evaluation Dataset

The repo includes a script to create and seed the Langfuse dataset
`it-support/end-to-end` with 20 end-to-end IT support cases.

Seed the dataset:

```powershell
& D:\mse\nguyen_sy_hung_codebases\ai-ops\.ovenv\Scripts\python.exe -m app.seed_langfuse_dataset --dataset it-support/end-to-end --create-only
```

The seeded dataset items use this shape:

```json
{
  "input": {
    "ticket_text": "I cannot access the VPN from home."
  },
  "expected_output": {
    "next_agent": "support_resolution_agent",
    "must_include_any": ["VPN", "network", "MFA", "credentials"]
  },
  "metadata": {
    "case_type": "access_support",
    "difficulty": "easy"
  }
}
```

### Dataset Experiments

Run the app against the Langfuse dataset and create a dataset run:

```powershell
& D:\mse\nguyen_sy_hung_codebases\ai-ops\.ovenv\Scripts\python.exe -m app.langfuse_experiment_runner --dataset it-support/end-to-end --retriever-backend tfidf --router-backend heuristic --run-name local-smoke --limit 3
```

For a full local run, remove `--limit 3`:

```powershell
& D:\mse\nguyen_sy_hung_codebases\ai-ops\.ovenv\Scripts\python.exe -m app.langfuse_experiment_runner --dataset it-support/end-to-end --retriever-backend tfidf --router-backend heuristic --run-name local-full
```

The experiment runner defaults to `--response-backend template` so local dataset
runs stay fast and repeatable. Use `--response-backend auto` or
`--response-backend openrouter` only when you intentionally want hosted LLM
responses during evaluation.

The experiment output stored in Langfuse includes:

* `output.selected_agent`
* `output.router`
* `output.retrieved_docs`
* `output.retrieved_context_count`
* `output.rendered_prompt`
* `output.final_response`

### LLM-as-a-Judge Evaluation

LLM-as-a-judge evaluators are configured on the Langfuse server side. The
client app only needs to produce traces and dataset experiment runs with stable
input/output fields.

For the existing server-side judge, map variables to:

* Input: dataset item `input.ticket_text`
* Output: experiment `output.final_response`
* Expected output or ground truth: dataset item `expected_output`
* Optional context: experiment `output.retrieved_docs`

The server-side evaluator can target either:

* Live traces named `it-support-ticket`
* Dataset runs under `it-support/end-to-end`

Use dataset runs for regression comparison after code, prompt, router, or
retrieval changes.
