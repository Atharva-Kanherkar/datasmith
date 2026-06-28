# Research Notes

This repo is inspired by Meta FAIR's Autodata paper, `Autodata: An agentic data scientist to
create high quality synthetic data`, arXiv:2606.25996v2.

## What We Implement

The paper describes Autodata as an agentic data scientist that creates data, analyzes quality and
performance, extracts learnings, and updates the generation recipe. Agentic Self-Instruct is the
paper's practical weak-vs-strong implementation:

- challenger generates a candidate task
- weak solver attempts it
- strong solver attempts it
- judge evaluates candidate quality and weak/strong separation
- accepted examples should be solvable by the strong solver while exposing weaknesses in the weak
  solver

This package implements that inner loop as a provider-agnostic SDK.

The important implementation boundary is the weak/strong pressure test. The SDK can produce and
preserve candidate examples, solver attempts, judge feedback, rejection reasons, and accepted
examples. It does not train the weak solver or optimize the data scientist agent itself.

## What We Do Not Claim

- This is not an official Meta implementation.
- This does not reproduce Meta's training results.
- This does not include the paper's full meta-optimization of the data scientist agent.
- This is not tied to AgentClash.
- This does not guarantee generated examples are safe to train on without domain review.
- This does not redact production traces; users must remove secrets and personal data before
  ingestion.

## Improvements for Developers

Compared with classic Self-Instruct style scripts, this package focuses on:

- typed artifacts for accepted/rejected examples
- pluggable model interfaces instead of hard-coded providers
- OTLP and span JSONL ingestion
- deterministic local models for tests and tutorials
- CLI outputs that can feed fine-tuning, eval, RL, or custom review workflows

## Practical Gaps Found While Translating The Paper

The paper's loop assumes several components that are not universal in developer environments:

- A reliable judge. In practice, judge output is another model boundary. The SDK validates score
  ranges, quality labels, and tag shape before accepting a candidate.
- A known weak model. Many teams do not have a formal weak solver, but can use the deployed model,
  a cheaper model, a lower-compute decoding mode, or an older prompt.
- A strong solver budget. Strong solving can be expensive, so callers should tune rollouts and
  acceptance policy for the cost of their domain.
- Safe traces. OpenTelemetry spans can contain prompts, completions, headers, user IDs, and other
  sensitive values. The SDK preserves metadata by design; redaction belongs upstream.
- Dataset-level quality. This package scores one candidate at a time. Diversity, deduplication,
  contamination checks, and train/test split hygiene still need a downstream workflow.

## Use Cases That Fit The Loop

- Production agent evals: convert failed or suspicious traces into new eval cases, then generate
  nearby examples that expose the weak solver's current failure mode.
- Customer support automation: target refund, policy, entitlement, escalation, or tool-ordering
  cases where a deployed support bot gives plausible but wrong answers.
- Legal and compliance reasoning: create examples that are neither pure recall nor impossible legal
  hypotheticals, using a judge rubric and weak-rollout variance as quality signals.
- Coding assistants and tool-use agents: generate tasks where a strong solver can follow hidden
  constraints or tool results while a weaker model takes the obvious but incomplete path.
- Research and document QA: ground examples in papers, manuals, specs, or runbooks, then select
  prompts where stronger reasoning actually changes the answer.

## Related Work to Compare Against

- Self-Instruct and AgentInstruct style repos are useful for simple data bootstrapping, but usually
  lack weak/strong solver pressure testing.
- Eval harnesses can score data after creation, but are often not designed as a generative loop.
- Observability pipelines collect real production traces; this repo turns those traces into seed
  examples so synthetic data can target actual product failure modes.
