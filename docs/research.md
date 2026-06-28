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

## What We Do Not Claim

- This is not an official Meta implementation.
- This does not reproduce Meta's training results.
- This does not include the paper's full meta-optimization of the data scientist agent.
- This is not tied to AgentClash.

## Improvements for Developers

Compared with classic Self-Instruct style scripts, this package focuses on:

- typed artifacts for accepted/rejected examples
- pluggable model interfaces instead of hard-coded providers
- OTLP and span JSONL ingestion
- deterministic local models for tests and tutorials
- CLI outputs that can feed fine-tuning, eval, RL, or custom review workflows

## Related Work to Compare Against

- Self-Instruct and AgentInstruct style repos are useful for simple data bootstrapping, but usually
  lack weak/strong solver pressure testing.
- Eval harnesses can score data after creation, but are often not designed as a generative loop.
- Observability pipelines collect real production traces; this repo turns those traces into seed
  examples so synthetic data can target actual product failure modes.
