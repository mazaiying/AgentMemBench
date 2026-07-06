# MemDialogue v2 Dataset Card

MemDialogue v2 is a retrieval benchmark for persistent agent-memory systems.
Each row is a normalized memory event with a retrieval question and a concise
reference answer. The release does not include source conversation turns.

## Source and license

- Source: `allenai/WildChat-4.8M`, public non-toxic release
- Pinned source revision:
  `c827c6df8fcf008219ffaffa4d1dd77491099367`
- Source license: ODC-By-1.0
- Benchmark code license: MIT

Users must retain WildChat attribution and comply with ODC-By-1.0. See
`DATA_USE.md` for the attribution notice and usage conditions. The source
license does not replace privacy, ethics, or institutional-review obligations.

## Record schema

Each JSONL row contains:

- `record_id`: deterministic hash of the source identifier and normalized event
- `session_id`: benchmark session identifier
- `source_id`: source conversation hash used for grouping, not source text
- `source_dataset`, `source_license`
- `annotator_model`, `annotator_backend`, `prompt_version`, `verifier_version`
- `memory_events[0].event_type`: `PERSONAL_FACT` or `TASK_REQUEST`
- `memory_events[0].raw_text`: normalized memory beginning with `The user`
- `memory_events[0].query`: non-binary retrieval question
- `memory_events[0].ground_truth`: concise answer
- `memory_events[0].evidence_turn_indices`: indices in the non-released source
- `memory_events[0].release_verified`: binary release-gate result

The exact event counts, rejection counts, source revision, annotator mixture,
and filter configuration are recorded in `data/memdialogue_v2_meta.json`.
The deterministic release audit and file checksums are published as
`data/memdialogue_v2_audit.json` and `data/SHA256SUMS`.

## Construction

The builder streams English, non-toxic WildChat conversations and preselects
turns with signals for benign preferences, project context, or reusable task
requests. A local language model proposes normalized events. Deterministic
filters then reject:

- direct identifiers and identifier-like numeric strings;
- health, financial, legal, religious, political, sexual, employment,
  education, relationship, family, location, and emotional-distress content;
- prompt injection, jailbreak text, role-play, fiction, and adult content;
- unsupported or assistant-only evidence;
- binary or unanswerable questions, invalid categories, and duplicate events.

A separate model call verifies support, later-session usefulness, safety, and
answerability. Task-request answers are deterministically derived from the
normalized request so that the benchmark does not copy an assistant's solution
into the reference answer.

## Recommended splits and evaluation

Split and sample by `source_id`. Do not place records derived from the same
source conversation in different train, validation, or test partitions.
AgentMemBench retrieval evaluation uses a fixed-seed, source-unique,
event-stratified sample and releases per-example judge decisions.

## Intended uses

- retrieval and omission evaluation for agent-memory middleware;
- temporal-update, deletion, isolation, concurrency, and scale workloads;
- comparison of memory architectures under controlled model and hardware
  configurations.

The dataset is not intended for identifying WildChat users, reconstructing
source conversations, training user-profiling systems, or making decisions
about individuals.

## Limitations and residual risk

The data is English-only and model-normalized. Automated filters can miss
sensitive context, and local annotators can introduce linguistic or
model-specific bias. Source hashes and evidence indices support auditing but
do not make the released records equivalent to human annotations. Users should
report questionable records through the repository issue tracker; maintainers
should remove confirmed unsafe records and publish a new versioned checksum.

Nine task records retain public technical or website URLs needed to describe
the requested operation. The release scan found no email addresses, telephone
numbers, credentials, or direct contact fields in those records. One additional
record contains the wildcard development address `0.0.0.0`.

This GitHub release includes deterministic filtering and independent model
verification but does not claim a completed human audit. A future archival
dataset version should add a two-author stratified audit reporting agreement,
answerability, support, and safety rates.
