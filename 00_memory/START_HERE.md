# Start Here: Persistent Research Memory

This file is the required entry point for every model, skill, or sub-agent continuing this research.

## Current Project Snapshot

- Workspace root: `/home/johanericka/phenology-fuzzy`
- Project name: `phenology-fuzzy`
- Current phase: existing simulation-code project initialized as a structured research workspace.
- Working topic: phenology-aware fuzzy irrigation control for paddy rice using BMKG weather data, dynamic phenological soil-moisture target bands, and closed-loop simulation with AquaCrop/water-balance support.
- Target output: journal-style research article or dissertation module after evidence, experiment protocol, and result artifacts are aligned.
- Working language: Bahasa Indonesia for coordination; English for manuscript-facing artifacts unless requested otherwise.
- RAG toolkit: `/home/johanericka/RAG`
- RAG collection: `research_papers_v1`
- RAG query command: `python /home/johanericka/RAG/rag_ask.py "QUESTION" --collection research_papers_v1 --top-k 6`
- RAG retrieval-only command: `python /home/johanericka/RAG/rag_ask.py "QUESTION" --collection research_papers_v1 --retrieve-only --top-k 10`
- RAG ingest command for PDFs: `python /home/johanericka/RAG/rag_ingest_pdf.py PDF_PATH --collection research_papers_v1 --tags "rice,irrigation,phenology,fuzzy-control,aquacrop"`

## Required Read Order

1. `00_memory/global-context.md`
2. `00_memory/agent-handoff-protocol.md`
3. `00_memory/memory-index.yaml`
4. `00_memory/researcher-style.md`
5. `research-session.json`
6. `08_logs/coordinator-summary.md`
7. Relevant phase guides from the phase guide map below

## Phase Guide Map

- Intake and global scope: `01_topic/topic.md`
- Literature search, screening, and evidence status: `02_literature/literature.md`
- Subtopic clustering and drill-down history: `03_subtopics/subtopics.md`
- Candidate gaps and selected gap rationale: `04_gaps/gaps.md`
- Experiment design: `05_experiments/plans/current-plan.md`
- Results and analysis: `05_experiments/analysis/results-summary.md`
- Manuscript planning and claim-to-evidence map: `06_manuscript/contribution-statement.md`
- Coordinator cross-phase summary: `08_logs/coordinator-summary.md`

## Update Contract

- If a decision changes the whole project, update `00_memory/global-context.md` and `08_logs/coordinator-summary.md`.
- If a change belongs to one phase, update that phase guide and then summarize the change in `08_logs/coordinator-summary.md`.
- If a task creates an artifact, record its path in the relevant phase guide and coordinator summary.
- If a task identifies missing data, blocked access, weak evidence, or unresolved methodology, record it as a blocker or open decision.
