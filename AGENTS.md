# Persistent Research Memory Protocol

This workspace is a long-running research project. Any Codex model, skill, or sub-agent that works here must treat workspace files as persistent memory.

Before substantial work, read these files in order:

1. `00_memory/START_HERE.md`
2. `00_memory/global-context.md`
3. `research-session.json`
4. `08_logs/coordinator-summary.md`
5. Relevant phase guides listed in `00_memory/memory-index.yaml`

Operating rules:

- Do not rely on chat history as the source of truth when workspace memory files exist.
- Update the relevant phase guide whenever scope, evidence, decisions, assumptions, blockers, artifacts, or next actions change.
- Update `08_logs/coordinator-summary.md` after each important integration, delegation, or scientific decision.
- Update `00_memory/global-context.md` only when the cross-phase project state changes.
- For literature work, query local RAG before online discovery.
- Newly acquired full texts must be tracked, normalized, and ingested into the local retrieval workflow before they are treated as stable manuscript evidence.
- Existing runnable code at the repository root remains the canonical simulation package unless a deliberate migration is approved.
