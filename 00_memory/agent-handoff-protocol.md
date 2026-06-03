# Agent Handoff Protocol

Any sub-agent or future session working in this repository must:

1. Read `00_memory/START_HERE.md`, `00_memory/global-context.md`, `research-session.json`, and `08_logs/coordinator-summary.md` before making decisions.
2. Read the phase guide related to its task.
3. Write outputs into the standard workspace folders, not into ad hoc paths.
4. Report exact artifact paths in its final handoff.
5. Update its owned phase guide before returning.
6. Update `08_logs/coordinator-summary.md` if it changes research state, claim framing, experiment status, or next actions.

## Evidence Rules

- Literature claims require local RAG or stored paper evidence.
- Simulation claims require local run artifacts or reproducible commands.
- Manuscript claims must map to either literature evidence, experiment artifacts, or explicitly labeled inference.

## Default Language

- Coordination: Bahasa Indonesia.
- Manuscript-facing prose: academic English unless requested otherwise.
