# AGENTS.md

## Agent skills

### Issue tracker

Issues live as GitHub issues, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, with label strings matching their names. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Servers

Shared lab GPU servers: which hosts, how to connect, GPU-sharing rules, remote-operation pitfalls, where envs/data live, how large files get in, MATLAB. Read before touching any server. See `docs/agents/servers.md`. How code gets onto servers and where results go: `docs/agents/experiments.md`.

### Experiments

Running an experiment (writing anything under `runs/<id>/`: preds, intermediate results, TensorBoard logs, notes): first read `workbench/RECORDS.md` and follow it. Don't read `workbench/README.md` for this; it documents the workbench UI.

Running anything on a server (code, branches, worktrees, launching, wrapping up a ticket): read `docs/agents/experiments.md`. Code must be committed before it runs; never run from `git archive` copies.
