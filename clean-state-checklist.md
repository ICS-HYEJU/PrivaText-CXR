# Clean State Checklist

- [ ] `CXR_dp_medclip` container is running and reachable (`./init.sh` step 1).
- [ ] The standard startup path (`./init.sh`) still passes end-to-end.
- [ ] No `run_feature_eval.py` process is still running against a shared `--out_dir`
      (check `docker exec CXR_dp_medclip ps aux`) — never leave two writers on the
      same `eval_summary.json`.
- [ ] GPUs (`nvidia-smi`) are free of orphaned generation/eval processes from this
      session.
- [ ] Current progress is recorded in `claude-progress.md`.
- [ ] Feature state in `feature_list.json` reflects what is actually `passing` versus
      still `not_started`/`blocked` — no feature marked `passing` without a
      corresponding `evidence` entry.
- [ ] No half-finished step (e.g. "generated but not yet evaluated") is left
      undocumented in `session-handoff.md`.
- [ ] The next session can continue without manual repair — no stray temp scripts
      left only inside the container or only in `/tmp` scratch space if they're
      needed again (see `session-handoff.md` "Unverified path" for an open example
      of this from 2026-08-10).
