# Notebook Hygiene

Notebooks store more than code: outputs, execution counts, and the kernel and Python version of whoever saved last. Committing these creates huge diffs and merge conflicts, so we strip them automatically.

## One-time setup (each person, each computer)

```bash
python -m pip install -r requirements.txt
nbstripout --install
git config core.autocrlf false
git config core.eol lf
```

`nbstripout --install` writes to your local `.git/config`, so every teammate must run it once. The `.gitattributes` file in the repo already tells Git which files to filter.

Editor settings:

- **VS Code:** set `files.eol` to `\n`.
- **PyCharm:** Settings > Editor > Code Style > Line separator: Unix and macOS (`\n`).

## What this prevents

| Problem | How it is prevented |
|---|---|
| Different kernel or Python version in each commit | `nbstripout` removes `kernelspec` and `language_info` metadata |
| Huge diffs from outputs and execution counts | `nbstripout` removes outputs when you commit |
| Phantom modified files from CRLF vs LF | `.gitattributes` forces LF; `core.autocrlf false` |
| Editor checkpoint files in Git | `.ipynb_checkpoints/` is ignored and no longer tracked |

Your local notebook still shows its outputs. Only the committed copy is stripped.

## Daily habits

1. Before working: `git fetch`, `git status`, then `git pull --ff-only` on `main`.
2. Create a task branch. Never commit on `main`.
3. Only one person edits a given notebook at a time. Put reusable logic in `src/`.
4. Keep a local clone on your own disk instead of working inside the Shared Drive.
5. If Git shows a notebook as modified and you did not change it, run `git diff` first. If the diff is empty, it is a stale timestamp, not a real change.
