---
name: git-workflow
description: "Use when a teammate asks an agent to safely inspect Git status, sync main, create a branch, commit, push, prepare a pull request, undo local changes, or resolve a Git conflict in this repository."
---

# Safe Git Workflow

Use this skill for Git operations in the Stock_Price_DL repository. The user should understand what will happen before Git state changes.

## Required Safety Checks

Before any mutating Git command:

```bash
git status --short --branch
git remote -v
git branch --show-current
```

Explain the current branch and whether there are uncommitted changes. Stop and ask before discarding, stashing, resetting, rebasing, amending, deleting, or overwriting user work. Never force-push. Never push directly to `main`.

Before a commit:

```bash
git status
git diff --staged
```

Confirm that only intended files are staged and that there are no secrets, data files, artifacts, environments, or unrelated edits. After every operation, run `git status` and summarize the result.

## Common Requests

### Inspect status

Run `git status --short --branch`, explain the output in plain language, and make no changes unless requested.

### Safely update local main

First inspect status and current branch. If there are uncommitted changes, stop and explain that pulling may create complications. With a clean working tree:

```bash
git switch main
git pull --ff-only origin main
```

Use `--ff-only` so the agent does not silently create a merge commit. If it fails, report the reason and stop.

### Create a task branch

After updating `main`:

```bash
git switch -c <type>/<short-description>
```

Use `feature/`, `fix/`, `docs/`, or `experiment/`. Confirm the new branch with `git status --short --branch`.

### Commit changes

Inspect the diff and stage only requested files:

```bash
git diff
git add <specific-files>
git diff --staged
git commit -m "<imperative message>"
```

Do not use `git add .` without reviewing the resulting status first. Do not commit if the staged content contains secrets, generated data, artifacts, or unrelated work.

### Push a branch

Confirm the current branch is not `main`, then inspect commits that are ahead of the remote. For a new branch:

```bash
git push -u origin <branch-name>
```

For an existing branch:

```bash
git push
```

Report the branch name and remote after pushing. Do not create or merge a pull request unless the user requests it.

### Prepare a pull request

Run:

```bash
git status
git log --oneline origin/main..HEAD
```

Summarize the commits, tests, and files changed. Tell the user to open a pull request into `main`, request the teammate's review, and merge only after approval.

### Clean up after merge

Only after the user confirms that the pull request was merged:

```bash
git switch main
git pull --ff-only origin main
git branch -d <branch-name>
git fetch --prune
```

If the local branch cannot be deleted because it contains unmerged commits, stop and ask for confirmation. Never use `-D` automatically.

### Undo or recover

For uncommitted changes, explain that `git restore <file>` discards edits and ask for confirmation first. To unstage without discarding edits, use `git restore --staged <file>`. Do not reset or clean the repository automatically.

For conflicts, run `git status`, list the conflicted files, and explain the conflict markers. Do not choose one side automatically. Help the user inspect and resolve each file, then stage the resolution and finish the operation Git requests. Offer `git merge --abort` only when the user wants to cancel the merge.

## Platform Notes

Git commands are the same in Windows PowerShell, Git Bash, macOS Terminal, and Linux shells. Do not assume a Unix-only command is available. Prefer Git commands and repository-relative paths. If a command depends on a shell, identify the Windows, macOS, or Linux version before running it.

## Response Format

For each request:

1. State what the requested Git operation will do.
2. Show or summarize the safety checks.
3. Run only the approved operation.
4. Report the result, current branch, and any next step.

Refer the user to `docs/git-workflow.md` for the complete beginner guide, VS Code and PyCharm instructions, and platform notes.
