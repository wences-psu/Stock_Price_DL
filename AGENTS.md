# Shared Project Instructions

These instructions apply to agents working in this repository. They are shared by both teammates so that VS Code and PyCharm work follows the same conventions.

## Communication and Style

- Do not use emojis in code, documentation, commit messages, or agent responses.
- Explain Git and project changes in beginner-friendly language.
- Keep changes focused on the requested task. Do not rewrite unrelated files.
- Prefer existing project patterns and preserve public names unless a change is necessary.
- Use Markdown headings, tables, and code blocks when they make instructions easier to scan.
- Mention assumptions and test results briefly after completing work.

## Project Practices

- The default branch is `main`.
- Work on a short-lived task branch such as `feature/model-evaluation`, `fix/data-loading`, or `docs/git-guide`.
- Do not edit `main` directly and do not commit directly to `main`.
- Use pull requests for merging into `main`; one teammate reviews before merge.
- Avoid simultaneous edits to the same notebook. Keep notebook execution order aligned with the README.
- Do not commit downloaded data, model outputs, generated artifacts, virtual environments, secrets, or API keys. Check `.gitignore` before adding unfamiliar files.
- Keep changes compatible with Windows, macOS, and Linux where practical.

## Safe Git Operations

When asked to perform Git work:

1. Identify the repository and run `git status --short --branch`.
2. Report the current branch and whether there are uncommitted changes before changing Git state.
3. For sync, pull, branch creation, commit, or push requests, state the proposed commands in plain language.
4. Never discard, overwrite, stash, reset, rebase, amend, or delete work without explicit approval when it could affect user changes.
5. Never force-push. Do not rewrite shared history or push `main` unless the user explicitly approves a specific exception.
6. Before committing, show the files that will be included and verify that no secrets, data, artifacts, or unrelated edits are staged.
7. Before pushing, confirm the branch name, remote, and commits that will be uploaded.
8. After the operation, run `git status` and summarize what happened.
9. If a conflict or unexpected state appears, stop and explain it instead of guessing.

For the complete beginner workflow and platform-specific instructions, read `docs/git-workflow.md`. For an on-demand Git assistant workflow, read `.github/skills/git-workflow/SKILL.md`.

## Validation

Use the narrowest relevant check after editing. For Markdown-only changes, verify links and headings. For notebook changes, inspect the diff and run the affected notebook or cells when practical. Report checks that could not be run.
