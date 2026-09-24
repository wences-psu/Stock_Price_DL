# Git Workflow for Our Two-Person Team

This guide keeps collaboration predictable. The short version is: update your local copy, create a branch for one task, save small commits, push the branch, open a pull request, get a review, and then merge into `main`.

## Our Team Rules

- `main` should always be in a usable state. Do not work directly on it.
- Use one short-lived branch per task, such as `feature/add-model-metrics` or `docs/update-readme`.
- Pull requests are the normal way to merge work into `main`.
- One teammate reviews the pull request before it is merged.
- Keep commits focused and describe what changed.
- Do not commit passwords, API keys, local environments, downloaded data, model files, or generated artifacts. Check `.gitignore` when unsure.
- Avoid editing the same notebook at the same time. Notebook conflicts are harder to resolve than ordinary text-file conflicts.
- Never use `git push --force` on a shared branch.

If Git reports something unexpected, stop and check `git status` before running another command.

## First-Time Setup

### Install Git

- **Windows:** Install Git from [git-scm.com](https://git-scm.com/download/win). Git Bash and PowerShell both work.
- **macOS:** Install Xcode Command Line Tools with `xcode-select --install`, or install Git from [git-scm.com](https://git-scm.com/download/mac).
- **Linux:** Install Git with your distribution's package manager, for example `sudo apt install git` on Ubuntu/Debian.

Git commands are almost the same on Windows, macOS, and Linux. The main differences are the terminal and file paths. The examples below work in PowerShell, Git Bash, macOS Terminal, and most Linux shells.

### Clone the project

Clone only once on each computer:

```bash
git clone https://github.com/wences-psu/Stock_Price_DL.git
cd Stock_Price_DL
```

Confirm that Git knows the remote repository and current branch:

```bash
git remote -v
git status
```

The remote is usually named `origin`, and the default branch is `main`.

## The Normal Daily Workflow

### 1. Start from an up-to-date `main`

Do this before starting a new task:

```bash
git switch main
git pull origin main
```

`git switch main` moves you to the shared base branch. `git pull origin main` downloads the latest changes and applies them to your local `main`.

### 2. Create your task branch

Create a branch from the updated `main`:

```bash
git switch -c feature/short-description
```

Use a clear prefix such as `feature/`, `fix/`, `docs/`, or `experiment/`. The branch belongs to your task, not to a person.

### 3. Work and inspect your changes

Edit the project, run the relevant notebook or checks, then inspect what changed:

```bash
git status
git diff
```

`git status` shows changed and untracked files. `git diff` shows the actual edits that are not staged yet.

### 4. Save a small commit

Stage only the files that belong to this task:

```bash
git add README.md docs/git-workflow.md
git status
git commit -m "docs: add team Git workflow"
```

A commit is a named checkpoint in your local history. Use an imperative message that says what the commit does, such as `fix: handle missing price values` or `docs: explain notebook setup`.

To stage all intended changes in the current directory, you may use `git add .`, but always review `git status` before committing.

### 5. Push the branch

The first push connects your local branch to its remote branch:

```bash
git push -u origin feature/short-description
```

Later pushes from the same branch can use:

```bash
git push
```

### 6. Open and review a pull request

On GitHub, open a pull request from your branch into `main`. Explain:

- What changed.
- How you tested it.
- Anything the reviewer should pay attention to.

The other teammate reviews the files and leaves comments if needed. Make requested edits on the same branch, commit them, and push again. The pull request updates automatically.

### 7. Merge and clean up

After approval, merge the pull request on GitHub, preferably using squash merge for a clean history. Then update your local copy and delete the finished branch:

```bash
git switch main
git pull origin main
git branch -d feature/short-description
git fetch --prune
```

The `-d` option refuses to delete a branch that Git believes has unmerged work. That is a safety check.

## Useful Commands

| Command | Meaning |
|---|---|
| `git status` | Show the current branch and uncommitted changes. |
| `git branch` | List local branches; the current one has `*`. |
| `git switch main` | Move to the local `main` branch. |
| `git switch -c name` | Create and switch to a new branch. |
| `git log --oneline -5` | Show the five latest commits briefly. |
| `git diff` | Show unstaged edits. |
| `git diff --staged` | Show edits prepared for the next commit. |
| `git fetch --prune` | Download remote updates and remove stale remote-branch names locally. |
| `git pull origin main` | Update local `main` from GitHub. |
| `git push` | Upload commits from the current branch. |

## Safe Recovery

### I changed a file but have not committed it

To discard changes in one file, only after confirming you do not need them:

```bash
git restore path/to/file
```

To unstage a file while keeping its edits:

```bash
git restore --staged path/to/file
```

If the work matters, make a commit or copy it elsewhere before restoring. These commands can remove local edits.

### I committed locally but have not pushed

The simplest option is usually to keep the commit and continue. If the commit message is the only problem:

```bash
git commit --amend -m "better commit message"
```

Ask before changing commits that someone else may already have pulled.

### Git says there are conflicts

1. Run `git status` to see the conflicted files.
2. Open each file and choose the correct content between the conflict markers.
3. Remove all conflict markers (`<<<<<<<`, `=======`, and `>>>>>>>`).
4. Stage the resolved files: `git add path/to/file`.
5. Finish the operation with `git commit`, or use the command Git prints.
6. Run the relevant checks before pushing.

If the conflict is confusing, do not guess. Save a copy of your work and ask your teammate or an agent for help.

To cancel an in-progress merge when appropriate:

```bash
git merge --abort
```

## VS Code Source Control

1. Open the cloned `Stock_Price_DL` folder.
2. Select the Source Control icon in the Activity Bar.
3. Use the branch indicator in the bottom-left corner to switch to `main`, then use **Sync Changes** to pull updates.
4. Use **Create Branch** to create a task branch before editing.
5. Review changed files and the inline diff in Source Control.
6. Enter a commit message, stage the intended files with the `+` button, and select **Commit**.
7. Select **Publish Branch** or **Sync Changes** to push the branch.
8. Use the GitHub Pull Requests extension or GitHub in a browser to create and review the pull request.

The graphical actions run the same Git operations as the terminal commands. When unsure, check the Source Control view and `git status` in the integrated terminal.

## PyCharm Git

1. Open the cloned `Stock_Price_DL` folder as a project.
2. Use the branch widget in the bottom-right corner to check out `main`, then choose **Update Project** to pull changes.
3. Use the branch widget and choose **New Branch** before starting a task.
4. Open the **Commit** tool window to review diffs, select only the intended files, and enter a commit message.
5. Select **Commit and Push** to upload the branch, or commit first and push later from the **Git** menu.
6. Use **Git > GitHub > Create Pull Request**, if available, or open the pull request in a browser.
7. After merging, check out `main`, update the project, and delete the local finished branch from the branch widget.

PyCharm may ask which files to include in a commit. Treat that list like `git status`: do not select generated data, artifacts, or unrelated changes.

## Working Across Windows, macOS, and Linux

The Git workflow is shared across all three systems. Use the terminal built into your editor if that is easiest. Be aware of these small differences:

- Windows PowerShell commonly uses paths like `C:\Users\Name\Stock_Price_DL`; Git commands still use forward slashes in repository paths when convenient.
- macOS and Linux commonly use paths like `/Users/name/Stock_Price_DL` or `/home/name/Stock_Price_DL`.
- Windows users may use Git Bash for Unix-like commands. PowerShell commands such as `Get-Location` are not Git commands and are not needed for this guide.
- Do not commit virtual environments. Keep environment setup local to each computer.
- Run notebooks in the same order documented in the root README. Keep large downloaded data and generated artifacts out of Git unless the team explicitly decides otherwise.

## Asking an Agent for Git Help

You can ask an agent to explain or perform a Git step. Be specific about the intended operation and repository. Examples:

- `Please show my Git status and explain what each changed file means. Do not modify anything.`
- `Please safely update my local main from origin/main. First inspect my status and stop if I have uncommitted work.`
- `Create a branch named docs/update-readme from the latest origin/main, but ask before discarding or stashing anything.`
- `Review the staged files and commit them with a suitable message. Do not push.`
- `Push my current branch and tell me the pull request steps. Do not push main.`
- `Help me resolve this merge conflict, but show me the conflicting files before changing them.`

The shared `AGENTS.md` file and the Git skill at `.github/skills/git-workflow/SKILL.md` describe the safety rules agents should follow. An agent can run commands, but you remain responsible for reviewing the result before accepting or pushing changes.
