# GitHub Copilot Loop 🔄

![GitHub stars](https://img.shields.io/github/stars/kayba-ai/agentic-context-engine?style=social)
[![Discord](https://img.shields.io/discord/1429935408145236131?label=Discord&logo=discord&logoColor=white&color=5865F2)](https://discord.gg/mqCqH7sTyK)
[![Twitter Follow](https://img.shields.io/twitter/follow/kaybaai?style=social)](https://twitter.com/kaybaai)
[![PyPI version](https://badge.fury.io/py/ace-framework.svg)](https://badge.fury.io/py/ace-framework)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**GitHub Copilot CLI that learns from itself**

Run GitHub Copilot CLI in a continuous loop. After each run, ACE (our open-source framework for agents that learn from execution feedback) analyzes what worked and what failed, then injects those learnings into the next iteration. Walk away and come back to finished work.

---

## 🚀 Quick Start

Simple setup: clone the repo, add your API key, write a prompt, and run.

### 1. Clone

```bash
git clone https://github.com/kayba-ai/agentic-context-engine.git
cd agentic-context-engine/examples/copilot-loop
```

### 2. Setup

```bash
# Install GitHub Copilot CLI extension
gh extension install github/gh-copilot

# Add your API key for ACE learning to .env.copilot
./setup.sh  # Initialize workspace
```

`setup.sh` copies `workspace_template/` to `workspace/` and initializes it as a git repo. Copilot runs inside `workspace/` and is constrained to that directory. If you want to work on an existing codebase, put it in `workspace_template/` first.

### 3. Define Your Task

Edit `prompt.md` with your task (see [Prompt Tips](#-prompt-tips) for guidance).

### 4. Run

```bash
uv run python copilot_loop.py
```

GitHub Copilot starts working in `workspace/` and learned skills get stored in `skillbook/`.

You can stop anytime with `Ctrl+C` and resume later with `uv run python copilot_loop.py` - it picks up where it left off. We recommend leaving it running until stall detection kicks in (no new commits for 4 iterations) or you're happy with the result.

### 5. Reset

Run this when starting a new task or trying a different prompt (workspace and skillbook get archived to logs).

```bash
./setup.sh
```

---

## 💳 What You Need

- **GitHub Copilot:** GitHub Copilot subscription or access via GitHub account
- **Learning loop:** API key for ACE learning model (~$0.01-0.05 per iteration with gpt-4o-mini)

---

## 💡 Prompt Tips

**Example prompt that worked well:**

- **Task definition:** "Your job is to [task]" - describe what you want accomplished
- **Commit after edits:** "Make a commit after every single file edit" - enables stall detection (loop stops after 4 iterations with no commits)
- **.agent/ directory:** "Use .agent/ as scratchpad. Store long term plans and todos there" - Copilot tracks its own progress
- **.env file:** "The .env file contains API keys" - add keys to `workspace_template/.env` if Copilot needs them to test your task
- **Time allocation:** "Spend 80% on X, 20% on Y" - specify focus split to balance implementation and verification
- **Continuation:** "When done, improve code quality" - keeps the loop productive instead of stopping early

```markdown
Your job is to create a REST API with authentication using Flask.

Make a commit after every single file edit.

Use .agent/ directory as scratchpad for your work. Store long term plans and todo lists there.

The .env file contains API keys for running examples.

Spend 80% of time on implementation, 20% on testing.

When implementation is complete, improve code quality and fix any issues.
```

---

## 🔄 How It Works

```
Run → Reflect → Learn → Loop
 │       │         │       │
 │       │         │       └── Restart with learned skills
 │       │         └── SkillManager updates skillbook
 │       └── Reflector analyzes execution trace
 └── Copilot executes prompt.md
```

Each iteration builds on previous work. Skills compound over time.

---

## 📁 Files

| File                  | What it does                              |
| --------------------- | ----------------------------------------- |
| `.env.copilot`        | Your API key (edit this)                  |
| `prompt.md`           | Your task (edit this)                     |
| `copilot_loop.py`     | Main loop script                          |
| `workspace_template/` | Your codebase + .env (copied on reset)    |
| `workspace/`          | Where Copilot works                       |
| `.data/skillbooks/`   | Learned strategies (archived on reset)    |
| `setup.sh`            | Initialize/reset workspace                |

---

## ⚙️ Environment Variables

Set in `.env.copilot` file:

| Variable            | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` | Required: API key for ACE learning model |
| `AUTO_MODE`         | `true` (default) runs fully automatic, `false` prompts between iterations |
| `ACE_MODEL`         | Model for learning (default: gpt-4o-mini)                  |

---

## 🤝 Contributing

Contributions welcome! See our [Contributing Guide](../../CONTRIBUTING.md).

---

## 📝 License

MIT License - see [LICENSE](../../LICENSE) for details.

---

## 🔗 Links

- [ACE Framework Documentation](https://github.com/Kayba-ai/agentic-context-engine)
- [GitHub Copilot CLI Documentation](https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli)
- [Discord Community](https://discord.gg/mqCqH7sTyK)
