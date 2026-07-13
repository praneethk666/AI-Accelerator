# Contributing — standards & workflow

Read this once before you push. The goal: many small, pluggable tools that snap
together because everyone follows the same shapes and rules.

## Branches & PRs
- Never commit to `main`. Branch: `feat/<yourname>-<short-task>` (e.g. `feat/alex-word-export`).
- Small, frequent commits with clear messages ("add pdf page profiler").
- Open a Pull Request into `main` with: what it does, how to run, a sample input + output.

## Secrets (this repo is PUBLIC)
- Never commit keys. Put them in a local `.env` (gitignored). `.env.example` lists the variables.
- No secrets, URLs, or magic numbers hardcoded in source — read them from config / `.env`.

## Code structure
- One module = one tool / one concern. Put it in the right `backend/<area>/` folder.
- Do NOT fork the shared contracts in `backend/core/` (schemas, Tool, config). Build to them.
- A tool is a function/class with **typed input and output** matching `backend/core/schemas.py`.

## Python standards
- Python 3.11+. Type hints on every public function. A one-line docstring (what / args / returns).
- Format with **black**; lint with **ruff**. Run both before pushing.
- Naming: `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE` constants.
- Add your dependencies to `requirements.txt` in your PR.

## Behavior
- Fail gracefully: append to `state["errors"]`, never raise in a way that kills the whole document.
- Keep external (vision/LLM) calls minimal, downscaled, cached, and only when needed.

## Tests
- Ship at least one runnable test or demo per tool, under `tests/`.
- The smoke test must keep passing: `python tests/test_smoke.py`.

## How tools talk
- Tools never call each other. They read what they need from `PipelineState` and write results back.
- The graph (config-driven) decides order and routing. Adding a tool = add a node + one config line.
