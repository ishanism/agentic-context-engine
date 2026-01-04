# GitHub Copilot Instructions for ACE Framework

This file provides guidance to GitHub Copilot when working with code in this repository.

## Repository Overview

This is the Agentic Context Engine (ACE) framework - a Python library that enables AI agents to learn from their execution feedback. The framework is based on the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618).

### Core Concepts

- **Skillbook**: Structured context store containing skills (strategy entries) with helpful/harmful counters
  - Uses TOON (Token-Oriented Object Notation) format for 16-62% token savings
  - `skillbook.as_prompt()` returns TOON format (for LLM consumption)
  - `str(skillbook)` returns markdown format (for human debugging)
- **Three Agentic Roles** sharing the same base LLM:
  - **Agent**: Produces answers using the current skillbook
  - **Reflector**: Analyzes errors and classifies skill contributions
  - **SkillManager**: Emits update operations to update the skillbook
- **Update Operations**: Incremental updates to the skillbook (ADD, UPDATE, TAG, REMOVE)

## Development Setup

### Package Installation

```bash
# For end users
pip install ace-framework

# For contributors (recommended - uses UV package manager)
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync  # Installs all dependencies (10-100x faster than pip)

# Run tests
uv run pytest

# Run examples
uv run python examples/litellm/simple_ace_example.py
```

## Code Style and Conventions

### Python Style

- **Python Version**: 3.12 (required for development)
- **Line Length**: 88 characters (Black default)
- **Formatting**: Use Black for automatic formatting
- **Type Hints**: Use type hints where possible
- **Docstrings**: Required for all public functions and classes
- **Imports**: Group standard library, third-party, and local imports

### Testing

- **Test Framework**: pytest and unittest
- **Test Location**: `tests/` directory
- **Test Naming**: `test_*.py` for files, `test_*` for functions
- **Markers**: Use `@pytest.mark.unit` and `@pytest.mark.integration`
- **Coverage**: Minimum 25% coverage (enforced in CI)

### Commands

```bash
# Format code
uv run black ace/ tests/ examples/

# Type checking
uv run mypy ace/

# Run tests
uv run pytest                        # All tests
uv run pytest tests/test_skillbook.py  # Specific file
uv run pytest -m unit                # Only unit tests
uv run pytest -m integration         # Only integration tests

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

## Module Structure

### Core Library (`ace/`)

- `skillbook.py`: Skill and Skillbook classes for context storage (TOON format)
- `updates.py`: UpdateOperation and UpdateBatch for incremental updates
- `roles.py`: Agent, Reflector, SkillManager implementations
- `adaptation.py`: OfflineACE and OnlineACE orchestration loops
- `llm.py`: LLMClient interface with DummyLLMClient and TransformersLLMClient
- `prompts.py`: Default prompt templates (v1.0 - simple, for tutorials)
- `prompts_v2_1.py`: State-of-the-art prompts (v2.1 - RECOMMENDED for production)
- `features.py`: Centralized optional dependency detection

### LLM Providers (`ace/llm_providers/`)

- `litellm_client.py`: LiteLLM integration (100+ model providers)
- `langchain_client.py`: LangChain integration
- `instructor_client.py`: Instructor wrapper for robust JSON parsing

### Integrations (`ace/integrations/`)

Wrappers for external agentic frameworks:

- `base.py`: Base integration pattern and utilities
- `browser_use.py`: ACEAgent - browser automation with learning
- `claude_code.py`: ACEClaudeCode - Claude Code CLI with learning
- `langchain.py`: ACELangChain - wrap LangChain chains/agents
- `litellm.py`: ACELiteLLM - simple conversational agent

### Observability (`ace/observability/`)

- `opik_integration.py`: Enterprise-grade monitoring with Opik
- `tracers.py`: Automatic tracing decorators for all role interactions
- Automatic token usage and cost tracking for all LLM calls

## Key Implementation Patterns

### 1. Full ACE Pipeline (for new agents)

```python
# Sample → Agent → Environment → Reflector → SkillManager → Skillbook
from ace import OfflineACE, Agent, Reflector, SkillManager, Skillbook

skillbook = Skillbook()
agent = Agent(llm)
reflector = Reflector(llm)
skill_manager = SkillManager(llm)

adapter = OfflineACE(skillbook, agent, reflector, skill_manager)
results = adapter.run(samples, environment, epochs=3)
```

### 2. Integration Pattern (for existing agents)

```python
# External agent → Reflector → SkillManager
from ace import ACELangChain

ace_agent = ACELangChain(chain)
result = ace_agent.run("task description")  # Auto-learns from execution
```

### 3. Prompt Version Guidance

- **v1.0** (`prompts.py`): Simple, minimal - use for tutorials
- **v2.1** (`prompts_v2_1.py`): RECOMMENDED for production (+17% success rate)

```python
from ace.prompts_v2_1 import PromptManager

prompt_mgr = PromptManager()
agent = Agent(llm, prompt_template=prompt_mgr.get_agent_prompt())
```

### 4. Optional Dependencies

Check for optional dependencies before using:

```python
from ace.features import has_opik, has_instructor, get_available_features

if has_opik():
    # Use Opik observability
    pass

if has_instructor():
    # Use Instructor for robust JSON parsing
    pass
```

## Async Learning Mode

ACE supports asynchronous learning where the Agent returns immediately while Reflector and SkillManager process in the background:

```python
adapter = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True,           # Enable async mode
    max_reflector_workers=3,       # Parallel Reflector threads
)

results = adapter.run(samples, environment, epochs=3)
```

## Checkpoint Saving

OfflineACE supports automatic checkpoint saving during training:

```python
results = adapter.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=10,  # Save every 10 samples
    checkpoint_dir="./checkpoints"
)
```

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(llm): add support for new LLM provider
fix(adapter): resolve memory leak in online mode
docs(readme): update installation instructions
```

## Environment Variables

Set API keys for LLM providers:

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-api-key"
```

## Testing Guidelines

- Write tests for new features
- Ensure all tests pass before submitting PR
- Use meaningful test names
- Keep tests focused and independent
- Mock external API calls in unit tests

## Common Patterns to Follow

### Error Handling

Use tenacity for retries with LLM calls:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def llm_call():
    # Make LLM call
    pass
```

### Pydantic Models

Use Pydantic for data validation:

```python
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    field: str = Field(..., description="Field description")
```

### Optional Dependency Imports

Always check for optional dependencies:

```python
from ace.features import has_opik

if has_opik():
    from opik import Opik
else:
    # Fallback behavior
    pass
```

## Files to NOT Modify

- `uv.lock`: Auto-generated dependency lock file (only update via `uv lock`)
- `.python-version`: Python version specification for UV
- Build artifacts in `dist/`, `build/`, `*.egg-info`

## Documentation

When adding new features:

1. Update relevant docstrings
2. Add example to `examples/` directory
3. Update README.md if it's a major feature
4. Update CLAUDE.md for AI coding assistants
5. Update this file if it affects development workflow
