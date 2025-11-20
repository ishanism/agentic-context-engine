# 🧪 ACE Framework Testing Guide

Complete guide for testing ACE agents and validating performance.

## Testing Philosophy

ACE testing focuses on three key areas:
1. **Correctness**: Does the agent produce accurate answers?
2. **Learning**: Does the playbook improve over time?
3. **Robustness**: Does the system handle edge cases?

---

## Running Tests

### Quick Start

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_adaptation.py

# Run integration tests
uv run pytest tests/test_integration.py
```

### Using unittest

```bash
# Run all tests
python -m unittest discover -s tests

# Run specific test file
python -m unittest tests.test_adaptation

# Verbose output
python -m unittest discover -s tests -v
```

---

## Unit Testing

### Testing Playbook Operations

```python
import unittest
from ace import Playbook

class TestPlaybook(unittest.TestCase):
    def test_add_and_retrieve_bullet(self):
        playbook = Playbook()

        bullet = playbook.add_bullet(
            section="Test",
            content="Test strategy"
        )

        retrieved = playbook.get_bullet(bullet.id)
        self.assertEqual(retrieved.content, "Test strategy")

    def test_save_and_load(self):
        playbook = Playbook()
        playbook.add_bullet("Section", "Content")

        playbook.save_to_file("test.json")
        loaded = Playbook.load_from_file("test.json")

        self.assertEqual(len(loaded.bullets()), 1)
```

### Testing Generator

```python
from ace import Generator, DummyLLMClient, Playbook

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.client = DummyLLMClient()
        self.generator = Generator(self.client)
        self.playbook = Playbook()

    def test_generate_with_empty_playbook(self):
        output = self.generator.generate(
            question="What is 2+2?",
            context="",
            playbook=self.playbook
        )

        self.assertIsNotNone(output.final_answer)
        self.assertIsNotNone(output.reasoning)

    def test_generate_uses_playbook(self):
        bullet = self.playbook.add_bullet(
            section="Math",
            content="Show step-by-step work"
        )

        output = self.generator.generate(
            question="What is 10*5?",
            context="",
            playbook=self.playbook
        )

        # Generator should cite the bullet
        self.assertIn(bullet.id, output.bullet_ids)
```

### Testing Reflector & Curator

```python
from ace import Reflector, Curator, GeneratorOutput

class TestReflectorCurator(unittest.TestCase):
    def setUp(self):
        self.client = DummyLLMClient()
        self.reflector = Reflector(self.client)
        self.curator = Curator(self.client)
        self.playbook = Playbook()

    def test_reflection(self):
        gen_output = GeneratorOutput(
            reasoning="Solved math problem",
            final_answer="4",
            bullet_ids=[],
            raw={}
        )

        reflection = self.reflector.reflect(
            question="What is 2+2?",
            generator_output=gen_output,
            playbook=self.playbook,
            feedback="Correct answer"
        )

        self.assertIsNotNone(reflection.analysis)

    def test_curation(self):
        reflection = self.reflector.reflect(...)

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context="Math problem",
            progress="Task 1/10"
        )

        self.assertIsNotNone(curator_output.delta)
```

---

## Integration Testing

### End-to-End Learning Cycle

```python
from ace import OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult

class SimpleEnvironment(TaskEnvironment):
    def evaluate(self, sample, output):
        correct = sample.ground_truth in output.final_answer
        return EnvironmentResult(
            feedback="Correct" if correct else "Wrong",
            ground_truth=sample.ground_truth
        )

class TestLearningCycle(unittest.TestCase):
    def test_offline_adaptation(self):
        # Setup
        client = DummyLLMClient()
        generator = Generator(client)
        reflector = Reflector(client)
        curator = Curator(client)
        adapter = OfflineAdapter(generator, reflector, curator)

        # Training samples
        samples = [
            Sample("What is 2+2?", "", "4"),
            Sample("What is 3+3?", "", "6")
        ]

        # Run adaptation
        results = adapter.run(samples, SimpleEnvironment(), epochs=2)

        # Verify learning occurred
        self.assertGreater(len(adapter.playbook.bullets()), 0)
```

### Testing Checkpoints

```python
def test_checkpoint_saving(self):
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = OfflineAdapter(generator, reflector, curator)

        results = adapter.run(
            samples,
            environment,
            checkpoint_interval=2,
            checkpoint_dir=tmpdir
        )

        # Verify checkpoints exist
        checkpoints = os.listdir(tmpdir)
        self.assertGreater(len(checkpoints), 0)
        self.assertIn("latest.json", checkpoints)
```

---

## Testing Without API Calls

Use `DummyLLMClient` to avoid real API calls during tests:

```python
from ace import DummyLLMClient

# Returns predefined responses
client = DummyLLMClient()

# Use in tests
generator = Generator(client)
output = generator.generate("test question", "", playbook)
```

**Benefits:**
- No API costs
- Deterministic test results
- Fast execution
- No rate limits

---

## Performance Testing

### Benchmark Learning Speed

```python
import time

def test_learning_performance(self):
    start = time.time()

    results = adapter.run(samples, environment, epochs=3)

    duration = time.time() - start
    avg_per_sample = duration / len(samples)

    print(f"Processed {len(samples)} samples in {duration:.2f}s")
    print(f"Average: {avg_per_sample:.2f}s per sample")

    # Assert reasonable performance
    self.assertLess(avg_per_sample, 5.0)  # Less than 5s per sample
```

### Measure Playbook Growth

```python
def test_playbook_growth(self):
    initial_bullets = len(adapter.playbook.bullets())

    results = adapter.run(samples, environment)

    final_bullets = len(adapter.playbook.bullets())
    growth = final_bullets - initial_bullets

    print(f"Playbook grew by {growth} bullets")

    # Verify learning occurred
    self.assertGreater(growth, 0)
```

---

## Common Test Patterns

### Fixture Setup

```python
class TestACEComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run once for all tests"""
        cls.client = DummyLLMClient()

    def setUp(self):
        """Run before each test"""
        self.playbook = Playbook()
        self.generator = Generator(self.client)
        self.reflector = Reflector(self.client)
        self.curator = Curator(self.client)

    def tearDown(self):
        """Run after each test"""
        # Clean up temporary files
        import os
        if os.path.exists("test.json"):
            os.remove("test.json")
```

### Mocking External Services

```python
from unittest.mock import Mock, patch

def test_with_mock_llm(self):
    mock_client = Mock()
    mock_client.complete.return_value = Mock(text='{"answer": "42"}')

    generator = Generator(mock_client)
    output = generator.generate("test", "", playbook)

    mock_client.complete.assert_called_once()
```

---

## Continuous Integration

ACE includes pytest configuration and GitHub Actions workflow.

### Local CI Simulation

```bash
# Run full test suite
uv run pytest

# With coverage
uv run pytest --cov=ace --cov-report=html

# Type checking
uv run mypy ace/

# Code formatting check
uv run black --check ace/ tests/
```

### GitHub Actions

Tests run automatically on:
- Push to main
- Pull requests
- Release tags

See `.github/workflows/test.yml` for configuration.

---

## Troubleshooting Tests

### Tests Timeout

```python
# Reduce epochs/samples
results = adapter.run(samples[:5], environment, epochs=1)

# Use DummyLLMClient
client = DummyLLMClient()  # Instead of real LLM
```

### Flaky Tests

```python
# Use deterministic data
samples = [Sample("2+2", "", "4")]  # Not random

# Set random seeds
import random
random.seed(42)
```

### Import Errors

```bash
# Install test dependencies
uv sync  # Installs dev dependencies

# Or manually
pip install pytest pytest-cov
```

---

## Best Practices

1. **Use DummyLLMClient** for unit tests (fast, no API costs)
2. **Test one thing per test** (easier to debug failures)
3. **Clean up temp files** in tearDown()
4. **Mock external services** (databases, APIs)
5. **Set timeouts** for long-running tests
6. **Run tests frequently** during development

---

## Resources

- **Test Suite:** `tests/` directory
- **Integration Tests:** `tests/test_integration.py` (10 comprehensive tests)
- **CI Configuration:** `.github/workflows/test.yml`
- **Coverage Reports:** Run `uv run pytest --cov=ace`

---

**Need help with testing?** Join our [Discord](https://discord.gg/mqCqH7sTyK) or open a [GitHub issue](https://github.com/kayba-ai/agentic-context-engine/issues).
