#!/usr/bin/env python3
"""
Quick start example using Ollama with ACE framework.

This shows the minimal code needed to use ACE with local Ollama models.
Requires Ollama to be installed and running locally.
"""

import subprocess
from ace import (
    LiteLLMClient,
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
)


class SimpleEnvironment(TaskEnvironment):
    """Minimal environment for testing."""

    def evaluate(self, sample, generator_output):
        correct = sample.ground_truth.lower() in generator_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )


def check_ollama_running():
    """Check if Ollama is running and has models available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout
        return False, "No models found. Run 'ollama pull llama2' to get started."
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Ollama not found. Install from https://ollama.ai"
    except Exception as e:
        return False, f"Error checking Ollama: {e}"


def main():
    # Check if Ollama is available
    is_running, message = check_ollama_running()
    if not is_running:
        print(f"❌ {message}")
        print("\nTo get started:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull a model: ollama pull llama2")
        print("3. Verify: ollama list")
        return

    print("✅ Ollama is running")
    print("Available models:")
    print(message)

    # 1. Create Ollama client via LiteLLM
    llm = LiteLLMClient(model="ollama/llama2")

    # 2. Create ACE components
    adapter = OfflineAdapter(
        playbook=Playbook(),
        generator=Generator(llm),
        reflector=Reflector(llm),
        curator=Curator(llm),
    )

    # 3. Create training samples
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]

    # 4. Run adaptation
    print("\n🚀 Running ACE adaptation with Ollama...")
    environment = SimpleEnvironment()
    results = adapter.run(samples, environment, epochs=1)

    # 5. Check results
    print(f"\n📊 Trained on {len(results)} samples")
    print(f"📚 Playbook now has {len(adapter.playbook.bullets())} strategies")

    # Show a few learned strategies
    if adapter.playbook.bullets():
        print("\n💡 Learned strategies:")
        for bullet in adapter.playbook.bullets()[:2]:
            print(f"  • {bullet.content[:80]}...")


if __name__ == "__main__":
    main()