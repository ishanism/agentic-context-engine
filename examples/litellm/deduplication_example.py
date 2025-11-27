#!/usr/bin/env python3
"""
ACELiteLLM Deduplication Example

Demonstrates bullet deduplication with ACELiteLLM:
- Uses vector embeddings to detect similar bullets
- Prevents playbook bloat from redundant strategies
- Shows deduplication stats and results

Requires: OPENAI_API_KEY (for embeddings and LLM)
"""

import os
import time
from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment, DeduplicationConfig

load_dotenv()


def run_without_dedup():
    """Run learning WITHOUT deduplication (baseline)."""
    print("\n" + "=" * 60)
    print("WITHOUT DEDUPLICATION (baseline)")
    print("=" * 60)

    agent = ACELiteLLM(model="gpt-4o-mini")

    # Similar questions that will likely create similar bullets
    samples = [
        # Math variations
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What is 2 plus 2?", ground_truth="4"),
        Sample(question="Calculate two plus two", ground_truth="4"),
        # Color variations
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="What is the color of the sky?", ground_truth="blue"),
        # Capital variations
        Sample(question="Capital of France?", ground_truth="Paris"),
        Sample(question="What is the capital of France?", ground_truth="Paris"),
        Sample(question="What city is the capital of France?", ground_truth="Paris"),
    ]

    environment = SimpleEnvironment()

    print(f"\nProcessing {len(samples)} samples (many are similar)...")
    start = time.time()

    results = agent.learn(samples, environment, epochs=1)

    elapsed = time.time() - start
    bullets = agent.playbook.bullets()

    print(f"\nResults:")
    print(f"  - Time: {elapsed:.2f}s")
    print(f"  - Samples processed: {len(results)}")
    print(f"  - Bullets in playbook: {len(bullets)}")

    if bullets:
        print(f"\nPlaybook bullets (may have duplicates):")
        for i, bullet in enumerate(bullets[:5], 1):
            content = (
                bullet.content[:60] + "..."
                if len(bullet.content) > 60
                else bullet.content
            )
            print(f"  {i}. [{bullet.id}] {content}")
        if len(bullets) > 5:
            print(f"  ... and {len(bullets) - 5} more")

    return len(bullets)


def run_with_dedup():
    """Run learning WITH deduplication enabled."""
    print("\n" + "=" * 60)
    print("WITH DEDUPLICATION")
    print("=" * 60)

    # Configure deduplication
    dedup_config = DeduplicationConfig(
        enabled=True,
        similarity_threshold=0.80,  # Bullets with >80% similarity are flagged
        embedding_model="text-embedding-3-small",  # OpenAI embedding model
        within_section_only=True,  # Only compare within same section
    )

    agent = ACELiteLLM(
        model="gpt-4o-mini",
        dedup_config=dedup_config,  # Enable deduplication!
    )

    # Same similar questions
    samples = [
        # Math variations
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What is 2 plus 2?", ground_truth="4"),
        Sample(question="Calculate two plus two", ground_truth="4"),
        # Color variations
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="What is the color of the sky?", ground_truth="blue"),
        # Capital variations
        Sample(question="Capital of France?", ground_truth="Paris"),
        Sample(question="What is the capital of France?", ground_truth="Paris"),
        Sample(question="What city is the capital of France?", ground_truth="Paris"),
    ]

    environment = SimpleEnvironment()

    print(f"\nProcessing {len(samples)} samples with deduplication...")
    print(f"  - Similarity threshold: {dedup_config.similarity_threshold}")
    print(f"  - Embedding model: {dedup_config.embedding_model}")
    start = time.time()

    results = agent.learn(samples, environment, epochs=1)

    elapsed = time.time() - start
    bullets = agent.playbook.bullets()

    print(f"\nResults:")
    print(f"  - Time: {elapsed:.2f}s")
    print(f"  - Samples processed: {len(results)}")
    print(f"  - Bullets in playbook: {len(bullets)}")

    if bullets:
        print(f"\nPlaybook bullets (deduplicated):")
        for i, bullet in enumerate(bullets[:5], 1):
            content = (
                bullet.content[:60] + "..."
                if len(bullet.content) > 60
                else bullet.content
            )
            print(f"  {i}. [{bullet.id}] {content}")
        if len(bullets) > 5:
            print(f"  ... and {len(bullets) - 5} more")

    return len(bullets)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    print("=" * 60)
    print("ACELiteLLM DEDUPLICATION DEMO")
    print("=" * 60)
    print("\nThis demo shows how deduplication prevents playbook bloat")
    print("by detecting and consolidating similar bullets.")
    print("\nWe'll process similar questions (variations of the same concept)")
    print("and compare bullet counts with and without deduplication.")

    # Run without dedup first
    baseline_bullets = run_without_dedup()

    # Run with dedup
    dedup_bullets = run_with_dedup()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nWithout deduplication: {baseline_bullets} bullets")
    print(f"With deduplication:    {dedup_bullets} bullets")

    if dedup_bullets < baseline_bullets:
        reduction = ((baseline_bullets - dedup_bullets) / baseline_bullets) * 100
        print(f"\nDeduplication reduced playbook size by {reduction:.0f}%")
    elif dedup_bullets == baseline_bullets:
        print("\nNote: No duplicate bullets detected (results may vary)")
    else:
        print("\nNote: More bullets with dedup (Curator may have different strategy)")


if __name__ == "__main__":
    main()
