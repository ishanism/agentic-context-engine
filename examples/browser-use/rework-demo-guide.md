# ACE Browser-Use Integration Rework Guide

This guide provides step-by-step instructions for transitioning from the old ACE integration pattern to the new, simplified ACEAgent integration.

## Summary of Changes

The new ACE integration dramatically simplifies browser-use examples:
- **70-80% code reduction**: From 700+ lines to ~200 lines
- **Automatic learning**: No manual role setup required
- **Better performance**: Uses v2.1 prompts automatically (+17% success rate)
- **Clean API**: Drop-in replacement for browser-use Agent

## Before vs After Comparison

### Old Pattern (Complex - 700+ lines)
```python
# Complex manual setup
from ace import Generator, Reflector, Curator, OnlineAdapter, Playbook
from ace.llm_providers import LiteLLMClient
from ace.prompts_v2_1 import PromptManager
from browser_use import Agent, ChatBrowserUse

# Manual role creation (50+ lines)
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)
prompt_mgr = PromptManager()
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
playbook = Playbook()

# Custom environment class (100+ lines)
class CustomEnvironment(TaskEnvironment):
    def __init__(self, browser_llm):
        self.browser_llm = browser_llm

    async def evaluate(self, question, generator_output):
        # Complex browser execution logic...
        # Manual trace extraction...
        # Custom feedback parsing...

# Manual adaptation loop (100+ lines)
adapter = OnlineAdapter(playbook, generator, reflector, curator)
results = []
for task in tasks:
    result = await adapter.adapt_single(task, environment)
    results.append(result)
```

### New Pattern (Simple - 200 lines)
```python
# Simple one-line setup
from ace import ACEAgent
from browser_use import ChatBrowserUse

# Automatic everything!
agent = ACEAgent(
    llm=ChatBrowserUse(),                    # Browser automation LLM
    ace_model="gpt-4o-mini",                 # ACE learning LLM
    ace_max_tokens=2048,                     # Token limit
    playbook_path="expert.json",             # Load existing knowledge
    max_steps=25,                            # Browser steps
    calculate_cost=True                      # Track usage
)

# Direct execution with automatic learning
for task in tasks:
    history = await agent.run(task=task)
    # Learning happens automatically!

# Save learned strategies
agent.save_playbook("expert.json")
```

## Step-by-Step Transition Instructions

### Step 1: Update Imports

**Remove:**
```python
from ace import Generator, Reflector, Curator, OnlineAdapter, Playbook
from ace.llm_providers import LiteLLMClient
from ace.prompts_v2_1 import PromptManager
from ace.environments import TaskEnvironment
from browser_use import Agent
import asyncio
import json
from pathlib import Path
```

**Replace with:**
```python
from ace import ACEAgent
from ace.observability import configure_opik  # Optional
from browser_use import ChatBrowserUse
import asyncio
from pathlib import Path
```

### Step 2: Remove Manual Role Setup

**Remove entire sections like:**
```python
# Manual LLM setup (DELETE THIS)
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)

# Manual prompt setup (DELETE THIS)
prompt_mgr = PromptManager()
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())

# Manual playbook setup (DELETE THIS)
playbook_path = Path("expert.json")
if playbook_path.exists():
    playbook = Playbook.load_from_file(str(playbook_path))
else:
    playbook = Playbook()

# Manual browser LLM (DELETE THIS)
browser_llm = ChatBrowserUse()
```

### Step 3: Delete Custom Environment Classes

**Remove entire custom environment implementations:**
```python
class GroceryShoppingEnvironment(TaskEnvironment):  # DELETE ENTIRE CLASS
    def __init__(self, browser_llm):
        self.browser_llm = browser_llm

    async def evaluate(self, question, generator_output):
        # 100+ lines of complex logic - DELETE ALL
        pass

class DomainCheckEnvironment(TaskEnvironment):  # DELETE ENTIRE CLASS
    def __init__(self, browser_llm):
        self.browser_llm = browser_llm

    async def evaluate(self, question, generator_output):
        # 100+ lines of complex logic - DELETE ALL
        pass
```

### Step 4: Replace with Simple ACEAgent Setup

**Add this clean setup:**
```python
async def main():
    # Configure observability (optional)
    try:
        configure_opik(project_name="your-project-name")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    # Setup playbook persistence
    playbook_path = Path(__file__).parent / "expert_playbook.json"

    # Create ACE agent - handles everything automatically!
    agent = ACEAgent(
        llm=ChatBrowserUse(),                    # Browser automation LLM
        ace_model="gpt-4o-mini",                 # ACE learning LLM (or claude-haiku-4-5-20251001)
        ace_max_tokens=2048,                     # Enough for analysis
        playbook_path=str(playbook_path) if playbook_path.exists() else None,
        max_steps=25,                            # Browser automation steps
        calculate_cost=True                      # Track usage
    )

    # Show current knowledge
    if playbook_path.exists():
        print(f"📚 Loaded {len(agent.playbook.bullets())} learned strategies")
    else:
        print("🆕 Starting with empty playbook - learning from scratch")
```

### Step 5: Simplify Task Execution

**Remove complex adaptation loops:**
```python
# OLD: Complex manual adaptation (DELETE THIS)
adapter = OnlineAdapter(playbook, generator, reflector, curator)
environment = CustomEnvironment(browser_llm)

results = []
for i, task in enumerate(tasks):
    print(f"Task {i+1}/{len(tasks)}: {task}")

    try:
        result = await adapter.adapt_single(task, environment)
        results.append(result)

        # Manual result processing...
        # Manual playbook saving...
        # Manual error handling...

    except Exception as e:
        # Complex error handling...
        pass
```

**Replace with simple execution:**
```python
# NEW: Simple direct execution
results = []
for i, task in enumerate(tasks, 1):
    print(f"\n{'='*20} TASK {i}/{len(tasks)} {'='*20}")

    try:
        # Execute with automatic learning
        history = await agent.run(task=task)

        # Extract results (automatic)
        output = history.final_result() if hasattr(history, "final_result") else ""
        steps = history.number_of_steps() if hasattr(history, "number_of_steps") else 0

        result = {
            "task": task,
            "success": True,
            "steps": steps,
            "output": output
        }
        results.append(result)

        # Show immediate results
        print(f"✅ Completed in {steps} steps")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        results.append({
            "task": task,
            "success": False,
            "error": str(e)
        })

    # Small delay between tasks
    if i < len(tasks):
        await asyncio.sleep(2)

# Save learned strategies (automatic)
agent.save_playbook(str(playbook_path))
```

### Step 6: Update Task Templates

**Transform complex task generators into simple string templates:**

**Old:**
```python
def generate_shopping_task(items):
    # Complex task generation logic
    base_task = "Navigate to grocery website..."
    # Multiple string concatenations...
    # Complex formatting...
    return final_task

def generate_domain_task(domain):
    # Complex domain checking logic
    # Multiple conditions...
    # Complex validation...
    return domain_task
```

**New:**
```python
# Simple string templates
SHOPPING_TASK_TEMPLATE = """
You are a browser agent. For every step, first think, then act.
Use exactly this format:
Thought: describe what you want to do next
Action: <browser-use-tool with JSON args>
I will reply with Observation: … after each action.
Repeat Thought → Action → Observation until you can answer.
When you are done, write Final: with the result.

Task: {task_description}

Remember: Focus on accuracy and efficiency. Use learned strategies to improve your approach.
"""

DOMAIN_CHECK_TASK_TEMPLATE = """
You are a browser agent. For every step, first think, then act.
Use exactly this format:
Thought: describe what you want to do next
Action: <browser-use-tool with JSON args>
I will reply with Observation: … after each action.
Repeat Thought → Action → Observation until you can answer.
When you are done, write Final: with the result.

Task: Check if the domain "{domain}" is available.

IMPORTANT: Do NOT navigate to {domain} directly. Instead:
1. Go to a domain checking website (like whois.net, namecheap.com, or godaddy.com)
2. In the search bar type "{domain}" on that website
3. Read the availability status from the results

Output format (exactly one of these):
AVAILABLE: {domain}
TAKEN: {domain}
ERROR: <reason>

Remember: Focus on accuracy and efficiency. Use learned strategies to improve your approach.
"""

# Usage:
task = DOMAIN_CHECK_TASK_TEMPLATE.format(domain="example.com")
history = await agent.run(task=task)
```

### Step 7: Simplify Result Processing

**Remove complex result parsing:**
```python
# OLD: Complex result processing (DELETE THIS)
def parse_shopping_result(result):
    # Complex parsing logic...
    # Multiple conditions...
    # Error handling...
    pass

def analyze_domain_result(result):
    # Complex analysis...
    # Validation logic...
    # Format conversion...
    pass
```

**Replace with simple extraction:**
```python
# NEW: Simple result extraction
def parse_domain_result(output: str, domain: str) -> dict:
    """Parse domain check result from agent output."""
    if not output:
        return {"status": "ERROR", "reason": "No output"}

    output_upper = output.upper()
    domain_upper = domain.upper()

    if f"AVAILABLE: {domain_upper}" in output_upper:
        return {"status": "AVAILABLE"}
    elif f"TAKEN: {domain_upper}" in output_upper:
        return {"status": "TAKEN"}
    else:
        return {"status": "ERROR", "reason": f"Could not parse result: {output[:100]}..."}

# Usage:
result = parse_domain_result(history.final_result(), domain)
```

### Step 8: Update Statistics and Reporting

**Simplify statistics collection:**

**Old:**
```python
# Complex metrics collection (SIMPLIFY THIS)
def calculate_detailed_metrics(results):
    # 50+ lines of complex calculations...
    # Multiple nested loops...
    # Complex aggregations...
    pass

def generate_comprehensive_report(metrics):
    # Complex report generation...
    # Multiple output formats...
    # Complex formatting...
    pass
```

**New:**
```python
# Simple metrics collection
def show_final_results(results):
    """Show final results summary."""
    print(f"\n{'='*60}")
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'#':<3} {'Task':<30} {'Status':<12} {'Steps':<6} {'Result'}")
    print("-" * 60)

    total_steps = 0
    successful_tasks = 0

    for i, result in enumerate(results, 1):
        status_icon = "✅" if result.get("success", False) else "❌"
        steps = result.get("steps", 0)
        total_steps += steps
        if result.get("success", False):
            successful_tasks += 1

        task_name = result.get("task", "Unknown")[:28]
        status = "SUCCESS" if result.get("success", False) else "FAILED"

        print(f"{i:<3} {task_name:<30} {status:<12} {steps:<6} {status_icon}")

    # Show summary statistics
    success_rate = (successful_tasks / len(results)) * 100 if results else 0
    avg_steps = total_steps / len(results) if results else 0

    print(f"\n📈 SUMMARY:")
    print("-" * 30)
    print(f"✅ Success Rate: {successful_tasks}/{len(results)} ({success_rate:.1f}%)")
    print(f"📊 Total Steps: {total_steps}")
    print(f"⚡ Avg Steps/Task: {avg_steps:.1f}")

# Usage:
show_final_results(results)
```

### Step 9: Add Strategy Learning Visualization

**Add this new section to show learned strategies:**
```python
def show_learned_strategies(agent):
    """Show learned strategies from the playbook."""
    strategies = agent.playbook.bullets()
    print(f"\n🎯 LEARNED STRATEGIES: {len(strategies)} total")
    print("-" * 60)

    if strategies:
        # Show recent strategies (last 5)
        recent_strategies = strategies[-5:] if len(strategies) > 5 else strategies

        for i, bullet in enumerate(recent_strategies, 1):
            helpful = bullet.helpful
            harmful = bullet.harmful
            effectiveness = "✅" if helpful > harmful else "⚠️" if helpful == harmful else "❌"
            print(f"{i}. {effectiveness} {bullet.content}")
            print(f"   (+{helpful}/-{harmful})")

        if len(strategies) > 5:
            print(f"   ... and {len(strategies) - 5} older strategies")

        print(f"\n💾 Strategies saved automatically")
        print("🔄 Next run will use these learned strategies automatically!")
    else:
        print("No strategies learned yet (tasks may have failed)")

# Usage at end of main():
show_learned_strategies(agent)
```

## Complete Example Transformation

### Before (Old Domain Checker - 794 lines)
```python
#!/usr/bin/env python3
"""Complex domain checker with manual ACE setup."""

# 50+ import lines...
from ace import Generator, Reflector, Curator, OnlineAdapter, Playbook
from ace.llm_providers import LiteLLMClient
from ace.prompts_v2_1 import PromptManager
# Many more imports...

class DomainCheckEnvironment(TaskEnvironment):
    """Custom environment for domain checking."""

    def __init__(self, browser_llm):
        # 20+ lines of initialization...

    async def evaluate(self, question, generator_output):
        # 100+ lines of complex evaluation logic...

    def _parse_domain_result(self, output):
        # 50+ lines of parsing logic...

    def _validate_domain_format(self, domain):
        # 30+ lines of validation...

async def setup_ace_components():
    # 50+ lines of manual role setup...

async def run_domain_checks():
    # 200+ lines of complex adaptation logic...

def analyze_results():
    # 100+ lines of result analysis...

# 794 total lines
```

### After (New Domain Checker - 235 lines)
```python
#!/usr/bin/env python3
"""
ACE + Browser-Use Domain Checker Demo

Simple demo showing ACE learning to improve at checking domain availability.
Uses the new ACEAgent integration for clean, automatic learning.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add parent directories to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import ACE framework with new integration
from ace import ACEAgent
from ace.observability import configure_opik
from browser_use import ChatBrowserUse

# Import common utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from domain_utils import get_test_domains

# Domain checking task definition
DOMAIN_CHECK_TASK_TEMPLATE = """
You are a browser agent. For every step, first think, then act.
Use exactly this format:
Thought: describe what you want to do next
Action: <browser-use-tool with JSON args>
I will reply with Observation: … after each action.
Repeat Thought → Action → Observation until you can answer.
When you are done, write Final: with the result.

Task: Check if the domain "{domain}" is available.

IMPORTANT: Do NOT navigate to {domain} directly. Instead:
1. Go to a domain checking website (like whois.net, namecheap.com, or godaddy.com)
2. In the search bar type "{domain}" on that website
3. Read the availability status from the results

Output format (exactly one of these):
AVAILABLE: {domain}
TAKEN: {domain}
ERROR: <reason>

Remember: Focus on accuracy and efficiency. Use learned strategies to improve your approach.
"""

def parse_domain_result(output: str, domain: str) -> dict:
    """Parse domain check result from agent output."""
    if not output:
        return {"status": "ERROR", "reason": "No output"}

    output_upper = output.upper()
    domain_upper = domain.upper()

    if f"AVAILABLE: {domain_upper}" in output_upper:
        return {"status": "AVAILABLE"}
    elif f"TAKEN: {domain_upper}" in output_upper:
        return {"status": "TAKEN"}
    else:
        return {"status": "ERROR", "reason": f"Could not parse result: {output[:100]}..."}

async def check_single_domain(agent: ACEAgent, domain: str) -> dict:
    """Check a single domain and return results with metrics."""
    print(f"🔍 Checking domain: {domain}")

    try:
        # Create task for this specific domain
        task = DOMAIN_CHECK_TASK_TEMPLATE.format(domain=domain)

        # Run domain check with ACE learning
        history = await agent.run(task=task, max_steps=25)

        # Extract results
        output = history.final_result() if hasattr(history, "final_result") else ""
        steps = history.number_of_steps() if hasattr(history, "number_of_steps") else 0

        # Parse domain check result
        result = parse_domain_result(output, domain)

        return {
            "domain": domain,
            "status": result["status"],
            "success": result["status"] != "ERROR",
            "steps": steps,
            "output": output,
            "error": result.get("reason"),
        }

    except Exception as e:
        print(f"❌ Error checking {domain}: {str(e)}")
        return {
            "domain": domain,
            "status": "ERROR",
            "success": False,
            "steps": 0,
            "output": "",
            "error": f"Exception: {str(e)}",
        }

async def main():
    """Run domain checking with ACE learning."""

    # Configure observability
    try:
        configure_opik(project_name="ace-domain-checker")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🔍 ACE + Browser-Use Domain Checker")
    print("🧠 Automated domain checking with learning!")
    print("=" * 60)

    # Setup playbook persistence
    playbook_path = Path(__file__).parent / "ace_domain_checker_playbook.json"

    # Create ACE agent - handles everything automatically!
    agent = ACEAgent(
        llm=ChatBrowserUse(),                    # Browser automation LLM
        ace_model="claude-haiku-4-5-20251001",   # ACE learning LLM
        ace_max_tokens=4096,                     # Enough for domain check analysis
        playbook_path=str(playbook_path) if playbook_path.exists() else None,
        max_steps=25,                            # Browser automation steps
        calculate_cost=True                      # Track usage
    )

    # Show current knowledge
    if playbook_path.exists():
        print(f"📚 Loaded {len(agent.playbook.bullets())} learned strategies")
    else:
        print("🆕 Starting with empty playbook - learning from scratch")

    # Get test domains
    domains = get_test_domains()
    print(f"\n📋 Testing {len(domains)} domains:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    print(f"\n🎯 Each domain check will teach ACE new strategies")
    print("💡 ACE learns automatically after each execution\n")

    # Run domain checks with learning
    results = []
    for i, domain in enumerate(domains, 1):
        print(f"\n{'='*20} DOMAIN CHECK {i}/{len(domains)} {'='*20}")

        result = await check_single_domain(agent, domain)
        results.append(result)

        # Show immediate results
        status_icon = "✅" if result["success"] else "❌"
        print(f"{status_icon} {domain}: {result['status']} ({result['steps']} steps)")

        if result["error"]:
            print(f"   Error: {result['error']}")

        # Small delay between checks to avoid rate limits
        if i < len(domains):
            print(f"⏳ Waiting 2 seconds before next check...")
            await asyncio.sleep(2)

    # Save learned strategies
    agent.save_playbook(str(playbook_path))

    # Show final results
    print(f"\n{'='*60}")
    print("📊 DOMAIN CHECK RESULTS")
    print("=" * 60)
    print(f"{'#':<3} {'Domain':<25} {'Status':<12} {'Steps':<6} {'Result'}")
    print("-" * 60)

    total_steps = 0
    successful_checks = 0

    for i, result in enumerate(results, 1):
        status_icon = "✅" if result["success"] else "❌"
        total_steps += result["steps"]
        if result["success"]:
            successful_checks += 1

        print(f"{i:<3} {result['domain']:<25} {result['status']:<12} {result['steps']:<6} {status_icon}")

    # Show summary statistics
    success_rate = (successful_checks / len(results)) * 100 if results else 0
    avg_steps = total_steps / len(results) if results else 0

    print(f"\n📈 SUMMARY:")
    print("-" * 30)
    print(f"✅ Success Rate: {successful_checks}/{len(results)} ({success_rate:.1f}%)")
    print(f"📊 Total Steps: {total_steps}")
    print(f"⚡ Avg Steps/Domain: {avg_steps:.1f}")

    # Show learned strategies
    strategies = agent.playbook.bullets()
    print(f"\n🎯 LEARNED STRATEGIES: {len(strategies)} total")
    print("-" * 60)

    if strategies:
        # Show recent strategies (last 5)
        recent_strategies = strategies[-5:] if len(strategies) > 5 else strategies

        for i, bullet in enumerate(recent_strategies, 1):
            helpful = bullet.helpful
            harmful = bullet.harmful
            effectiveness = "✅" if helpful > harmful else "⚠️" if helpful == harmful else "❌"
            print(f"{i}. {effectiveness} {bullet.content}")
            print(f"   (+{helpful}/-{harmful})")

        if len(strategies) > 5:
            print(f"   ... and {len(strategies) - 5} older strategies")

        print(f"\n💾 Strategies saved to: {playbook_path}")
        print("🔄 Next run will use these learned strategies automatically!")
    else:
        print("No new strategies learned (tasks may have failed)")

    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
# 235 total lines - 70% reduction!
```

## Key Benefits of the New Pattern

### 1. Dramatic Code Reduction
- **Online Shopping**: 886 → 156 lines (82% reduction)
- **Domain Checker**: 794 → 235 lines (70% reduction)
- **Average**: 75% less code to maintain

### 2. Automatic Best Practices
- ✅ **v2.1 prompts**: Automatic +17% performance improvement
- ✅ **Proper trace extraction**: Full browser execution history captured
- ✅ **Citation tracking**: Automatic bullet ID extraction from agent thoughts
- ✅ **Error handling**: Graceful degradation on failures
- ✅ **Cost tracking**: Automatic token usage and cost monitoring

### 3. Simplified Maintenance
- ✅ **No custom environments**: Direct task execution
- ✅ **No manual role setup**: ACEAgent handles everything
- ✅ **No complex result parsing**: Simple output extraction
- ✅ **No adaptation loops**: Sequential execution with automatic learning

### 4. Better User Experience
- ✅ **Real-time feedback**: Immediate task results
- ✅ **Strategy visualization**: See what ACE learned
- ✅ **Progress tracking**: Clear step counts and timing
- ✅ **Automatic persistence**: Playbooks saved automatically

## Common Pitfalls to Avoid

### 1. Don't Keep Old Environment Classes
❌ **Wrong:**
```python
# Don't keep these even if "simplified"
class SimplifiedShoppingEnvironment(TaskEnvironment):
    async def evaluate(self, question, generator_output):
        # Even "simple" environments are unnecessary
        pass
```

✅ **Correct:**
```python
# Direct execution - no environment needed
history = await agent.run(task=task)
```

### 2. Don't Manually Create Roles
❌ **Wrong:**
```python
# Don't do this anymore
llm = LiteLLMClient(model="gpt-4o-mini")
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)
```

✅ **Correct:**
```python
# ACEAgent handles all roles automatically
agent = ACEAgent(llm=ChatBrowserUse(), ace_model="gpt-4o-mini")
```

### 3. Don't Complex Result Processing
❌ **Wrong:**
```python
def complex_result_processor(generator_output, environment_result):
    # 50+ lines of complex processing logic
    # Multiple validation steps
    # Complex error handling
    pass
```

✅ **Correct:**
```python
# Simple extraction from history
output = history.final_result() if hasattr(history, "final_result") else ""
steps = history.number_of_steps() if hasattr(history, "number_of_steps") else 0
```

### 4. Don't Forget Playbook Persistence
❌ **Wrong:**
```python
# Missing playbook management
agent = ACEAgent(llm=ChatBrowserUse())
# No loading previous knowledge
# No saving learned strategies
```

✅ **Correct:**
```python
# Proper playbook management
playbook_path = Path(__file__).parent / "expert_playbook.json"
agent = ACEAgent(
    llm=ChatBrowserUse(),
    playbook_path=str(playbook_path) if playbook_path.exists() else None
)
# Learning persists automatically
agent.save_playbook(str(playbook_path))  # Save at the end
```

## Migration Checklist

Use this checklist to ensure complete transition:

### Code Structure ✅
- [ ] Removed all custom `TaskEnvironment` classes
- [ ] Removed manual `Generator`, `Reflector`, `Curator` setup
- [ ] Removed `OnlineAdapter` usage
- [ ] Replaced with single `ACEAgent` instance
- [ ] Updated imports to use `ACEAgent` and `ChatBrowserUse`

### Task Execution ✅
- [ ] Converted complex task generators to simple string templates
- [ ] Replaced adaptation loops with direct `agent.run()` calls
- [ ] Simplified result extraction using `history` methods
- [ ] Added automatic learning through ACEAgent

### Configuration ✅
- [ ] Added playbook persistence with proper path handling
- [ ] Configured ACE learning LLM (separate from browser LLM)
- [ ] Set appropriate `ace_max_tokens` (2048-4096)
- [ ] Enabled cost tracking with `calculate_cost=True`
- [ ] Added observability with `configure_opik()` (optional)

### Output & Monitoring ✅
- [ ] Added strategy visualization with `agent.playbook.bullets()`
- [ ] Simplified metrics collection and display
- [ ] Added real-time progress feedback
- [ ] Removed complex report generation logic

### Testing ✅
- [ ] Verified examples run with new pattern
- [ ] Confirmed playbook loading/saving works
- [ ] Tested automatic learning after task execution
- [ ] Validated strategy accumulation over multiple runs

## File Size Expectations

After applying this migration guide:

| Example Type | Before | After | Reduction |
|-------------|--------|--------|----------|
| Online Shopping | 886 lines | 156 lines | 82% |
| Domain Checker | 794 lines | 235 lines | 70% |
| Form Filler | ~800 lines | ~200 lines | 75% |
| **Average** | **~800 lines** | **~200 lines** | **75%** |

## Performance Improvements

The new integration provides automatic performance benefits:

1. **+17% Success Rate**: v2.1 prompts vs v1.0 prompts
2. **Better Learning**: Full execution traces captured for Reflector
3. **Citation Tracking**: Automatic bullet ID extraction from agent thoughts
4. **Cost Efficiency**: Automatic token usage tracking and optimization
5. **Faster Development**: 75% less code to write and maintain

## Next Steps

1. **Apply this guide** to migrate your existing ACE browser-use examples
2. **Test thoroughly** to ensure functionality is preserved
3. **Monitor performance** using the built-in observability features
4. **Leverage learned strategies** by reusing playbooks across similar tasks
5. **Contribute improvements** back to the community

---

This guide represents the complete transition from complex manual ACE setup to the simplified ACEAgent integration. The new pattern maintains all the learning capabilities while dramatically reducing code complexity and maintenance burden.