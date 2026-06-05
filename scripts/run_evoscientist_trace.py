#!/usr/bin/env python3
"""Run EvoScientist against vllm-hust and capture a workload trace.

This script:
1. Configures EvoScientist to use the local vllm-hust service (Qwen3-32B on port 18000)
   via the custom-openai provider
2. Attaches a WorkloadTraceHandler callback to capture all LLM interactions
3. Runs a representative research prompt non-interactively
4. Saves the trace to scripts/traces/evoscientist-research-trace.jsonl

Prerequisites:
- vllm-hust serving Qwen3-32B on localhost:18000
- EvoScientist installed in the current environment (pip install -e /path/to/EvoScientist)

Usage:
    python scripts/run_evoscientist_trace.py [--prompt "your prompt"] [--output traces/out.jsonl]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ============================================================================
# Default configuration
# ============================================================================

DEFAULT_VLLM_BASE_URL = "http://localhost:18000/v1"
DEFAULT_VLLM_API_KEY = "dummy"
DEFAULT_MODEL = "Qwen3-32B"
DEFAULT_PROVIDER = "custom-openai"

DEFAULT_PROMPT = (
    "You are a research scientist. Conduct a brief literature analysis on "
    "'LLM inference optimization techniques for production serving systems'. "
    "Focus on: (1) KV-cache management strategies, (2) batching and scheduling "
    "algorithms, (3) speculative decoding approaches. For each area, identify "
    "2-3 key papers or techniques, summarize their core ideas, and note their "
    "trade-offs. Conclude with a table comparing the approaches on latency, "
    "throughput, and memory efficiency. Keep the total output under 2000 words."
)

DEFAULT_OUTPUT = Path(__file__).parent / "traces" / "evoscientist-research-trace.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture EvoScientist workload trace against vllm-hust"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=DEFAULT_PROMPT,
        help="Research prompt to send to EvoScientist",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL trace file path",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_VLLM_BASE_URL,
        help="vllm-hust OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_VLLM_API_KEY,
        help="API key for vllm-hust (usually 'dummy')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name served by vllm-hust",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum LangGraph recursion iterations",
    )
    return parser.parse_args()


def check_service(base_url: str) -> bool:
    """Check if vllm-hust is reachable."""
    import urllib.request
    import urllib.error

    try:
        # Use urllib to bypass SOCKS proxy issues with httpx
        req = urllib.request.Request(f"{base_url}/models")
        # Install a proxy handler that ignores proxies for localhost
        no_proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(no_proxy_handler)
        resp = opener.open(req, timeout=5)
        if resp.status == 200:
            import json as _json
            data = _json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", [])]
            print(f"[OK] vllm-hust is running. Available models: {models}")
            return True
        else:
            print(f"[WARN] vllm-hust responded with status {resp.status}")
            return False
    except Exception as e:
        print(f"[ERROR] Cannot reach vllm-hust at {base_url}: {e}")
        return False


def setup_environment(args: argparse.Namespace) -> None:
    """Configure environment for EvoScientist to use vllm-hust."""
    os.environ["CUSTOM_OPENAI_BASE_URL"] = args.base_url
    os.environ["CUSTOM_OPENAI_API_KEY"] = args.api_key
    # Disable interactive features
    os.environ.setdefault("EVOSCIENTIST_DEFAULT_MODE", "run")
    # Unset proxy env vars to prevent SOCKS issues with localhost
    for key in ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def run_trace(args: argparse.Namespace) -> int:
    """Execute EvoScientist with trace capture and return entry count."""
    # Import after env setup so EvoScientist picks up the config
    from langchain_core.messages import HumanMessage, SystemMessage

    # Add scripts dir to path so we can import the handler
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from evoscientist_trace_handler import WorkloadTraceHandler

    # Create the trace handler
    handler = WorkloadTraceHandler(args.output)
    print(f"[INFO] Trace output: {args.output}")

    # Try the full agent first; fall back to direct LLM if agent graph has issues
    try:
        entries = _run_full_agent(args, handler)
    except Exception as e:
        print(f"[WARN] Full agent failed ({e}), falling back to direct LLM multi-turn trace")
        handler.close()
        # Re-open for the fallback path
        handler = WorkloadTraceHandler(args.output)
        entries = _run_direct_llm(args, handler)

    handler.close()
    return entries


def _run_full_agent(args: argparse.Namespace, handler) -> int:
    """Run the full EvoScientist agent graph."""
    from langchain_core.messages import HumanMessage
    from EvoScientist.config import EvoScientistConfig, apply_config_to_env
    from EvoScientist.EvoScientist import _ensure_config, create_cli_agent

    config = EvoScientistConfig(
        provider=DEFAULT_PROVIDER,
        model=args.model,
        custom_openai_api_key=args.api_key,
        custom_openai_base_url=args.base_url,
        auto_approve=True,
        enable_ask_user=False,
    )
    apply_config_to_env(config)
    _ensure_config(config)

    print("[INFO] Building EvoScientist agent...")
    agent = create_cli_agent(config=config)

    thread_config = {
        "configurable": {"thread_id": "trace-capture-001"},
        "callbacks": [handler],
        "recursion_limit": args.max_iterations,
    }

    print(f"[INFO] Sending prompt ({len(args.prompt)} chars)...")
    start = time.time()

    for state in agent.stream(
        {"messages": [HumanMessage(content=args.prompt)]},
        config=thread_config,
        stream_mode="values",
    ):
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            msg_type = getattr(last, "type", "unknown")
            content_preview = ""
            if hasattr(last, "content") and isinstance(last.content, str):
                content_preview = last.content[:80]
            print(f"  [{msg_type}] {content_preview}...")

    elapsed = time.time() - start
    print(f"\n[DONE] Full agent trace complete in {elapsed:.1f}s")
    return handler.total_entries


def _run_direct_llm(args: argparse.Namespace, handler) -> int:
    """Simulate a realistic EvoScientist multi-agent workload via direct LLM calls.

    A real research agent iterates heavily:
    - Plan -> research -> re-plan based on findings
    - Code -> debug -> fix -> verify loops
    - Write -> review -> revise cycles
    - Multiple sub-agent dispatches per phase

    This produces 35 LLM calls representative of a real autonomous research session.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from EvoScientist.llm import get_chat_model
    from EvoScientist.prompts import get_system_prompt

    print("[INFO] Running realistic multi-agent trace (simulating iterative research workflow)...")

    model = get_chat_model(model=args.model, provider=DEFAULT_PROVIDER)

    try:
        system_prompt = get_system_prompt()
    except Exception:
        system_prompt = (
            "You are EvoScientist, an AI research assistant. "
            "You help conduct scientific research by planning experiments, "
            "reviewing literature, writing code, and analyzing results."
        )

    # =========================================================================
    # Realistic multi-phase research workflow with iteration loops
    # =========================================================================
    research_turns = _build_research_turns(system_prompt, args.prompt)

    start = time.time()
    total_entries = 0
    num_turns = len(research_turns)

    for i, turn in enumerate(research_turns, 1):
        agent_name = turn["agent"]
        messages = [
            SystemMessage(content=turn["system"]),
            HumanMessage(content=turn["user"]),
        ]

        print(f"  [{i}/{num_turns}] {agent_name}: {turn['user'][:70]}...")

        try:
            response = model.invoke(messages, config={"callbacks": [handler]})
            completion_preview = ""
            if hasattr(response, "content") and isinstance(response.content, str):
                completion_preview = response.content[:80]
            print(f"    -> {completion_preview}...")
            total_entries += 1
        except Exception as e:
            print(f"    [ERROR] {agent_name} call failed: {e}")
            continue

    elapsed = time.time() - start
    print(f"\n[DONE] Realistic multi-agent trace complete.")
    print(f"  Entries: {handler.total_entries}")
    print(f"  Turns: {num_turns}")
    print(f"  Duration: {elapsed:.1f}s")

    return handler.total_entries


def _build_research_turns(system_prompt: str, user_prompt: str) -> list[dict]:
    """Build the full multi-phase research turn sequence (35 turns)."""
    sys_planner = system_prompt[:2000]
    sys_research = (
        "You are a research sub-agent specialized in literature review. "
        "Search thoroughly, summarize precisely, and note limitations."
    )
    sys_code = (
        "You are a code sub-agent. Write clean, documented Python code. "
        "Follow best practices and include type hints."
    )
    sys_debug = (
        "You are a debugging sub-agent. Review code for bugs, edge cases, "
        "and performance issues. Suggest fixes with explanations."
    )
    sys_analysis = (
        "You are a data analysis sub-agent. Analyze experimental results, "
        "compute statistics, and create clear summaries."
    )
    sys_writing = (
        "You are a scientific writing sub-agent. Write clear, publication-quality "
        "text with proper structure, citations, and technical depth."
    )
    sys_review = (
        "You are a peer review sub-agent. Critically evaluate scientific writing "
        "for clarity, accuracy, completeness, and logical flow."
    )

    return [
        # -- Phase 1: Initial Planning (3 calls) --
        {"agent": "planner", "system": sys_planner, "user": user_prompt},
        {"agent": "planner", "system": sys_planner, "user": (
            "Break down the research plan into concrete sub-tasks. For each sub-task, "
            "specify: (1) which agent should handle it, (2) what inputs it needs, "
            "(3) expected outputs, (4) success criteria. Output as a structured plan."
        )},
        {"agent": "planner", "system": sys_planner, "user": (
            "Identify the top 5 most relevant search queries to find papers on: "
            "KV-cache optimization, speculative decoding, and LLM batch scheduling. "
            "For each query, explain what gap in knowledge it addresses."
        )},
        # -- Phase 2: Literature Research (8 calls) --
        {"agent": "research-agent", "system": sys_research, "user": (
            "Search for recent papers (2023-2024) on PagedAttention and KV-cache "
            "management for LLM serving. Find 3-4 key papers, summarize each one's "
            "core contribution, methodology, evaluation setup, and key results."
        )},
        {"agent": "research-agent", "system": sys_research, "user": (
            "Search for papers on vAttention, RadixAttention, and DistKV-Cache. "
            "For each system, describe: memory layout strategy, cache eviction policy, "
            "and reported speedup over PagedAttention baseline."
        )},
        {"agent": "research-agent", "system": sys_research, "user": (
            "Find papers on speculative decoding: Medusa, Eagle, SpecInfer, Draft&Verify. "
            "For each, explain: draft model architecture, verification strategy, "
            "acceptance rate, and observed speedup on different model sizes."
        )},
        {"agent": "research-agent", "system": sys_research, "user": (
            "Research batch scheduling systems: Orca, Sarathi, FastServe, DeepSpeed-FastGen. "
            "Compare their scheduling algorithms, preemption strategies, and how they handle "
            "heterogeneous request lengths."
        )},
        {"agent": "research-agent", "system": sys_research, "user": (
            "Find papers combining multiple optimization techniques (e.g., speculative "
            "decoding + continuous batching, or KV-cache compression + paged attention). "
            "What are the challenges of combining these techniques? Any interference effects?"
        )},
        {"agent": "research-agent", "system": sys_research, "user": (
            "Search for quantization-aware KV-cache techniques: KIVI, KVQuant, Gear. "
            "How do they reduce memory while maintaining quality? What are the "
            "accuracy-memory trade-offs at different bit widths (2-bit, 4-bit, 8-bit)?"
        )},
        {"agent": "research-agent", "system": sys_research, "user": (
            "Look up benchmark results from the most recent LLM inference papers. "
            "What hardware setups do they use? What models do they test on? "
            "What are the standard metrics (TTFT, TBT, throughput, memory)? "
            "Compile a cross-paper comparison."
        )},
        {"agent": "research-agent", "system": sys_research, "user": (
            "What are the open problems and future directions in LLM inference? "
            "Identify 3-5 unsolved challenges that current papers acknowledge. "
            "What approaches are being proposed but not yet validated?"
        )},
        # -- Phase 3: Re-planning after research (2 calls) --
        {"agent": "planner", "system": sys_planner, "user": (
            "Based on the literature findings, revise the research plan. "
            "We found that KV-cache quantization is a rapidly growing area. "
            "Should we expand coverage of KIVI/KVQuant? Also, the speculative "
            "decoding section has many variants - should we focus on the top 3? "
            "Update the outline and task assignments."
        )},
        {"agent": "planner", "system": sys_planner, "user": (
            "Design the experimental comparison methodology. What metrics should "
            "we compare? What baselines? What model sizes? Propose a table structure "
            "that would be informative for practitioners choosing between approaches."
        )},
        # -- Phase 4: Code generation & debugging (7 calls) --
        {"agent": "code-agent", "system": sys_code, "user": (
            "Write a Python script that benchmarks the throughput difference between "
            "naive sequential generation and a simple speculative decoding implementation. "
            "Use a mock model for demonstration. Include timing, token counting, and "
            "a formatted comparison table."
        )},
        {"agent": "debug-agent", "system": sys_debug, "user": (
            "Review this benchmark code for issues:\n"
            "```python\n"
            "import time\n"
            "def benchmark_sequential(model, prompts, max_tokens=100):\n"
            "    results = []\n"
            "    for p in prompts:\n"
            "        start = time.time()\n"
            "        tokens = model.generate(p, max_tokens)\n"
            "        elapsed = time.time() - start\n"
            "        results.append({'tokens': len(tokens), 'time': elapsed})\n"
            "    return results\n"
            "```\n"
            "Issues to check: (1) Does it handle warmup? (2) Is timing accurate "
            "for GPU ops? (3) Should we exclude first iteration? (4) Memory measurement?"
        )},
        {"agent": "code-agent", "system": sys_code, "user": (
            "Rewrite the benchmark script incorporating the debug feedback: "
            "add GPU warmup (3 iterations), use torch.cuda.Event for timing, "
            "add memory tracking with torch.cuda.max_memory_allocated(), "
            "exclude warmup from results, add statistical reporting (mean, std, p95)."
        )},
        {"agent": "code-agent", "system": sys_code, "user": (
            "Write a KV-cache simulator in Python that models the memory behavior of "
            "PagedAttention vs traditional pre-allocated attention. Simulate 100 requests "
            "with varying sequence lengths (128-4096 tokens). Track memory fragmentation, "
            "waste, and effective utilization for both approaches."
        )},
        {"agent": "debug-agent", "system": sys_debug, "user": (
            "The KV-cache simulator is producing negative memory waste values for short "
            "sequences. The formula is: waste = allocated - used. But with paged attention, "
            "'allocated' should round up to page boundaries. Debug this: what's the formula "
            "for page-aligned allocation? How should we handle the last partial page?"
        )},
        {"agent": "code-agent", "system": sys_code, "user": (
            "Write a visualization script that takes the simulation results and creates "
            "matplotlib charts: (1) memory usage over time for both approaches, "
            "(2) fragmentation ratio vs sequence length, (3) throughput vs batch size. "
            "Save as publication-quality PDF."
        )},
        {"agent": "code-agent", "system": sys_code, "user": (
            "Write unit tests for the KV-cache simulator: test page allocation alignment, "
            "test memory accounting correctness, test that total allocated >= total used, "
            "test edge cases (sequence length = 1, sequence length = page_size exactly, "
            "sequence length = max_seq_len). Use pytest."
        )},
        # -- Phase 5: Data Analysis (4 calls) --
        {"agent": "data-analysis-agent", "system": sys_analysis, "user": (
            "Analyze these benchmark results from our KV-cache simulation:\n"
            "- PagedAttention (page=16): avg_util=0.89, frag=0.11, waste=1.2GB, peak=14.1GB\n"
            "- PagedAttention (page=64): avg_util=0.94, frag=0.06, waste=0.8GB, peak=14.5GB\n"
            "- PagedAttention (page=256): avg_util=0.97, frag=0.03, waste=0.4GB, peak=15.1GB\n"
            "- Pre-allocated: avg_util=0.45, frag=0.0, waste=8.2GB, peak=22.0GB\n\n"
            "Compute relative improvements. What's the optimal page size? What's the "
            "trade-off between page size and internal fragmentation vs memory efficiency?"
        )},
        {"agent": "data-analysis-agent", "system": sys_analysis, "user": (
            "Compare speculative decoding results across different draft-target model pairs:\n"
            "- Draft=68M, Target=7B: acceptance_rate=0.72, speedup=1.8x, overhead=12%\n"
            "- Draft=160M, Target=7B: acceptance_rate=0.81, speedup=2.1x, overhead=18%\n"
            "- Draft=1.3B, Target=7B: acceptance_rate=0.88, speedup=1.9x, overhead=35%\n"
            "- Medusa-2heads, Target=7B: acceptance_rate=0.64, speedup=1.6x, overhead=8%\n"
            "- Medusa-5heads, Target=7B: acceptance_rate=0.79, speedup=2.3x, overhead=15%\n"
            "- Eagle, Target=7B: acceptance_rate=0.85, speedup=2.5x, overhead=20%\n\n"
            "Which approach has the best speedup-to-overhead ratio? When does a larger "
            "draft model stop being worth it? Plot the Pareto frontier."
        )},
        {"agent": "data-analysis-agent", "system": sys_analysis, "user": (
            "Perform a sensitivity analysis: how does batch size affect the benefit of "
            "each optimization technique?\n"
            "At batch=1: SpecDecode helps most (2.5x), KV-cache irrelevant, scheduling N/A\n"
            "At batch=8: SpecDecode helps less (1.4x), KV-cache saves 30% memory\n"
            "At batch=32: SpecDecode overhead dominates (-0.8x), KV-cache critical, scheduling key\n"
            "At batch=128: Only scheduling + KV-cache matter, all speculative methods hurt\n\n"
            "What's the crossover point? Provide guidance for practitioners."
        )},
        {"agent": "data-analysis-agent", "system": sys_analysis, "user": (
            "Create a comprehensive comparison table for the survey paper. Columns: "
            "Technique, Category, TTFT Impact, TBT Impact, Throughput Gain, Memory Delta, "
            "Best For (use-case), Limitations. Fill in for all techniques covered: "
            "PagedAttention, vAttention, RadixAttention, KIVI, Medusa, Eagle, SpecInfer, "
            "Orca, Sarathi, FastServe, Continuous Batching. Use relative numbers vs baseline."
        )},
        # -- Phase 6: Writing first draft (4 calls) --
        {"agent": "writing-agent", "system": sys_writing, "user": (
            "Write the Introduction section (approx 400 words) for a survey titled "
            "'LLM Inference Optimization: A Comparative Analysis'. Cover: "
            "(1) explosive growth of LLM deployment, (2) inference cost as bottleneck, "
            "(3) three main optimization axes (memory, compute, scheduling), "
            "(4) scope and contributions of this survey. Use citations like [Kwon et al., 2023]."
        )},
        {"agent": "writing-agent", "system": sys_writing, "user": (
            "Write Section 2: KV-Cache Management (approx 600 words). Cover: "
            "PagedAttention [Kwon et al., 2023], vAttention [Patel et al., 2024], "
            "RadixAttention [Zheng et al., 2024], and KV-cache quantization (KIVI, KVQuant). "
            "Explain memory layout, cache eviction, and the trade-offs of each approach. "
            "Include a mini comparison table."
        )},
        {"agent": "writing-agent", "system": sys_writing, "user": (
            "Write Section 3: Scheduling and Batching (approx 500 words). Cover: "
            "continuous batching [Yu et al., 2022], Orca [Yu et al., 2022], "
            "Sarathi [Agrawal et al., 2024], FastServe [Wu et al., 2023], "
            "and chunked prefill. Explain iteration-level scheduling, "
            "preemption strategies, and how they handle mixed-length requests."
        )},
        {"agent": "writing-agent", "system": sys_writing, "user": (
            "Write Section 4: Speculative Decoding (approx 500 words). Cover: "
            "original speculative decoding [Leviathan et al., 2023], Medusa [Cai et al., 2024], "
            "Eagle [Li et al., 2024], SpecInfer [Miao et al., 2024]. Explain draft-verify, "
            "tree-structured speculation, and when speculative decoding helps vs hurts. "
            "Include the batch-size crossover analysis."
        )},
        # -- Phase 7: Review and Revision (4 calls) --
        {"agent": "review-agent", "system": sys_review, "user": (
            "Review the full survey draft for: (1) logical flow between sections, "
            "(2) consistency of terminology, (3) missing important references, "
            "(4) claims without evidence, (5) redundancy. The paper covers KV-cache, "
            "scheduling, and speculative decoding for LLM inference. Provide specific "
            "actionable feedback organized by section."
        )},
        {"agent": "writing-agent", "system": sys_writing, "user": (
            "Revise the Introduction based on review feedback: "
            "(1) Add a clearer definition of 'inference optimization' vs 'training optimization', "
            "(2) Mention that these techniques are complementary not competing, "
            "(3) Add a brief roadmap paragraph at the end of the introduction, "
            "(4) Strengthen the motivation with specific cost numbers (inference = 90% of LLM costs)."
        )},
        {"agent": "writing-agent", "system": sys_writing, "user": (
            "Write the Conclusion and Future Directions section (approx 300 words). "
            "Summarize key findings, highlight that the optimal strategy depends on "
            "deployment scenario (batch size, latency requirements, hardware), "
            "identify 3 open problems: (1) combining techniques without interference, "
            "(2) automatic optimization selection, (3) long-context inference scaling."
        )},
        {"agent": "planner", "system": sys_planner, "user": (
            "Final check: review the complete research output. Is anything missing? "
            "Verify: (1) all 3 topic areas covered, (2) comparison table complete, "
            "(3) code experiments documented, (4) limitations acknowledged, "
            "(5) future work identified. Provide a final quality assessment (1-10) "
            "and list any remaining gaps."
        )},
    ]


def main() -> None:
    args = parse_args()

    # Setup environment first (clears proxies)
    setup_environment(args)

    # Check service
    if not check_service(args.base_url):
        print("\n[FATAL] vllm-hust service is not available.")
        print("Please start vllm-hust with Qwen3-32B on port 18000 first.")
        print("The trace infrastructure is ready - rerun this script when the service is up.")
        sys.exit(1)

    # Run trace capture
    entries = run_trace(args)

    if entries == 0:
        print("[WARN] No trace entries captured. Check model/service configuration.")
        sys.exit(1)

    print(f"\n[SUCCESS] Captured {entries} LLM interactions.")
    print("Next step: python scripts/convert_trace_to_workload.py")


if __name__ == "__main__":
    main()
