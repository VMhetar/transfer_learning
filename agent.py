import os
import json
import httpx
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Iterable
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("127.0.0.1")

api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Content-type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

last_call_time = 0
RATE_LIMIT_INTERVAL = 0.5 

async def throttle():
    global last_call_time
    now = time.time()
    if now - last_call_time < RATE_LIMIT_INTERVAL:
        await asyncio.sleep(RATE_LIMIT_INTERVAL - (now - last_call_time))
    last_call_time = time.time()

@mcp.tool()
async def agent(prompt: str, retries=3, backoff=2.0) -> str:
    """
    Resilient agent call with retry, throttling and fallback
    """
    global last_call_time

    data = {
        "model": "mistralai/devstral-2512:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(retries):
        await throttle()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=data)

            result = response.json()

            # handle rate-limit
            if response.status_code == 429:
                print(" Too many requests, backing off...")
                await asyncio.sleep(backoff ** (attempt + 1))
                continue

            # malformed result struct
            if "choices" not in result:
                print(f" Malformed response: {result}")
                continue

            return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Agent error attempt {attempt}: {e}")
            await asyncio.sleep(backoff ** (attempt + 1))

    return "[AGENT FAILURE] No response after retries."


@dataclass
class BenchmarkTask:
    """A task with expected performance metrics"""
    name: str
    description: str
    source_domain: str
    target_domain: str
    task_category: str 
    difficulty: str 
    
@dataclass
class BenchmarkResult:
    """Results from applying a principle to a task"""
    task: BenchmarkTask
    principle: str
    principle_strength: float
    applicability_score: float  
    success: bool 
    reasoning: str
    timestamp: float

class BenchmarkSuite:
    """Collection of standard benchmark tasks for transfer learning"""
    
    def __init__(self):
        self.tasks: List[BenchmarkTask] = [
            BenchmarkTask(
                name="immune_network_design",
                description="Design a distributed network that detects and responds to anomalies",
                source_domain="Biology",
                target_domain="Computer Science",
                task_category="design",
                difficulty="medium"
            ),
            BenchmarkTask(
                name="evolving_algorithm",
                description="Create an algorithm that improves through iterative mutation and selection",
                source_domain="Biology",
                target_domain="Computer Science",
                task_category="optimization",
                difficulty="hard"
            ),
            BenchmarkTask(
                name="self_repair_system",
                description="Build a system that detects and fixes its own errors at runtime",
                source_domain="Biology",
                target_domain="Computer Science",
                task_category="design",
                difficulty="hard"
            ),
            
            # Economics → ML
            BenchmarkTask(
                name="resource_allocation",
                description="Optimize allocation of limited compute to competing tasks",
                source_domain="Economics",
                target_domain="Machine Learning",
                task_category="optimization",
                difficulty="medium"
            ),
            BenchmarkTask(
                name="incentive_alignment",
                description="Design rewards so multiple agents pursue a shared goal",
                source_domain="Economics",
                target_domain="Machine Learning",
                task_category="design",
                difficulty="hard"
            ),
            BenchmarkTask(
                name="market_equilibrium",
                description="Train agents to reach stable competitive equilibrium",
                source_domain="Economics",
                target_domain="Machine Learning",
                task_category="optimization",
                difficulty="hard"
            ),
        ]
    
    def get_tasks(self, source: str, target: str) -> List[BenchmarkTask]:
        """Get all tasks for a specific transfer"""
        return [t for t in self.tasks if t.source_domain == source and t.target_domain == target]


# ============ 1. SEMANTIC ABSTRACTION PRINCIPLES ============

@dataclass
class SemanticAbstraction:
    """Text-based principle that captures transfer"""
    principle: str
    source_domain: str
    target_domain: str
    strength: float
    reasoning: str
    constraints: List[str]
    counterexamples: List[str]

    def __repr__(self):
        return f"{self.source_domain}→{self.target_domain}: {self.principle}"


class SemanticAbstractionMiner:
    """Extract meaningful transfer principles via reasoning"""
    
    def __init__(self):
        self.abstractions: Dict[str, List[SemanticAbstraction]] = {}
        self.reasoning_chains: List[str] = []

    async def extract_principles(
        self, 
        source_domain: str,
        target_domain: str,
        task: str
    ) -> List[SemanticAbstraction]:
        """Extract what actually transfers between domains"""
        
        analysis = await agent(
            f"For the task '{task}':\n"
            f"What are the CORE MECHANISMS in {source_domain} that could transfer to {target_domain}?\n"
            f"Be specific about:\n"
            f"1. What principle/mechanism is at play in {source_domain}\n"
            f"2. How does that same mechanism appear in {target_domain}\n"
            f"3. What would break this analogy\n"
            f"List 3 specific transfers."
        )
        
        self.reasoning_chains.append(analysis)
        principles = await self._parse_principles(analysis, source_domain, target_domain)
        
        key = f"{source_domain}→{target_domain}"
        self.abstractions[key] = principles
        
        return principles

    async def _parse_principles(
        self, 
        analysis: str, 
        source: str, 
        target: str
    ) -> List[SemanticAbstraction]:
        """Convert analysis into structured principles"""
        
        parse_prompt = (
            f"From this analysis, extract the transfer principles as bullet points:\n{analysis}\n"
            f"Format each as: 'PRINCIPLE: [statement] | REASONING: [why] | CONSTRAINTS: [when breaks]'"
        )
        
        parsed = await agent(parse_prompt)
        principles = []
        
        for line in parsed.split('\n'):
            if 'PRINCIPLE:' in line:
                try:
                    parts = line.split('|')
                    principle = parts[0].split('PRINCIPLE:')[1].strip() if len(parts) > 0 else ""
                    reasoning = parts[1].split('REASONING:')[1].strip() if len(parts) > 1 else ""
                    constraints_str = parts[2].split('CONSTRAINTS:')[1].strip() if len(parts) > 2 else ""
                    
                    if principle:
                        abs_obj = SemanticAbstraction(
                            principle=principle,
                            source_domain=source,
                            target_domain=target,
                            strength=0.6,
                            reasoning=reasoning,
                            constraints=[c.strip() for c in constraints_str.split(',') if c.strip()],
                            counterexamples=[]
                        )
                        principles.append(abs_obj)
                except Exception as e:
                    print(f"Parse error: {e}")
                    continue
        
        return principles


# ============ 2. ADVERSARIAL CRITIQUE ============

class SemanticAdversary:
    """Attack transfer principles with counter-reasoning"""
    
    def __init__(self):
        self.critiques: List[Dict] = []
        self.vulnerabilities: List[str] = []

    async def generate_counterargument(
        self, 
        principle: SemanticAbstraction
    ) -> Optional[str]:
        """Find logical flaws in a transfer principle"""
        
        critique_prompt = (
            f"Critically evaluate this transfer claim:\n"
            f"'{principle.principle}'\n"
            f"From {principle.source_domain} to {principle.target_domain}\n"
            f"Reasoning: {principle.reasoning}\n\n"
            f"What are 3 specific ways this transfer could FAIL? "
            f"Give concrete counterexamples where this principle breaks down."
        )
        
        counterargument = await agent(critique_prompt)
        
        if len(counterargument) > 50:
            self.critiques.append({
                "principle": principle.principle,
                "counterargument": counterargument
            })
            return counterargument
        
        return None

    async def find_breaking_cases(
        self,
        principle: SemanticAbstraction
    ) -> List[str]:
        """Identify specific scenarios where principle fails"""
        
        prompt = (
            f"For this transfer principle:\n"
            f"'{principle.principle}'\n"
            f"List 5 specific test cases/scenarios where this fails:\n"
            f"Each should be a concrete task that breaks this principle."
        )
        
        cases = await agent(prompt)
        breaking_cases = [c.strip() for c in cases.split('\n') if c.strip()]
        
        return breaking_cases[:5]

    async def repair_principle(
        self,
        principle: SemanticAbstraction,
        breaking_case: str
    ) -> Optional[SemanticAbstraction]:
        """Modify principle to handle the breaking case"""
        
        repair_prompt = (
            f"This principle breaks on: '{breaking_case}'\n"
            f"Original: '{principle.principle}'\n\n"
            f"Propose a REFINED principle that:\n"
            f"1. Still captures the core transfer\n"
            f"2. Explicitly handles this breaking case\n"
            f"3. Clarifies when it applies vs doesn't apply"
        )
        
        refined = await agent(repair_prompt)
        
        return SemanticAbstraction(
            principle=refined,
            source_domain=principle.source_domain,
            target_domain=principle.target_domain,
            strength=principle.strength,
            reasoning=principle.reasoning,
            constraints=principle.constraints + [breaking_case],
            counterexamples=principle.counterexamples + [breaking_case]
        )


# ============ 3. BENCHMARK EVALUATION ============

class BenchmarkEvaluator:
    """Test principles against real benchmark tasks"""
    
    def __init__(self, benchmark_suite: BenchmarkSuite):
        self.suite = benchmark_suite
        self.results: List[BenchmarkResult] = []
        self.principle_scores: Dict[str, List[float]] = {}

    async def evaluate_principle_on_task(
        self,
        principle: SemanticAbstraction,
        task: BenchmarkTask
    ) -> BenchmarkResult:
        """Apply a principle to a benchmark task"""
        
        eval_prompt = (
            f"Given this transfer principle:\n"
            f"'{principle.principle}'\n"
            f"Reasoning: {principle.reasoning}\n\n"
            f"Apply it to solve this task:\n"
            f"Task: {task.description}\n"
            f"Category: {task.task_category} | Difficulty: {task.difficulty}\n\n"
            f"Questions:\n"
            f"1. How well does this principle apply (0-100)?\n"
            f"2. Does it help solve the task? (YES/NO)\n"
            f"3. Brief reasoning\n\n"
            f"Format: SCORE: X | SUCCESS: YES/NO | REASON: ..."
        )
        
        result_text = await agent(eval_prompt)
        
        try:
            score_part = result_text.split('SCORE:')[1].split('|')[0].strip()
            applicability = int(score_part) / 100.0
            
            success_part = result_text.split('SUCCESS:')[1].split('|')[0].strip().upper()
            success = 'YES' in success_part
            
            reason_part = result_text.split('REASON:')[1].strip() if 'REASON:' in result_text else ""
        except:
            applicability = 0.5
            success = False
            reason_part = "Parse error"
        
        benchmark_result = BenchmarkResult(
            task=task,
            principle=principle.principle,
            principle_strength=principle.strength,
            applicability_score=applicability,
            success=success,
            reasoning=reason_part,
            timestamp=time.time()
        )
        
        self.results.append(benchmark_result)
        
        if principle.principle not in self.principle_scores:
            self.principle_scores[principle.principle] = []
        self.principle_scores[principle.principle].append(applicability)
        
        return benchmark_result

    async def evaluate_principles_on_benchmarks(
        self,
        principles: List[SemanticAbstraction],
        source_domain: str,
        target_domain: str
    ) -> Dict[str, any]:
        """Run full benchmark suite"""
        
        print(f"\n Running benchmark suite for {source_domain}→{target_domain}")
        
        tasks = self.suite.get_tasks(source_domain, target_domain)
        print(f"   Found {len(tasks)} benchmark tasks\n")
        
        for principle in principles:
            for task in tasks:
                print(f"   Testing '{principle.principle[:40]}...' on '{task.name}'")
                result = await self.evaluate_principle_on_task(principle, task)
                
                status = "right" if result.success else "wrong"
                print(f"   {status} Score: {result.applicability_score:.1%}")
        
        return self._summarize_results()

    def _summarize_results(self) -> Dict:
        """Aggregate benchmark results"""
        
        total_tests = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        avg_applicability = sum(r.applicability_score for r in self.results) / total_tests if total_tests > 0 else 0
        
        by_principle = {}
        for principle, scores in self.principle_scores.items():
            by_principle[principle] = {
                "avg_score": sum(scores) / len(scores),
                "tests": len(scores)
            }
        
        return {
            "total_benchmark_tests": total_tests,
            "successful_applications": successful,
            "success_rate": successful / total_tests if total_tests > 0 else 0,
            "avg_applicability": avg_applicability,
            "by_principle": by_principle
        }

    def get_best_principles(self, top_n: int = 3) -> List[tuple[str, float]]:
        """Return highest-performing principles"""
        
        ranked = [
            (principle, scores)
            for principle, scores in self.principle_scores.items()
        ]
        ranked.sort(key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
        
        return [(p, sum(s) / len(s)) for p, s in ranked[:top_n]]


# ============ 4. SEMANTIC COMPOSITION ============

class SemanticComposer:
    """Combine principles into coherent frameworks"""
    
    def __init__(self):
        self.compositions: List[str] = []

    async def compose_principles(
        self,
        principles: List[SemanticAbstraction]
    ) -> str:
        """Synthesize multiple principles into unified understanding"""
        
        principles_text = "\n".join([
            f"- {p.principle} (strength: {p.strength:.2f})"
            for p in principles
        ])
        
        compose_prompt = (
            f"Integrate these transfer principles into a COHERENT FRAMEWORK:\n"
            f"{principles_text}\n\n"
            f"Create a unified conceptual model that:\n"
            f"1. Explains how these principles relate\n"
            f"2. Identifies common patterns\n"
            f"3. Shows where they conflict\n"
            f"4. Provides a meta-principle that unifies them"
        )
        
        framework = await agent(compose_prompt)
        self.compositions.append(framework)
        
        return framework

    async def identify_contradictions(
        self,
        abstractions: List[SemanticAbstraction]
    ) -> List[tuple[SemanticAbstraction, SemanticAbstraction, str]]:
        """Find where principles contradict"""
        
        contradictions = []
        
        for i, abs1 in enumerate(abstractions):
            for abs2 in abstractions[i+1:]:
                conflict_prompt = (
                    f"Do these principles contradict?\n"
                    f"1. '{abs1.principle}'\n"
                    f"2. '{abs2.principle}'\n\n"
                    f"If yes, explain the conflict. If no, say 'NO CONFLICT'."
                )
                
                result = await agent(conflict_prompt)
                
                if "NO CONFLICT" not in result.upper():
                    contradictions.append((abs1, abs2, result))
        
        return contradictions


# ============ INTEGRATED SYSTEM ============

class SemanticTransferLearningSystem:
    """Complete system with benchmark evaluation"""
    
    def __init__(self):
        self.miner = SemanticAbstractionMiner()
        self.adversary = SemanticAdversary()
        self.composer = SemanticComposer()
        self.benchmark_suite = BenchmarkSuite()
        self.evaluator = BenchmarkEvaluator(self.benchmark_suite)
        self.all_principles: List[SemanticAbstraction] = []
        self.benchmark_results: Dict[str, any] = {}

    async def extract_and_validate(
        self,
        source_domain: str,
        target_domain: str,
        task: str
    ) -> List[SemanticAbstraction]:
        """Extract, critique, repair, and benchmark principles"""
        
        print(f"\n{'='*70}")
        print(f" SEMANTIC TRANSFER: {source_domain} → {target_domain}")
        print(f"{'='*70}")

        print(f"\n Extracting transfer principles...")
        principles = await self.miner.extract_principles(source_domain, target_domain, task)
        print(f"   Found {len(principles)} candidate principles")
        for p in principles[:3]:
            print(f"   - {p.principle[:60]}...")

        print(f"\n Generating adversarial critiques...")
        refined_principles = []
        
        for principle in principles:
            counter = await self.adversary.generate_counterargument(principle)
            if counter:
                print(f" Critique found")
                breaking_cases = await self.adversary.find_breaking_cases(principle)
                principle.counterexamples = breaking_cases
                
                if breaking_cases:
                    repaired = await self.adversary.repair_principle(principle, breaking_cases[0])
                    print(f"   ✓ Repaired principle")
                    refined_principles.append(repaired)
                else:
                    refined_principles.append(principle)
            else:
                refined_principles.append(principle)

        print(f"\n  Running benchmark tests...")
        benchmark_summary = await self.evaluator.evaluate_principles_on_benchmarks(
            refined_principles, source_domain, target_domain
        )
        
        key = f"{source_domain}→{target_domain}"
        self.benchmark_results[key] = benchmark_summary
        
        print(f"\n Benchmark Results:")
        print(f"      Total tests: {benchmark_summary['total_benchmark_tests']}")
        print(f"      Success rate: {benchmark_summary['success_rate']:.1%}")
        print(f"      Avg applicability: {benchmark_summary['avg_applicability']:.1%}")
        
        best = self.evaluator.get_best_principles(top_n=3)
        print(f"\n Top performers:")
        for principle, score in best:
            print(f"      [{score:.1%}] {principle[:50]}...")
        
        self.all_principles.extend(refined_principles)
        return refined_principles

    async def generate_report(self) -> str:
        """Comprehensive report with benchmark results"""
        
        report = "\n" + "="*70 + "\n"
        report += "SEMANTIC TRANSFER LEARNING - BENCHMARK REPORT\n"
        report += "="*70 + "\n"
        
        report += f"\n OVERVIEW:\n"
        report += f"  Total principles extracted: {len(self.all_principles)}\n"
        report += f"  Adversarial critiques generated: {len(self.adversary.critiques)}\n"
        report += f"  Benchmark transfers tested: {len(self.benchmark_results)}\n"
        
        report += f"\n BENCHMARK RESULTS BY TRANSFER:\n"
        for transfer, results in self.benchmark_results.items():
            report += f"\n  {transfer}:\n"
            report += f"    Tests run: {results['total_benchmark_tests']}\n"
            report += f"    Success rate: {results['success_rate']:.1%}\n"
            report += f"    Avg applicability: {results['avg_applicability']:.1%}\n"
        
        report += f"\n TOP PRINCIPLES (by benchmark performance):\n"
        top = sorted(self.all_principles, key=lambda p: p.strength, reverse=True)[:5]
        for i, p in enumerate(top, 1):
            report += f"  {i}. {p.principle}\n"
        
        overall_success = sum(r['success_rate'] for r in self.benchmark_results.values()) / len(self.benchmark_results) if self.benchmark_results else 0
        report += f"\n OVERALL BENCHMARK SUCCESS RATE: {overall_success:.1%}\n"
        
        return report


# ============ MAIN EXECUTION ============

@mcp.tool()
async def run_semantic_transfer_system():
    """Execute complete semantic transfer learning with benchmarks"""
    
    system = SemanticTransferLearningSystem()
    
    # Transfer 1: Biology → Computer Science
    await system.extract_and_validate(
        source_domain="Biology",
        target_domain="Computer Science",
        task="Design adaptive systems"
    )
    
    # Transfer 2: Economics → Machine Learning
    await system.extract_and_validate(
        source_domain="Economics",
        target_domain="Machine Learning",
        task="Optimize resource allocation"
    )
    
    # Generate final report
    report = await system.generate_report()
    print(report)
    
    return {
        "total_principles": len(system.all_principles),
        "benchmark_transfers": len(system.benchmark_results),
        "overall_success_rate": sum(r['success_rate'] for r in system.benchmark_results.values()) / len(system.benchmark_results) if system.benchmark_results else 0,
        "report_excerpt": report[:500]
    }


if __name__ == "__main__":
    asyncio.run(run_semantic_transfer_system())