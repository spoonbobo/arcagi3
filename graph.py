import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv(dotenv_path=".env-example", override=False)
load_dotenv(dotenv_path=".env", override=True)

logger = logging.getLogger(__name__)


class SwarmGraphState(TypedDict):
    """State for the swarm orchestration graph"""
    task: str
    swarm_count: int
    worker_results: List[Dict[str, Any]]
    summary: Optional[str]
    completed_workers: int
    total_score: int
    execution_start_time: float
    execution_end_time: Optional[float]


class LLMWorker:
    """LLM-powered worker"""
    
    def __init__(self, worker_id: int, task: str, llm_model: str = "deepseek-chat"):
        self.worker_id = worker_id
        self.task = task
        self.llm = ChatDeepSeek(
            model=llm_model,
            temperature=0.7,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    
    async def execute_async(self) -> Dict[str, Any]:
        """Execute LLM task"""
        start_time = time.time()
        
        try:
            prompt = f"""Task: {self.task}
            
As Worker #{self.worker_id}, provide a solution and rate your confidence (1-100).

Response format:
{{"solution": "your solution", "confidence": 85}}"""
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            try:
                result = json.loads(response.content)
                success = True
                score = result.get("confidence", 50)
                solution = result.get("solution", "No solution")
            except json.JSONDecodeError:
                success = True
                score = 50
                solution = response.content[:100] + "..."
            
            return {
                "worker_id": self.worker_id,
                "success": success,
                "score": score,
                "solution": solution,
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "worker_id": self.worker_id,
                "success": False,
                "score": 0,
                "solution": f"Error: {str(e)}",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }


class SwarmOrchestrator:
    """LangGraph-based orchestrator for LLM workers"""
    
    def __init__(self, llm_model: str = "deepseek-chat"):
        self.llm_model = llm_model
        self.summary_llm = ChatDeepSeek(
            model=llm_model,
            temperature=0.3,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(SwarmGraphState)
        
        workflow.add_node("initialize_workers", self._initialize_workers)
        workflow.add_node("execute_workers", self._execute_workers_async)
        workflow.add_node("summarize_results", self._summarize_results)
        
        workflow.set_entry_point("initialize_workers")
        workflow.add_edge("initialize_workers", "execute_workers")
        workflow.add_edge("execute_workers", "summarize_results") 
        workflow.add_edge("summarize_results", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _initialize_workers(self, state: SwarmGraphState) -> SwarmGraphState:
        """Initialize workers"""
        logger.info(f"Starting {state['swarm_count']} workers")
        
        state["worker_results"] = []
        state["completed_workers"] = 0
        state["total_score"] = 0
        state["execution_start_time"] = time.time()
        state["execution_end_time"] = None
        
        return state
    
    async def _execute_single_worker_async(self, worker_id: int, state: SwarmGraphState) -> Dict[str, Any]:
        """Execute single worker"""
        worker = LLMWorker(worker_id, state["task"], self.llm_model)
        return await worker.execute_async()
    
    def _execute_workers_async(self, state: SwarmGraphState) -> SwarmGraphState:
        """Execute all workers asynchronously"""
        
        async def run_workers():
            semaphore = asyncio.Semaphore(8)  # Max 8 concurrent calls
            
            async def run_worker_with_semaphore(worker_id: int):
                async with semaphore:
                    return await self._execute_single_worker_async(worker_id, state)
            
            tasks = [
                run_worker_with_semaphore(worker_id) 
                for worker_id in range(state["swarm_count"])
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "worker_id": i,
                        "success": False,
                        "score": 0,
                        "solution": f"Error: {str(result)}",
                        "execution_time": 0,
                        "timestamp": time.time()
                    })
                else:
                    processed_results.append(result)
                    state["total_score"] += result.get("score", 0)
                
                state["completed_workers"] += 1
            
            return processed_results
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_workers())
            loop.close()
        except Exception as e:
            logger.error(f"Worker execution failed: {e}")
            results = []
        
        state["worker_results"] = results
        state["execution_end_time"] = time.time()
        logger.info(f"Completed {state['swarm_count']} workers")
        return state
    
    def _summarize_results(self, state: SwarmGraphState) -> SwarmGraphState:
        """Generate simple summary"""
        successful = [r for r in state["worker_results"] if r.get("success", False)]
        failed = len(state["worker_results"]) - len(successful)
        
        total_time = state["execution_end_time"] - state["execution_start_time"]
        avg_score = state["total_score"] / max(len(successful), 1)
        
        # Get top solutions
        top_solutions = sorted(successful, key=lambda x: x['score'], reverse=True)[:3]
        solutions_text = "\n".join([f"- {s['solution'][:100]}" for s in top_solutions])
        
        prompt = f"""Summarize this swarm task execution:

Task: {state['task']}
Workers: {state['swarm_count']} total, {len(successful)} successful, {failed} failed
Average Score: {avg_score:.1f}
Execution Time: {total_time:.1f}s

Top Solutions:
{solutions_text}

Provide a brief 2-3 sentence summary."""
        
        try:
            response = self.summary_llm.invoke([HumanMessage(content=prompt)])
            state["summary"] = response.content
        except Exception as e:
            state["summary"] = f"Summary failed: {e}"
        
        return state
    
    def run(self, task: str, swarm_count: int = 8) -> Dict[str, Any]:
        """Run the swarm orchestration"""
        initial_state = SwarmGraphState(
            task=task,
            swarm_count=swarm_count,
            worker_results=[],
            summary=None,
            completed_workers=0,
            total_score=0,
            execution_start_time=0.0,
            execution_end_time=None
        )
        
        config = {"configurable": {"thread_id": f"swarm_{int(time.time())}"}}
        return self.graph.invoke(initial_state, config)


def main():
    """Run swarm example"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    
    orchestrator = SwarmOrchestrator()
    
    results = orchestrator.run(
        task="Design a sustainable city transportation system",
        swarm_count=6
    )
    
    print(f"\n🚀 SWARM RESULTS")
    print(f"Task: {results['task']}")
    print(f"Workers: {results['completed_workers']}/{results['swarm_count']}")
    print(f"Score: {results['total_score']}")
    print(f"Time: {results['execution_end_time'] - results['execution_start_time']:.1f}s")
    print(f"\nSummary: {results['summary']}")
    
    successful = [r for r in results['worker_results'] if r.get('success')]
    if successful:
        top = sorted(successful, key=lambda x: x['score'], reverse=True)[0]
        print(f"\nTop Solution (Score: {top['score']}): {top['solution']}")


if __name__ == "__main__":
    main()
