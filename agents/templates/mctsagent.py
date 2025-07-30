from datetime import datetime
from typing import Any
from collections import OrderedDict
import os
import json
import logging
from pathlib import Path
from enum import Enum

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from utils import need_to_reset, need_to_terminate, parse_knowledge_sequence, write_knowledge_response

# LangGraph imports
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Optional

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, frame_data: FrameData, description: str = ""):
        self.frame_data = frame_data
        self.state_id = str(hash(str(frame_data.frame)))
        self.description = description
        self.edges = {}
        self.is_expanded = False
        self.is_terminal = frame_data.state in [GameState.WIN, GameState.GAME_OVER]
        
    def add_edge(self, action: GameAction, child_state_id: str, prior: float):
        edge = Edge(self.state_id, child_state_id, action, prior)
        self.edges[action.value] = edge
        return edge

class Edge:
    def __init__(self, parent_state_id: str, child_state_id: str, action: GameAction, prior: float):
        self.parent_state_id = parent_state_id
        self.child_state_id = child_state_id
        self.action = action
        self.N = 0
        self.W = 0.0
        self.P = prior
        
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0
        
    def update(self, value: float):
        self.N += 1
        self.W += value

class KnowledgeTree:
    def __init__(self, root_frame: FrameData, root_description: str = "", state_limit: int = 10000):
        root_node = Node(root_frame, root_description)
        self.nodes = {root_node.state_id: root_node}
        self.root_id = root_node.state_id
        self.states = StateRegistry(state_limit)
        self.states.register(root_frame)
        
    def add_node(self, frame_data: FrameData, description: str = "") -> Node:
        state_id = self.states.register(frame_data)
        if state_id not in self.nodes:
            node = Node(frame_data, description)
            self.nodes[state_id] = node
        return self.nodes[state_id]
    
    def get_node(self, state_id: str) -> Node:
        return self.nodes.get(state_id)
    
    def get_frame(self, state_id: str) -> FrameData:
        return self.states.get_frame(state_id)
    
    def get_all_states(self) -> dict[str, FrameData]:
        return self.states._states.copy()
    
    def state_count(self) -> int:
        return self.states.count()
    
    def get_edge(self, parent_state_id: str, action: GameAction) -> Edge:
        node = self.get_node(parent_state_id)
        return node.edges.get(action.value) if node else None
    
    def get_statistics(self, state_id: str, action: GameAction) -> tuple[int, float, float, float]:
        edge = self.get_edge(state_id, action)
        if edge:
            return edge.N, edge.W, edge.P, edge.Q
        return 0, 0.0, 0.0, 0.0

class StateRegistry:
    def __init__(self, limit: int = 10000):
        self._states = OrderedDict()
        self.limit = limit
        
    def register(self, frame_data: FrameData) -> str:
        state_id = str(hash(str(frame_data.frame)))
        
        if state_id in self._states:
            self._states.move_to_end(state_id)
        else:
            self._states[state_id] = frame_data
            if len(self._states) > self.limit:
                self._states.popitem(last=False)
                
        return state_id
    
    def get_frame(self, state_id: str) -> FrameData:
        if state_id in self._states:
            self._states.move_to_end(state_id)
            return self._states[state_id]
        return FrameData()
    
    def count(self) -> int:
        return len(self._states)

class ActionRegistry:
    ACTIONS = [
        GameAction.ACTION1,
        GameAction.ACTION2, 
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
        GameAction.ACTION6
    ]
    
    def __init__(self):
        self.action_confidences = {
            GameAction.ACTION1.value: ActionConfidenceLevel.UNKNOWN,
            GameAction.ACTION2.value: ActionConfidenceLevel.UNKNOWN,
            GameAction.ACTION3.value: ActionConfidenceLevel.UNKNOWN,
            GameAction.ACTION4.value: ActionConfidenceLevel.UNKNOWN,
            GameAction.ACTION5.value: ActionConfidenceLevel.UNKNOWN,
            GameAction.ACTION6.value: ActionConfidenceLevel.UNKNOWN,
        }
        
        for action in self.ACTIONS:
            action.reasoning = "unknown meaning"
    
    def get_all(self) -> list[GameAction]:
        return self.ACTIONS.copy()
    
    def get_by_id(self, action_id: int) -> GameAction:
        for action in self.ACTIONS:
            if action.value == action_id:
                return action
        return GameAction.ACTION1
    
    def update_action_understanding(self, action_updates: Dict[str, Dict[str, Any]]) -> None:
        """Update action reasoning and confidence based on AI analysis"""
        for action_name, update_data in action_updates.items():
            action = getattr(GameAction, action_name, None)
            if action and action in self.ACTIONS:
                # Update reasoning (natural language description)
                if "reasoning" in update_data:
                    action.reasoning = update_data["reasoning"]
                
                # Update confidence level
                if "confidence" in update_data:
                    confidence_name = update_data["confidence"].upper()
                    if hasattr(ActionConfidenceLevel, confidence_name):
                        confidence_level = getattr(ActionConfidenceLevel, confidence_name)
                        self.action_confidences[action.value] = confidence_level
                        logger.info(f"[UPDATE] {action_name}: {action.reasoning} (confidence: {confidence_level.name})")

# Add new template for action analysis
ACTION_ANALYSIS_TEMPLATE = """
You are an expert in analyzing ARC-AGI-3 game actions by observing before/after frame changes.

CURRENT ACTION UNDERSTANDING:
Action: {action_name}
Current Reasoning: "{current_reasoning}"
Current Confidence: {current_confidence}
Action Data: {action_data}

BEFORE FRAME:
{before_frame}

AFTER FRAME:
{after_frame}

{expectation_section}

TASK: Analyze what this action did by comparing the before and after frames{expectation_task}.

Based on your analysis:
1. Confirm or update the reasoning for what this action does
2. Adjust the confidence level based on your observation
3. If an expectation was provided, determine if the actual result matches the expectation

Output your analysis in the following JSON format:
{{
    "{action_name}": {{
        "reasoning": "Updated natural language description of what this action does (e.g., 'moves player up one cell', 'places a block at coordinates', 'rotates the grid clockwise')",
        "confidence": "CONFIDENCE_LEVEL",
        "expectation_matched": "MATCH_STATUS",
        "expectation_explanation": "Explanation of how well the actual result matched the expectation (or 'No expectation provided' if none)"
    }}
}}

CONFIDENCE LEVELS:
- UNKNOWN (0.0): No clear understanding of what the action does
- LOW (0.2): Some indication but uncertain
- MEDIUM (0.5): Reasonable understanding with some doubt
- HIGH (0.7): Good understanding with minor uncertainty
- ALMOST_SURE (0.9): Very confident understanding
- ASSERTIVE (1.0): Completely certain about the action's function

EXPECTATION MATCH STATUS:
- FULL_MATCH: The actual result exactly matches the expectation
- PARTIAL_MATCH: The actual result partially matches the expectation
- NO_MATCH: The actual result does not match the expectation
- NO_EXPECTATION: No expectation was provided to compare against

CRITICAL: Return ONLY valid JSON, no additional text or explanation.
"""

class ActionAnalysisOrchestrator:
    """LangGraph-based orchestrator for action analysis"""
    
    def __init__(self, llm_model: str = "deepseek-chat"):
        self.llm_model = llm_model
        self.llm = ChatDeepSeek(
            model=llm_model,
            temperature=0.3,  # Lower temperature for more consistent analysis
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    
    def analyze_action_effect(self, action_name: str, action_data: dict, before_frame: FrameData, after_frame: FrameData, expectation: str = "", current_reasoning: str = "unknown meaning", current_confidence: str = "UNKNOWN") -> Dict[str, Dict[str, Any]]:
        """Analyze what an action did by comparing before/after frames"""
        
        # Format frames for analysis
        before_frame_str = json.dumps(before_frame.frame, indent=2)
        after_frame_str = json.dumps(after_frame.frame, indent=2)
        action_data_str = json.dumps(action_data, indent=2)
        
        # Handle expectation
        expectation_section = ""
        expectation_task = ""
        
        if expectation and expectation.strip():
            try:
                # Try to parse expectation as JSON grid
                expected_grid = json.loads(expectation) if isinstance(expectation, str) else expectation
                expectation_str = json.dumps(expected_grid, indent=2)
                expectation_section = f"EXPECTED FRAME (from knowledge acquisition):\n{expectation_str}\n"
                expectation_task = " and validate if the result matches the expectation"
            except (json.JSONDecodeError, TypeError):
                # If expectation is not valid JSON, treat as text description
                expectation_section = f"EXPECTED OUTCOME (from knowledge acquisition): {expectation}\n"
                expectation_task = " and validate if the result matches the expected outcome"
        
        prompt = ACTION_ANALYSIS_TEMPLATE.format(
            action_name=action_name,
            current_reasoning=current_reasoning,
            current_confidence=current_confidence,
            before_frame=before_frame_str,
            action_data=action_data_str,
            after_frame=after_frame_str,
            expectation_section=expectation_section,
            expectation_task=expectation_task
        )
        
        # Add debug logging
        logger.debug(f"[DEBUG] Action analysis prompt for {action_name}:")
        logger.debug(f"[DEBUG] Prompt length: {len(prompt)} characters")
        logger.debug(f"[DEBUG] Has expectation: {bool(expectation and expectation.strip())}")
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Add debug logging for LLM response
            logger.debug(f"[DEBUG] Raw LLM response for {action_name}: '{response.content}'")
            logger.debug(f"[DEBUG] Response length: {len(response.content)} characters")
            
            # Check for empty response
            if not response.content or not response.content.strip():
                logger.error(f"[ANALYSIS] Empty response from LLM for {action_name}")
                return {}
            
            # Clean the response - remove any markdown formatting or extra text
            cleaned_content = response.content.strip()
            
            # Try to extract JSON if wrapped in markdown
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.replace("```json", "").replace("```", "").strip()
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content.replace("```", "").strip()
            
            # Try to parse the JSON response
            try:
                analysis_result = json.loads(cleaned_content)
                logger.info(f"[ANALYSIS] Successfully analyzed {action_name}")
                logger.debug(f"[DEBUG] Parsed analysis result: {analysis_result}")
                return analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"[ANALYSIS] Failed to parse JSON for {action_name}: {e}")
                logger.error(f"[ANALYSIS] Cleaned content was: '{cleaned_content}'")
                
                # Try to fix common JSON issues
                try:
                    # Remove any trailing commas
                    fixed_content = cleaned_content.rstrip(',')
                    # Try parsing again
                    analysis_result = json.loads(fixed_content)
                    logger.info(f"[ANALYSIS] Successfully parsed after cleaning for {action_name}")
                    return analysis_result
                except json.JSONDecodeError as e2:
                    logger.error(f"[ANALYSIS] Still failed after cleaning: {e2}")
                    return {}
                
        except Exception as e:
            logger.error(f"[ANALYSIS] LLM call failed for {action_name}: {e}")
            return {}

class Memory:
    def __init__(self, state_limit: int = 10000):
        self.hypothesized_game_purpose: str = ""
        self.knowledge_tree: KnowledgeTree = None
        self.actions: ActionRegistry = ActionRegistry()
        self.current_plan: str = ""
        self._state_limit = state_limit
        
    def initialize_knowledge_tree(self, root_frame: FrameData, root_description: str = "") -> KnowledgeTree:
        self.knowledge_tree = KnowledgeTree(root_frame, root_description, self._state_limit)
        return self.knowledge_tree

class KnowledgeAcquisitionState(TypedDict):
    """State for the knowledge acquisition LangGraph workflow"""
    frame_data: FrameData
    actions: Dict[str, Any]
    max_steps: int
    proposed_sequence: Optional[List[Dict[str, Any]]]
    analysis_complete: bool

class KnowledgeAcquisitionOrchestrator:
    """LangGraph-based orchestrator for knowledge acquisition"""
    
    def __init__(self, llm_model: str = "deepseek-chat"):
        self.llm_model = llm_model
        self.llm = ChatDeepSeek(
            model=llm_model,
            temperature=0.7,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for knowledge acquisition"""
        workflow = StateGraph(KnowledgeAcquisitionState)
        
        workflow.add_node("analyze_context", self._analyze_context)
        workflow.add_node("generate_sequence", self._generate_sequence)
        workflow.add_node("validate_sequence", self._validate_sequence)
        
        workflow.set_entry_point("analyze_context")
        workflow.add_edge("analyze_context", "generate_sequence")
        workflow.add_edge("generate_sequence", "validate_sequence")
        workflow.add_edge("validate_sequence", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _analyze_context(self, state: KnowledgeAcquisitionState) -> KnowledgeAcquisitionState:
        """Analyze the current game context"""
        logger.info("Analyzing game context for knowledge acquisition")
                
        state["analysis_complete"] = True
        return state
    
    def _generate_sequence(self, state: KnowledgeAcquisitionState) -> KnowledgeAcquisitionState:
        """Generate action sequence using LLM"""
        logger.info("Generating knowledge acquisition sequence")
        
        # Format the frame data for the prompt
        frame_str = json.dumps(state["frame_data"].__dict__, default=str, indent=2)
        actions_str = json.dumps(state["actions"], indent=2)
        
        prompt = KNOWLEDGE_ACQUISITION_PROPOSAL_TEMPLATE.format(
            frame_data=frame_str,
            actions=actions_str,
            max_steps=state["max_steps"]
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Add debug logging for the raw response
            logger.debug(f"[DEBUG] Raw knowledge acquisition response: '{response.content}'")
            logger.debug(f"[DEBUG] Response length: {len(response.content)} characters")
            
            # Check for empty response
            if not response.content or not response.content.strip():
                logger.error("[KNOWLEDGE] Empty response from LLM")
                state["proposed_sequence"] = []
                print("\n❌ Empty response from LLM, using fallback")
                return state
            
            # Clean the response - remove any markdown formatting or extra text
            cleaned_content = response.content.strip()
            
            # Try to extract JSON if wrapped in markdown
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.replace("```json", "").replace("```", "").strip()
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content.replace("```", "").strip()
            
            # Remove any leading/trailing text that might not be JSON
            lines = cleaned_content.split('\n')
            json_start = -1
            json_end = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('['):
                    json_start = i
                    break
            
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().endswith(']'):
                    json_end = i
                    break
            
            if json_start != -1 and json_end != -1:
                json_content = '\n'.join(lines[json_start:json_end + 1])
            else:
                json_content = cleaned_content
            
            # Try to parse the JSON response
            try:
                proposed_sequence = json.loads(json_content)
                if isinstance(proposed_sequence, list):
                    state["proposed_sequence"] = proposed_sequence
                    logger.info(f"[KNOWLEDGE] Successfully generated {len(proposed_sequence)} actions")
                    print(f"\n✅ Knowledge Acquisition - Generated {len(proposed_sequence)} actions:")
                    print(json.dumps(proposed_sequence, indent=2))
                else:
                    # Fallback to empty sequence if parsing fails
                    state["proposed_sequence"] = []
                    logger.error("[KNOWLEDGE] Response is not a list")
                    print("\n❌ Response is not a list, using fallback")
                    
            except json.JSONDecodeError as e:
                logger.error(f"[KNOWLEDGE] Failed to parse JSON: {e}")
                logger.error(f"[KNOWLEDGE] Cleaned content was: '{json_content}'")
                
                # Try to fix common JSON issues
                try:
                    # Remove any trailing commas
                    fixed_content = json_content.rstrip(',')
                    # Try parsing again
                    proposed_sequence = json.loads(fixed_content)
                    if isinstance(proposed_sequence, list):
                        state["proposed_sequence"] = proposed_sequence
                        logger.info(f"[KNOWLEDGE] Successfully parsed after cleaning: {len(proposed_sequence)} actions")
                        print(f"\n✅ Knowledge Acquisition - Generated {len(proposed_sequence)} actions (after cleaning):")
                        print(json.dumps(proposed_sequence, indent=2))
                    else:
                        state["proposed_sequence"] = []
                        print("\n❌ Fixed content is not a list, using fallback")
                        
                except json.JSONDecodeError as e2:
                    state["proposed_sequence"] = []
                    logger.error(f"[KNOWLEDGE] Still failed after cleaning: {e2}")
                    print(f"\n❌ Failed to parse JSON even after cleaning: {e2}")
                
        except Exception as e:
            state["proposed_sequence"] = []
            logger.error(f"[KNOWLEDGE] LLM call failed: {e}")
            print(f"\n❌ LLM call failed: {e}")
        
        return state
    
    def _validate_sequence(self, state: KnowledgeAcquisitionState) -> KnowledgeAcquisitionState:
        """Validate and potentially adjust the generated sequence"""
        logger.info("Validating generated sequence")
        
        if not state["proposed_sequence"]:
            # Create a simple fallback sequence
            state["proposed_sequence"] = [
                {"action": "ACTION1", "expectation": ""},
                {"action": "ACTION2", "expectation": ""}
            ]
        
        # Ensure sequence doesn't exceed max_steps
        if len(state["proposed_sequence"]) > state["max_steps"]:
            state["proposed_sequence"] = state["proposed_sequence"][:state["max_steps"]]
        
        return state
    
    def run_knowledge_acquisition(
        self, 
        frame_data: FrameData, 
        actions: Dict[str, Any], 
        max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """Run the knowledge acquisition workflow"""
        
        initial_state = KnowledgeAcquisitionState(
            frame_data=frame_data,
            actions=actions,
            max_steps=max_steps,
            proposed_sequence=None,
            analysis_complete=False
        )
        
        config = {"configurable": {"thread_id": f"knowledge_acq_{int(datetime.now().timestamp())}"}}
        result = self.graph.invoke(initial_state, config)
        
        return result.get("proposed_sequence", [])

class ActionConfidenceLevel(Enum):
    UNKNOWN = 0.0
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.7
    ALMOST_SURE = 0.9
    ASSERTIVE = 1.0

# Updated template with better JSON parsing instructions
KNOWLEDGE_ACQUISITION_PROPOSAL_TEMPLATE = """
You are an expert in playing ARC-AGI-3 games.
ARC-AGI-3 games are turn-based systems where agents interact with 2D grid environments through a standardized action interface.

Available actions:
- RESET: Initialize or restart the game state
- ACTION1: Simple action - varies by game
- ACTION2: Simple action - varies by game  
- ACTION3: Simple action - varies by game
- ACTION4: Simple action - varies by game
- ACTION5: Simple action - varies by game
- ACTION6: Complex action requiring x,y coordinates (0-63 range)

Current frame data:
{frame_data}

Your current understanding of the actions (including confidence levels):
{actions}

TASK: Purpose an action sequence up to {max_steps} steps to understand these actions.

SITUATION 1: If all actions have confidence 0.0 and reasoning "unknown meaning":
- Create a random exploration sequence to discover what each action does
- No specific expectations needed since actions are completely unknown

SITUATION 2: If some actions have confidence > 0.0 or reasoning beyond "unknown meaning":
- Use your existing knowledge to create a targeted sequence
- Provide clear expectations as grid states showing what you expect the grid to look like after the action

SITUATION 2 EXAMPLES:
If current grid is:
[[0,0,0,0],
 [0,1,0,0],
 [0,0,0,0],
 [0,0,0,0]]

And ACTION1 is known to move player (1) up, expectation would be:
[[0,1,0,0],
 [0,0,0,0],
 [0,0,0,0],
 [0,0,0,0]]

If ACTION6 is known to place a block at coordinates, expectation for ACTION6 at (2,1) would be:
[[0,0,0,0],
 [0,1,2,0],
 [0,0,0,0],
 [0,0,0,0]]

CRITICAL: Return ONLY valid JSON array, no additional text or explanation. Format:
[
    {{"action": "ACTION1", "expectation": "grid state after action or empty if unknown"}},
    {{"action": "ACTION2", "expectation": "grid state after action or empty if unknown"}},
    {{"action": "ACTION6", "expectation": "grid state after action or empty if unknown", "coordinates": [x, y]}}
]
"""

class MCTSAgent(Agent):
    MAX_ACTIONS = 20
    KNOWLEDGE_ACQUISITION_MAX_STEPS = 10
    PURPOSE_FINDING_MAX_STEPS = 10
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory = Memory(state_limit=10000)
        self.knowledge_orchestrator = KnowledgeAcquisitionOrchestrator()
        self.action_analyzer = ActionAnalysisOrchestrator()
        self.action_analysis_enabled = True
        self.last_frame = None  # Store frame before action for comparison
        self.knowledge_sequence = None  # Store knowledge sequence for expectation lookup
        self.sequence_executed = False  # Track if we've executed the full sequence

    @property
    def name(self) -> str:
        return f"{super().name}.mcts.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return need_to_terminate(frames, latest_frame)

    def run_knowledge_acquisition(self, latest_frame: FrameData) -> List[Dict[str, Any]]:
        """Run knowledge acquisition using LangGraph orchestrator"""
        logger.info("Running knowledge acquisition with LangGraph")
        
        # Prepare actions data for the orchestrator
        actions_data = {}
        for action in self.memory.actions.get_all():
            action_confidence = self.memory.actions.action_confidences.get(action.value, ActionConfidenceLevel.UNKNOWN)
            actions_data[action.name] = {
                "name": action.name,
                "reasoning": getattr(action, 'reasoning', 'unknown meaning'),
                "confidence": action_confidence.value  # Use enum value for JSON serialization
            }
        
        return self.knowledge_orchestrator.run_knowledge_acquisition(
            frame_data=latest_frame,
            actions=actions_data,
            max_steps=self.KNOWLEDGE_ACQUISITION_MAX_STEPS
        )
    
    def analyze_action_effect(self, action: GameAction, before_frame: FrameData, after_frame: FrameData) -> None:
        """Analyze the effect of an action and update action registry"""
        if not self.action_analysis_enabled or not before_frame or not after_frame:
            return
            
        logger.info(f"[ANALYSIS] Analyzing effect of {action.name}")
        
        # Get action data for analysis
        action_data = action.action_data.model_dump() if hasattr(action.action_data, 'model_dump') else {}
        
        # Get current understanding of the action
        current_reasoning = getattr(action, 'reasoning', 'unknown meaning')
        current_confidence_level = self.memory.actions.action_confidences.get(action.value, ActionConfidenceLevel.UNKNOWN)
        current_confidence = current_confidence_level.name
        
        # Look for expectation from knowledge sequence
        expectation = ""
        if self.knowledge_sequence:
            for item in self.knowledge_sequence:
                if item.get("action", "").upper() == action.name:
                    expectation = item.get("expectation", "")
                    break
        
        # Analyze the action with current understanding and expectation
        analysis_result = self.action_analyzer.analyze_action_effect(
            action.name, action_data, before_frame, after_frame, expectation, 
            current_reasoning, current_confidence
        )
        
        # Update action registry with analysis
        if analysis_result:
            self.memory.actions.update_action_understanding(analysis_result)
            
        # Note: Individual action analysis files are no longer written
        # Only sequence analysis will be written

    def analyze_action_sequence(self, sequence: List[Dict[str, Any]], initial_frame: FrameData) -> None:
        """Execute all actions first, then batch analyze the before/after frames"""
        logger.info(f"[SEQUENCE] Starting analysis of {len(sequence)} actions")
        
        frames_sequence = [initial_frame]
        executed_actions = []
        current_frame = initial_frame
        
        # Phase 1: Execute all actions and collect frames
        logger.info(f"[SEQUENCE] Phase 1: Executing all {len(sequence)} actions")
        for i, action_item in enumerate(sequence):
            action_name = action_item.get("action", "").upper()
            expectation = action_item.get("expectation", "")
            coordinates = action_item.get("coordinates", None)
            
            logger.info(f"[SEQUENCE] Executing step {i+1}: {action_name}")
            
            try:
                # Create GameAction from name
                if hasattr(GameAction, action_name):
                    game_action = getattr(GameAction, action_name)
                    
                    # Set coordinates if provided
                    if coordinates and action_name == "ACTION6":
                        action_data = {"coordinates": coordinates}
                        game_action.set_data(action_data)
                    
                    # Execute the action
                    before_frame = current_frame
                    after_frame = super(MCTSAgent, self).take_action(game_action)
                    
                    if after_frame:
                        frames_sequence.append(after_frame)
                        current_frame = after_frame
                        
                        # Store action execution info for later analysis
                        executed_actions.append({
                            "step": i + 1,
                            "action_name": action_name,
                            "game_action": game_action,
                            "expectation": expectation,
                            "before_frame": before_frame,
                            "after_frame": after_frame,
                            "success": True
                        })
                        
                        logger.info(f"[SEQUENCE] Step {i+1} executed successfully")
                    else:
                        logger.error(f"[SEQUENCE] Step {i+1}: Action {action_name} returned no frame")
                        executed_actions.append({
                            "step": i + 1,
                            "action_name": action_name,
                            "success": False,
                            "error": "No frame returned"
                        })
                        
                else:
                    logger.error(f"[SEQUENCE] Step {i+1}: Unknown action {action_name}")
                    executed_actions.append({
                        "step": i + 1,
                        "action_name": action_name,
                        "success": False,
                        "error": f"Unknown action {action_name}"
                    })
                    
            except Exception as e:
                logger.error(f"[SEQUENCE] Step {i+1}: Failed to execute {action_name}: {e}")
                executed_actions.append({
                    "step": i + 1,
                    "action_name": action_name,
                    "success": False,
                    "error": str(e)
                })
        
        # Phase 2: Batch analyze all successful action effects
        logger.info(f"[SEQUENCE] Phase 2: Batch analyzing {len([a for a in executed_actions if a['success']])} successful actions")
        analysis_results = []
        
        successful_actions = [action for action in executed_actions if action.get("success", False)]
        
        if successful_actions:
            # Batch analyze all actions at once
            batch_analysis_results = self.batch_analyze_actions(successful_actions)
            
            # Map results back to original sequence order
            result_index = 0
            for action in executed_actions:
                if action.get("success", False):
                    if result_index < len(batch_analysis_results):
                        analysis_results.append(batch_analysis_results[result_index])
                        result_index += 1
                    else:
                        analysis_results.append({})
                else:
                    analysis_results.append({})
            
            # Update action registry with all analysis results
            for analysis_result in batch_analysis_results:
                if analysis_result:
                    self.memory.actions.update_action_understanding(analysis_result)
        else:
            # No successful actions to analyze
            analysis_results = [{}] * len(executed_actions)
        
        # Write comprehensive sequence analysis
        from utils import write_sequence_analysis
        write_sequence_analysis(sequence, frames_sequence, analysis_results)
        
        logger.info(f"[SEQUENCE] Analysis complete. Executed {len(executed_actions)} actions, analyzed {len([r for r in analysis_results if r])} successfully")

    def batch_analyze_actions(self, successful_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch analyze multiple action effects for efficiency"""
        if not successful_actions:
            return []
        
        logger.info(f"[BATCH_ANALYSIS] Analyzing {len(successful_actions)} actions in batch")
        
        # For now, analyze each action individually but in a more efficient manner
        # This can be extended to true batch processing with LLM batch APIs
        batch_results = []
        
        for action_info in successful_actions:
            try:
                logger.info(f"[BATCH_ANALYSIS] Analyzing step {action_info['step']}: {action_info['action_name']}")
                
                # Get current understanding of the action
                game_action = action_info['game_action']
                current_reasoning = getattr(game_action, 'reasoning', 'unknown meaning')
                current_confidence_level = self.memory.actions.action_confidences.get(game_action.value, ActionConfidenceLevel.UNKNOWN)
                current_confidence = current_confidence_level.name
                
                analysis_result = self.action_analyzer.analyze_action_effect(
                    action_info['action_name'],
                    action_info['game_action'].action_data.model_dump() if hasattr(action_info['game_action'].action_data, 'model_dump') else {},
                    action_info['before_frame'],
                    action_info['after_frame'],
                    action_info.get('expectation', ''),
                    current_reasoning,
                    current_confidence
                )
                
                batch_results.append(analysis_result)
                logger.info(f"[BATCH_ANALYSIS] Step {action_info['step']} analyzed successfully")
                
            except Exception as e:
                logger.error(f"[BATCH_ANALYSIS] Failed to analyze step {action_info['step']}: {e}")
                batch_results.append({})
        
        logger.info(f"[BATCH_ANALYSIS] Batch analysis complete. {len([r for r in batch_results if r])} successful analyses")
        return batch_results

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if need_to_reset(frames, latest_frame):
            action = GameAction.RESET
            return action
        
        # Store current frame for comparison after action
        self.last_frame = latest_frame
        
        if self.memory.knowledge_tree is None:
            self.memory.initialize_knowledge_tree(latest_frame, "Fresh ARC-AGI-3 problem")
            
            # Run knowledge acquisition using LangGraph for the first time
            knowledge_acquisition_sequence = self.run_knowledge_acquisition(latest_frame)
            
            # Store the sequence for expectation lookup
            self.knowledge_sequence = knowledge_acquisition_sequence
            
            # Parse the actions from the sequence and write knowledge response files
            from utils import parse_knowledge_sequence, write_knowledge_response
            parsed_actions = parse_knowledge_sequence(knowledge_acquisition_sequence)
            write_knowledge_response(knowledge_acquisition_sequence, latest_frame, parsed_actions)
            
            # Execute and analyze the entire sequence for comprehensive understanding
            if not self.sequence_executed and self.action_analysis_enabled:
                logger.info("[SEQUENCE] Starting comprehensive sequence analysis")
                self.analyze_action_sequence(knowledge_acquisition_sequence, latest_frame)
                self.sequence_executed = True
            
            # Return the first parsed action if available, otherwise RESET
            if parsed_actions:
                return parsed_actions[0]
        
        # Just return RESET for now since you only want to see the knowledge acquisition
        return GameAction.RESET
    
    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Override to add action analysis after taking action"""
        # Take the action using parent method
        result_frame = super().take_action(action)
        
        # Note: Individual action analysis is disabled to avoid duplicate files
        # All analysis is done in analyze_action_sequence() method
        
        return result_frame
