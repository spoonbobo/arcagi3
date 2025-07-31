import random
import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import math

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


class MeaningHypothesis:
    """Represents a hypothesis about what the game wants."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.evidence = 0.0
        self.confidence = 0.0
        self.tests_performed = 0
        self.supporting_observations = []
        self.contradicting_observations = []
        
    def add_evidence(self, observation: Dict[str, Any], supports: bool, strength: float = 1.0):
        """Add evidence for or against this hypothesis."""
        self.tests_performed += 1
        
        if supports:
            self.evidence += strength
            self.supporting_observations.append(observation)
        else:
            self.evidence -= strength * 0.5  # Weigh contradictions less heavily
            self.contradicting_observations.append(observation)
        
        # Update confidence based on consistency
        total_obs = len(self.supporting_observations) + len(self.contradicting_observations)
        if total_obs > 0:
            support_ratio = len(self.supporting_observations) / total_obs
            test_confidence = min(1.0, self.tests_performed / 10.0)
            self.confidence = support_ratio * test_confidence
    
    def get_priority(self) -> float:
        """Get priority for testing this hypothesis (higher = test sooner)."""
        if self.tests_performed == 0:
            return 10.0  # Untested hypotheses have highest priority
        
        # Balance between promising hypotheses and uncertain ones
        uncertainty = 1.0 - self.confidence
        promise = max(0, self.evidence)
        
        return promise + uncertainty * 0.5


class ActiveMeaningSearcher:
    """Actively searches for meaning by forming and testing hypotheses."""
    
    def __init__(self):
        self.hypotheses = {}
        self.current_experiment = None
        self.experiment_step = 0
        
        # Initialize meaning hypotheses
        self._initialize_hypotheses()
        
        # Frame analysis for meaning detection
        self.frame_history = deque(maxlen=50)
        self.score_events = []  # Track when scores change and context
        
    def _initialize_hypotheses(self):
        """Initialize hypotheses about game meaning."""
        
        hypotheses_list = [
            ("player_movement", "There's a player/avatar that moves with directional actions"),
            ("object_collection", "Goal is to collect specific objects or items"),
            ("spatial_puzzle", "Need to arrange objects in specific spatial patterns"),
            ("pathfinding", "Navigate through obstacles to reach a target"),
            ("pattern_matching", "Create or match specific visual patterns"),
            ("resource_management", "Manage limited resources (energy, items, etc.)"),
            ("sequence_completion", "Complete actions in a specific sequence"),
            ("transformation_puzzle", "Transform objects from one state to another"),
            ("territorial_control", "Control or fill specific areas of the grid"),
            ("logical_constraints", "Satisfy logical rules or constraints"),
            ("timing_based", "Actions must be timed correctly"),
            ("memory_game", "Remember and repeat patterns or sequences")
        ]
        
        for name, description in hypotheses_list:
            self.hypotheses[name] = MeaningHypothesis(name, description)
    
    def observe_transition(self, before_frame: List[List[List[int]]], 
                          after_frame: List[List[List[int]]], 
                          action: GameAction, score_change: float,
                          action_signature: Dict[str, Any]):
        """Observe a transition and update hypothesis evidence."""
        
        # Store frame transition
        transition = {
            'before': before_frame,
            'after': after_frame,
            'action': action,
            'score_change': score_change,
            'signature': action_signature,
            'step': len(self.frame_history)
        }
        
        self.frame_history.append(transition)
        
        # Record score events
        if score_change != 0:
            self.score_events.append({
                'step': len(self.frame_history) - 1,
                'change': score_change,
                'action': action,
                'context': self._analyze_score_context(transition)
            })
            print(f"SCORE EVENT: {action.name} caused {score_change:+.1f} at step {len(self.frame_history)}")
        
        # Test hypotheses against this observation
        self._test_hypotheses_against_observation(transition)
        
        # Update current experiment if running
        if self.current_experiment:
            self._update_experiment(transition)
    
    def _analyze_score_context(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the context when a score change occurred."""
        
        before_analysis = self._analyze_frame_content(transition['before'])
        after_analysis = self._analyze_frame_content(transition['after'])
        
        return {
            'before_content': before_analysis,
            'after_content': after_analysis,
            'action': transition['action'].name,
            'signature': transition['signature']
        }
    
    def _analyze_frame_content(self, frame: List[List[List[int]]]) -> Dict[str, Any]:
        """Analyze frame content for meaning."""
        
        if not frame or not frame[0]:
            return {}
        
        grid = np.array(frame[0])
        
        analysis = {
            'unique_values': sorted(np.unique(grid).tolist()),
            'value_counts': {int(val): int(count) for val, count in 
                           zip(*np.unique(grid, return_counts=True))},
            'density': float(np.mean(grid != 0)),
            'shape': grid.shape,
            'clusters': self._detect_clusters(grid),
            'borders': self._analyze_borders(grid),
            'patterns': self._detect_visual_patterns(grid)
        }
        
        return analysis
    
    def _detect_clusters(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect clusters of similar values."""
        clusters = []
        
        for value in np.unique(grid):
            if value == 0:  # Skip background
                continue
                
            positions = list(zip(*np.where(grid == value)))
            if len(positions) > 0:
                # Calculate cluster properties
                positions_array = np.array(positions)
                center = np.mean(positions_array, axis=0)
                spread = np.std(positions_array, axis=0) if len(positions) > 1 else [0, 0]
                
                clusters.append({
                    'value': int(value),
                    'count': len(positions),
                    'center': (float(center[0]), float(center[1])),
                    'spread': (float(spread[0]), float(spread[1])),
                    'positions': positions[:10]  # Limit for memory
                })
        
        return clusters
    
    def _analyze_borders(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze border content."""
        h, w = grid.shape
        if h < 3 or w < 3:
            return {}
        
        top_border = grid[0, :]
        bottom_border = grid[-1, :]
        left_border = grid[:, 0]
        right_border = grid[:, -1]
        
        return {
            'top_values': np.unique(top_border).tolist(),
            'bottom_values': np.unique(bottom_border).tolist(),
            'left_values': np.unique(left_border).tolist(),
            'right_values': np.unique(right_border).tolist(),
            'border_density': float(np.mean(np.concatenate([top_border, bottom_border, left_border, right_border]) != 0))
        }
    
    def _detect_visual_patterns(self, grid: np.ndarray) -> Dict[str, bool]:
        """Detect visual patterns."""
        return {
            'symmetric_h': np.array_equal(grid, np.fliplr(grid)),
            'symmetric_v': np.array_equal(grid, np.flipud(grid)),
            'has_single_object': len(np.unique(grid)) == 2,  # background + one object
            'has_multiple_objects': len(np.unique(grid)) > 3,
            'sparse': np.mean(grid != 0) < 0.2,
            'dense': np.mean(grid != 0) > 0.8
        }
    
    def _test_hypotheses_against_observation(self, transition: Dict[str, Any]):
        """Test all hypotheses against new observation."""
        
        for hypothesis in self.hypotheses.values():
            supports, strength = self._evaluate_hypothesis(hypothesis, transition)
            if strength > 0:  # Only update if we have meaningful evidence
                hypothesis.add_evidence(transition, supports, strength)
    
    def _evaluate_hypothesis(self, hypothesis: MeaningHypothesis, 
                           transition: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate if transition supports a hypothesis."""
        
        name = hypothesis.name
        score_change = transition['score_change']
        action = transition['action']
        signature = transition['signature']
        
        # Player movement hypothesis
        if name == "player_movement":
            movement = signature.get('movement_vector', (0, 0))
            has_movement = abs(movement[0]) > 0.5 or abs(movement[1]) > 0.5
            
            if action.name in ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'] and has_movement:
                return True, 0.8  # Strong evidence for movement
            elif action.name in ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'] and not has_movement:
                return False, 0.3  # Weak contradiction
        
        # Object collection hypothesis
        elif name == "object_collection":
            before_content = self._analyze_frame_content(transition['before'])
            after_content = self._analyze_frame_content(transition['after'])
            
            # Check if objects disappeared (collected)
            before_objects = len(before_content.get('clusters', []))
            after_objects = len(after_content.get('clusters', []))
            
            if score_change > 0 and after_objects < before_objects:
                return True, 1.0  # Strong evidence for collection
            elif score_change > 0 and after_objects > before_objects:
                return False, 0.5  # Contradicts collection
        
        # Spatial puzzle hypothesis
        elif name == "spatial_puzzle":
            before_analysis = self._analyze_frame_content(transition['before'])
            after_analysis = self._analyze_frame_content(transition['after'])
            
            # Check if pattern completion occurred
            before_patterns = before_analysis.get('patterns', {})
            after_patterns = after_analysis.get('patterns', {})
            
            pattern_improved = (
                after_patterns.get('symmetric_h', False) and not before_patterns.get('symmetric_h', False) or
                after_patterns.get('symmetric_v', False) and not before_patterns.get('symmetric_v', False)
            )
            
            if score_change > 0 and pattern_improved:
                return True, 0.9  # Strong evidence for spatial puzzle
        
        # Pathfinding hypothesis
        elif name == "pathfinding":
            # Check for consistent directional movement toward goals
            movement = signature.get('movement_vector', (0, 0))
            has_consistent_movement = signature.get('movement_consistency', 0) > 0.7
            
            if score_change > 0 and has_consistent_movement:
                return True, 0.7  # Evidence for pathfinding
        
        # Pattern matching hypothesis
        elif name == "pattern_matching":
            signature_confidence = signature.get('confidence', 0)
            change_pattern = signature.get('common_change_pattern', '')
            
            if score_change > 0 and 'pattern' in change_pattern.lower():
                return True, 0.6
        
        # Resource management hypothesis
        elif name == "resource_management":
            # Look for gradual score decrease followed by increase
            if len(self.score_events) >= 2:
                recent_changes = [e['change'] for e in self.score_events[-3:]]
                if any(c < 0 for c in recent_changes) and score_change > 0:
                    return True, 0.5
        
        # Add more hypothesis evaluations...
        
        return False, 0.0  # No evidence
    
    def plan_active_experiment(self) -> Optional[Dict[str, Any]]:
        """Plan an active experiment to test hypotheses."""
        
        # Find hypothesis with highest priority
        best_hypothesis = max(self.hypotheses.values(), key=lambda h: h.get_priority())
        
        if best_hypothesis.get_priority() < 0.1:
            return None  # No experiments needed
        
        # Design experiment based on hypothesis
        experiment = self._design_experiment(best_hypothesis)
        
        if experiment:
            self.current_experiment = experiment
            self.experiment_step = 0
            print(f"STARTING EXPERIMENT: {experiment['name']} to test '{best_hypothesis.name}'")
        
        return experiment
    
    def _design_experiment(self, hypothesis: MeaningHypothesis) -> Optional[Dict[str, Any]]:
        """Design an experiment to test a specific hypothesis."""
        
        name = hypothesis.name
        
        if name == "player_movement":
            return {
                'name': f"test_directional_movement",
                'hypothesis': name,
                'steps': [
                    {'action': 'ACTION1', 'repeat': 3, 'expect': 'consistent_upward_movement'},
                    {'action': 'ACTION2', 'repeat': 3, 'expect': 'consistent_downward_movement'},
                    {'action': 'ACTION3', 'repeat': 3, 'expect': 'consistent_leftward_movement'},
                    {'action': 'ACTION4', 'repeat': 3, 'expect': 'consistent_rightward_movement'}
                ]
            }
        
        elif name == "object_collection":
            return {
                'name': f"test_object_interaction",
                'hypothesis': name,
                'steps': [
                    {'action': 'ACTION6', 'target': 'object_position', 'expect': 'object_disappears_score_increases'},
                    {'action': 'ACTION1', 'repeat': 2, 'expect': 'move_toward_objects'},
                    {'action': 'ACTION6', 'target': 'object_position', 'expect': 'object_disappears_score_increases'}
                ]
            }
        
        elif name == "spatial_puzzle":
            return {
                'name': f"test_pattern_creation",
                'hypothesis': name,
                'steps': [
                    {'action': 'ACTION6', 'target': 'symmetry_position', 'expect': 'pattern_completion'},
                    {'action': 'ACTION1', 'repeat': 1, 'expect': 'pattern_adjustment'},
                    {'action': 'ACTION6', 'target': 'symmetry_position', 'expect': 'pattern_completion'}
                ]
            }
        
        elif name == "pathfinding":
            return {
                'name': f"test_navigation",
                'hypothesis': name,
                'steps': [
                    {'action': 'ACTION1', 'repeat': 5, 'expect': 'progress_toward_goal'},
                    {'action': 'ACTION4', 'repeat': 5, 'expect': 'progress_toward_goal'},
                    {'action': 'ACTION2', 'repeat': 5, 'expect': 'progress_toward_goal'}
                ]
            }
        
        return None
    
    def get_next_experimental_action(self) -> Optional[Tuple[GameAction, Dict[str, Any]]]:
        """Get next action for current experiment."""
        
        if not self.current_experiment:
            return None
        
        steps = self.current_experiment['steps']
        if self.experiment_step >= len(steps):
            self._conclude_experiment()
            return None
        
        step = steps[self.experiment_step]
        action_name = step['action']
        
        try:
            action = getattr(GameAction, action_name)
            context = {
                'experiment': self.current_experiment['name'],
                'hypothesis': self.current_experiment['hypothesis'],
                'step': self.experiment_step,
                'expectation': step.get('expect', 'unknown')
            }
            
            self.experiment_step += 1
            return action, context
        except:
            self._conclude_experiment()
            return None
    
    def _update_experiment(self, transition: Dict[str, Any]):
        """Update current experiment with new observation."""
        
        if not self.current_experiment:
            return
        
        # Check if expectation was met
        # This would be more sophisticated in practice
        score_change = transition['score_change']
        
        if score_change > 0:
            print(f"EXPERIMENT SUCCESS: {self.current_experiment['name']} step {self.experiment_step-1} - score increased!")
        
        # Continue or conclude experiment based on results
        if self.experiment_step >= len(self.current_experiment['steps']):
            self._conclude_experiment()
    
    def _conclude_experiment(self):
        """Conclude current experiment."""
        if self.current_experiment:
            print(f"CONCLUDED EXPERIMENT: {self.current_experiment['name']}")
            self.current_experiment = None
            self.experiment_step = 0
    
    def get_best_hypothesis(self) -> Tuple[str, float]:
        """Get the most confident hypothesis."""
        if not self.hypotheses:
            return "unknown", 0.0
        
        best = max(self.hypotheses.values(), key=lambda h: h.confidence)
        return best.name, best.confidence
    
    def get_meaning_summary(self) -> Dict[str, Any]:
        """Get summary of discovered meaning."""
        
        return {
            'best_hypothesis': self.get_best_hypothesis(),
            'score_events': len(self.score_events),
            'hypotheses': {
                name: {
                    'confidence': h.confidence,
                    'evidence': h.evidence,
                    'tests': h.tests_performed
                }
                for name, h in self.hypotheses.items()
                if h.tests_performed > 0 or h.confidence > 0.1
            },
            'current_experiment': self.current_experiment['name'] if self.current_experiment else None
        }


class ArcAgent(Agent):
    """
    Active meaning search agent that forms hypotheses and designs experiments.
    """

    MAX_ACTIONS = 500

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32))
        
        # Active meaning searcher
        self.meaning_searcher = ActiveMeaningSearcher()
        
        # Keep previous frame analysis capabilities
        self.frame_analyzer = FrameChangeAnalyzer()
        self.mh_selector = MetropolisHastingsActionSelector(list(GameAction))
        
        # State tracking
        self.last_action: Optional[GameAction] = None
        self.last_frame: Optional[List[List[List[int]]]] = None
        self.last_score: int = 0
        self.last_coordinates: Optional[Tuple[int, int]] = None
        
        # Experiment tracking
        self.experiments_run = 0
        self.hypotheses_tested = 0

    @property
    def name(self) -> str:
        return f"active-meaning-search.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any([
            latest_frame.state is GameState.WIN,
            latest_frame.state is GameState.GAME_OVER,
        ])

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose action using active meaning search."""
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        
        try:
            current_score = latest_frame.score
            current_frame = latest_frame.frame
            
            # Learn from previous transition
            if self.last_action is not None and self.last_frame is not None:
                self._analyze_and_learn(current_frame, current_score)
            
            # Select action using active meaning search
            action, reasoning_context = self._select_meaningful_action(current_frame)
            
            # Configure action
            self._configure_meaningful_action(action, current_frame, current_score, reasoning_context)
            
            # Update state
            self.last_action = action
            self.last_frame = current_frame
            self.last_score = current_score
            
            return action
            
        except Exception as e:
            print(f"Error in active meaning search: {e}")
            return self._fallback_action()
    
    def _analyze_and_learn(self, current_frame: List[List[List[int]]], current_score: int):
        """Analyze transition and update meaning understanding."""
        
        # Frame change analysis
        change_analysis = self.frame_analyzer.analyze_frame_change(
            self.last_frame, current_frame, self.last_action
        )
        
        # Action signature
        signature = self.frame_analyzer.get_action_signature(self.last_action)
        
        # Update meaning searcher
        score_change = current_score - self.last_score
        self.meaning_searcher.observe_transition(
            self.last_frame, current_frame, self.last_action, score_change, signature
        )
        
        # Update MH selector
        frame_change_quality = self._evaluate_frame_change_quality(change_analysis, current_score)
        self.mh_selector.update_action_utility(
            self.last_action, frame_change_quality, signature.get("confidence", 0.0)
        )
    
    def _select_meaningful_action(self, current_frame: List[List[List[int]]]) -> Tuple[GameAction, Dict[str, Any]]:
        """Select action to actively search for meaning."""
        
        # Check if we should run an experiment
        experimental_action = self.meaning_searcher.get_next_experimental_action()
        if experimental_action:
            action, context = experimental_action
            context['selection_type'] = 'experiment'
            return action, context
        
        # Plan new experiment if needed
        if self.action_counter % 20 == 0:  # Every 20 steps, consider new experiment
            experiment = self.meaning_searcher.plan_active_experiment()
            if experiment:
                self.experiments_run += 1
                experimental_action = self.meaning_searcher.get_next_experimental_action()
                if experimental_action:
                    action, context = experimental_action
                    context['selection_type'] = 'new_experiment'
                    return action, context
        
        # Use Metropolis-Hastings for general exploration
        current_features = self._extract_state_features(current_frame)
        action = self.mh_selector.metropolis_hastings_step(current_features)
        
        context = {
            'selection_type': 'mh_exploration',
            'hypothesis': self.meaning_searcher.get_best_hypothesis()[0]
        }
        
        return action, context
    
    def _extract_state_features(self, frame: List[List[List[int]]]) -> Dict[str, Any]:
        """Extract features for decision making."""
        if not frame or not frame[0]:
            return {"empty": True}
        
        grid = np.array(frame[0])
        return {
            "density": float(np.mean(grid != 0)),
            "unique_values": len(np.unique(grid)),
            "center_mass": self._get_center_mass(grid)
        }
    
    def _get_center_mass(self, grid: np.ndarray) -> Tuple[float, float]:
        """Get center of mass of non-zero elements."""
        non_zero = np.where(grid != 0)
        if len(non_zero[0]) == 0:
            return (0.5, 0.5)
        return (float(np.mean(non_zero[1]) / grid.shape[1]), 
                float(np.mean(non_zero[0]) / grid.shape[0]))
    
    def _evaluate_frame_change_quality(self, change_analysis: Dict[str, Any], current_score: int) -> float:
        """Evaluate quality of frame change."""
        changed_pixels = change_analysis.get("changed_pixels", 0)
        if changed_pixels == 0:
            return 0.0
        
        base_quality = min(1.0, changed_pixels / 20.0)
        score_bonus = 2.0 if current_score > self.last_score else 0.0
        return base_quality + score_bonus
    
    def _configure_meaningful_action(self, action: GameAction, current_frame: List[List[List[int]]], 
                                   current_score: int, reasoning_context: Dict[str, Any]):
        """Configure action with meaning-search reasoning."""
        
        if action.is_simple():
            action.reasoning = self._generate_meaning_reasoning(action, current_score, reasoning_context)
            
        elif action.is_complex():
            coordinates = self._select_meaningful_coordinates(action, current_frame, reasoning_context)
            action.set_data({"x": coordinates[0], "y": coordinates[1]})
            self.last_coordinates = coordinates
            
            action.reasoning = {
                "action_type": action.name,
                "coordinates": coordinates,
                "meaning_reasoning": self._generate_meaning_reasoning(action, current_score, reasoning_context),
                "selection_type": reasoning_context.get('selection_type', 'unknown'),
                "hypothesis": reasoning_context.get('hypothesis', 'unknown')
            }
    
    def _select_meaningful_coordinates(self, action: GameAction, 
                                     current_frame: List[List[List[int]]], 
                                     reasoning_context: Dict[str, Any]) -> Tuple[int, int]:
        """Select coordinates based on meaning search."""
        
        # Use experiment-specific coordinate selection if in experiment
        if reasoning_context.get('selection_type') == 'experiment':
            return self._select_experimental_coordinates(action, current_frame, reasoning_context)
        
        # Use learned action signature
        signature = self.frame_analyzer.get_action_signature(action)
        return self._select_signature_coordinates(action, current_frame, signature)
    
    def _select_experimental_coordinates(self, action: GameAction, 
                                       current_frame: List[List[List[int]]], 
                                       context: Dict[str, Any]) -> Tuple[int, int]:
        """Select coordinates for experimental purposes."""
        
        expectation = context.get('expectation', '')
        
        if 'object_position' in expectation:
            # Try to target objects
            return self._find_object_position(current_frame)
        elif 'symmetry_position' in expectation:
            # Try to create symmetry
            return self._find_symmetry_position(current_frame)
        else:
            # Default systematic exploration
            return (32, 32)
    
    def _find_object_position(self, current_frame: List[List[List[int]]]) -> Tuple[int, int]:
        """Find position of an object to target."""
        if not current_frame or not current_frame[0]:
            return (32, 32)
        
        grid = np.array(current_frame[0])
        non_zero_positions = list(zip(*np.where(grid != 0)))
        
        if non_zero_positions:
            return random.choice(non_zero_positions)
        return (32, 32)
    
    def _find_symmetry_position(self, current_frame: List[List[List[int]]]) -> Tuple[int, int]:
        """Find position that would create symmetry."""
        # Simplified symmetry position finding
        return (32, 32)  # Center for now
    
    def _select_signature_coordinates(self, action: GameAction, 
                                    current_frame: List[List[List[int]]], 
                                    signature: Dict[str, Any]) -> Tuple[int, int]:
        """Select coordinates based on learned action signature."""
        
        # Use movement vector if confident
        movement = signature.get("movement_vector", (0, 0))
        confidence = signature.get("movement_consistency", 0.0)
        
        if confidence > 0.5:
            center_mass = self._get_center_mass(np.array(current_frame[0]) if current_frame and current_frame[0] else np.zeros((1, 1)))
            target_x = center_mass[0] * 63 - movement[0] * 10
            target_y = center_mass[1] * 63 - movement[1] * 10
            
            return (int(np.clip(target_x, 0, 63)), int(np.clip(target_y, 0, 63)))
        
        return (random.randint(0, 63), random.randint(0, 63))
    
    def _generate_meaning_reasoning(self, action: GameAction, current_score: int, 
                                  context: Dict[str, Any]) -> str:
        """Generate reasoning focused on meaning search."""
        
        selection_type = context.get('selection_type', 'unknown')
        hypothesis = context.get('hypothesis', 'unknown')
        best_hyp, confidence = self.meaning_searcher.get_best_hypothesis()
        
        parts = [f"MEANING: {action.name}"]
        
        if selection_type == 'experiment':
            experiment_name = context.get('experiment', 'unknown')
            parts.append(f"[EXP:{experiment_name}]")
        elif selection_type == 'new_experiment':
            parts.append("[NEW-EXP]")
        else:
            parts.append("[EXPLORE]")
        
        # Show best hypothesis
        if confidence > 0.3:
            parts.append(f"[{best_hyp}:{confidence:.1%}]")
        else:
            parts.append("[seeking-meaning]")
        
        parts.append(f"score={current_score}")
        
        return " ".join(parts)
    
    def _fallback_action(self) -> GameAction:
        """Fallback action."""
        actions = [a for a in GameAction if a != GameAction.RESET]
        action = random.choice(actions)
        
        if action.is_simple():
            action.reasoning = f"MEANING-FALLBACK: {action.name}"
        elif action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"fallback": True, "meaning_search": True}
        
        return action
    
    def cleanup(self, scorecard=None):
        """Enhanced cleanup with meaning search results."""
        try:
            print(f"\n=== ACTIVE MEANING SEARCH SUMMARY ===")
            
            meaning_summary = self.meaning_searcher.get_meaning_summary()
            best_hypothesis, confidence = meaning_summary['best_hypothesis']
            
            print(f"Best hypothesis: {best_hypothesis} ({confidence:.1%} confidence)")
            print(f"Score events observed: {meaning_summary['score_events']}")
            print(f"Experiments run: {self.experiments_run}")
            
            print("\nHypothesis Evidence:")
            for name, data in meaning_summary['hypotheses'].items():
                conf = data['confidence']
                evidence = data['evidence']
                tests = data['tests']
                print(f"  {name}: {conf:.1%} confidence, {evidence:.1f} evidence, {tests} tests")
            
            if meaning_summary['current_experiment']:
                print(f"\nActive experiment: {meaning_summary['current_experiment']}")
            
            # Also show technical details
            mh_summary = self.mh_selector.get_action_relationship_summary()
            print(f"\nMCMC acceptance rate: {mh_summary['acceptance_rate']:.1%}")
            
        except Exception as e:
            print(f"Error in meaning search cleanup: {e}")
        
        super().cleanup(scorecard)

# Import and update frame analyzer from previous implementation
class FrameChangeAnalyzer:
    """Analyzes specific frame changes caused by actions."""
    
    def __init__(self):
        self.change_patterns = defaultdict(list)
        
    def analyze_frame_change(self, before_frame: List[List[List[int]]], 
                           after_frame: List[List[List[int]]], 
                           action: GameAction) -> Dict[str, Any]:
        """Analyze frame changes."""
        if not before_frame or not after_frame or not before_frame[0] or not after_frame[0]:
            return {}
        
        before_grid = np.array(before_frame[0])
        after_grid = np.array(after_frame[0])
        
        diff_mask = (before_grid != after_grid)
        changed_positions = list(zip(*np.where(diff_mask)))
        
        change_analysis = {
            "changed_pixels": len(changed_positions),
            "movement_vector": self._detect_movement(before_grid, after_grid),
            "change_pattern": "major_change" if len(changed_positions) > 10 else "few_pixels"
        }
        
        self.change_patterns[action].append(change_analysis)
        if len(self.change_patterns[action]) > 30:
            self.change_patterns[action] = self.change_patterns[action][-20:]
        
        return change_analysis
    
    def _detect_movement(self, before: np.ndarray, after: np.ndarray) -> Tuple[float, float]:
        """Detect movement between frames."""
        before_nonzero = np.where(before != 0)
        after_nonzero = np.where(after != 0)
        
        if len(before_nonzero[0]) == 0 or len(after_nonzero[0]) == 0:
            return (0.0, 0.0)
        
        before_center = (np.mean(before_nonzero[0]), np.mean(before_nonzero[1]))
        after_center = (np.mean(after_nonzero[0]), np.mean(after_nonzero[1]))
        
        return (after_center[1] - before_center[1], after_center[0] - before_center[0])
    
    def get_action_signature(self, action: GameAction) -> Dict[str, Any]:
        """Get action signature."""
        if action not in self.change_patterns or not self.change_patterns[action]:
            return {"confidence": 0.0, "movement_vector": (0, 0)}
        
        patterns = self.change_patterns[action]
        movements = [p.get("movement_vector", (0, 0)) for p in patterns]
        
        avg_movement_x = np.mean([m[0] for m in movements])
        avg_movement_y = np.mean([m[1] for m in movements])
        
        return {
            "movement_vector": (avg_movement_x, avg_movement_y),
            "confidence": min(1.0, len(patterns) / 10.0),
            "attempts": len(patterns)
        }


class MetropolisHastingsActionSelector:
    """Simplified MH selector for meaning search."""
    
    def __init__(self, actions: List[GameAction]):
        self.actions = [a for a in actions if a != GameAction.RESET]
        self.action_utilities = {action: 0.0 for action in self.actions}
        self.current_action = None
        self.temperature = 1.0
        
    def update_action_utility(self, action: GameAction, utility: float, confidence: float):
        """Update action utility."""
        alpha = 0.3
        self.action_utilities[action] = (1 - alpha) * self.action_utilities[action] + alpha * utility
    
    def metropolis_hastings_step(self, features: Dict[str, Any]) -> GameAction:
        """Simple MH step."""
        if self.current_action is None:
            self.current_action = random.choice(self.actions)
        
        proposed = random.choice(self.actions)
        
        current_utility = self.action_utilities[self.current_action]
        proposed_utility = self.action_utilities[proposed]
        
        if random.random() < min(1.0, math.exp((proposed_utility - current_utility) / self.temperature)):
            self.current_action = proposed
        
        return self.current_action
    
    def get_action_relationship_summary(self) -> Dict[str, Any]:
        """Get summary."""
        return {
            "action_utilities": dict(self.action_utilities),
            "acceptance_rate": 0.5,  # Simplified
            "temperature": self.temperature
        }

