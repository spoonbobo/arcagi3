import random
import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import math

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


class FrameChangeAnalyzer:
    """Analyzes specific frame changes caused by actions."""
    
    def __init__(self):
        self.change_patterns = defaultdict(list)  # action -> list of change patterns
        self.frame_diffs = defaultdict(list)      # action -> list of frame differences
        
    def analyze_frame_change(self, before_frame: List[List[List[int]]], 
                           after_frame: List[List[List[int]]], 
                           action: GameAction) -> Dict[str, Any]:
        """Analyze the specific changes an action caused."""
        
        if not before_frame or not after_frame or not before_frame[0] or not after_frame[0]:
            return {}
        
        before_grid = np.array(before_frame[0])
        after_grid = np.array(after_frame[0])
        
        # Calculate pixel-level differences
        diff_mask = (before_grid != after_grid)
        
        if not np.any(diff_mask):
            return {"change_type": "no_change", "changed_pixels": 0}
        
        # Extract change details
        changed_positions = list(zip(*np.where(diff_mask)))
        
        change_analysis = {
            "change_type": "pixels_changed",
            "changed_pixels": len(changed_positions),
            "changed_positions": changed_positions[:20],  # Limit for memory
            "before_values": [int(before_grid[pos]) for pos in changed_positions[:20]],
            "after_values": [int(after_grid[pos]) for pos in changed_positions[:20]],
            "change_pattern": self._identify_change_pattern(before_grid, after_grid, diff_mask),
            "movement_vector": self._detect_movement(before_grid, after_grid),
            "value_transformations": self._analyze_value_changes(before_grid, after_grid, diff_mask)
        }
        
        # Store this pattern for the action
        self.change_patterns[action].append(change_analysis)
        
        # Keep bounded history
        if len(self.change_patterns[action]) > 30:
            self.change_patterns[action] = self.change_patterns[action][-20:]
        
        return change_analysis
    
    def _identify_change_pattern(self, before: np.ndarray, after: np.ndarray, 
                                diff_mask: np.ndarray) -> str:
        """Identify the type of change pattern."""
        
        # Count types of changes
        total_changes = np.sum(diff_mask)
        
        if total_changes == 0:
            return "no_change"
        elif total_changes == 1:
            return "single_pixel"
        elif total_changes <= 4:
            return "few_pixels"
        elif total_changes <= 20:
            return "block_change"
        else:
            return "major_change"
    
    def _detect_movement(self, before: np.ndarray, after: np.ndarray) -> Tuple[float, float]:
        """Detect if there's a movement pattern."""
        
        # Simple center of mass comparison
        before_nonzero = np.where(before != 0)
        after_nonzero = np.where(after != 0)
        
        if len(before_nonzero[0]) == 0 or len(after_nonzero[0]) == 0:
            return (0.0, 0.0)
        
        before_center = (np.mean(before_nonzero[0]), np.mean(before_nonzero[1]))
        after_center = (np.mean(after_nonzero[0]), np.mean(after_nonzero[1]))
        
        movement_y = after_center[0] - before_center[0]
        movement_x = after_center[1] - before_center[1]
        
        return (movement_x, movement_y)
    
    def _analyze_value_changes(self, before: np.ndarray, after: np.ndarray, 
                              diff_mask: np.ndarray) -> Dict[int, List[int]]:
        """Analyze what values changed to what other values."""
        
        transformations = defaultdict(list)
        
        changed_positions = np.where(diff_mask)
        for i in range(len(changed_positions[0])):
            pos_y, pos_x = changed_positions[0][i], changed_positions[1][i]
            before_val = int(before[pos_y, pos_x])
            after_val = int(after[pos_y, pos_x])
            transformations[before_val].append(after_val)
        
        return dict(transformations)
    
    def get_action_signature(self, action: GameAction) -> Dict[str, Any]:
        """Get the characteristic signature of what this action does."""
        
        if action not in self.change_patterns or not self.change_patterns[action]:
            return {"signature": "unknown", "confidence": 0.0}
        
        patterns = self.change_patterns[action]
        
        # Analyze movement consistency
        movements = [p.get("movement_vector", (0, 0)) for p in patterns]
        avg_movement_x = np.mean([m[0] for m in movements])
        avg_movement_y = np.mean([m[1] for m in movements])
        movement_consistency = self._calculate_movement_consistency(movements)
        
        # Analyze change pattern consistency
        change_types = [p.get("change_pattern", "unknown") for p in patterns]
        most_common_change = max(set(change_types), key=change_types.count)
        change_consistency = change_types.count(most_common_change) / len(change_types)
        
        # Analyze value transformations
        all_transformations = defaultdict(list)
        for pattern in patterns:
            for from_val, to_vals in pattern.get("value_transformations", {}).items():
                all_transformations[from_val].extend(to_vals)
        
        signature = {
            "movement_vector": (avg_movement_x, avg_movement_y),
            "movement_consistency": movement_consistency,
            "common_change_pattern": most_common_change,
            "change_consistency": change_consistency,
            "value_transformations": dict(all_transformations),
            "attempts": len(patterns),
            "confidence": min(movement_consistency, change_consistency)
        }
        
        return signature
    
    def _calculate_movement_consistency(self, movements: List[Tuple[float, float]]) -> float:
        """Calculate how consistent the movement vectors are."""
        if len(movements) < 2:
            return 0.5
        
        movements_array = np.array(movements)
        std_x = np.std(movements_array[:, 0])
        std_y = np.std(movements_array[:, 1])
        
        # Lower standard deviation = higher consistency
        avg_std = (std_x + std_y) / 2
        consistency = 1.0 / (1.0 + avg_std)
        
        return consistency


class MetropolisHastingsActionSelector:
    """Uses Metropolis-Hastings to sample actions based on frame change relationships."""
    
    def __init__(self, actions: List[GameAction], temperature: float = 1.0):
        self.actions = [a for a in actions if a != GameAction.RESET]
        self.temperature = temperature
        self.current_action = None
        
        # Track action effectiveness for MCMC
        self.action_utilities = {action: 0.0 for action in self.actions}
        self.action_confidences = {action: 0.0 for action in self.actions}
        
        # MCMC parameters
        self.acceptance_count = 0
        self.total_proposals = 0
        self.target_acceptance_rate = 0.234  # Optimal for random walk MH
        
    def update_action_utility(self, action: GameAction, frame_change_quality: float, 
                             signature_confidence: float):
        """Update utility based on frame change analysis."""
        
        # Utility = quality of frame changes + confidence in understanding action
        new_utility = frame_change_quality + 0.5 * signature_confidence
        
        # Exponential moving average
        alpha = 0.3
        self.action_utilities[action] = (1 - alpha) * self.action_utilities[action] + alpha * new_utility
        self.action_confidences[action] = (1 - alpha) * self.action_confidences[action] + alpha * signature_confidence
    
    def propose_action(self, current_state_features: Dict[str, Any]) -> GameAction:
        """Propose next action using informed proposal distribution."""
        
        # Create proposal weights based on action utilities
        proposal_weights = []
        for action in self.actions:
            utility = self.action_utilities[action]
            confidence = self.action_confidences[action]
            
            # Weight = utility + exploration bonus for uncertain actions
            exploration_bonus = (1.0 - confidence) * 0.5
            weight = max(0.1, utility + exploration_bonus)
            proposal_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(proposal_weights)
        if total_weight > 0:
            proposal_weights = [w / total_weight for w in proposal_weights]
        else:
            proposal_weights = [1.0 / len(self.actions)] * len(self.actions)
        
        # Sample from proposal distribution
        proposed_action = np.random.choice(self.actions, p=proposal_weights)
        return proposed_action
    
    def metropolis_hastings_step(self, current_state_features: Dict[str, Any]) -> GameAction:
        """Execute one Metropolis-Hastings step."""
        
        if self.current_action is None:
            # Initialize with random action
            self.current_action = random.choice(self.actions)
            return self.current_action
        
        # Propose new action
        proposed_action = self.propose_action(current_state_features)
        self.total_proposals += 1
        
        # Calculate acceptance probability
        current_utility = self.action_utilities[self.current_action]
        proposed_utility = self.action_utilities[proposed_action]
        
        # Metropolis-Hastings acceptance ratio
        utility_diff = (proposed_utility - current_utility) / self.temperature
        acceptance_prob = min(1.0, math.exp(utility_diff))
        
        # Accept or reject
        if random.random() < acceptance_prob:
            self.current_action = proposed_action
            self.acceptance_count += 1
        
        # Adapt temperature based on acceptance rate
        if self.total_proposals % 10 == 0:
            self._adapt_temperature()
        
        return self.current_action
    
    def _adapt_temperature(self):
        """Adapt temperature to maintain target acceptance rate."""
        if self.total_proposals > 0:
            current_acceptance_rate = self.acceptance_count / self.total_proposals
            
            if current_acceptance_rate < self.target_acceptance_rate * 0.8:
                # Too low acceptance, increase temperature (more exploration)
                self.temperature *= 1.1
            elif current_acceptance_rate > self.target_acceptance_rate * 1.2:
                # Too high acceptance, decrease temperature (more exploitation)
                self.temperature *= 0.9
            
            # Keep temperature in reasonable bounds
            self.temperature = max(0.1, min(5.0, self.temperature))
    
    def get_action_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of learned action relationships."""
        return {
            "action_utilities": dict(self.action_utilities),
            "action_confidences": dict(self.action_confidences),
            "acceptance_rate": self.acceptance_count / max(1, self.total_proposals),
            "temperature": self.temperature,
            "total_proposals": self.total_proposals
        }


class ArcAgent(Agent):
    """
    Metropolis-Hastings agent that builds explicit action ↔ frame change relationships.
    """

    MAX_ACTIONS = 1000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32))
        
        # Frame change analyzer
        self.frame_analyzer = FrameChangeAnalyzer()
        
        # Metropolis-Hastings action selector
        self.mh_selector = MetropolisHastingsActionSelector(list(GameAction))
        
        # State tracking
        self.last_action: Optional[GameAction] = None
        self.last_frame: Optional[List[List[List[int]]]] = None
        self.last_score: int = 0
        self.last_coordinates: Optional[Tuple[int, int]] = None
        
        # Learning metrics
        self.total_meaningful_changes = 0
        self.action_discovery_steps = {}

    @property
    def name(self) -> str:
        return f"mh-frame-relations.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any([
            latest_frame.state is GameState.WIN,
            latest_frame.state is GameState.GAME_OVER,
        ])

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose action using Metropolis-Hastings with frame change relationships."""
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        
        try:
            current_score = latest_frame.score
            current_frame = latest_frame.frame
            
            # Analyze previous action's effect
            if self.last_action is not None and self.last_frame is not None:
                self._analyze_and_learn(current_frame, current_score)
            
            # Extract current state features
            current_features = self._extract_state_features(current_frame)
            
            # Select action using Metropolis-Hastings
            action = self.mh_selector.metropolis_hastings_step(current_features)
            
            # Configure action
            self._configure_mh_action(action, current_frame, current_score)
            
            # Update state
            self.last_action = action
            self.last_frame = current_frame
            self.last_score = current_score
            
            return action
            
        except Exception as e:
            print(f"Error in MH frame relations: {e}")
            return self._fallback_action()
    
    def _analyze_and_learn(self, current_frame: List[List[List[int]]], current_score: int):
        """Analyze the frame change and update action relationships."""
        
        # Analyze frame change caused by last action
        change_analysis = self.frame_analyzer.analyze_frame_change(
            self.last_frame, current_frame, self.last_action
        )
        
        # Calculate frame change quality
        frame_change_quality = self._evaluate_frame_change_quality(change_analysis, current_score)
        
        # Get action signature confidence
        signature = self.frame_analyzer.get_action_signature(self.last_action)
        signature_confidence = signature.get("confidence", 0.0)
        
        # Update Metropolis-Hastings utilities
        self.mh_selector.update_action_utility(
            self.last_action, frame_change_quality, signature_confidence
        )
        
        # Track meaningful changes
        if change_analysis.get("changed_pixels", 0) > 0:
            self.total_meaningful_changes += 1
            
            # Record when we first discovered this action causes changes
            if self.last_action not in self.action_discovery_steps:
                self.action_discovery_steps[self.last_action] = self.action_counter
                print(f"DISCOVERED: {self.last_action.name} causes {change_analysis.get('change_pattern', 'unknown')} changes at step {self.action_counter}")
    
    def _evaluate_frame_change_quality(self, change_analysis: Dict[str, Any], 
                                     current_score: int) -> float:
        """Evaluate the quality/usefulness of a frame change."""
        
        changed_pixels = change_analysis.get("changed_pixels", 0)
        change_pattern = change_analysis.get("change_pattern", "no_change")
        
        if changed_pixels == 0:
            return 0.0  # No change is not useful
        
        # Base quality on amount and type of change
        base_quality = min(1.0, changed_pixels / 20.0)  # Normalize to [0, 1]
        
        # Bonus for consistent change patterns
        pattern_bonus = {
            "single_pixel": 0.3,
            "few_pixels": 0.5,
            "block_change": 0.7,
            "major_change": 0.4  # Major changes might be less controlled
        }.get(change_pattern, 0.1)
        
        # Huge bonus for score changes
        score_bonus = 2.0 if current_score > self.last_score else 0.0
        
        return base_quality + pattern_bonus + score_bonus
    
    def _extract_state_features(self, frame: List[List[List[int]]]) -> Dict[str, Any]:
        """Extract features describing current state."""
        
        if not frame or not frame[0]:
            return {"empty": True}
        
        grid = np.array(frame[0])
        
        return {
            "density": float(np.mean(grid != 0)),
            "unique_values": len(np.unique(grid)),
            "grid_shape": grid.shape,
            "center_mass": self._get_center_mass(grid),
            "has_borders": self._check_borders(grid)
        }
    
    def _get_center_mass(self, grid: np.ndarray) -> Tuple[float, float]:
        """Get center of mass of non-zero elements."""
        non_zero = np.where(grid != 0)
        if len(non_zero[0]) == 0:
            return (0.5, 0.5)
        return (float(np.mean(non_zero[1]) / grid.shape[1]), 
                float(np.mean(non_zero[0]) / grid.shape[0]))
    
    def _check_borders(self, grid: np.ndarray) -> bool:
        """Check if there are elements near borders."""
        h, w = grid.shape
        if h < 3 or w < 3:
            return False
        
        borders = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        return np.any(borders != 0)
    
    def _configure_mh_action(self, action: GameAction, current_frame: List[List[List[int]]], 
                            current_score: int):
        """Configure action with Metropolis-Hastings reasoning."""
        
        signature = self.frame_analyzer.get_action_signature(action)
        mh_summary = self.mh_selector.get_action_relationship_summary()
        
        if action.is_simple():
            action.reasoning = self._generate_mh_reasoning(
                action, current_score, signature, mh_summary
            )
            
        elif action.is_complex():
            coordinates = self._select_mh_coordinates(action, current_frame, signature)
            action.set_data({"x": coordinates[0], "y": coordinates[1]})
            self.last_coordinates = coordinates
            
            action.reasoning = {
                "action_type": action.name,
                "coordinates": coordinates,
                "mh_reasoning": self._generate_mh_reasoning(
                    action, current_score, signature, mh_summary
                ),
                "movement_vector": signature.get("movement_vector", (0, 0)),
                "change_consistency": signature.get("change_consistency", 0.0),
                "utility": mh_summary["action_utilities"].get(action, 0.0)
            }
    
    def _select_mh_coordinates(self, action: GameAction, 
                              current_frame: List[List[List[int]]], 
                              signature: Dict[str, Any]) -> Tuple[int, int]:
        """Select coordinates based on learned action signature."""
        
        try:
            # Use movement vector if we've learned one
            movement_vector = signature.get("movement_vector", (0, 0))
            confidence = signature.get("movement_consistency", 0.0)
            
            if confidence > 0.5 and (abs(movement_vector[0]) > 0.5 or abs(movement_vector[1]) > 0.5):
                # Use predicted movement
                current_features = self._extract_state_features(current_frame)
                center_mass = current_features.get("center_mass", (0.5, 0.5))
                
                # Predict where to click based on movement vector
                target_x = center_mass[0] * 63 - movement_vector[0] * 10
                target_y = center_mass[1] * 63 - movement_vector[1] * 10
                
                x = int(np.clip(target_x, 0, 63))
                y = int(np.clip(target_y, 0, 63))
                
                return (x, y)
            
            # Fallback to systematic exploration
            attempt = signature.get("attempts", 0)
            systematic_coords = [
                (32, 32), (16, 16), (48, 48), (16, 48), (48, 16),
                (0, 0), (63, 63), (0, 63), (63, 0), (32, 0), (32, 63)
            ]
            
            if attempt < len(systematic_coords):
                return systematic_coords[attempt]
            else:
                return (random.randint(0, 63), random.randint(0, 63))
                
        except:
            return (random.randint(0, 63), random.randint(0, 63))
    
    def _generate_mh_reasoning(self, action: GameAction, current_score: int,
                              signature: Dict[str, Any], mh_summary: Dict[str, Any]) -> str:
        """Generate Metropolis-Hastings reasoning."""
        
        utility = mh_summary["action_utilities"].get(action, 0.0)
        confidence = signature.get("confidence", 0.0)
        attempts = signature.get("attempts", 0)
        movement = signature.get("movement_vector", (0, 0))
        
        parts = [f"MH: {action.name}"]
        
        # Show learned signature
        if confidence > 0.3:
            if abs(movement[0]) > 0.5 or abs(movement[1]) > 0.5:
                parts.append(f"[moves({movement[0]:.1f},{movement[1]:.1f})]")
            else:
                change_pattern = signature.get("common_change_pattern", "unknown")
                parts.append(f"[{change_pattern}]")
        
        # Show utility and confidence
        parts.append(f"(U={utility:.2f},C={confidence:.2f})")
        
        # Show discovery info
        if action in self.action_discovery_steps:
            discovery_step = self.action_discovery_steps[action]
            parts.append(f"[discovered@{discovery_step}]")
        elif attempts == 0:
            parts.append("[untested]")
        
        # MCMC info
        parts.append(f"T={mh_summary['temperature']:.2f}")
        parts.append(f"score={current_score}")
        
        return " ".join(parts)
    
    def _fallback_action(self) -> GameAction:
        """Fallback action."""
        actions = [a for a in GameAction if a != GameAction.RESET]
        action = random.choice(actions)
        
        if action.is_simple():
            action.reasoning = f"MH-FALLBACK: {action.name}"
        elif action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"fallback": True, "mh_learning": True}
        
        return action
    
    def cleanup(self, scorecard=None):
        """Enhanced cleanup with action relationship analysis."""
        try:
            print(f"\n=== METROPOLIS-HASTINGS FRAME RELATIONSHIP SUMMARY ===")
            print(f"Total meaningful frame changes: {self.total_meaningful_changes}")
            
            mh_summary = self.mh_selector.get_action_relationship_summary()
            print(f"MCMC acceptance rate: {mh_summary['acceptance_rate']:.1%}")
            print(f"Final temperature: {mh_summary['temperature']:.2f}")
            
            print("\nLearned Action Signatures:")
            for action in self.frame_analyzer.change_patterns.keys():
                signature = self.frame_analyzer.get_action_signature(action)
                attempts = signature.get("attempts", 0)
                confidence = signature.get("confidence", 0.0)
                movement = signature.get("movement_vector", (0, 0))
                change_pattern = signature.get("common_change_pattern", "unknown")
                
                print(f"  {action.name}: {attempts} attempts, {confidence:.2f} confidence")
                print(f"    → Pattern: {change_pattern}")
                if abs(movement[0]) > 0.1 or abs(movement[1]) > 0.1:
                    print(f"    → Movement: ({movement[0]:.2f}, {movement[1]:.2f})")
                
                # Show when discovered
                if action in self.action_discovery_steps:
                    step = self.action_discovery_steps[action]
                    print(f"    → Discovered at step: {step}")
            
            print("\nAction Utilities (higher = more promising):")
            for action_name in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]:
                action = getattr(GameAction, action_name)
                utility = mh_summary["action_utilities"].get(action, 0.0)
                confidence = mh_summary["action_confidences"].get(action, 0.0)
                print(f"  {action_name}: utility={utility:.3f}, confidence={confidence:.3f}")
                
        except Exception as e:
            print(f"Error in MH cleanup: {e}")
        
        super().cleanup(scorecard)
