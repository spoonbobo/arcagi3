# Prompts for ARC-AGI-3 Agents

ACTION_ANALYSIS_TEMPLATE = """
Analyze this ARC-AGI-3 action by comparing before/after frames.

CURRENT UNDERSTANDING: {current_reasoning} (confidence: {current_confidence})

BEFORE: {before_frame}
AFTER: {after_frame}
{expectation_section}

ANALYSIS GUIDELINES:
1. Focus on GENERAL BEHAVIORAL PATTERNS, not specific grid positions
2. Look for TRANSFORMATIONS that could apply to different puzzles
3. Identify RULES that work regardless of grid content or size
4. Avoid overly specific position-based descriptions

GOOD EXAMPLES of generalizable reasoning:
- "Swaps two specific values throughout the entire grid"
- "Rotates the grid 90 degrees clockwise"
- "Mirrors the grid horizontally"
- "Fills empty spaces (0s) with a specific pattern"
- "Removes all instances of a particular value"
- "Shifts all non-zero values one position in a direction"

BAD EXAMPLES (too specific):
- "Changes value 15 to 3 at position (2,4)"
- "Modifies the third row only"
- "Affects values in specific coordinates"

If confidence is not UNKNOWN, provide a grid example showing the expected result.

JSON output - IMPORTANT: Use EXACTLY the action name "{action_name}" as the key:
{{
    "{action_name}": {{
        "name": "Descriptive name (e.g. 'Move Player Up', 'Place Block')",
        "reasoning": "What this action does",
        "confidence": "UNKNOWN|LOW|MEDIUM|HIGH|ALMOST_SURE|ASSERTIVE",
        "expectation_matched": "FULL_MATCH|PARTIAL_MATCH|NO_MATCH|NO_EXPECTATION",
        "expectation_explanation": "How well actual result matched expectation"
    }}
}}

CRITICAL: The JSON key MUST be exactly "{action_name}" - do not change it to anything else.
Return ONLY valid JSON.
"""

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

LEARNING OBJECTIVE: Discover GENERALIZABLE action behaviors that work across different ARC puzzles.
Focus on learning TRANSFORMATION RULES, not grid-specific patterns.

SITUATION 1: If all actions have confidence 0.0 and reasoning "unknown meaning":
- Create a diverse exploration sequence to discover what each action does
- Test different actions on the current grid to see general behavioral patterns
- No specific expectations needed since actions are completely unknown

SITUATION 2: If some actions have confidence > 0.0 or reasoning beyond "unknown meaning":
- Use your existing knowledge to create a targeted sequence that validates generalizable behaviors
- **CRITICAL**: For actions with learned reasoning, you MUST provide specific expectations
- **CRITICAL**: Use the reasoning text to predict what the grid should look like after the action
- Focus on testing whether the learned behavior generalizes to this specific grid
- Provide clear expectations as grid states showing what you expect the grid to look like after the action
- Actions with higher confidence should have more detailed and accurate expectations

EXPECTATION QUALITY GUIDELINES:
- Base expectations on the learned GENERAL BEHAVIOR, applied to current grid
- If action "swaps values A and B", expect all As to become Bs and vice versa
- If action "rotates grid", expect the entire grid to be rotated
- If action "mirrors horizontally", expect grid to be horizontally mirrored
- Avoid expectations based on specific positions unless the action is inherently position-based

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
