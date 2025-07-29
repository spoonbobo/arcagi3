from datetime import datetime
from typing import Any

from ..agent import Agent

class MCTSAgent(Agent):
    """An agent that uses MCTS to play games."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert "SLAVE_ID" in kwargs, "SLAVE_ID is required"

    @property
    def name(self) -> str:
        return f"{super().name}.{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

