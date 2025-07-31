"""Rich-based logging system for ARC-AGI-3-Agents."""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich.status import Status
from rich import box


@dataclass
class TrainingStats:
    """Training statistics for display."""
    total_updates: int = 0
    avg_loss: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    learning_rate: float = 0.0
    buffer_size: int = 0
    gradient_norm: float = 0.0
    
    def update(self, **kwargs):
        """Update stats with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class PerformanceStats:
    """Performance statistics for episodes."""
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    best_score: int = 0
    avg_score: float = 0.0
    total_score: int = 0
    episode_times: List[float] = field(default_factory=list)
    
    @property
    def total_episodes(self) -> int:
        return self.wins + self.losses + self.timeouts
    
    @property
    def win_rate(self) -> float:
        return (self.wins / max(1, self.total_episodes)) * 100
    
    @property
    def avg_episode_time(self) -> float:
        return sum(self.episode_times) / max(1, len(self.episode_times))


@dataclass
class MCTSStats:
    """MCTS-specific statistics."""
    simulations_per_move: int = 0
    exploration_rate: float = 0.0
    tree_depth: int = 0
    nodes_expanded: int = 0
    avg_simulation_time: float = 0.0


@dataclass
class TaskStats:
    """Task-specific statistics for ARC-AGI problems."""
    task_id: str = ""
    total_attempts: int = 0
    successful_patterns: int = 0
    failed_patterns: int = 0
    learning_iterations: int = 0
    pattern_discoveries: int = 0
    confidence_improvements: int = 0
    
    @property
    def success_rate(self) -> float:
        total = self.successful_patterns + self.failed_patterns
        return (self.successful_patterns / max(1, total)) * 100
    
    @property
    def learning_efficiency(self) -> float:
        return (self.pattern_discoveries / max(1, self.learning_iterations)) * 100

@dataclass
class LearningStats:
    """On-the-fly learning statistics."""
    patterns_learned: int = 0
    avg_confidence_gain: float = 0.0
    knowledge_updates: int = 0
    mcts_improvements: int = 0
    adaptation_rate: float = 0.0
    
    def update(self, **kwargs):
        """Update learning stats with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RichLogger:
    """Rich-based logger for ARC-AGI task solving with MCTS."""
    
    def __init__(self, agent_name: str = "Agent", show_progress: bool = True):
        self.console = Console()
        self.agent_name = agent_name
        self.show_progress = show_progress
        
        # Statistics - updated for ARC-AGI
        self.training_stats = TrainingStats()
        self.task_stats = TaskStats()
        self.learning_stats = LearningStats()
        self.mcts_stats = MCTSStats()
        
        # Timing
        self.task_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None
        
        # Progress tracking
        self.current_progress: Optional[Progress] = None
        self.current_status: Optional[Status] = None
    
    def print_welcome(self, config: Dict[str, Any]):
        """Print a beautiful welcome panel with agent configuration."""
        welcome_text = Text(f"{self.agent_name}", style="bold blue")
        welcome_text.append("\n🚀 Aggressive Online Learning Mode", style="bold green")
        
        for key, value in config.items():
            if key == "gpu":
                welcome_text.append(f"\n🎯 GPU: {value}", style="cyan")
            elif key == "simulations":
                welcome_text.append(f"\n🧠 Simulations: {value}", style="yellow")
            elif key == "exploration":
                welcome_text.append(f"\n🔍 Exploration: {value}", style="magenta")
            elif key == "learning_rate":
                welcome_text.append(f"\n📚 Learning Rate: {value}", style="green")
            elif key == "hidden_size":
                welcome_text.append(f"\n🏗️ Network Size: {value}", style="blue")
            else:
                welcome_text.append(f"\n• {key}: {value}", style="white")
        
        panel = Panel(welcome_text, title="🤖 Agent Initialization", border_style="green")
        self.console.print(panel)
    
    def start_episode(self, episode_num: int):
        """Start timing a new episode."""
        self.episode_start_time = time.time()
        self.console.print(f"\n[bold blue]🎮 Episode {episode_num} Starting...[/bold blue]")
    
    def start_task(self, task_id: str):
        """Start timing a new ARC task."""
        self.task_start_time = time.time()
        self.task_stats.task_id = task_id
        self.console.print(f"\n[bold blue]🧩 ARC Task {task_id} Starting...[/bold blue]")
    
    def create_mcts_progress(self, total_simulations: int, step_num: int) -> Progress:
        """Create and return MCTS progress bar."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True  # Remove after completion
        )
        
        task = progress.add_task(
            f"🔍 MCTS Simulations (Step {step_num})", 
            total=total_simulations
        )
        
        self.current_progress = progress
        return progress, task
    
    def update_mcts_progress(self, task_id, advance: int = 1):
        """Update MCTS progress."""
        if self.current_progress:
            self.current_progress.update(task_id, advance=advance)
    
    def finish_mcts_progress(self):
        """Finish MCTS progress."""
        if self.current_progress:
            self.current_progress.stop()
            self.current_progress = None
    
    def training_status(self, message: str):
        """Show training status with spinner."""
        if self.current_status:
            self.current_status.stop()
        
        self.current_status = self.console.status(
            f"[bold blue]{message}[/bold blue]", 
            spinner="dots"
        )
        return self.current_status
    
    def stop_status(self):
        """Stop current status spinner."""
        if self.current_status:
            self.current_status.stop()
            self.current_status = None
    
    def update_training_stats(self, **kwargs):
        """Update training statistics."""
        self.training_stats.update(**kwargs)
    
    def log_training_batch(self, epoch: int, total_epochs: int, loss: float, experiences: int):
        """Log training batch progress."""
        self.console.print(
            f"[dim]Training epoch {epoch+1}/{total_epochs}, "
            f"Loss: {loss:.4f}, "
            f"Experiences: {experiences}[/dim]"
        )
    
    def get_training_table(self) -> Table:
        """Create a rich table with training statistics."""
        table = Table(title="🧠 Training Statistics", show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="yellow", width=12)
        
        stats = self.training_stats
        table.add_row("Total Updates", str(stats.total_updates))
        table.add_row("Avg Loss", f"{stats.avg_loss:.4f}")
        table.add_row("Value Loss", f"{stats.value_loss:.4f}")
        table.add_row("Policy Loss", f"{stats.policy_loss:.4f}")
        table.add_row("Learning Rate", f"{stats.learning_rate:.6f}")
        table.add_row("Buffer Size", str(stats.buffer_size))
        table.add_row("Gradient Norm", f"{stats.gradient_norm:.4f}")
        
        return table
    
    def get_performance_table(self) -> Table:
        """Create a rich table with performance statistics."""
        table = Table(title="📊 Performance Statistics", show_header=True, header_style="bold green", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="yellow", width=12)
        
        stats = self.performance_stats
        table.add_row("Episodes", str(stats.total_episodes))
        table.add_row("Wins", f"[green]{stats.wins}[/green]")
        table.add_row("Losses", f"[red]{stats.losses}[/red]")
        table.add_row("Timeouts", f"[yellow]{stats.timeouts}[/yellow]")
        table.add_row("Win Rate", f"{stats.win_rate:.1f}%")
        table.add_row("Best Score", str(stats.best_score))
        table.add_row("Avg Score", f"{stats.avg_score:.1f}")
        table.add_row("Avg Time", f"{stats.avg_episode_time:.1f}s")
        
        return table
    
    def get_mcts_table(self) -> Table:
        """Create a rich table with MCTS statistics."""
        table = Table(title="🌳 MCTS Statistics", show_header=True, header_style="bold blue", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="yellow", width=12)
        
        stats = self.mcts_stats
        table.add_row("Simulations", str(stats.simulations_per_move))
        table.add_row("Exploration", f"{stats.exploration_rate:.3f}")
        table.add_row("Tree Depth", str(stats.tree_depth))
        table.add_row("Nodes Expanded", str(stats.nodes_expanded))
        table.add_row("Avg Sim Time", f"{stats.avg_simulation_time:.3f}ms")
        
        return table
    
    def log_episode_complete(self, episode_num: int, result: str, score: int, 
                           steps: int, reward: float, style: str = "white"):
        """Log episode completion with detailed stats."""
        # Calculate episode time
        episode_time = 0.0
        if self.episode_start_time:
            episode_time = time.time() - self.episode_start_time
            self.performance_stats.episode_times.append(episode_time)
        
        # Update performance stats
        if result == "WIN":
            self.performance_stats.wins += 1
            status_emoji = "🏆"
        elif result == "LOSS":
            self.performance_stats.losses += 1
            status_emoji = "💥"
        else:  # TIMEOUT
            self.performance_stats.timeouts += 1
            status_emoji = "⏰"
        
        self.performance_stats.total_score += score
        self.performance_stats.best_score = max(self.performance_stats.best_score, score)
        self.performance_stats.avg_score = (
            self.performance_stats.total_score / max(1, self.performance_stats.total_episodes)
        )
        
        # Create episode summary table
        episode_table = Table(
            title=f"{status_emoji} Episode {episode_num} Complete", 
            show_header=False, 
            box=box.ROUNDED
        )
        episode_table.add_column("", style="cyan", width=12)
        episode_table.add_column("", style="white", width=15)
        
        episode_table.add_row("Status:", f"[{style}]{result}[/{style}]")
        episode_table.add_row("Score:", str(score))
        episode_table.add_row("Steps:", str(steps))
        episode_table.add_row("Time:", f"{episode_time:.1f}s")
        episode_table.add_row("Reward:", f"{reward:.3f}")
        episode_table.add_row("Win Rate:", f"{self.performance_stats.win_rate:.1f}%")
        
        self.console.print(episode_table)
    
    def log_pattern_discovery(self, pattern_type: str, confidence: float):
        """Log discovery of a new pattern."""
        self.learning_stats.patterns_learned += 1
        self.task_stats.pattern_discoveries += 1
        self.console.print(
            f"[green]🔍 Pattern discovered: {pattern_type} (confidence: {confidence:.2f})[/green]"
        )
    
    def log_learning_iteration(self, iteration: int, improvements: int, confidence_gain: float):
        """Log on-the-fly learning progress."""
        self.learning_stats.knowledge_updates += 1
        self.learning_stats.avg_confidence_gain = (
            self.learning_stats.avg_confidence_gain * 0.9 + confidence_gain * 0.1
        )
        self.task_stats.learning_iterations += 1
        
        if improvements > 0:
            self.learning_stats.mcts_improvements += improvements
            self.task_stats.confidence_improvements += improvements
            
        self.console.print(
            f"[dim]Learning iteration {iteration}: "
            f"improvements: {improvements}, "
            f"confidence gain: {confidence_gain:.3f}[/dim]"
        )
    
    def get_task_table(self) -> Table:
        """Create a rich table with task-specific statistics."""
        table = Table(title="🧩 Task Progress", show_header=True, header_style="bold blue", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=18)
        table.add_column("Value", style="yellow", width=15)
        
        stats = self.task_stats
        table.add_row("Task ID", stats.task_id)
        table.add_row("Total Attempts", str(stats.total_attempts))
        table.add_row("Success Rate", f"{stats.success_rate:.1f}%")
        table.add_row("Patterns Found", str(stats.pattern_discoveries))
        table.add_row("Learning Iterations", str(stats.learning_iterations))
        table.add_row("Learning Efficiency", f"{stats.learning_efficiency:.1f}%")
        
        return table
    
    def get_learning_table(self) -> Table:
        """Create a rich table with learning statistics."""
        table = Table(title="🧠 On-the-fly Learning", show_header=True, header_style="bold green", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=18)
        table.add_column("Value", style="yellow", width=15)
        
        stats = self.learning_stats
        table.add_row("Patterns Learned", str(stats.patterns_learned))
        table.add_row("Knowledge Updates", str(stats.knowledge_updates))
        table.add_row("MCTS Improvements", str(stats.mcts_improvements))
        table.add_row("Avg Confidence Gain", f"{stats.avg_confidence_gain:.3f}")
        table.add_row("Adaptation Rate", f"{stats.adaptation_rate:.3f}")
        
        return table
    
    def log_task_complete(self, task_id: str, result: str, score: int, 
                         attempts: int, patterns_found: int):
        """Log task completion with detailed stats."""
        # Calculate task time
        task_time = 0.0
        if self.task_start_time:
            task_time = time.time() - self.task_start_time
        
        # Update task stats
        self.task_stats.total_attempts = attempts
        if result == "SOLVED":
            self.task_stats.successful_patterns += patterns_found
            status_emoji = "🏆"
            style = "bold green"
        else:
            self.task_stats.failed_patterns += 1
            status_emoji = "❌"
            style = "bold red"
        
        # Create task summary table
        task_table = Table(
            title=f"{status_emoji} Task {task_id} Complete", 
            show_header=False, 
            box=box.ROUNDED
        )
        task_table.add_column("", style="cyan", width=15)
        task_table.add_column("", style="white", width=18)
        
        task_table.add_row("Status:", f"[{style}]{result}[/{style}]")
        task_table.add_row("Score:", str(score))
        task_table.add_row("Attempts:", str(attempts))
        task_table.add_row("Time:", f"{task_time:.1f}s")
        task_table.add_row("Patterns Found:", str(patterns_found))
        task_table.add_row("Success Rate:", f"{self.task_stats.success_rate:.1f}%")
        
        self.console.print(task_table)
    
    def show_combined_stats(self):
        """Show all statistics in a combined view for ARC-AGI."""
        # Create columns with all stats tables
        columns = Columns([
            self.get_task_table(),
            self.get_learning_table(),
            self.get_mcts_table()
        ], equal=True, expand=True)
        
        self.console.print(columns)
    
    def log_step_action(self, step_num: int, action: str, confidence: float = None):
        """Log action selection for a step."""
        confidence_text = f" (confidence: {confidence:.2f})" if confidence else ""
        self.console.print(
            f"[dim]Step {step_num}: [bold]{action}[/bold]{confidence_text}[/dim]"
        )
    
    def log_error(self, message: str, exception: Exception = None):
        """Log error messages."""
        error_text = f"[bold red]❌ Error: {message}[/bold red]"
        if exception:
            error_text += f"\n[red]{str(exception)}[/red]"
        
        panel = Panel(error_text, title="⚠️  Error", border_style="red")
        self.console.print(panel)
    
    def log_warning(self, message: str):
        """Log warning messages."""
        self.console.print(f"[bold yellow]⚠️  Warning: {message}[/bold yellow]")
    
    def log_info(self, message: str):
        """Log info messages."""
        self.console.print(f"[blue]ℹ️  {message}[/blue]")
    
    def log_success(self, message: str):
        """Log success messages."""
        self.console.print(f"[bold green]✅ {message}[/bold green]")
    
    def update_mcts_stats(self, **kwargs):
        """Update MCTS statistics."""
        for key, value in kwargs.items():
            if hasattr(self.mcts_stats, key):
                setattr(self.mcts_stats, key, value)
    
    def update_learning_stats(self, **kwargs):
        """Update learning statistics."""
        self.learning_stats.update(**kwargs)
    
    def log_mcts_decision(self, step_num: int, action: str, confidence: float, 
                         simulations: int, tree_depth: int):
        """Log MCTS decision with learning context."""
        confidence_text = f"confidence: {confidence:.2f}, sims: {simulations}, depth: {tree_depth}"
        self.console.print(
            f"[dim]Step {step_num}: [bold]{action}[/bold] ({confidence_text})[/dim]"
        )
    
    def create_live_dashboard(self) -> Live:
        """Create a live updating dashboard for ARC-AGI."""
        layout = Layout()
        
        # Split layout
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header
        header_text = Text(f"{self.agent_name} ARC-AGI Dashboard", style="bold blue")
        header_text.append(f" - {datetime.now().strftime('%H:%M:%S')}", style="dim")
        layout["header"].update(Panel(header_text, border_style="blue"))
        
        # Body with stats
        layout["body"].split_row(
            Layout(self.get_task_table(), name="task"),
            Layout(self.get_learning_table(), name="learning"),
            Layout(self.get_mcts_table(), name="mcts")
        )
        
        # Footer
        footer_text = Text("ARC-AGI On-the-fly Learning • Press Ctrl+C to stop", style="dim")
        layout["footer"].update(Panel(footer_text, border_style="dim"))
        
        return Live(layout, console=self.console, refresh_per_second=2)


# Global logger instance
_global_logger: Optional[RichLogger] = None


def get_logger(agent_name: str = "Agent") -> RichLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = RichLogger(agent_name)
    return _global_logger


def set_logger(logger: RichLogger):
    """Set global logger instance."""
    global _global_logger
    _global_logger = logger
