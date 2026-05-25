"""
GPU Memory Monitoring Utilities

Provides functions for tracking CUDA memory usage during navigation.
"""

import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger("[GPU Memory]")


def log_gpu_memory(tag: str = "", log_level: str = "info") -> dict:
    """
    Log current GPU memory usage.
    
    Args:
        tag: Label for identifying the log point
        log_level: Logging level ('debug', 'info', 'warning')
        
    Returns:
        dict with memory stats (allocated_gb, reserved_gb, peak_gb)
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    msg = f"[{tag}] Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={peak:.2f}GB"
    
    if log_level == "debug":
        logger.debug(msg)
    elif log_level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "peak_gb": peak
    }


def reset_peak_memory():
    """Reset peak memory stats for fresh tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def clear_gpu_cache():
    """Clear the GPU cache to free unreferenced memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def gpu_memory_tracker(name: str, verbose: bool = True):
    """
    Context manager to track memory delta for a code block.
    
    Usage:
        with gpu_memory_tracker("mast3r_inference"):
            # ... code that uses GPU ...
    
    Args:
        name: Label for the code block
        verbose: If True, logs the memory delta
        
    Yields:
        dict that will be populated with memory stats after the block completes
    """
    stats = {}
    
    if not torch.cuda.is_available():
        yield stats
        return
    
    torch.cuda.synchronize()
    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    
    yield stats
    
    torch.cuda.synchronize()
    end_allocated = torch.cuda.memory_allocated()
    end_reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    
    delta_allocated = (end_allocated - start_allocated) / 1e6
    delta_reserved = (end_reserved - start_reserved) / 1e6
    peak_during = peak / 1e9
    
    stats.update({
        "delta_allocated_mb": delta_allocated,
        "delta_reserved_mb": delta_reserved,
        "peak_during_gb": peak_during,
        "start_allocated_gb": start_allocated / 1e9,
        "end_allocated_gb": end_allocated / 1e9
    })
    
    if verbose:
        logger.info(f"[{name}] Memory delta: {delta_allocated:+.1f}MB allocated, "
                   f"{delta_reserved:+.1f}MB reserved, peak={peak_during:.2f}GB")


class GPUMemoryMonitor:
    """
    Class for tracking GPU memory across multiple steps.
    Useful for identifying which steps cause memory spikes.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.step_history = []
        self.current_step = -1
        self.step_checkpoints = {}
    
    def start_step(self, step: int):
        """Start tracking a new navigation step."""
        if not self.enabled:
            return
        
        self.current_step = step
        self.step_checkpoints = {}
        reset_peak_memory()
        self.checkpoint("step_start")
    
    def checkpoint(self, name: str) -> dict:
        """Record a memory checkpoint within the current step."""
        if not self.enabled or not torch.cuda.is_available():
            return {}
        
        stats = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9
        }
        self.step_checkpoints[name] = stats
        return stats
    
    def end_step(self) -> dict:
        """End tracking for current step and return summary."""
        if not self.enabled:
            return {}
        
        self.checkpoint("step_end")
        
        step_summary = {
            "step": self.current_step,
            "checkpoints": self.step_checkpoints.copy(),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        }
        self.step_history.append(step_summary)
        
        return step_summary
    
    def print_step_summary(self):
        """Print memory summary for the current step."""
        if not self.enabled or not self.step_checkpoints:
            return
        
        logger.info(f"=== Step {self.current_step} Memory Summary ===")
        prev_alloc = 0
        for name, stats in self.step_checkpoints.items():
            delta = stats["allocated_gb"] - prev_alloc
            logger.info(f"  {name}: {stats['allocated_gb']:.2f}GB (delta: {delta*1000:+.1f}MB)")
            prev_alloc = stats["allocated_gb"]
        logger.info(f"  Peak during step: {self.step_checkpoints.get('step_end', {}).get('peak_gb', 0):.2f}GB")
    
    def get_history(self) -> list:
        """Return full history of all steps."""
        return self.step_history
    
    def find_memory_spikes(self, threshold_gb: float = 0.5) -> list:
        """Find steps where memory increased above threshold."""
        spikes = []
        for entry in self.step_history:
            checkpoints = entry.get("checkpoints", {})
            values = [cp["allocated_gb"] for cp in checkpoints.values()]
            if len(values) >= 2:
                max_delta = max(values) - min(values)
                if max_delta > threshold_gb:
                    spikes.append({
                        "step": entry["step"],
                        "max_delta_gb": max_delta,
                        "checkpoints": checkpoints
                    })
        return spikes
