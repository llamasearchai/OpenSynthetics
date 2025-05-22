"""Monitoring and observability for OpenSynthetics."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator, Callable

from loguru import logger


@dataclass
class Metric:
    """Metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class Monitor:
    """Central monitoring system."""

    def __init__(self) -> None:
        """Initialize monitor."""
        self.metrics: List[Metric] = []
        self.handlers: List[Callable[[Metric], None]] = []

    def add_metric(self, metric: Metric) -> None:
        """Add a metric.

        Args:
            metric: Metric to add
        """
        self.metrics.append(metric)
        
        # Call handlers
        for handler in self.handlers:
            try:
                handler(metric)
            except Exception as e:
                logger.error(f"Error in metric handler: {e}")

    def add_handler(self, handler: Callable[[Metric], None]) -> None:
        """Add a metric handler.

        Args:
            handler: Handler function
        """
        self.handlers.append(handler)

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> Generator[None, None, None]:
        """Measure execution time of a block.

        Args:
            name: Metric name
            tags: Metric tags

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.add_metric(Metric(
                name=f"{name}_duration_seconds",
                value=duration,
                tags=tags or {},
            ))

    def count(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Count an event.

        Args:
            name: Metric name
            value: Metric value
            tags: Metric tags
        """
        self.add_metric(Metric(
            name=name,
            value=value,
            tags=tags or {},
        ))

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge value.

        Args:
            name: Metric name
            value: Metric value
            tags: Metric tags
        """
        self.add_metric(Metric(
            name=name,
            value=value,
            tags=tags or {},
        ))


# Global monitor instance
monitor = Monitor()


# Default handlers
def log_handler(metric: Metric) -> None:
    """Log metrics.

    Args:
        metric: Metric
    """
    tags_str = " ".join(f"{k}={v}" for k, v in metric.tags.items())
    logger.debug(f"METRIC {metric.name}={metric.value} {tags_str}")


# Register default handlers
monitor.add_handler(log_handler)