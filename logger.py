"""
Logging and Monitoring System
Tracks application events, errors, and performance metrics
"""
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import sys


class StructuredLogger:
    """Structured logging with JSON output."""

    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # JSON logs
        self.json_log_file = self.log_dir / f"{name}_structured.jsonl"

    def _write_json_log(self, level: str, message: str, data: Dict[str, Any] = None):
        """Write structured JSON log."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'data': data or {}
        }

        with open(self.json_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def info(self, message: str, data: Dict[str, Any] = None):
        """Log info message."""
        self.logger.info(message)
        self._write_json_log('INFO', message, data)

    def warning(self, message: str, data: Dict[str, Any] = None):
        """Log warning message."""
        self.logger.warning(message)
        self._write_json_log('WARNING', message, data)

    def error(self, message: str, data: Dict[str, Any] = None):
        """Log error message."""
        self.logger.error(message)
        self._write_json_log('ERROR', message, data)

    def debug(self, message: str, data: Dict[str, Any] = None):
        """Log debug message."""
        self.logger.debug(message)
        self._write_json_log('DEBUG', message, data)


class PerformanceMonitor:
    """Monitor application performance."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, name: str):
        """Start timing a metric."""
        self.start_times[name] = time.time()

    def end_timer(self, name: str, success: bool = True):
        """End timing and record metric."""
        if name not in self.start_times:
            return None

        elapsed = (time.time() - self.start_times[name]) * 1000  # ms

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'elapsed_ms': elapsed,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })

        del self.start_times[name]

        self.logger.debug(f"Timer '{name}': {elapsed:.2f}ms", {
            'metric': name,
            'elapsed_ms': elapsed,
            'success': success
        })

        return elapsed

    def record_metric(self, name: str, value: float, unit: str = ''):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        })

    def get_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metrics:
            return {}

        values = [m.get('elapsed_ms', m.get('value')) for m in self.metrics[metric_name]]

        if not values:
            return {}

        import numpy as np

        return {
            'count': len(values),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99)),
        }

    def print_summary(self):
        """Print performance summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        for metric_name in self.metrics.keys():
            summary = self.get_summary(metric_name)
            if summary:
                print(f"\n{metric_name}:")
                for key, value in summary.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")


class RequestLogger:
    """Log HTTP requests and responses."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def log_request(self, method: str, path: str, client_ip: str = None):
        """Log incoming request."""
        self.logger.info(f"{method} {path}", {
            'type': 'request',
            'method': method,
            'path': path,
            'client_ip': client_ip
        })

    def log_response(self, method: str, path: str, status_code: int, duration_ms: float):
        """Log outgoing response."""
        level = 'info' if status_code < 400 else 'warning'
        method_func = getattr(self.logger, level)

        method_func(f"{method} {path} {status_code}", {
            'type': 'response',
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration_ms': duration_ms
        })

    def log_prediction(self, predicted_digit: int, confidence: float, duration_ms: float,
                      from_cache: bool = False):
        """Log prediction."""
        self.logger.info(f"Prediction: digit={predicted_digit}, confidence={confidence:.2%}", {
            'type': 'prediction',
            'digit': predicted_digit,
            'confidence': confidence,
            'duration_ms': duration_ms,
            'from_cache': from_cache
        })

    def log_error(self, error_type: str, message: str, details: Dict[str, Any] = None):
        """Log error."""
        self.logger.error(f"{error_type}: {message}", {
            'type': 'error',
            'error_type': error_type,
            'message': message,
            'details': details or {}
        })


def create_loggers():
    """Create and configure all loggers."""
    main_logger = StructuredLogger("mnist_app")
    request_logger = RequestLogger(main_logger)
    performance_monitor = PerformanceMonitor(main_logger)

    return {
        'main': main_logger,
        'request': request_logger,
        'performance': performance_monitor
    }


# Global logger instance
_loggers = None


def get_loggers():
    """Get or create global logger instance."""
    global _loggers
    if _loggers is None:
        _loggers = create_loggers()
    return _loggers
