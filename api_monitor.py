"""
API Monitor - Track API requests, responses, and performance
"""
import time
import json
from datetime import datetime
from pathlib import Path
from functools import wraps
from collections import defaultdict


class APIMonitor:
    """Monitor API endpoints."""

    def __init__(self, log_dir: str = "api_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.requests = []
        self.stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'max_time': 0,
            'min_time': float('inf')
        })

    def log_request(self, method: str, endpoint: str, status_code: int,
                   duration_ms: float, success: bool = True):
        """Log API request."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration_ms': duration_ms,
            'success': success
        }

        self.requests.append(entry)

        # Update stats
        key = f"{method} {endpoint}"
        self.stats[key]['count'] += 1
        self.stats[key]['total_time'] += duration_ms
        if not success:
            self.stats[key]['errors'] += 1
        self.stats[key]['max_time'] = max(self.stats[key]['max_time'], duration_ms)
        self.stats[key]['min_time'] = min(self.stats[key]['min_time'], duration_ms)

        # Keep last 1000 requests
        if len(self.requests) > 1000:
            self.requests = self.requests[-1000:]

    def get_stats(self):
        """Get endpoint statistics."""
        stats_dict = {}
        for endpoint, data in self.stats.items():
            avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
            error_rate = (data['errors'] / data['count'] * 100) if data['count'] > 0 else 0

            stats_dict[endpoint] = {
                'requests': data['count'],
                'avg_time_ms': round(avg_time, 2),
                'max_time_ms': round(data['max_time'], 2),
                'min_time_ms': round(data['min_time'], 2) if data['min_time'] != float('inf') else 0,
                'errors': data['errors'],
                'error_rate_percent': round(error_rate, 2)
            }

        return stats_dict

    def save_logs(self):
        """Save logs to file."""
        log_file = self.log_dir / f"requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(log_file, 'w') as f:
            for req in self.requests:
                f.write(json.dumps(req) + '\n')
        return log_file

    def print_stats(self):
        """Print endpoint statistics."""
        stats = self.get_stats()
        print("\n" + "="*80)
        print("API MONITORING STATISTICS")
        print("="*80)
        print(f"{'Endpoint':<30} {'Requests':<12} {'Avg (ms)':<12} {'Errors':<10}")
        print("-"*80)

        for endpoint, data in stats.items():
            print(f"{endpoint:<30} {data['requests']:<12} {data['avg_time_ms']:<12.2f} {data['errors']:<10}")

        print("="*80 + "\n")


def monitor_endpoint(monitor: APIMonitor):
    """Decorator to monitor Flask endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                # Extract status code from response
                if isinstance(result, tuple):
                    status_code = result[1] if len(result) > 1 else 200
                else:
                    status_code = 200
                success = status_code < 400

                duration = (time.time() - start_time) * 1000
                endpoint = f.__name__
                monitor.log_request('POST', endpoint, status_code, duration, success)
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                monitor.log_request('POST', f.__name__, 500, duration, False)
                raise

        return wrapper
    return decorator
