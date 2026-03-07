#!/usr/bin/env python3
import os
import sys
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.performance_monitor import PerformanceMonitor

def run_analysis():
    print(f"[{datetime.now()}] Starting performance analysis...")
    try:
        monitor = PerformanceMonitor()
        report = monitor.analyze_performance(last_n_cycles=15)
        if "error" in report:
            print(f"[ERROR] {report['error']}")
            sys.exit(1)
        print(f"[OK] Performance analysis completed. Report saved to {monitor.performance_file}")
    except Exception as e:
        print(f"[CRITICAL] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_analysis()
