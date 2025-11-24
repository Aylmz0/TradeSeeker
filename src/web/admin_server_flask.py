"""
Flask-based Admin Server for Alpha Arena DeepSeek Trading Bot
Modern web interface with RESTful API endpoints
"""
import os
import sys
import json
import fcntl
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

# --- Path Configuration ---
# Resolve paths relative to this script file
# Script is in: src/web/admin_server_flask.py
CURRENT_FILE = Path(__file__).resolve()
WEB_DIR = CURRENT_FILE.parent
PROJECT_ROOT = WEB_DIR.parent.parent
TEMPLATE_DIR = WEB_DIR / 'templates'

# Add project root to sys.path to allow imports from src
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask
# We set static_folder to PROJECT_ROOT to allow serving JSON files directly from there if needed,
# though we primarily serve them via API.
app = Flask(__name__, static_folder=str(PROJECT_ROOT), template_folder=str(TEMPLATE_DIR))

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Utility Functions ---

def get_file_path(filename: str) -> str:
    """Get absolute path for a file in the project root."""
    return str(PROJECT_ROOT / filename)

def safe_file_read(filename: str, default_data: Any = None) -> Any:
    """Safely read data from a JSON file using file locking."""
    file_path = get_file_path(filename)
    try:
        if not os.path.exists(file_path):
            return default_data
            
        with open(file_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
            data = json.load(f)
            return data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return default_data

def safe_file_write(filename: str, data: Any):
    """Safely write data to a JSON file using file locking."""
    file_path = get_file_path(filename)
    try:
        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        raise

# --- API Routes ---

@app.route('/')
def serve_index():
    """Serve the main admin panel interface."""
    return send_from_directory(str(TEMPLATE_DIR), 'index.html')

@app.route('/api/portfolio')
def get_portfolio():
    """Get current portfolio state."""
    data = safe_file_read('portfolio_state.json', {})
    return jsonify(data)

@app.route('/api/trades')
def get_trades():
    """Get trade history."""
    data = safe_file_read('trade_history.json', [])
    return jsonify(data)

@app.route('/api/cycles')
def get_cycles():
    """Get AI cycle history."""
    data = safe_file_read('cycle_history.json', [])
    return jsonify(data)

@app.route('/api/alerts')
def get_alerts():
    """Get system alerts."""
    data = safe_file_read('alerts.json', [])
    return jsonify(data)

@app.route('/api/performance')
def get_performance():
    """Get performance analysis report."""
    reports = safe_file_read('performance_report.json', [])
    
    if isinstance(reports, dict):
        return jsonify(reports)
    
    if isinstance(reports, list) and len(reports) > 0:
        for report in reversed(reports):
            if isinstance(report, dict) and "reset_reason" not in report:
                return jsonify(report)
        return jsonify(reports[-1] if reports else {})
    
    return jsonify({})

@app.route('/api/performance/refresh', methods=['POST'])
def refresh_performance():
    """Trigger a new performance analysis."""
    try:
        # Import here to avoid circular imports and ensure path is set
        from src.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        report = monitor.analyze_performance(last_n_cycles=10)
        
        return jsonify({
            "status": "success",
            "message": "Performance analysis completed",
            "report": report
        })
        
    except Exception as e:
        logger.error(f"Error refreshing performance: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/force-close', methods=['POST'])
def force_close_position():
    """Force close a specific position."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
            
        coin_to_close = data.get('coin')
        if not coin_to_close:
            return jsonify({"status": "error", "message": "Coin not specified"}), 400

        logger.info(f"🔔 MANUAL CLOSE REQUEST RECEIVED for: {coin_to_close}")

        override_command = {
            "timestamp": datetime.now().isoformat(),
            "decisions": {
                coin_to_close: {
                    "signal": "close_position",
                    "justification": "Manually closed via admin panel."
                }
            }
        }
        
        safe_file_write("manual_override.json", override_command)
        
        return jsonify({
            "status": "success", 
            "message": f"Close command sent for {coin_to_close}."
        })
        
    except Exception as e:
        logger.error(f"Error processing force-close request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/bot-control', methods=['POST'])
def set_bot_control():
    """Set bot control status (pause/resume/stop)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        action = data.get('action')
        if not action or action not in ['pause', 'resume', 'stop']:
            return jsonify({"status": "error", "message": "Invalid action"}), 400
        
        status_map = {
            'pause': 'paused',
            'resume': 'running',
            'stop': 'stopped'
        }
        
        control_data = {
            "status": status_map[action],
            "last_updated": datetime.now().isoformat(),
            "action": action
        }
        
        safe_file_write("bot_control.json", control_data)
        logger.info(f"🔔 Bot control: {action.upper()} command sent successfully")
        
        return jsonify({
            "status": "success",
            "message": f"Bot {action} command sent successfully.",
            "bot_status": status_map[action]
        })
        
    except Exception as e:
        logger.error(f"Error setting bot control: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/bot-control', methods=['GET'])
def get_bot_control():
    """Get current bot control status."""
    try:
        control = safe_file_read("bot_control.json", {"status": "unknown", "last_updated": None})
        return jsonify(control)
    except Exception as e:
        logger.error(f"Error reading bot control: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/<path:filename>', methods=['GET'])
def serve_static_files(filename):
    """Serve static files (JSON data files, etc.) from PROJECT ROOT."""
    if filename.startswith('api/'):
        return jsonify({"status": "error", "message": "Endpoint not found"}), 404
    
    # Security check: prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return jsonify({"status": "error", "message": "Invalid filename"}), 400

    # Serve from project root
    try:
        response = send_from_directory(str(PROJECT_ROOT), filename)
        if filename.endswith('.json'):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response
    except Exception:
        return jsonify({"status": "error", "message": "File not found"}), 404

# --- Error Handlers ---

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

# --- Main Application ---

if __name__ == '__main__':
    PORT = 8002
    HOST = '0.0.0.0'
    
    logger.info(f"🚀 Flask Admin Panel Server starting on {HOST}:{PORT}...")
    logger.info(f"   Project Root: {PROJECT_ROOT}")
    logger.info(f"   Template Dir: {TEMPLATE_DIR}")
    logger.info("   Don't forget to start your bot (src/main.py) in a separate terminal.")
    logger.info(f"   Access the UI at http://localhost:{PORT}")
    
    app.run(host=HOST, port=PORT, debug=False)
