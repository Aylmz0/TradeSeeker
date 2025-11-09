"""
Flask-based Admin Server for Alpha Arena DeepSeek Trading Bot
Modern web interface with RESTful API endpoints
"""
import os
import json
import fcntl
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, send_from_directory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')

# --- Utility Functions ---

def safe_file_read(file_path: str, default_data: Any = None) -> Any:
    """Safely read data from a JSON file using file locking."""
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

def safe_file_write(file_path: str, data: Any):
    """Safely write data to a JSON file using file locking."""
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
    return send_from_directory('.', 'index.html')

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
    data = safe_file_read('performance_report.json', {})
    return jsonify(data)

@app.route('/api/performance/refresh', methods=['POST'])
def refresh_performance():
    """Trigger a new performance analysis."""
    try:
        # Import and run performance monitor
        from performance_monitor import PerformanceMonitor
        
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

        # Create the override command file
        override_command = {
            "timestamp": datetime.now().isoformat(),
            "decisions": {
                coin_to_close: {
                    "signal": "close_position",
                    "justification": "Manually closed via admin panel."
                }
            }
        }
        
        # Safely write the file for the bot to read
        safe_file_write("manual_override.json", override_command)
        
        return jsonify({
            "status": "success", 
            "message": f"Close command sent for {coin_to_close}."
        })
        
    except Exception as e:
        logger.error(f"Error processing force-close request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve static files (JSON data files, CSS, JS, etc.)."""
    if filename.endswith('.json'):
        # Add cache control headers for JSON files
        response = send_from_directory('.', filename)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return send_from_directory('.', filename)

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
    logger.info("   Don't forget to start your bot (alpha_arena_deepseek.py) in a separate terminal.")
    logger.info(f"   Access the UI at http://localhost:{PORT} (or your codeserver forwarded address).")
    
    app.run(host=HOST, port=PORT, debug=False)
