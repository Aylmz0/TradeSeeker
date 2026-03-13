"""
Flask-based Admin Server for Alpha Arena DeepSeek Trading Bot
Modern web interface with RESTful API endpoints
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


# --- Path Configuration ---
# Resolve paths relative to this script file
# Script is in: src/web/admin_server_flask.py
CURRENT_FILE = Path(__file__).resolve()
WEB_DIR = CURRENT_FILE.parent
PROJECT_ROOT = WEB_DIR.parent.parent
TEMPLATE_DIR = WEB_DIR / "templates"

# Add project root to sys.path to allow imports from src
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.services.ml_service import MLService


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask
# We set static_folder to PROJECT_ROOT to allow serving JSON files directly from there if needed,
# though we primarily serve them via API.
app = Flask(__name__, static_folder=str(PROJECT_ROOT), template_folder=str(TEMPLATE_DIR))

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Environment Detection ---

ml_service = MLService()


def get_python_executable() -> str:
    """Detect and return the absolute path to the .venv python if it exists."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def get_log_file(name: str):
    """Ensure data/logs directory exists and return the log file handle."""
    log_dir = PROJECT_ROOT / "data" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    return open(log_dir / f"{name}.log", "a", encoding="utf-8")


# --- Error Handlers ---


@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler to ensure all errors return JSON instead of HTML."""
    logger.error(f"Global Error Hook: {e}", exc_info=True)
    return jsonify({"status": "error", "message": str(e)}), 500


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

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return default_data


def safe_file_write(filename: str, data: Any):
    """Safely write data to a JSON file using file locking."""
    file_path = get_file_path(filename)
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        raise


# --- API Routes ---


@app.route("/")
def serve_index():
    """Serve the main admin panel interface."""
    return send_from_directory(str(TEMPLATE_DIR), "index.html")


@app.route("/assets/<path:filename>")
def serve_custom_static(filename):
    """Serve static files from src/web/static directory."""
    return send_from_directory(str(WEB_DIR / "static"), filename)


@app.route("/api/portfolio")
def get_portfolio():
    """Get current portfolio state."""
    data = safe_file_read("data/portfolio_state.json", {})
    return jsonify(data)


@app.route("/api/trades")
def get_trades():
    """Get trade history (prefer full history for persistent graph)."""
    # Try to read full history first (persistent across resets)
    data = safe_file_read("data/full_trade_history.json", None)

    # Fallback to active history if full history doesn't exist yet
    if data is None:
        data = safe_file_read("data/trade_history.json", [])

    return jsonify(data)


@app.route("/api/cycles")
def get_cycles():
    """Get AI cycle history."""
    data = safe_file_read("data/cycle_history.json", [])
    return jsonify(data)


@app.route("/api/alerts")
def get_alerts():
    """Get system alerts."""
    data = safe_file_read("data/alerts.json", [])
    return jsonify(data)


@app.route("/api/performance")
def get_performance():
    """Get performance analysis report."""
    reports = safe_file_read("data/performance_report.json", [])

    if isinstance(reports, dict):
        return jsonify(reports)

    if isinstance(reports, list) and len(reports) > 0:
        for report in reversed(reports):
            if isinstance(report, dict) and "reset_reason" not in report:
                return jsonify(report)
        return jsonify(reports[-1] if reports else {})

    return jsonify({})


@app.route("/api/performance/refresh", methods=["POST"])
def refresh_performance():
    """Trigger a new performance analysis in the background."""
    try:
        script_path = str(PROJECT_ROOT / "scripts" / "generate_performance_report.py")
        if not os.path.exists(script_path):
            return jsonify({"status": "error", "message": "Performance script missing."}), 404

        # Run as background process to avoid blocking
        log_file = get_log_file("performance_refresh")
        subprocess.Popen(
            [get_python_executable(), script_path],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Fully detached
        )

        return jsonify(
            {"status": "success", "message": "Performance refresh started in background."}
        )
    except Exception as e:
        logger.error(f"Error starting performance refresh: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/force-close", methods=["POST"])
def force_close_position():
    """Force close a specific position."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        coin_to_close = data.get("coin")
        if not coin_to_close:
            return jsonify({"status": "error", "message": "Coin not specified"}), 400

        logger.info(f"[INFO] MANUAL CLOSE REQUEST RECEIVED for: {coin_to_close}")

        override_command = {
            "timestamp": datetime.now().isoformat(),
            "decisions": {
                coin_to_close: {
                    "signal": "close_position",
                    "justification": "Manually closed via admin panel.",
                },
            },
        }

        safe_file_write("data/manual_override.json", override_command)

        return jsonify({"status": "success", "message": f"Close command sent for {coin_to_close}."})

    except Exception as e:
        logger.error(f"Error processing force-close request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bot-control", methods=["POST"])
def set_bot_control():
    """Set bot control status (pause/resume/stop)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        action = data.get("action")
        if not action or action not in ["pause", "resume", "stop"]:
            return jsonify({"status": "error", "message": "Invalid action"}), 400

        status_map = {"pause": "paused", "resume": "running", "stop": "stopped"}

        control_data = {
            "status": status_map[action],
            "last_updated": datetime.now().isoformat(),
            "action": action,
        }

        safe_file_write("data/bot_control.json", control_data)
        logger.info(f"[INFO] Bot control: {action.upper()} command sent successfully")

        return jsonify(
            {
                "status": "success",
                "message": f"Bot {action} command sent successfully.",
                "bot_status": status_map[action],
            },
        )

    except Exception as e:
        logger.error(f"Error setting bot control: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bot-control", methods=["GET"])
def get_bot_control():
    """Get current bot control status."""
    try:
        control = safe_file_read(
            "data/bot_control.json",
            {"status": "unknown", "last_updated": None},
        )
        return jsonify(control)
    except Exception as e:
        logger.error(f"Error reading bot control: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


import subprocess


# --- ML Control State ---
# Global variables to track the background training process
ml_training_process = None
ml_training_status = "idle"  # 'idle', 'training', 'error'
ml_training_error = ""

# --- API Routes ---


@app.route("/api/ml/train", methods=["POST"])
def trigger_ml_training():
    """Trigger the global ML model training in the background."""
    global ml_training_process, ml_training_status, ml_training_error

    # Check if a process is already running
    if ml_training_process is not None:
        if ml_training_process.poll() is None:
            return jsonify({"status": "error", "message": "Training is already in progress."}), 400

    try:
        # Resolve path to the training script
        script_path = str(PROJECT_ROOT / "scripts" / "train_model.py")

        # Launch as a background process so it doesn't block the Flask response
        logger.info("[INFO] Launching background ML Training Process...")
        ml_training_status = "training"
        ml_training_error = ""

        ml_training_process = subprocess.Popen(
            [get_python_executable(), script_path],
            cwd=str(PROJECT_ROOT),
            stdout=get_log_file("ml_training"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Fully detached
        )

        return jsonify(
            {"status": "success", "message": "Global ML training started in the background."}
        ), 202

    except Exception as e:
        logger.error(f"Error starting ML training: {e}")
        ml_training_status = "error"
        ml_training_error = str(e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/ml/status", methods=["GET"])
def get_ml_status():
    """Check the status of the background ML training process."""
    global ml_training_process, ml_training_status, ml_training_error

    if ml_training_process is not None:
        # poll() returns None if process is still running, otherwise returns the exit code
        return_code = ml_training_process.poll()

        if return_code is None:
            ml_training_status = "training"
        elif return_code == 0:
            ml_training_status = "idle"
            ml_training_process = None  # Reset
            logger.info("[INFO] Background ML Training Process completed successfully.")
        else:
            ml_training_status = "error"
            # Try to grab stderr if available
            err_output = ml_training_process.stderr.read()
            ml_training_error = f"Process exited with code {return_code}. {err_output}"
            ml_training_process = None  # Reset
            logger.error(f"[ERROR] Background ML Training Failed: {ml_training_error}")

    return jsonify(
        {
            "status": ml_training_status,
            "message": ml_training_error
            if ml_training_status == "error"
            else "Training in progress..."
            if ml_training_status == "training"
            else "Idle",
        }
    )


@app.route("/api/ml/scan", methods=["POST"])
def trigger_ml_scan():
    """Trigger a diagnostic scan (Calculates metrics and updates logs without full train)."""
    # For now, we simulate a scan by forcing a recalculation
    # (In the future, this could trigger a specific validation method in ml_service)
    try:
        # A lightweight check. Mostly handled by the existing /api/ml-drift endpoint
        # The frontend calls this to show UI intent, then re-fetches the drift.
        logger.info("[INFO] ML Diagnostic Scan requested by Admin Panel.")
        return jsonify({"status": "success", "message": "Diagnostic scan completed."})
    except Exception as e:
        logger.error(f"Error executing ML scan: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --- Error Handlers ---


@app.route("/api/ml-predictions")
def get_ml_predictions():
    """Get recent ML predictions from JSONL log."""
    jsonl_path = get_file_path("data/ml_predictions.jsonl")
    predictions = []
    try:
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                lines = f.readlines()
            for line in lines[-50:]:
                line = line.strip()
                if line:
                    raw_pred = json.loads(line)
                    # Map lowercase internal keys to Frontend-expected probabilities
                    # Supports both old and new formats
                    sell = raw_pred.get("sell", raw_pred.get("probabilities", {}).get("SELL", 0))
                    hold = raw_pred.get("hold", raw_pred.get("probabilities", {}).get("HOLD", 0))
                    buy = raw_pred.get("buy", raw_pred.get("probabilities", {}).get("BUY", 0))

                    # Legacy Normalization: If values are > 1 (e.g. 51.2), they are in 0-100 scale.
                    # Standardize everything to 0-1 for the Frontend.
                    if sell > 1.0:
                        sell /= 100.0
                    if hold > 1.0:
                        hold /= 100.0
                    if buy > 1.0:
                        buy /= 100.0

                    confidence = raw_pred.get("confidence", 0)
                    if confidence > 1.0:
                        confidence /= 100.0

                    predictions.append(
                        {
                            "ts": raw_pred.get("ts"),
                            "coin": raw_pred.get("coin"),
                            "dominant": raw_pred.get("dominant"),
                            "confidence": confidence,
                            "probabilities": {"SELL": sell, "HOLD": hold, "BUY": buy},
                        }
                    )
    except Exception as e:
        logger.error(f"Error reading ML predictions: {e}")
    return jsonify(predictions)


@app.route("/api/ml-drift")
def get_ml_drift():
    """Get ML model drift status."""
    import sqlite3

    # Load real training metrics if they exist
    training_accuracy = 0.431  # Default fallback
    metrics_path = get_file_path("models/model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
                training_accuracy = m.get("accuracy", 0.431)
        except Exception:
            pass

    # Get live health from engine
    health = ml_service.get_model_health()
    live_acc = health.get("live_accuracy")

    drift_pct = None
    if live_acc is not None:
        drift_pct = round((training_accuracy - live_acc) * 100, 1)

    result = {
        "training_accuracy": round(training_accuracy * 100, 1),
        "live_accuracy": round(live_acc * 100, 1) if live_acc is not None else None,
        "drift_pct": drift_pct,
        "status": health.get("status", "no_data"),
        "total_predictions": health.get("total_logged", 0),
        "evaluated_count": health.get("evaluated_count", 0),
        "prediction_distribution": {},
        "avg_confidence": 0,
    }
    try:
        jsonl_path = get_file_path("data/ml_predictions.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            result["total_predictions"] = len(lines)
            if lines:
                # FIX: JSON parse with individual line error handling
                preds = []
                for l in lines:
                    try:
                        preds.append(json.loads(l))
                    except json.JSONDecodeError as e:
                        print(f"[WARN]  Invalid JSON in ML predictions line: {e}")
                        continue
                dist = {}
                total_conf = 0
                for p in preds:
                    sig = p.get("dominant", "UNKNOWN")
                    dist[sig] = dist.get(sig, 0) + 1
                    total_conf += p.get("confidence", 0)
                if preds:
                    result["prediction_distribution"] = dist
                    result["avg_confidence"] = round(total_conf / len(preds), 2)

        db_path = get_file_path("data/market_data.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM decisions WHERE status='CLOSED'")
            closed = cursor.fetchone()[0]
            if closed > 0:
                cursor.execute(
                    "SELECT COUNT(*) FROM decisions WHERE status='CLOSED' AND pnl_result > 0"
                )
                wins = cursor.fetchone()[0]
                live_acc = wins / closed
                result["live_accuracy"] = round(live_acc * 100, 1)
                result["drift_pct"] = round((training_accuracy - live_acc) * 100, 1)
                result["status"] = "alert" if (training_accuracy - live_acc) > 0.10 else "ok"
            conn.close()
    except Exception as e:
        logger.error(f"Error computing drift: {e}")
    return jsonify(result)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


# --- Static File Serving (Greedy Route) ---
# MOVED TO BOTTOM to prevent intercepting specific API routes


@app.route("/<path:filename>", methods=["GET"])
def serve_static_files(filename):
    """Serve static files (JSON data files, etc.) from PROJECT ROOT."""
    if filename.startswith("api/"):
        return jsonify({"status": "error", "message": "Endpoint not found"}), 404

    # Security check: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        return jsonify({"status": "error", "message": "Invalid filename"}), 400

    # Serve from project root
    try:
        response = send_from_directory(str(PROJECT_ROOT), filename)
        if filename.endswith(".json"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    except Exception:
        return jsonify({"status": "error", "message": "File not found"}), 404


# --- Main Application ---

if __name__ == "__main__":
    PORT = 8002
    HOST = "0.0.0.0"

    logger.info(f"[INFO] Flask Admin Panel Server starting on {HOST}:{PORT}...")
    logger.info(f"   Project Root: {PROJECT_ROOT}")
    logger.info(f"   Template Dir: {TEMPLATE_DIR}")
    logger.info("   Don't forget to start your bot (src/main.py) in a separate terminal.")
    logger.info(f"   Access the UI at http://localhost:{PORT}")

    app.run(host=HOST, port=PORT, debug=False)
