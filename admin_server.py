import http.server
import socketserver
import json
import os
import fcntl # For file locking on Unix-like systems
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from typing import Dict, Any

# --- Utility for safe JSON file writing (to prevent conflicts with the bot) ---
def safe_file_write(file_path: str, data: Any):
    """Safely write data to a JSON file using file locking."""
    try:
        with open(file_path, 'w') as f:
            # Get an exclusive lock, preventing the bot from reading while writing
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=4)
            # Lock is automatically released when the file is closed
    except Exception as e:
        print(f"❌ Critical Error saving file {file_path}: {e}")

class AdminPanelHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom handler to serve the UI and process manual override commands.
    """
    MANUAL_OVERRIDE_FILE = "manual_override.json"
    
    def do_POST(self):
        """Handles POST requests (for manual override)."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/force-close':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                body = json.loads(post_data.decode('utf-8'))
                coin_to_close = body.get('coin')
                
                if not coin_to_close:
                    self._send_response(400, {"status": "error", "message": "Coin not specified"})
                    return

                print(f"🔔 MANUAL CLOSE REQUEST RECEIVED for: {coin_to_close}")

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
                safe_file_write(self.MANUAL_OVERRIDE_FILE, override_command)
                
                self._send_response(200, {"status": "success", "message": f"Close command sent for {coin_to_close}."})

            except Exception as e:
                print(f"❌ Error processing POST request: {e}")
                self._send_response(500, {"status": "error", "message": str(e)})
        else:
            # Endpoint not found for POST requests
            self._send_response(404, {"status": "error", "message": "Endpoint not found"})

    def do_GET(self):
        """Handles GET requests (serving index.html and .json data files)."""
        # Prevent caching for JSON files
        if self.path.endswith('.json'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            # Add anti-caching headers
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            try:
                # Safely read the file (get a shared lock to avoid conflict with bot writing)
                file_path = self.path.lstrip('/')
                if not os.path.exists(file_path):
                     # If the file doesn't exist yet (e.g., cycle_log before first run), send empty JSON
                     self.wfile.write(b'{}') 
                     return
                
                with open(file_path, 'rb') as f:
                    fcntl.flock(f, fcntl.LOCK_SH) # Shared lock allows multiple readers
                    self.wfile.write(f.read())
                    # Lock is released when 'with' block ends
            except FileNotFoundError:
                 self.wfile.write(b'{}') # Send empty JSON if file not found
            except Exception as e:
                print(f"❌ Error serving GET for {self.path}: {e}")
                self.wfile.write(b'{"error": "Could not read file"}')
        else:
            # For all other files (like index.html), use the default handler
            super().do_GET()

    def _send_response(self, code, content: Dict):
        """Sends a JSON response."""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(content).encode('utf-8'))

# --- Server Initialization ---
PORT = 8000
# Bind to 0.0.0.0 to allow access from outside the container (like in codeserver)
BIND_ADDRESS = "0.0.0.0" 

Handler = AdminPanelHandler

try:
    # Use ThreadingTCPServer for potentially better handling of multiple requests
    # Although SimpleHTTPServer is usually single-threaded
    with socketserver.TCPServer((BIND_ADDRESS, PORT), Handler) as httpd:
        print(f"🚀 Admin Panel Server starting on {BIND_ADDRESS}:{PORT}...")
        print("   Don't forget to start your bot (alpha_arena_deepseek.py) in a separate terminal.")
        print(f"   Access the UI at http://localhost:{PORT} (or your codeserver forwarded address).")
        httpd.serve_forever()
except OSError as e:
    if e.errno == 98: # Address already in use
        print(f"❌ ERROR: Port {PORT} is already in use.")
        print("   Please stop the other program or choose a different port.")
    else:
        # Re-raise other OS errors
        raise 
except KeyboardInterrupt:
    print("\n⏹️ Server stopped by user.")

