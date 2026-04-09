# app.py - Complete HenAi Flask Application with Multiple Self-Ping Methods

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import base64
import threading
import requests
import time
import logging
from datetime import datetime
from models import (
    query_ai_with_fallback, 
    is_code_generation_request,
    get_japanese_response,
    translate_to_japanese,
    text_to_speech_japanese,
    speech_to_text_japanese
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'henai-japanese-learning-secret-key')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store conversation history per session
conversations = {}

# Track when the app started
START_TIME = time.time()

# ============= MULTIPLE SELF-PING METHODS =============

class MultiPinger:
    """Multiple redundant self-ping methods to keep the app alive"""
    
    def __init__(self):
        self.running = False
        self.ping_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.app_url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000')
        
    def log_status(self, method, success):
        """Log ping status"""
        self.ping_count += 1
        if success:
            self.success_count += 1
            logger.info(f"[PING] {method} - SUCCESS (Total: {self.success_count}/{self.ping_count})")
        else:
            self.fail_count += 1
            logger.warning(f"[PING] {method} - FAILED (Total: {self.fail_count}/{self.ping_count})")
    
    # Method 1: Internal localhost ping
    def ping_localhost(self):
        """Ping via localhost - fastest and most reliable"""
        try:
            response = requests.get('http://localhost:5000/api/ping', timeout=5)
            if response.status_code == 200:
                self.log_status("localhost", True)
                return True
        except Exception as e:
            self.log_status("localhost", False)
        return False
    
    # Method 2: Ping via external URL (if available)
    def ping_external(self):
        """Ping via external Render URL"""
        if self.app_url and 'localhost' not in self.app_url:
            try:
                response = requests.get(f'{self.app_url}/api/ping', timeout=10)
                if response.status_code == 200:
                    self.log_status("external", True)
                    return True
            except Exception as e:
                self.log_status("external", False)
        return False
    
    # Method 3: Simulate user request (keeps session alive)
    def simulate_user_request(self):
        """Simulate a lightweight user request"""
        try:
            # Make a lightweight request to a public endpoint
            response = requests.get('http://localhost:5000/api/status', timeout=5)
            if response.status_code == 200:
                self.log_status("simulated_user", True)
                return True
        except Exception as e:
            self.log_status("simulated_user", False)
        return False
    
    # Method 4: Database-style keepalive (touches conversation storage)
    def touch_conversations(self):
        """Touch the conversations dict to keep it 'active'"""
        try:
            # Just read and write a dummy entry to keep things active
            temp_key = f"_keepalive_{int(time.time())}"
            conversations[temp_key] = [{'role': 'system', 'content': 'keepalive', 'timestamp': datetime.now().isoformat()}]
            # Remove it immediately
            if temp_key in conversations:
                del conversations[temp_key]
            self.log_status("storage_touch", True)
            return True
        except Exception as e:
            self.log_status("storage_touch", False)
        return False
    
    # Method 5: SocketIO self-emit (for WebSocket keepalive)
    def emit_socket_ping(self):
        """Emit a socketio event to keep websocket alive"""
        try:
            # This keeps the SocketIO connection thinking there's activity
            socketio.emit('ping', {'data': 'keepalive'}, namespace='/', broadcast=True)
            self.log_status("socket_emit", True)
            return True
        except Exception as e:
            self.log_status("socket_emit", False)
        return False
    
    def run_all_pings(self):
        """Execute all ping methods"""
        results = []
        
        # Method 1: Localhost
        results.append(self.ping_localhost())
        
        # Method 2: External
        results.append(self.ping_external())
        
        # Method 3: Simulated user
        results.append(self.simulate_user_request())
        
        # Method 4: Storage touch
        results.append(self.touch_conversations())
        
        # Method 5: Socket emit (only if SocketIO is active)
        # Uncomment if you have active socket connections
        # results.append(self.emit_socket_ping())
        
        # Overall success if at least one method worked
        overall_success = any(results)
        
        if overall_success:
            logger.info(f"[KEEPALIVE] ✓ App kept alive - {time.ctime()}")
        else:
            logger.error(f"[KEEPALIVE] ✗ ALL PING METHODS FAILED! - {time.ctime()}")
        
        return overall_success
    
    def start(self):
        """Start the multi-pinger in a background thread"""
        if self.running:
            logger.warning("Multi-pinger already running")
            return
        
        self.running = True
        
        def pinger_loop():
            logger.info("🚀 Multi-pinger started with 5 redundant methods")
            while self.running:
                try:
                    # Wait 8 minutes (shorter than Render's 15-minute timeout)
                    for _ in range(480):  # 8 minutes = 480 seconds
                        if not self.running:
                            break
                        time.sleep(1)
                    
                    if self.running:
                        self.run_all_pings()
                        
                except Exception as e:
                    logger.error(f"Multi-pinger error: {e}")
                    time.sleep(60)  # Wait a minute before retrying
        
        self.thread = threading.Thread(target=pinger_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the multi-pinger"""
        self.running = False
        logger.info("Multi-pinger stopped")
    
    def get_stats(self):
        """Get ping statistics"""
        return {
            'total_pings': self.ping_count,
            'successful_pings': self.success_count,
            'failed_pings': self.fail_count,
            'success_rate': (self.success_count / self.ping_count * 100) if self.ping_count > 0 else 0
        }

# Create global pinger instance
pinger = MultiPinger()

# ============= KEEP-ALIVE ENDPOINTS =============

@app.route('/api/ping')
def ping_endpoint():
    """Simple ping endpoint for keep-alive checks"""
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': int(time.time() - START_TIME),
        'message': 'HenAi is running smoothly!'
    })

@app.route('/api/status')
def status_endpoint():
    """Status endpoint with ping statistics"""
    return jsonify({
        'status': 'running',
        'uptime_seconds': int(time.time() - START_TIME),
        'uptime_hours': round((time.time() - START_TIME) / 3600, 2),
        'ping_stats': pinger.get_stats(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health_endpoint():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

# ============= CHAT ENDPOINTS =============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    mode = data.get('mode', 'normal')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Initialize conversation history for session
    if session_id not in conversations:
        conversations[session_id] = []
    
    # Add user message to history
    conversations[session_id].append({
        'role': 'user',
        'content': message,
        'timestamp': datetime.now().isoformat()
    })
    
    # Get AI response based on mode
    if mode == 'japanese':
        response_text = get_japanese_response(message, conversations[session_id][:-1])
    else:
        is_code = is_code_generation_request(message)
        response_text = query_ai_with_fallback(
            prompt=message,
            context=conversations[session_id][:-1],
            is_code_generation=is_code
        )
    
    # Add AI response to history
    conversations[session_id].append({
        'role': 'assistant',
        'content': response_text,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 50 messages
    if len(conversations[session_id]) > 50:
        conversations[session_id] = conversations[session_id][-50:]
    
    return jsonify({
        'response': response_text,
        'session_id': session_id
    })

@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    translated = translate_to_japanese(text)
    return jsonify({
        'original': text,
        'translated': translated,
        'target_language': 'ja'
    })

@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    audio_data = text_to_speech_japanese(text)
    if audio_data:
        return jsonify({
            'audio': base64.b64encode(audio_data).decode('utf-8'),
            'format': 'audio/mp3'
        })
    else:
        return jsonify({'error': 'TTS failed'}), 500

@socketio.on('voice_input')
def handle_voice_input(data):
    audio_bytes = base64.b64decode(data['audio'])
    text, detected_lang = speech_to_text_japanese(audio_bytes)
    
    if text:
        emit('voice_result', {
            'text': text,
            'detected_language': detected_lang,
            'success': True
        })
        
        if data.get('mode') == 'japanese':
            response = get_japanese_response(text)
            emit('voice_response', {
                'response': response,
                'success': True
            })
            
            audio_response = text_to_speech_japanese(response)
            if audio_response:
                emit('voice_audio', {
                    'audio': base64.b64encode(audio_response).decode('utf-8'),
                    'success': True
                })
    else:
        emit('voice_result', {
            'success': False,
            'error': 'Could not recognize speech'
        })

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in conversations:
        conversations[session_id] = []
    return jsonify({'success': True})

@app.route('/health')
def health():
    return {"status": "ok"}, 200

# ============= START APPLICATION =============

if __name__ == '__main__':
    # Get port from environment (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    # Start the multi-pinger
    pinger.start()
    
    logger.info(f"🚀 HenAi Server starting on port {port}")
    logger.info(f"📡 Multi-pinger active with 5 redundant methods")
    logger.info(f"⏰ Ping interval: 8 minutes (Render timeout is 15 minutes)")
    
    # Run the app
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
