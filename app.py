# app.py - HTTP Voice Version (Faster, No WebSocket Issues)

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
import base64
import threading
import requests
import time
import logging
from datetime import datetime
import tempfile
import io

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

# Store conversation history per session
conversations = {}

# Track when the app started
START_TIME = time.time()

# Cache for TTS audio to reduce generation time
tts_cache = {}

# ============= SELF-PINGER (Keeps App Awake) =============

class SelfPinger:
    """Self-ping to keep Render app alive"""
    
    def __init__(self):
        self.running = False
        self.ping_count = 0
        self.app_url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000')
    
    def start(self):
        """Start the pinger"""
        if self.running:
            return
        
        self.running = True
        
        def pinger_loop():
            logger.info("🚀 Self-pinger started (every 8 minutes)")
            while self.running:
                time.sleep(480)  # 8 minutes
                try:
                    response = requests.get(f'{self.app_url}/api/ping', timeout=10)
                    self.ping_count += 1
                    logger.info(f"[Keep-Alive] Ping #{self.ping_count} - Status: {response.status_code}")
                except Exception as e:
                    logger.warning(f"[Keep-Alive] Ping failed: {e}")
        
        self.thread = threading.Thread(target=pinger_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False

pinger = SelfPinger()

# ============= ENDPOINTS =============

@app.route('/api/ping')
def ping():
    return jsonify({'status': 'alive', 'timestamp': datetime.now().isoformat()})

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'uptime_seconds': int(time.time() - START_TIME),
        'ping_count': pinger.ping_count
    })

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
    
    # Keep only last 30 messages to save memory
    if len(conversations[session_id]) > 30:
        conversations[session_id] = conversations[session_id][-30:]
    
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
    """Text to Speech - with caching for speed"""
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Check cache first (faster)
    text_hash = hash(text)
    if text_hash in tts_cache:
        return jsonify({
            'audio': tts_cache[text_hash],
            'format': 'audio/mp3',
            'cached': True
        })
    
    # Generate new TTS
    audio_data = text_to_speech_japanese(text)
    if audio_data:
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        # Cache for future use (limit cache size)
        if len(tts_cache) < 100:
            tts_cache[text_hash] = audio_base64
        return jsonify({
            'audio': audio_base64,
            'format': 'audio/mp3',
            'cached': False
        })
    else:
        return jsonify({'error': 'TTS failed'}), 500

@app.route('/api/voice', methods=['POST'])
def voice():
    """HTTP endpoint for voice input - faster than WebSocket"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    mode = request.form.get('mode', 'japanese')
    
    # Read audio bytes
    audio_bytes = audio_file.read()
    
    # Convert speech to text
    text, detected_lang = speech_to_text_japanese(audio_bytes)
    
    if not text:
        return jsonify({
            'success': False,
            'error': 'Could not recognize speech'
        }), 400
    
    result = {
        'success': True,
        'text': text,
        'detected_language': detected_lang
    }
    
    # If in Japanese mode, get AI response
    if mode == 'japanese':
        response_text = get_japanese_response(text)
        result['response'] = response_text
        
        # Generate TTS for response (fast)
        audio_data = text_to_speech_japanese(response_text)
        if audio_data:
            result['response_audio'] = base64.b64encode(audio_data).decode('utf-8')
    
    return jsonify(result)

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in conversations:
        conversations[session_id] = []
    return jsonify({'success': True})

# ============= START APPLICATION =============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # Start the self-pinger
    pinger.start()
    
    logger.info(f"🚀 HenAi Server starting on port {port}")
    logger.info(f"📡 HTTP Voice API enabled (faster than WebSocket)")
    logger.info(f"💾 TTS Cache enabled for faster responses")
    
    # Use waitress for production (faster than Flask dev server)
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=4)
    except ImportError:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
