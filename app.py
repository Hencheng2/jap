# app.py - Complete HenAi Flask Application with Multilingual Support

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import time
import logging
import signal
from datetime import datetime
from models import (
    query_ai_with_fallback,
    is_code_generation_request,
    get_multilingual_response,
    translate_to_english,
    text_to_speech_multilingual,
    speech_to_text_multilingual
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'henai-multilingual-secret-key')
app.config['TIMEOUT'] = 120  # Increase timeout
CORS(app)

# Store conversation history per session with full context
conversations = {}

# Track when the app started
START_TIME = time.time()

# Language codes mapping
LANGUAGE_CODES = {
    'japanese': 'ja',
    'english': 'en',
    'swahili': 'sw',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'chinese': 'zh',
    'korean': 'ko',
    'hindi': 'hi',
    'arabic': 'ar',
    'russian': 'ru',
    'portuguese': 'pt'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    target_language = data.get('language', 'japanese')
    history = data.get('history', [])
    
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
    
    try:
        # Get AI response in target language with timeout
        response_text = get_multilingual_response(
            user_message=message,
            target_language=target_language,
            conversation_history=conversations[session_id][:-1]
        )
        
        # Clean response of markdown
        response_text = clean_markdown(response_text)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response_text = get_fallback_response(target_language)
    
    # Add AI response to history
    conversations[session_id].append({
        'role': 'assistant',
        'content': response_text,
        'timestamp': datetime.now().isoformat(),
        'language': target_language
    })
    
    # Keep only last 100 messages for context
    if len(conversations[session_id]) > 100:
        conversations[session_id] = conversations[session_id][-100:]
    
    return jsonify({
        'response': response_text,
        'session_id': session_id,
        'language': target_language
    })

def get_fallback_response(language):
    """Return fallback response without API call"""
    fallbacks = {
        'japanese': '申し訳ありません。現在応答を生成できません。もう一度お試しください。',
        'english': "I'm having trouble connecting right now. Please try again in a moment.",
        'swahili': "Nina shida kuunganisha kwa sasa. Tafadhali jaribu tena.",
        'spanish': "Estoy teniendo problemas para conectar. Por favor, inténtalo de nuevo.",
        'french': "J'ai des problèmes de connexion. Veuillez réessayer.",
        'german': "Ich habe gerade Verbindungsprobleme. Bitte versuchen Sie es erneut.",
        'chinese': "我现在连接有问题。请稍后再试。",
        'korean': "지금 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.",
        'hindi': "मुझे अभी कनेक्ट करने में समस्या हो रही है। कृपया पुनः प्रयास करें।",
        'arabic': "أواجه مشكلة في الاتصال الآن. يرجى المحاولة مرة أخرى.",
        'russian': "У меня проблемы с подключением. Пожалуйста, попробуйте еще раз.",
        'portuguese': "Estou tendo problemas de conexão. Por favor, tente novamente."
    }
    return fallbacks.get(language, fallbacks['english'])

def clean_markdown(text):
    """Remove markdown formatting from text"""
    import re
    if not text:
        return text
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove inline code markers
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # Remove extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    target = data.get('target', 'english')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    translated = translate_to_english(text)
    return jsonify({
        'original': text,
        'translated': translated,
        'target_language': target
    })

@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'japanese')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    lang_code = LANGUAGE_CODES.get(language, 'ja')
    audio_data = text_to_speech_multilingual(text, lang_code)
    
    if audio_data:
        return jsonify({
            'audio': base64.b64encode(audio_data).decode('utf-8'),
            'format': 'audio/mp3'
        })
    else:
        return jsonify({'error': 'TTS failed'}), 500

@app.route('/api/stt', methods=['POST'])
def stt():
    """Speech to text endpoint"""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    text, detected_lang = speech_to_text_multilingual(audio_bytes)
    
    if text:
        return jsonify({
            'success': True,
            'text': text,
            'detected_language': detected_lang
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Could not recognize speech'
        }), 400

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in conversations:
        conversations[session_id] = []
    return jsonify({'success': True})

@app.route('/api/ping')
def ping_endpoint():
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': int(time.time() - START_TIME)
    })

@app.route('/api/status')
def status_endpoint():
    return jsonify({
        'status': 'running',
        'uptime_seconds': int(time.time() - START_TIME),
        'active_sessions': len(conversations),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    return {"status": "ok"}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 HenAi Multilingual Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
