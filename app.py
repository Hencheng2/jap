# app.py
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import openai
import os
import base64
import tempfile
import uuid
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
import requests
import time

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-please-change-in-production')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# System prompts
JAPANESE_SYSTEM_PROMPT = """あなたは日本語学習アシスタントのHenAiです。以下のルールに厳密に従ってください：
1. ユーザーがどんな言語で話しかけても、必ず日本語で返答してください。
2. 返答は必ず日本語のみを使用し、他の言語を混ぜないでください。
3. 優しく、励ますような口調で、日本語学習者をサポートしてください。
4. 必要に応じて簡単な日本語や漢字にふりがなを提供しても構いません。
5. 会話は自然で教育的なものにしてください。"""

NORMAL_SYSTEM_PROMPT = """You are HenAi, a friendly and helpful Japanese learning assistant. 
Respond in the same language the user uses. If the user speaks English, respond in English. 
If the user speaks Japanese, respond in Japanese. Be warm, encouraging, and helpful."""

def get_openai_response(user_message, mode, session_id):
    """Get response from OpenAI API"""
    if not OPENAI_API_KEY:
        # Fallback mock responses if no API key
        if mode == 'japanese':
            return f"こんにちは！「{user_message}」についてお話ししましょう。日本語の学習を続けて頑張ってください！"
        else:
            return f"Hello! You said: '{user_message}'. I'm HenAi, your Japanese learning assistant. (Note: OpenAI API key not configured. Please add your API key to use full features.)"
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        system_prompt = JAPANESE_SYSTEM_PROMPT if mode == 'japanese' else NORMAL_SYSTEM_PROMPT
        
        # Simple in-memory conversation history (for demo)
        # In production, use a proper session store like Redis
        if not hasattr(app, 'conversations'):
            app.conversations = {}
        
        if session_id not in app.conversations:
            app.conversations[session_id] = []
        
        app.conversations[session_id].append({"role": "user", "content": user_message})
        
        # Keep last 10 messages for context
        messages = [{"role": "system", "content": system_prompt}] + app.conversations[session_id][-10:]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content
        app.conversations[session_id].append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    except Exception as e:
        print(f"OpenAI error: {e}")
        return "申し訳ありません。一時的なエラーが発生しました。もう一度お試しください。" if mode == 'japanese' else "Sorry, I encountered an error. Please try again."

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    mode = data.get('mode', 'japanese')
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    response_text = get_openai_response(user_message, mode, session_id)
    
    return jsonify({
        'response': response_text,
        'mode': mode
    })

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech and return base64 audio"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Use gTTS for Japanese TTS
        tts = gTTS(text=text, lang='ja', slow=False)
        
        # Save to bytes buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'audio': audio_base64
        })
    except Exception as e:
        print(f"TTS error: {e}")
        # Fallback: return empty audio
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stt', methods=['POST'])
def speech_to_text():
    """Convert uploaded audio to text using Google Speech Recognition"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Empty audio file'}), 400
    
    try:
        # Save audio temporarily
        temp_path = tempfile.gettempdir() + f"/audio_{uuid.uuid4().hex}.wav"
        audio_file.save(temp_path)
        
        # Use speech_recognition to convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            
        # Try to recognize with Google (free, requires internet)
        try:
            text = recognizer.recognize_google(audio_data, language='ja-JP')
        except sr.UnknownValueError:
            # Try English if Japanese fails
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
            except:
                text = ""
        except sr.RequestError:
            text = ""
        
        # Clean up temp file
        os.remove(temp_path)
        
        if text:
            return jsonify({
                'success': True,
                'text': text
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not recognize speech'
            }), 400
            
    except Exception as e:
        print(f"STT error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
