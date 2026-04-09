# models.py - AI Models with Multilingual Support and Timeout Handling

import os
import re
import requests
import tempfile
import subprocess
import sys
import json
import time

# Get API key from environment variable
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')

if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY not set in environment variables")

# Android compatibility
if hasattr(sys, 'getandroidtempdir'):
    TEMP_DIR = sys.getandroidtempdir()
else:
    TEMP_DIR = tempfile.gettempdir()

# List of reliable free models (prioritized)
RELIABLE_MODELS = [
    "openrouter/free",
    "microsoft/phi-3.5-mini-128k-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

def get_available_models():
    """Fetch available free models from OpenRouter"""
    if not OPENROUTER_API_KEY:
        return RELIABLE_MODELS
    
    try:
        url = "https://openrouter.ai/api/v1/models"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            free_models = []
            for model in models_data.get('data', []):
                pricing = model.get('pricing', {})
                if pricing.get('prompt') == '0' or pricing.get('request') == '0':
                    free_models.append(model['id'])
            return free_models[:5] if free_models else RELIABLE_MODELS
    except Exception as e:
        print(f"Error fetching models: {e}")
    
    return RELIABLE_MODELS

def query_openrouter(prompt, context=None, target_language='english'):
    """Query OpenRouter with full context and timeout handling"""
    if not OPENROUTER_API_KEY:
        print("❌ OpenRouter API key missing")
        return None
    
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://henai.onrender.com",
            "X-Title": "HenAi"
        }

        # Language-specific system prompts
        language_prompts = {
            'japanese': """あなたはHenAiという名前のAIアシスタントです。以下のルールに従ってください：
1. 必ず日本語のみで応答してください
2. ユーザーがどの言語で話しかけても日本語で返答してください
3. 自然で丁寧な日本語を使用してください
4. コードを生成する場合は、完全で実行可能なコードを提供してください
5. マークダウン記法（*、#、**など）は使用しないでください
6. 応答は簡潔で役立つものにしてください""",

            'english': """You are HenAi, an AI assistant. Follow these rules:
1. Always respond in English
2. Use natural, conversational English
3. Provide complete, working code when generating code
4. Do not use markdown formatting (*, #, **, etc.)
5. Keep responses concise and helpful""",

            'swahili': """Wewe ni HenAi, msaidizi wa AI. Fuata sheria hizi:
1. Jibu kwa Kiswahili tu
2. Tumia Kiswahili cha kawaida na cha asili
3. Toa code kamili inayofanya kazi wakati wa kutengeneza code
4. Usitumie muundo wa markdown (*, #, **, nk.)
5. Weka majibu mafupi na yenye manufaa""",

            'spanish': """Eres HenAi, un asistente de IA. Sigue estas reglas:
1. Responde siempre en español
2. Usa un español natural y conversacional
3. Proporciona código completo y funcional al generar código
4. No uses formato markdown (*, #, **, etc.)
5. Mantén las respuestas concisas y útiles""",

            'french': """Tu es HenAi, un assistant IA. Suis ces règles :
1. Réponds toujours en français
2. Utilise un français naturel et conversationnel
3. Fournis un code complet et fonctionnel lors de la génération de code
4. N'utilise pas de formatage markdown (*, #, **, etc.)
5. Garde les réponses concises et utiles""",

            'german': """Du bist HenAi, ein KI-Assistent. Befolge diese Regeln:
1. Antworte immer auf Deutsch
2. Verwende natürliches, konversationelles Deutsch
3. Liefere vollständigen, funktionierenden Code bei Code-Generierung
4. Verwende keine Markdown-Formatierung (*, #, **, etc.)
5. Halte Antworten prägnant und hilfreich""",

            'chinese': """你是HenAi，一个AI助手。请遵守以下规则：
1. 始终用中文回复
2. 使用自然、对话式的中文
3. 生成代码时提供完整可运行的代码
4. 不要使用markdown格式（*、#、**等）
5. 保持回复简洁有用""",

            'korean': """당신은 HenAi, AI 어시스턴트입니다. 다음 규칙을 따르세요:
1. 항상 한국어로 응답하세요
2. 자연스러운 대화체 한국어를 사용하세요
3. 코드 생성 시 완전하고 작동 가능한 코드를 제공하세요
4. 마크다운 형식(*, #, ** 등)을 사용하지 마세요
5. 응답을 간결하고 유용하게 유지하세요""",

            'hindi': """आप HenAi, एक AI सहायक हैं। इन नियमों का पालन करें:
1. हमेशा हिंदी में जवाब दें
2. प्राकृतिक, संवादात्मक हिंदी का उपयोग करें
3. कोड बनाते समय पूर्ण, काम करने वाला कोड प्रदान करें
4. मार्कडाउन फ़ॉर्मेटिंग (*, #, **, आदि) का उपयोग न करें
5. उत्तर संक्षिप्त और उपयोगी रखें""",

            'arabic': """أنت HenAi، مساعد ذكاء اصطناعي. اتبع هذه القواعد:
1. قم بالرد دائمًا باللغة العربية
2. استخدم اللغة العربية الطبيعية والمحادثة
3. قدم كودًا كاملاً وقابلاً للعمل عند إنشاء الكود
4. لا تستخدم تنسيق markdown (*, #, **، إلخ)
5. حافظ على الإجابات موجزة ومفيدة""",

            'russian': """Ты HenAi, AI-ассистент. Следуй этим правилам:
1. Всегда отвечай на русском языке
2. Используй естественный, разговорный русский
3. Предоставляй полный, рабочий код при генерации кода
4. Не используй markdown-форматирование (*, #, **, и т.д.)
5. Делай ответы краткими и полезными""",

            'portuguese': """Você é HenAi, um assistente de IA. Siga estas regras:
1. Responda sempre em português
2. Use português natural e conversacional
3. Forneça código completo e funcional ao gerar código
4. Não use formatação markdown (*, #, **, etc.)
5. Mantenha as respostas concisas e úteis"""
        }

        system_prompt = language_prompts.get(target_language, language_prompts['english'])

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            # Use last 15 messages for context (reduced for speed)
            for ctx_msg in context[-15:]:
                messages.append(ctx_msg)

        messages.append({"role": "user", "content": prompt})

        models = get_available_models()

        for model in models:
            try:
                print(f"  Trying model: {model} for language: {target_language}")
                
                # Determine max tokens based on request type
                is_code = is_code_generation_request(prompt)
                max_tokens = 6000 if is_code else 2000
                temperature = 0.5 if is_code else 0.7
                
                data = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                # Use shorter timeout for non-code requests
                timeout_seconds = 90 if is_code else 45
                
                response = requests.post(url, json=data, headers=headers, timeout=timeout_seconds)

                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        message_content = result['choices'][0]['message']['content']
                        print(f"✓ OpenRouter success with {model}")
                        return message_content
                elif response.status_code == 429:
                    print(f"  Rate limited for {model}, waiting 2 seconds...")
                    time.sleep(2)
                    continue
                elif response.status_code == 402:
                    print(f"  Model {model} requires payment, skipping...")
                    continue
                else:
                    print(f"  Error {response.status_code} for {model}")
                    continue

            except requests.exceptions.Timeout:
                print(f"  Timeout for {model}, trying next...")
                continue
            except requests.exceptions.ConnectionError:
                print(f"  Connection error for {model}, trying next...")
                continue
            except Exception as e:
                print(f"  Exception with {model}: {e}")
                continue

        print("❌ All models failed")
        return None

    except Exception as e:
        print(f"OpenRouter error: {e}")
        return None

def get_multilingual_response(user_message, target_language='japanese', conversation_history=None):
    """
    Get response in target language for any input language.
    """
    print(f"🌐 Multilingual Mode - Target: {target_language}, Message: {user_message[:50]}...")
    
    # Detect if this is a code generation request
    is_code = is_code_generation_request(user_message)
    
    if is_code:
        print("📝 Code generation detected")
        prompt = f"""Generate complete, production-ready {user_message}

Requirements:
- Provide FULL working code (no placeholders or abbreviations)
- Include all necessary imports, styling, and functionality
- Make it responsive and modern
- Add comments for important sections
- Return ONLY the code in a single code block
- Do not include any explanatory text outside the code block"""
    else:
        prompt = f"""User message: {user_message}

Please respond in {target_language} naturally and conversationally. 
Keep your response helpful, clear, and appropriate for the conversation context.
Do not use any markdown formatting like *, #, **, etc.
Just provide plain text response in {target_language}."""
    
    # Get AI response
    response = query_openrouter(
        prompt=prompt,
        context=conversation_history,
        target_language=target_language
    )
    
    if response:
        # Clean up response
        response = clean_markdown_from_response(response)
        return response
    else:
        # Fallback responses
        fallbacks = {
            'japanese': '申し訳ありません。応答を生成できませんでした。もう一度お試しください。',
            'english': "I'm sorry, I couldn't generate a response. Please try again.",
            'swahili': "Samahani, sikuweza kutoa jibu. Tafadhali jaribu tena.",
            'spanish': "Lo siento, no pude generar una respuesta. Por favor, inténtalo de nuevo.",
            'french': "Désolé, je n'ai pas pu générer de réponse. Veuillez réessayer.",
            'german': "Es tut mir leid, ich konnte keine Antwort generieren. Bitte versuchen Sie es erneut.",
            'chinese': "抱歉，我无法生成回复。请再试一次。",
            'korean': "죄송합니다. 응답을 생성할 수 없었습니다. 다시 시도해 주세요.",
            'hindi': "क्षमा करें, मैं प्रतिक्रिया उत्पन्न नहीं कर सका। कृपया पुनः प्रयास करें।",
            'arabic': "عذرًا، لم أتمكن من إنشاء رد. يرجى المحاولة مرة أخرى.",
            'russian': "Извините, не удалось сгенерировать ответ. Пожалуйста, попробуйте еще раз.",
            'portuguese': "Desculpe, não consegui gerar uma resposta. Por favor, tente novamente."
        }
        return fallbacks.get(target_language, fallbacks['english'])

def clean_markdown_from_response(text):
    """Clean markdown from response"""
    if not text:
        return text
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # Clean up extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def translate_to_english(text):
    """Translate text to English using Google Translate API"""
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'auto',
            'tl': 'en',
            'dt': 't',
            'q': text
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and len(result[0]) > 0:
                translated = ''.join([part[0] for part in result[0]])
                return translated
    except Exception as e:
        print(f"Translation error: {e}")
    return text

def text_to_speech_multilingual(text, lang_code='ja'):
    """Convert text to speech using gTTS"""
    try:
        from gtts import gTTS
        import io
        
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
        
    except Exception as e:
        print(f"TTS error: {e}")
        return None

def speech_to_text_multilingual(audio_bytes):
    """Convert speech to text using Google Speech Recognition"""
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        temp_audio_path = os.path.join(TEMP_DIR, 'temp_audio.wav')
        
        with open(temp_audio_path, 'wb') as temp_audio:
            temp_audio.write(audio_bytes)
        
        with sr.AudioFile(temp_audio_path) as source:
            audio = recognizer.record(source)
        
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        
        # Try multiple languages
        languages = ['ja-JP', 'en-US', 'sw', 'es-ES', 'fr-FR', 'de-DE', 'zh-CN', 'ko-KR', 'hi-IN', 'ar-SA', 'ru-RU', 'pt-PT']
        
        for lang in languages:
            try:
                text = recognizer.recognize_google(audio, language=lang)
                return text, lang.split('-')[0]
            except:
                continue
        
        # Auto-detect as last resort
        try:
            text = recognizer.recognize_google(audio)
            return text, 'auto'
        except:
            pass
        
        return None, None
        
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None, None

def is_code_generation_request(message):
    """Detect if message is asking for code generation"""
    message_lower = message.lower()
    
    # File analysis - NOT code generation
    file_analysis_phrases = ['summarize', 'explain', 'what is', 'tell me about', 'describe']
    if any(phrase in message_lower for phrase in file_analysis_phrases):
        return False
    
    # Code generation keywords
    code_keywords = [
        'create code', 'generate code', 'write code', 'build code',
        'write a program', 'create a program', 'generate a program',
        'write a script', 'create a script', 'generate a script',
        'implement', 'code for', 'program that', 'make a website',
        'build a website', 'create a website', 'marketing website'
    ]
    
    if any(keyword in message_lower for keyword in code_keywords):
        return True
    
    verbs = ['create', 'generate', 'write', 'build', 'develop', 'make', 'code']
    technologies = ['html', 'css', 'javascript', 'python', 'react', 'vue', 'angular', 'website']
    
    has_verb = any(verb in message_lower for verb in verbs)
    has_tech = any(tech in message_lower for tech in technologies)
    
    return has_verb and has_tech

def query_ai_with_fallback(prompt, context=None, is_code_generation=False, force_japanese=False):
    """Legacy function for compatibility"""
    target_lang = 'japanese' if force_japanese else 'english'
    return get_multilingual_response(prompt, target_lang, context)
