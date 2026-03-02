"""
OUR OWN SENTIMENT ANALYSIS SERVER
=================================
A custom Flask server with our own trained model.
No external APIs - everything runs locally on our server!

Features:
- Web UI for sentiment analysis
- REST API endpoints
- Our own ML model (TF-IDF + Logistic Regression)
- 100% self-owned and self-hosted
"""
import os
import json
import csv
import re
import html
import math
import hashlib
import secrets
import threading
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from dotenv import load_dotenv

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except Exception:
    Limiter = None  # type: ignore[assignment]
    get_remote_address = None  # type: ignore[assignment]

load_dotenv()
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import our own predictor
from predictor import OurSentimentPredictor

# ==================== CONFIGURATION ====================


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    logger.warning('Invalid boolean for %s=%r. Using default=%s', name, raw, default)
    return default


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning('Invalid integer for %s=%r. Using default=%s', name, raw, default)
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning('Invalid float for %s=%r. Using default=%s', name, raw, default)
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _resolve_secret_key() -> str:
    configured = os.getenv('SECRET_KEY', '').strip()
    if configured:
        return configured
    generated = secrets.token_urlsafe(48)
    logger.warning(
        'SECRET_KEY is not configured. Using a generated ephemeral key for this process.'
    )
    return generated

class Config:
    """Server configuration."""
    SECRET_KEY = _resolve_secret_key()
    DEBUG = _env_bool('DEBUG', False)
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = _env_int('PORT', 5000, minimum=1)
    TEMPLATES_AUTO_RELOAD = _env_bool('TEMPLATES_AUTO_RELOAD', True)
    DISABLE_HTML_CACHE = _env_bool('DISABLE_HTML_CACHE', True)
    
    # Model directory
    MODEL_DIR = Path(__file__).parent / 'models'
    
    # History settings
    HISTORY_FILE = Path(__file__).parent / 'data' / 'history.csv'
    HISTORY_LIMIT = 100
    DASHBOARD_HISTORY_WINDOW = _env_int('DASHBOARD_HISTORY_WINDOW', 500, minimum=1)
    DEFAULT_AUTO_TRANSLATE = _env_bool('DEFAULT_AUTO_TRANSLATE', True)

    # Google AI Studio settings (optional)
    GOOGLE_AI_STUDIO_API_KEY = os.getenv('GOOGLE_AI_STUDIO_API_KEY', '').strip()
    GOOGLE_AI_MODEL = os.getenv('GOOGLE_AI_MODEL', 'gemini-flash-latest')
    _legacy_fallback_models = [
        model.strip()
        for model in os.getenv(
            'GOOGLE_AI_FALLBACK_MODELS',
            'gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite'
        ).split(',')
        if model.strip()
    ]
    GOOGLE_AI_FALLBACK_MODEL = os.getenv(
        'GOOGLE_AI_FALLBACK_MODEL',
        _legacy_fallback_models[0] if _legacy_fallback_models else 'gemini-2.5-flash'
    ).strip()
    GOOGLE_AI_MAX_CONCURRENT_REQUESTS = _env_int(
        'GOOGLE_AI_MAX_CONCURRENT_REQUESTS',
        3,
        minimum=1
    )
    GOOGLE_AI_MAX_RETRIES = _env_int('GOOGLE_AI_MAX_RETRIES', 2, minimum=0)
    GOOGLE_AI_RETRY_BACKOFF_SECONDS = _env_float(
        'GOOGLE_AI_RETRY_BACKOFF_SECONDS',
        1.25,
        minimum=0.0
    )
    GOOGLE_AI_TIMEOUT = _env_int('GOOGLE_AI_TIMEOUT', 20, minimum=1)

    # API rate limiting
    RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '20 per minute').strip()
    RATE_LIMIT_REWRITE = os.getenv('RATE_LIMIT_REWRITE', '30 per minute').strip()
    RATE_LIMIT_STORAGE_URI = os.getenv('RATE_LIMIT_STORAGE_URI', 'memory://').strip()

    # API key system
    API_KEYS_FILE = Path(__file__).parent / 'data' / 'api_keys.json'
    ENFORCE_API_KEYS = _env_bool('ENFORCE_API_KEYS', False)
    ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', '').strip()
    TRUST_X_FORWARDED_FOR = _env_bool('TRUST_X_FORWARDED_FOR', False)

    # Request guardrails
    BATCH_MAX_ITEMS = _env_int('BATCH_MAX_ITEMS', 100, minimum=1)
    BATCH_MAX_TEXT_LENGTH = _env_int('BATCH_MAX_TEXT_LENGTH', 4000, minimum=1)


# ==================== FLASK APP ====================

app = Flask(__name__)
app.config.from_object(Config)
app.jinja_env.auto_reload = app.config.get('TEMPLATES_AUTO_RELOAD', True)


def _rate_limit_key() -> str:
    """Build client key with Render/proxy support."""
    forwarded_for = request.headers.get('X-Forwarded-For', '').strip()
    if Config.TRUST_X_FORWARDED_FOR and forwarded_for:
        return forwarded_for.split(',')[0].strip()
    if get_remote_address is not None:
        return str(get_remote_address())
    return request.remote_addr or 'unknown'


class _NoopLimiter:
    """Fallback limiter when flask-limiter is unavailable."""

    def limit(self, *_args, **_kwargs):
        def _decorator(func):
            return func
        return _decorator


if Limiter is not None:
    limiter = Limiter(
        key_func=_rate_limit_key,
        app=app,
        default_limits=[Config.RATE_LIMIT_DEFAULT] if Config.RATE_LIMIT_DEFAULT else [],
        storage_uri=Config.RATE_LIMIT_STORAGE_URI or 'memory://'
    )
else:
    limiter = _NoopLimiter()
    print(
        'Warning: flask-limiter is not installed. '
        'Rate limiting is disabled until dependency is installed.'
    )

google_api_semaphore = threading.Semaphore(Config.GOOGLE_AI_MAX_CONCURRENT_REQUESTS)

# Initialize our predictor
predictor = None
api_store_lock = threading.Lock()


@app.after_request
def apply_html_cache_headers(response):
    """Prevent stale HTML templates from being cached in browsers/proxies."""
    if app.config.get('DISABLE_HTML_CACHE', True):
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type.lower():
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
    return response

DEMO_EXAMPLES = [
    {
        'title': 'Happy Customer Review',
        'category': 'E-commerce',
        'text': "I absolutely love this product. The quality is fantastic and delivery was super fast!",
        'target_tone': 'positive'
    },
    {
        'title': 'Angry Complaint',
        'category': 'Support',
        'text': "This is unacceptable. The app keeps crashing and your support team is not responding.",
        'target_tone': 'professional'
    },
    {
        'title': 'Neutral Business Update',
        'category': 'Work',
        'text': "The meeting has been moved to Monday at 10 AM. Please review the attached report before attending.",
        'target_tone': 'formal'
    },
    {
        'title': 'Social Post Draft',
        'category': 'Social',
        'text': "just launched our new feature today and we are excited to see your feedback",
        'target_tone': 'friendly'
    },
    {
        'title': 'Hindi Example',
        'category': 'Multilingual',
        'text': "यह सेवा बहुत उपयोगी है और मुझे इसका अनुभव अच्छा लगा।",
        'target_tone': 'positive'
    },
]


class GoogleAIError(RuntimeError):
    """Structured Google AI error with status metadata for retry logic."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_status: Optional[str] = None,
        retryable: bool = False,
        model: Optional[str] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_status = error_status
        self.retryable = retryable
        self.model = model


def get_predictor():
    """Get or create our predictor instance."""
    global predictor
    if predictor is None:
        print("Loading our own sentiment model...")
        predictor = OurSentimentPredictor(str(Config.MODEL_DIR))
    return predictor


# ==================== HELPER FUNCTIONS ====================

def save_to_history(text: str, sentiment: str, confidence: float):
    """Save prediction to history file."""
    try:
        Config.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        file_exists = Config.HISTORY_FILE.exists()
        
        with open(Config.HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'text', 'sentiment', 'confidence'])
            writer.writerow([
                datetime.now().isoformat(),
                text[:200],
                sentiment,
                f"{confidence:.4f}"
            ])
    except Exception as e:
        print(f"Error saving to history: {e}")


def load_history(limit: int = 100):
    """Load prediction history."""
    predictions = []
    try:
        if Config.HISTORY_FILE.exists():
            with open(Config.HISTORY_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_records = list(reader)
                predictions = all_records[-limit:] if len(all_records) > limit else all_records
    except Exception as e:
        print(f"Error loading history: {e}")
    return predictions


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO format."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def _load_json_file(path: Path, default_value: Dict[str, Any]) -> Dict[str, Any]:
    """Load JSON data from disk with fallback defaults."""
    try:
        if not path.exists():
            return json.loads(json.dumps(default_value))
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return json.loads(json.dumps(default_value))


def _save_json_file(path: Path, data: Dict[str, Any]) -> None:
    """Safely write JSON data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _empty_api_store() -> Dict[str, Any]:
    return {'keys': [], 'usage_logs': []}


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()


def _mask_api_key(raw_key: str) -> str:
    if len(raw_key) <= 8:
        return '*' * len(raw_key)
    return f"{raw_key[:6]}...{raw_key[-4:]}"


def _find_api_key_record(store: Dict[str, Any], raw_key: str) -> Optional[Dict[str, Any]]:
    key_hash = _hash_api_key(raw_key)
    for record in store.get('keys', []):
        if record.get('active', True) and record.get('key_hash') == key_hash:
            return record
    return None


def create_api_key(name: str) -> Dict[str, str]:
    """Create and persist a new API key. Raw key is returned only once."""
    clean_name = (name or 'default-key').strip()[:80]
    raw_key = f"sk-{secrets.token_urlsafe(24)}"
    now = _utc_now_iso()
    record = {
        'id': secrets.token_hex(8),
        'name': clean_name,
        'key_hash': _hash_api_key(raw_key),
        'key_preview': _mask_api_key(raw_key),
        'created_at': now,
        'last_used_at': None,
        'usage_count': 0,
        'active': True
    }

    with api_store_lock:
        store = _load_json_file(Config.API_KEYS_FILE, _empty_api_store())
        store.setdefault('keys', []).append(record)
        _save_json_file(Config.API_KEYS_FILE, store)

    return {
        'id': record['id'],
        'name': record['name'],
        'api_key': raw_key,
        'key_preview': record['key_preview'],
        'created_at': record['created_at']
    }


def list_api_keys() -> List[Dict[str, Any]]:
    """List persisted API key metadata without exposing raw key data."""
    with api_store_lock:
        store = _load_json_file(Config.API_KEYS_FILE, _empty_api_store())
    keys = []
    for record in store.get('keys', []):
        keys.append({
            'id': record.get('id'),
            'name': record.get('name'),
            'key_preview': record.get('key_preview'),
            'created_at': record.get('created_at'),
            'last_used_at': record.get('last_used_at'),
            'usage_count': record.get('usage_count', 0),
            'active': bool(record.get('active', True))
        })
    return sorted(keys, key=lambda item: item.get('usage_count', 0), reverse=True)


def track_api_usage(endpoint: str) -> Optional[Tuple[Any, int]]:
    """
    Track API usage and optionally enforce API key validation.
    """
    raw_key = request.headers.get('X-API-Key', '').strip()
    now = _utc_now_iso()

    with api_store_lock:
        store = _load_json_file(Config.API_KEYS_FILE, _empty_api_store())
        record = _find_api_key_record(store, raw_key) if raw_key else None
        valid_key = bool(record)

        if record:
            record['usage_count'] = int(record.get('usage_count', 0)) + 1
            record['last_used_at'] = now

        usage_event = {
            'timestamp': now,
            'endpoint': endpoint,
            'api_key_id': record.get('id') if record else None,
            'api_key_name': record.get('name') if record else ('anonymous' if not raw_key else 'invalid'),
            'valid_key': valid_key,
            'ip': request.remote_addr or ''
        }
        store.setdefault('usage_logs', []).append(usage_event)
        store['usage_logs'] = store['usage_logs'][-10000:]
        _save_json_file(Config.API_KEYS_FILE, store)

    if Config.ENFORCE_API_KEYS and not valid_key:
        return jsonify({
            'success': False,
            'error': 'Valid X-API-Key required'
        }), 401

    return None


def _is_admin_authorized(data: Optional[Dict[str, Any]] = None) -> bool:
    """Validate admin access using ADMIN_TOKEN if configured."""
    provided = request.headers.get('X-Admin-Token', '').strip()
    if (not provided) and isinstance(data, dict):
        provided = str(data.get('admin_token', '')).strip()

    if Config.ADMIN_TOKEN:
        return bool(provided) and secrets.compare_digest(provided, Config.ADMIN_TOKEN)

    logger.warning('ADMIN_TOKEN is not configured. Admin endpoints are disabled.')
    return False


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse bool from JSON values safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'n', 'off'}:
            return False
    if value is None:
        return default
    return bool(value)


def detect_language(text: str) -> Dict[str, Any]:
    """
    Lightweight language detection for common languages/scripts.
    """
    if not text.strip():
        return {'code': 'en', 'name': 'English', 'confidence': 0.0}

    if re.search(r'[\u0900-\u097F]', text):
        return {'code': 'hi', 'name': 'Hindi', 'confidence': 0.99}
    if re.search(r'[\u4E00-\u9FFF]', text):
        return {'code': 'zh', 'name': 'Chinese', 'confidence': 0.99}
    if re.search(r'[\u0600-\u06FF]', text):
        return {'code': 'ar', 'name': 'Arabic', 'confidence': 0.99}
    if re.search(r'[\u0400-\u04FF]', text):
        return {'code': 'ru', 'name': 'Russian', 'confidence': 0.99}

    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return {'code': 'en', 'name': 'English', 'confidence': 0.5}

    hints = {
        'en': {'the', 'and', 'is', 'are', 'this', 'that', 'with', 'for'},
        'es': {'el', 'la', 'que', 'de', 'y', 'en', 'es', 'muy'},
        'fr': {'le', 'la', 'de', 'et', 'est', 'très', 'avec', 'pour'},
        'de': {'der', 'die', 'das', 'und', 'ist', 'mit', 'für', 'sehr'},
        'pt': {'o', 'a', 'de', 'e', 'é', 'com', 'muito', 'para'},
    }

    counts: Dict[str, int] = {}
    word_set = set(words)
    for code, tokens in hints.items():
        counts[code] = len(word_set.intersection(tokens))

    best_code = max(counts, key=lambda code: counts[code]) if counts else 'en'
    best_score = counts.get(best_code, 0)
    if best_score == 0:
        best_code = 'en'
        confidence = 0.55
    else:
        confidence = min(0.95, 0.55 + best_score * 0.1)

    language_names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'pt': 'Portuguese',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'ru': 'Russian'
    }
    return {
        'code': best_code,
        'name': language_names.get(best_code, best_code.upper()),
        'confidence': round(confidence, 2)
    }


def _parse_google_error_body(body: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract provider status/message from Google error JSON."""
    if not body:
        return None, None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None, None

    if not isinstance(parsed, dict):
        return None, None
    error_obj = parsed.get('error', {})
    if not isinstance(error_obj, dict):
        return None, None

    status = str(error_obj.get('status', '')).strip() or None
    message = str(error_obj.get('message', '')).strip() or None
    return status, message


def _is_retryable_google_error(
    status_code: Optional[int],
    status_text: Optional[str]
) -> bool:
    """Retry only temporary provider overload errors."""
    if status_code == 503:
        return True
    if status_text and status_text.upper() == 'UNAVAILABLE':
        return True
    return False


def _google_model_pair() -> Tuple[str, Optional[str]]:
    """Return (primary_model, fallback_model)."""
    primary_model = str(Config.GOOGLE_AI_MODEL or '').strip()
    fallback_model = str(Config.GOOGLE_AI_FALLBACK_MODEL or '').strip() or None
    if fallback_model == primary_model:
        fallback_model = None
    return primary_model, fallback_model


def _call_google_ai(
    prompt: str,
    max_output_tokens: int = 512,
    temperature: float = 0.2,
    model_name: Optional[str] = None
) -> str:
    """Call one Google model once and return generated text."""
    api_key = Config.GOOGLE_AI_STUDIO_API_KEY
    if not api_key:
        raise GoogleAIError(
            'GOOGLE_AI_STUDIO_API_KEY is not configured',
            status_code=503,
            error_status='NOT_CONFIGURED',
            retryable=False
        )

    selected_model = (model_name or Config.GOOGLE_AI_MODEL or '').strip()
    if not selected_model:
        raise GoogleAIError(
            'GOOGLE_AI_MODEL is not configured',
            status_code=503,
            error_status='NOT_CONFIGURED',
            retryable=False
        )

    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': temperature,
            'maxOutputTokens': max_output_tokens
        }
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{selected_model}:generateContent?key={api_key}"
    )
    req = urllib_request.Request(
        url=url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        with google_api_semaphore:
            with urllib_request.urlopen(req, timeout=Config.GOOGLE_AI_TIMEOUT) as response:
                raw_response = response.read().decode('utf-8')
    except urllib_error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='ignore')
        provider_status, provider_message = _parse_google_error_body(body)
        retryable = _is_retryable_google_error(exc.code, provider_status)
        message = provider_message or body[:300] or str(exc)
        if retryable:
            logger.warning(
                'Google AI temporary failure: model=%s code=%s status=%s message=%s',
                selected_model,
                exc.code,
                provider_status or '',
                message[:160]
            )
        else:
            logger.info(
                'Google AI non-retry failure: model=%s code=%s status=%s message=%s',
                selected_model,
                exc.code,
                provider_status or '',
                message[:160]
            )
        raise GoogleAIError(
            (
                f'Google AI request failed ({exc.code}) for model "{selected_model}": '
                f'{message}'
            ),
            status_code=exc.code,
            error_status=provider_status,
            retryable=retryable,
            model=selected_model
        ) from exc
    except Exception as exc:
        logger.warning(
            'Google AI network failure: model=%s error=%s',
            selected_model,
            str(exc)[:180]
        )
        raise GoogleAIError(
            f'Google AI request failed for model "{selected_model}": {str(exc)}',
            status_code=None,
            error_status='NETWORK_ERROR',
            retryable=False,
            model=selected_model
        ) from exc

    try:
        response_json = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise GoogleAIError(
            f'Google AI returned invalid JSON response from "{selected_model}"',
            status_code=502,
            error_status='INVALID_RESPONSE',
            retryable=True,
            model=selected_model
        ) from exc

    generated_text = (
        response_json.get('candidates', [{}])[0]
        .get('content', {})
        .get('parts', [{}])[0]
        .get('text', '')
        .strip()
    )
    if not generated_text:
        raise GoogleAIError(
            f'Google AI returned empty output from "{selected_model}"',
            status_code=502,
            error_status='EMPTY_OUTPUT',
            retryable=True,
            model=selected_model
        )
    return generated_text


def _call_google_ai_resilient(
    prompt: str,
    max_output_tokens: int = 512,
    temperature: float = 0.2
) -> Tuple[str, str]:
    """
    Call Google AI with retry + fallback models.
    Returns (generated_text, model_used).
    """
    primary_model, fallback_model = _google_model_pair()
    if not primary_model:
        raise GoogleAIError(
            'No Google AI model is configured',
            status_code=503,
            error_status='NOT_CONFIGURED',
            retryable=False
        )

    attempted_models = [primary_model]
    last_error: Optional[GoogleAIError] = None

    # Primary model with bounded exponential backoff retries.
    for attempt in range(Config.GOOGLE_AI_MAX_RETRIES + 1):
        try:
            result_text = _call_google_ai(
                prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                model_name=primary_model
            )
            return result_text, primary_model
        except GoogleAIError as exc:
            last_error = exc
            should_retry = (
                exc.retryable and attempt < Config.GOOGLE_AI_MAX_RETRIES
            )
            if not should_retry:
                break
            delay_seconds = (
                Config.GOOGLE_AI_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                + random.uniform(0.0, 0.5)
            )
            delay_seconds = min(8.0, max(0.0, delay_seconds))
            logger.warning(
                'Retrying Google AI primary model: model=%s attempt=%s/%s delay=%.2fs',
                primary_model,
                attempt + 1,
                Config.GOOGLE_AI_MAX_RETRIES,
                delay_seconds
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    should_try_fallback = bool(
        fallback_model and last_error and _is_retryable_google_error(
            last_error.status_code,
            last_error.error_status
        )
    )

    # Single fallback attempt only (no fan-out).
    if should_try_fallback and fallback_model:
        attempted_models.append(fallback_model)
        logger.warning(
            'Switching to Google AI fallback model: primary=%s fallback=%s',
            primary_model,
            fallback_model
        )
        try:
            result_text = _call_google_ai(
                prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                model_name=fallback_model
            )
            return result_text, fallback_model
        except GoogleAIError as exc:
            last_error = exc

    if last_error and _is_retryable_google_error(
        last_error.status_code,
        last_error.error_status
    ):
        logger.warning(
            'Google AI unavailable after retries. models=%s',
            ','.join(attempted_models)
        )
        raise GoogleAIError(
            (
                'Google AI service is temporarily busy. '
                f'Tried models: {", ".join(attempted_models)}. '
                'Please retry in a few seconds.'
            ),
            status_code=503,
            error_status='UNAVAILABLE',
            retryable=True,
            model=last_error.model
        ) from last_error

    if last_error is not None:
        raise last_error

    raise GoogleAIError('Google AI request did not run', status_code=502)


def translate_to_english(text: str, source_lang: str) -> Tuple[str, bool, Optional[str]]:
    """
    Translate text to English when source language is not English.
    Returns (translated_text, translated_flag, note_or_error).
    """
    if source_lang == 'en':
        return text, False, None

    prompt = (
        "Translate the following text into English.\n"
        "Return only the translated text with no extra commentary.\n\n"
        f"Text:\n{text}"
    )
    try:
        translated, _ = _call_google_ai_resilient(
            prompt,
            max_output_tokens=300,
            temperature=0.0
        )
        translated = translated.strip()
    except GoogleAIError as exc:
        if _is_retryable_google_error(exc.status_code, exc.error_status):
            return text, False, 'Auto-translation temporarily unavailable; analyzed original text.'
        if exc.error_status == 'NOT_CONFIGURED':
            return text, False, 'Auto-translation is not configured; analyzed original text.'
        return text, False, 'Auto-translation failed; analyzed original text.'

    if not translated:
        return text, False, 'Translation returned empty output'
    return translated, True, None


def _strip_markdown_code_fence(raw_text: str) -> str:
    """Remove surrounding markdown code fences if present."""
    cleaned = str(raw_text or '').strip()
    if not cleaned.startswith('```'):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines).strip()


def _parse_rewrite_output(raw_text: str) -> Optional[Dict[str, str]]:
    """
    Parse model rewrite output that may include markdown fences or surrounding text.
    """
    raw = str(raw_text or '').strip()
    if not raw:
        return None

    candidates: List[str] = [raw]
    fenced_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', raw, flags=re.IGNORECASE)
    for block in fenced_blocks:
        block = block.strip()
        if block:
            candidates.append(block)

    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start:end + 1].strip())

    seen: set = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized.lower().startswith('json'):
            candidate_lines = normalized.splitlines()
            if candidate_lines and candidate_lines[0].strip().lower() == 'json':
                normalized = '\n'.join(candidate_lines[1:]).strip()
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return {
                'corrected_text': str(parsed.get('corrected_text', '')).strip(),
                'rewritten_text': str(parsed.get('rewritten_text', '')).strip(),
                'notes': str(parsed.get('notes', '')).strip()
            }

    extracted: Dict[str, str] = {}
    for key in ('corrected_text', 'rewritten_text', 'notes'):
        match = re.search(
            rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"',
            raw,
            flags=re.IGNORECASE | re.DOTALL
        )
        if not match:
            continue
        value = match.group(1)
        try:
            extracted[key] = json.loads(f'"{value}"').strip()
        except Exception:
            extracted[key] = value.strip()

    if extracted:
        return {
            'corrected_text': str(extracted.get('corrected_text', '')).strip(),
            'rewritten_text': str(extracted.get('rewritten_text', '')).strip(),
            'notes': str(extracted.get('notes', '')).strip()
        }

    return None


def rewrite_text_with_google_ai(
    text: str,
    style: str = 'clear',
    target_tone: str = 'positive'
):
    """
    Check and rewrite text using Google AI Studio (Gemini API).
    """
    style_map = {
        'clear': 'clear and concise',
        'formal': 'formal and professional',
        'simple': 'simple and easy to understand',
        'friendly': 'friendly and conversational'
    }
    style_name = style.strip().lower() if isinstance(style, str) else 'clear'
    style_instruction = style_map.get(style_name, style_map['clear'])
    tone_name = target_tone.strip().lower() if isinstance(target_tone, str) else 'positive'
    tone_instruction = (
        "5) Rewrite every sentence with clearly positive, optimistic, and constructive language. "
        "If the original has negative words, replace them with solution-oriented phrasing while preserving facts."
        if tone_name in {'positive', 'very_positive', 'professional', 'friendly'}
        else (
            "5) Keep the tone respectfully critical and direct without abusive wording."
            if tone_name in {'negative', 'very_negative'}
            else "5) Keep tone balanced and clear."
        )
    )

    prompt = (
        "You are a writing assistant.\n"
        "Task:\n"
        "1) Correct grammar, spelling, and punctuation.\n"
        "2) Rewrite the text while preserving exact meaning.\n"
        f"3) Use this style: {style_instruction}.\n"
        f"4) Keep the emotional tone {tone_name} if possible without changing the core message.\n\n"
        f"{tone_instruction}\n\n"
        f"Text:\n{text}\n\n"
        "Return valid JSON with keys: corrected_text, rewritten_text, notes."
    )
    generated_text, model_used = _call_google_ai_resilient(
        prompt,
        max_output_tokens=512,
        temperature=0.2
    )
    if not generated_text:
        raise GoogleAIError('Google AI returned empty output', status_code=502)

    result = {
        'corrected_text': '',
        'rewritten_text': '',
        'notes': '',
        'model_used': model_used
    }

    parsed_output = _parse_rewrite_output(generated_text)
    if parsed_output is not None:
        result['corrected_text'] = parsed_output.get('corrected_text', '')
        result['rewritten_text'] = parsed_output.get('rewritten_text', '')
        result['notes'] = parsed_output.get('notes', '')
    else:
        # Fallback: if model did not return strict JSON, normalize raw output.
        result['rewritten_text'] = _strip_markdown_code_fence(generated_text)
        result['notes'] = 'Model output was not strict JSON; returned normalized text.'

    if not result['corrected_text']:
        result['corrected_text'] = text
    if not result['rewritten_text']:
        result['rewritten_text'] = result['corrected_text']

    return result


def rewrite_text_locally(
    text: str,
    style: str = 'clear',
    target_tone: str = 'positive'
) -> Dict[str, str]:
    """
    Local fallback rewrite path used when Google AI is unavailable.
    """
    raw_text = str(text or '').strip()
    if not raw_text:
        return {
            'corrected_text': '',
            'rewritten_text': '',
            'notes': 'No text provided for local rewrite fallback.',
            'model_used': 'local-rules-v2'
        }

    # Basic normalization and light grammar cleanup.
    corrected = re.sub(r'\s+', ' ', raw_text).strip()
    corrected = re.sub(r'\bi\b', 'I', corrected)
    if corrected:
        corrected = corrected[0].upper() + corrected[1:]
    if corrected and corrected[-1] not in '.!?':
        corrected += '.'

    rewritten = corrected
    tone_name = str(target_tone or 'positive').strip().lower()
    style_name = str(style or 'clear').strip().lower()
    request_seed = secrets.token_hex(4)

    def _split_sentences(value: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.findall(r'[^.!?]+[.!?]?', value or '')
            if sentence.strip()
        ]

    def _lower_first_if_safe(value: str) -> str:
        if not value:
            return value
        if value.startswith('I ') or value == 'I':
            return value
        return value[:1].lower() + value[1:]

    def _pick_variant(options: List[str], seed: str, fallback: str) -> str:
        if not options:
            return fallback
        digest = hashlib.sha256(seed.encode('utf-8')).hexdigest()
        index = int(digest[:8], 16) % len(options)
        return options[index]

    def _variant_seed(seed: str) -> str:
        return f'{request_seed}:{seed}'

    # Positive/confident rewrites for common negative phrasing.
    if tone_name in {'positive', 'very_positive', 'professional', 'friendly'}:
        phrase_replacements = [
            (r"\bnot good\b", "promising with clear room to improve"),
            (r"\bnot great\b", "improving steadily"),
            (r"\bnot working\b", "being improved to work reliably"),
            (r"\bdoes not work\b", "can work reliably after improvements"),
            (r"\bdo not work\b", "can work reliably after improvements"),
            (r"\bdid not work\b", "needed improvements to work reliably"),
            (r"\bnot work\b", "need improvements to work reliably"),
            (r"\bdoesn't work\b", "can work reliably with improvements"),
            (r"\bdidn't work\b", "is being improved for reliable results"),
            (r"\bkeeps crashing\b", "is being stabilized"),
            (r"\bkept crashing\b", "has been stabilized progressively"),
            (r"\bcrashing\b", "becoming more stable"),
            (r"\bvery slow\b", "improving in speed"),
            (r"\bwon't\b", "will"),
            (r"\bcan't\b", "can"),
            (r"\bcannot\b", "can"),
            (r"\btoo slow\b", "improving in speed"),
            (r"\btoo bad\b", "better with improvements"),
            (r"\bi hate how slow it is\b", "I want this to be much faster"),
            (r"\bi hate how\b", "I would like to improve how"),
            (r"\bui is confusing\b", "UI can be clearer with a better flow"),
            (r"\bis confusing\b", "can be made clearer"),
            (r"\bvery confusing\b", "ready to be simplified"),
        ]
        word_replacements = [
            (r"\bworst\b", "most challenging"),
            (r"\bterrible\b", "improving with focused support"),
            (r"\bawful\b", "improving with focused effort"),
            (r"\bhorrible\b", "improving with focused effort"),
            (r"\bbad\b", "improving"),
            (r"\bhate\b", "want to improve"),
            (r"\bangry\b", "motivated to resolve this"),
            (r"\bfrustrated\b", "highly motivated to improve this"),
            (r"\bproblem\b", "opportunity to improve"),
            (r"\bissue\b", "improvement area"),
            (r"\bbroken\b", "ready for repair"),
            (r"\bslow\b", "steadily improving in speed"),
            (r"\buseless\b", "able to improve"),
            (r"\bscam\b", "not trustworthy yet"),
            (r"\brefund\b", "resolution"),
            (r"\bcrash(?:ing)?\b", "stability issue being addressed"),
            (r"\berror(?:s)?\b", "issue"),
            (r"\bfailing\b", "still improving"),
            (r"\bfailed\b", "needed improvement"),
            (r"\bfail\b", "improve"),
            (r"\bdisappointed\b", "expecting better outcomes"),
            (r"\bannoying\b", "inconvenient"),
            (r"\bwaste\b", "learning investment"),
            (r"\bpoor\b", "below expectations but improvable"),
        ]
        positive_markers = re.compile(
            r"\b("
            r"confident|optimistic|improv|better|progress|strong|clear|positive|"
            r"reliable|effective|promising|supportive|constructive|steady|eager"
            r")\w*\b",
            re.IGNORECASE
        )
        suffix_options = [
            'and this can improve effectively',
            'and there is a clear path to improve it',
            'with the right steps, this can become reliable',
            'and this can be resolved constructively',
        ]
        first_prefix_options = ['I am confident that', 'It is encouraging that', 'I can see that']
        follow_prefix_options = ['We can ensure that', 'There is a clear path where', 'We can steadily improve so that']
        pronoun_followup_options = ['Additionally', 'Also', 'Going forward']
        if tone_name == 'very_positive':
            first_prefix_options = [
                'I am very optimistic that',
                'It is exciting that',
                'I am highly confident that'
            ]
            follow_prefix_options = [
                'We can confidently ensure that',
                'There is a strong path where',
                'We are well positioned so that'
            ]
        elif tone_name == 'friendly':
            first_prefix_options = [
                'I am happy to share that',
                'It is great to see that',
                'I am glad that'
            ]
            follow_prefix_options = [
                'We can happily ensure that',
                'We can make sure that',
                'We can support this so that'
            ]
            pronoun_followup_options = ['Also', 'On top of that', 'Moving ahead']
        elif tone_name == 'professional':
            first_prefix_options = [
                'I am confident that',
                'It is encouraging to note that',
                'The outlook shows that'
            ]
            follow_prefix_options = [
                'We can ensure that',
                'There is a practical path where',
                'The next step can ensure that'
            ]

        # Style-specific framing to ensure the same sentence rewrites differently
        # when users switch style (clear/formal/simple/friendly).
        if style_name == 'formal':
            first_prefix_options = [
                'It is evident that',
                'The assessment indicates that',
                'It is clear that'
            ]
            follow_prefix_options = [
                'Further, we can ensure that',
                'Additionally, there is a practical path where',
                'The next step is to ensure that'
            ]
            pronoun_followup_options = ['Additionally', 'Furthermore', 'In continuation']
        elif style_name == 'simple':
            first_prefix_options = [
                'I think',
                'It looks like',
                'This shows'
            ]
            follow_prefix_options = [
                'Next, we can make sure',
                'Also, we can help so',
                'Then we can improve so'
            ]
            pronoun_followup_options = ['Also', 'Next', 'Then']
        elif style_name == 'friendly':
            first_prefix_options = [
                'Good news,',
                'I am glad that',
                'Happy to share,'
            ]
            follow_prefix_options = [
                'Also, we can make sure',
                'On top of that, we can help so',
                'We can work together so'
            ]
            pronoun_followup_options = ['Also', 'On top of that', 'Together']

        pronoun_tails = {
            'positive': 'and I am confident we can improve this',
            'very_positive': 'and I am highly optimistic about the outcome',
            'professional': 'and I will improve this with a clear plan',
            'friendly': 'and I am happy to keep improving this together',
            'negative': 'and this still needs urgent attention',
            'very_negative': 'and this needs immediate corrective action'
        }
        tone_closers = {
            'positive': '',
            'professional': 'with a practical plan ahead',
            'friendly': 'and we can improve it together',
            'very_positive': 'and the outlook is very promising',
        }

        rewritten_sentences: List[str] = []
        for index, sentence in enumerate(_split_sentences(rewritten)):
            core = sentence[:-1] if sentence[-1] in '.!?' else sentence
            for pattern, replacement in phrase_replacements:
                core = re.sub(pattern, replacement, core, flags=re.IGNORECASE)
            for pattern, replacement in word_replacements:
                core = re.sub(pattern, replacement, core, flags=re.IGNORECASE)
            core = re.sub(r'\s+', ' ', core).strip(' ,;')
            if not core:
                core = 'this is moving in a better direction'
            if not positive_markers.search(core):
                suffix_seed = f'{tone_name}:{style_name}:{index}:{core.lower()}:suffix'
                suffix = _pick_variant(
                    suffix_options,
                    _variant_seed(suffix_seed),
                    suffix_options[0]
                )
                core = f"{core} {suffix}"

            tone_closer = tone_closers.get(tone_name, '')
            if tone_closer and tone_closer.lower() not in core.lower():
                core = f"{core} {tone_closer}"

            seed = f'{tone_name}:{style_name}:{index}:{core.lower()}'

            if re.match(r'^(i|we)\b', core, re.IGNORECASE):
                if index == 0:
                    tail = pronoun_tails.get(tone_name, pronoun_tails['positive'])
                    if not positive_markers.search(core):
                        core = f"{core} {tail}"
                    rewritten_sentences.append(core[0].upper() + core[1:] + '.')
                else:
                    connector = _pick_variant(
                        pronoun_followup_options,
                        _variant_seed(f'{seed}:connector'),
                        pronoun_followup_options[0]
                    )
                    rewritten_sentences.append(f"{connector}, {_lower_first_if_safe(core)}.")
                continue

            if index == 0:
                prefix = _pick_variant(
                    first_prefix_options,
                    _variant_seed(f'{seed}:first_prefix'),
                    first_prefix_options[0]
                )
            else:
                prefix = _pick_variant(
                    follow_prefix_options,
                    _variant_seed(f'{seed}:follow_prefix'),
                    follow_prefix_options[0]
                )
            rewritten_sentence = f"{prefix} {_lower_first_if_safe(core)}."
            rewritten_sentences.append(rewritten_sentence)

        if rewritten_sentences:
            rewritten = ' '.join(rewritten_sentences)

    elif tone_name in {'negative', 'very_negative'} and rewritten:
        core = rewritten[:-1] if rewritten[-1] in '.!?' else rewritten
        prefix = 'I am concerned that'
        if tone_name == 'very_negative':
            prefix = 'I am seriously concerned that'
        if style_name == 'formal':
            prefix = 'It is a concern that' if tone_name == 'negative' else 'It is a serious concern that'
        elif style_name == 'simple':
            prefix = 'I see a problem that' if tone_name == 'negative' else 'I see a big problem that'
        elif style_name == 'friendly':
            prefix = 'I feel concerned that' if tone_name == 'negative' else 'I feel very concerned that'
        rewritten = f"{prefix} {_lower_first_if_safe(core)}."

    if style_name == 'formal':
        formal_map = {
            r"\bdon't\b": 'do not',
            r"\bdoesn't\b": 'does not',
            r"\bdidn't\b": 'did not',
            r"\bit's\b": 'it is',
            r"\bcan't\b": 'cannot',
            r"\bwon't\b": 'will not',
        }
        for pattern, replacement in formal_map.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

    elif style_name == 'simple':
        simple_map = {
            r"\bapproximately\b": 'about',
            r"\butilize\b": 'use',
            r"\bassistance\b": 'help',
            r"\btherefore\b": 'so',
            r"\bhowever\b": 'but',
            r"\bassessment indicates\b": 'it shows',
            r"\bevident\b": 'clear',
            r"\bpractical path\b": 'clear way',
            r"\bconstructively\b": 'in a good way',
        }
        for pattern, replacement in simple_map.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

    notes = (
        'Google AI was unavailable, so local fallback rewriting was used. '
        'Set GOOGLE_AI_STUDIO_API_KEY for richer and more context-aware rewrites.'
    )
    return {
        'corrected_text': corrected,
        'rewritten_text': rewritten or corrected,
        'notes': notes,
        'model_used': 'local-rules-v2'
    }


def build_share_card_svg(
    original_text: str,
    sentiment: str,
    confidence: float,
    top_emotion: str
) -> str:
    """Generate an SVG card that can be shared/downloaded."""
    color_map = {
        'positive': '#16a34a',
        'negative': '#dc2626',
        'neutral': '#d97706',
    }
    color = color_map.get(sentiment, '#0284c7')
    safe_text = html.escape(original_text[:140])
    safe_emotion = html.escape(top_emotion or 'unknown')
    safe_sentiment = html.escape(sentiment.upper())
    confidence_pct = f"{confidence * 100:.1f}%"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1e293b"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <rect x="70" y="70" width="1060" height="490" rx="24" fill="#0b1220" stroke="#334155" stroke-width="2"/>
  <text x="110" y="150" fill="#93c5fd" font-family="Arial, sans-serif" font-size="34" font-weight="700">Custom Sentiment Server</text>
  <text x="110" y="215" fill="{color}" font-family="Arial, sans-serif" font-size="60" font-weight="700">{safe_sentiment}</text>
  <text x="110" y="270" fill="#cbd5e1" font-family="Arial, sans-serif" font-size="30">Confidence: {confidence_pct}</text>
  <text x="110" y="320" fill="#cbd5e1" font-family="Arial, sans-serif" font-size="30">Top Emotion: {safe_emotion}</text>
  <foreignObject x="110" y="360" width="980" height="170">
    <div xmlns="http://www.w3.org/1999/xhtml" style="color:#e2e8f0;font-family:Arial,sans-serif;font-size:28px;line-height:1.35;">
      "{safe_text}"
    </div>
  </foreignObject>
</svg>"""


def build_dashboard_stats() -> Dict[str, Any]:
    """Build live dashboard metrics from history and API usage data."""
    history = load_history(Config.DASHBOARD_HISTORY_WINDOW)
    predictor_instance = get_predictor()

    sentiment_counts: Counter = Counter()
    emotions_counter: Counter = Counter()
    hourly_counts: Counter = Counter()
    avg_confidence = 0.0

    for row in history:
        sentiment = str(row.get('sentiment', 'neutral')).lower()
        sentiment_counts[sentiment] += 1

        try:
            conf = float(row.get('confidence', 0))
            avg_confidence += conf
        except (TypeError, ValueError):
            pass

        timestamp = str(row.get('timestamp', ''))
        hour_label = timestamp[:13] if len(timestamp) >= 13 else timestamp[:10]
        hourly_counts[hour_label] += 1

        text_value = str(row.get('text', ''))
        emotion_data = predictor_instance.detect_emotions(text_value)
        top_emotion = emotion_data.get('top_emotion', 'neutral')
        if top_emotion:
            emotions_counter[top_emotion] += 1

    total_history = len(history)
    average_confidence_pct = round((avg_confidence / total_history) * 100, 2) if total_history else 0.0

    with api_store_lock:
        store = _load_json_file(Config.API_KEYS_FILE, _empty_api_store())
    usage_logs = store.get('usage_logs', [])
    endpoint_counts: Counter = Counter(log.get('endpoint', 'unknown') for log in usage_logs)
    active_key_count = sum(1 for key in store.get('keys', []) if key.get('active', True))

    return {
        'generated_at': _utc_now_iso(),
        'history_metrics': {
            'total_predictions': total_history,
            'average_confidence_pct': average_confidence_pct,
            'sentiment_counts': dict(sentiment_counts),
            'emotion_counts': dict(emotions_counter),
            'hourly_counts': dict(sorted(hourly_counts.items())[-24:])
        },
        'api_metrics': {
            'total_api_requests': len(usage_logs),
            'endpoint_counts': dict(endpoint_counts),
            'active_api_keys': active_key_count,
            'top_keys': list_api_keys()[:5]
        }
    }


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page - our sentiment analysis form."""
    prefill_text = request.args.get('text', '').strip()
    return render_template(
        'index.html',
        prefill_text=prefill_text,
        demo_examples=DEMO_EXAMPLES,
        default_auto_translate=Config.DEFAULT_AUTO_TRANSLATE
    )


@app.route('/examples')
def examples():
    """Public demo examples page."""
    return render_template('examples.html', demo_examples=DEMO_EXAMPLES)


@app.route('/dashboard')
def dashboard():
    """Live analytics dashboard page."""
    return render_template('dashboard.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment from web form."""
    try:
        text = request.form.get('text', '').strip()
        
        if not text:
            flash('Please enter some text to analyze.', 'error')
            return redirect(url_for('index'))
        
        # Use OUR predictor
        p = get_predictor()
        sentiment, confidence = p.predict(text)
        
        # Save to history
        save_to_history(text, sentiment, confidence)
        
        # Format confidence
        confidence_pct = f"{confidence * 100:.1f}%"
        
        return render_template('result.html', 
                             text=text,
                             sentiment=sentiment,
                             confidence=confidence_pct,
                             confidence_raw=confidence)
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """
    REST API - Our sentiment analysis endpoint.
    Returns JSON response.
    """
    try:
        access_error = track_api_usage('/api/sentiment')
        if access_error:
            return access_error

        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        text = data.get('text', '')
        if not isinstance(text, str):
            return jsonify({'success': False, 'error': 'Text must be a string'}), 400
        text = text.strip()

        compare_models = parse_bool(data.get('compare_models', False), default=False)
        auto_translate = parse_bool(
            data.get('auto_translate', Config.DEFAULT_AUTO_TRANSLATE),
            default=Config.DEFAULT_AUTO_TRANSLATE
        )
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        language = detect_language(text)
        analyzed_text = text
        translation = {
            'used': False,
            'source_language': language,
            'translated_text': None,
            'note': None
        }

        if auto_translate and language.get('code') != 'en':
            translated_text, translated, note = translate_to_english(text, language.get('code', 'en'))
            if translated:
                analyzed_text = translated_text
                translation['used'] = True
                translation['translated_text'] = translated_text
            else:
                translation['note'] = note
        
        # Use OUR predictor
        p = get_predictor()
        sentiment, confidence = p.predict(analyzed_text)
        
        # Get detailed prediction
        detailed = p.predict_detailed(analyzed_text)
        emotions = detailed.get('emotions', p.detect_emotions(analyzed_text))
        explainability = detailed.get('explainability', p.explain_prediction(analyzed_text))
        baseline = detailed.get('baseline_rule_model', p.rule_based_sentiment(analyzed_text))
        
        # Save to history
        save_to_history(text, sentiment, confidence)
        
        result: Dict[str, Any] = {
            'text': text,
            'analyzed_text': analyzed_text,
            'language': language,
            'translation': translation,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'confidence_percentage': f"{confidence * 100:.1f}%",
            'probabilities': detailed['probabilities'],
            'emotions': emotions,
            'top_emotion': emotions.get('top_emotion', 'neutral'),
            'explainability': explainability,
            'model': 'Our Own TF-IDF + Logistic Regression'
        }

        if compare_models:
            result['model_comparison'] = {
                'primary_model': {
                    'name': 'TF-IDF + Logistic Regression',
                    'sentiment': sentiment,
                    'confidence': round(confidence, 4)
                },
                'baseline_model': {
                    'name': 'Rule-based Lexicon',
                    'sentiment': baseline.get('sentiment'),
                    'confidence': baseline.get('confidence'),
                    'score': baseline.get('score')
                },
                'agreement': baseline.get('sentiment') == sentiment
            }

        # Return JSON response
        return jsonify({
            'success': True,
            'result': result
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sentiment/batch', methods=['POST'])
def api_batch_sentiment():
    """
    Batch API - Analyze multiple texts at once.
    """
    try:
        access_error = track_api_usage('/api/sentiment/batch')
        if access_error:
            return access_error

        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'success': False, 'error': 'texts array is required'}), 400
        if len(texts) > Config.BATCH_MAX_ITEMS:
            return jsonify({
                'success': False,
                'error': f'texts array exceeds limit ({Config.BATCH_MAX_ITEMS})'
            }), 400
        
        # Analyze each text with OUR model
        p = get_predictor()
        results = []
        
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                return jsonify({
                    'success': False,
                    'error': f'texts[{index}] must be a string'
                }), 400

            cleaned_text = text.strip()
            if cleaned_text:
                if len(cleaned_text) > Config.BATCH_MAX_TEXT_LENGTH:
                    return jsonify({
                        'success': False,
                        'error': (
                            f'texts[{index}] is too long '
                            f'(max {Config.BATCH_MAX_TEXT_LENGTH} characters)'
                        )
                    }), 400
                sentiment, confidence = p.predict(cleaned_text)
                emotions = p.detect_emotions(cleaned_text)
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': round(confidence, 4),
                    'top_emotion': emotions.get('top_emotion', 'neutral')
                })
        
        if not results:
            return jsonify({
                'success': False,
                'error': 'texts must contain at least one non-empty string'
            }), 400
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rewrite', methods=['POST'])
@limiter.limit(Config.RATE_LIMIT_REWRITE)
def api_rewrite():
    """
    Check and rewrite text using Google AI Studio.
    """
    try:
        access_error = track_api_usage('/api/rewrite')
        if access_error:
            return access_error

        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        text = data.get('text', '')
        style = data.get('style', 'clear')
        target_tone = data.get('target_tone', 'positive')

        if not isinstance(text, str):
            return jsonify({'success': False, 'error': 'Text must be a string'}), 400
        if not isinstance(style, str):
            return jsonify({'success': False, 'error': 'Style must be a string'}), 400
        if not isinstance(target_tone, str):
            return jsonify({'success': False, 'error': 'target_tone must be a string'}), 400

        text = text.strip()
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        if len(text) > 4000:
            return jsonify({'success': False, 'error': 'Text is too long (max 4000 characters)'}), 400

        rewrite_provider = 'Google AI Studio'
        rewritten = {}
        try:
            rewritten = rewrite_text_with_google_ai(text, style, target_tone)
        except GoogleAIError as ai_error:
            logger.warning(
                'Rewrite using local fallback rewriter. status=%s code=%s model=%s',
                ai_error.error_status or '',
                ai_error.status_code,
                ai_error.model or ''
            )
            rewritten = rewrite_text_locally(text, style, target_tone)
            fallback_note = (
                f' Google AI unavailable ({ai_error.error_status or ai_error.status_code or "unknown"}).'
            )
            rewritten['notes'] = (rewritten.get('notes', '').strip() + fallback_note).strip()
            rewrite_provider = 'Local Fallback Rewriter'

        rewritten_sentiment = 'neutral'
        rewritten_confidence = 0.0
        rewritten_emotions = {'scores': {}, 'top_emotion': 'neutral'}
        try:
            p = get_predictor()
            rewritten_sentiment, rewritten_confidence = p.predict(rewritten['rewritten_text'])
            rewritten_emotions = p.detect_emotions(rewritten['rewritten_text'])
        except Exception:
            rewritten['notes'] = (
                (rewritten.get('notes', '').strip() + ' Sentiment scoring unavailable right now.')
            ).strip()
        return jsonify({
            'success': True,
            'result': {
                'original_text': text,
                'style': style,
                'target_tone': target_tone,
                'corrected_text': rewritten['corrected_text'],
                'rewritten_text': rewritten['rewritten_text'],
                'notes': rewritten['notes'],
                'sentiment': rewritten_sentiment,
                'confidence': round(rewritten_confidence, 4),
                'confidence_percentage': f"{rewritten_confidence * 100:.1f}%",
                'emotions': rewritten_emotions,
                'provider': rewrite_provider,
                'model': rewritten.get('model_used', Config.GOOGLE_AI_MODEL)
            }
        }), 200

    except GoogleAIError as e:
        message = str(e)
        if e.error_status == 'NOT_CONFIGURED':
            return jsonify({'success': False, 'error': message}), 503
        if _is_retryable_google_error(e.status_code, e.error_status):
            return jsonify({'success': False, 'error': message}), 503
        return jsonify({'success': False, 'error': message}), 502
    except RuntimeError as e:
        message = str(e)
        if 'not configured' in message:
            return jsonify({'success': False, 'error': message}), 503
        return jsonify({'success': False, 'error': message}), 502
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/share-card', methods=['POST'])
def api_share_card():
    """Create a downloadable share card from analysis result."""
    try:
        access_error = track_api_usage('/api/share-card')
        if access_error:
            return access_error

        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        text = str(data.get('text', '')).strip()
        sentiment = str(data.get('sentiment', 'neutral')).strip().lower()
        raw_confidence = data.get('confidence', 0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'error': 'confidence must be a numeric value between 0 and 1'
            }), 400
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            return jsonify({
                'success': False,
                'error': 'confidence must be a numeric value between 0 and 1'
            }), 400
        top_emotion = str(data.get('top_emotion', 'neutral')).strip().lower()

        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        svg = build_share_card_svg(text, sentiment, confidence, top_emotion)
        return jsonify({
            'success': True,
            'result': {
                'filename': f"sentiment-card-{datetime.now().strftime('%Y%m%d-%H%M%S')}.svg",
                'svg': svg
            }
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Explicit model comparison endpoint."""
    try:
        access_error = track_api_usage('/api/compare')
        if access_error:
            return access_error

        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        text = data.get('text', '')
        if not isinstance(text, str):
            return jsonify({'success': False, 'error': 'Text must be a string'}), 400
        text = text.strip()
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        auto_translate = parse_bool(
            data.get('auto_translate', Config.DEFAULT_AUTO_TRANSLATE),
            default=Config.DEFAULT_AUTO_TRANSLATE
        )
        language = detect_language(text)
        analyzed_text = text
        translation_note = None
        translated_flag = False

        if auto_translate and language.get('code') != 'en':
            translated_text, translated_flag, translation_note = translate_to_english(
                text,
                language.get('code', 'en')
            )
            if translated_flag:
                analyzed_text = translated_text

        p = get_predictor()
        ml_sentiment, ml_confidence = p.predict(analyzed_text)
        baseline = p.rule_based_sentiment(analyzed_text)
        details = p.predict_detailed(analyzed_text)

        return jsonify({
            'success': True,
            'result': {
                'text': text,
                'analyzed_text': analyzed_text,
                'language': language,
                'translated': translated_flag,
                'translation_note': translation_note,
                'primary_model': {
                    'name': 'TF-IDF + Logistic Regression',
                    'sentiment': ml_sentiment,
                    'confidence': round(ml_confidence, 4),
                    'probabilities': details.get('probabilities', {})
                },
                'baseline_model': {
                    'name': 'Rule-based Lexicon',
                    'sentiment': baseline.get('sentiment'),
                    'confidence': baseline.get('confidence'),
                    'score': baseline.get('score')
                },
                'agreement': baseline.get('sentiment') == ml_sentiment,
                'emotions': details.get('emotions', p.detect_emotions(analyzed_text)),
                'explainability': details.get('explainability', p.explain_prediction(analyzed_text))
            }
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    """Return live dashboard analytics."""
    try:
        access_error = track_api_usage('/api/dashboard/stats')
        if access_error:
            return access_error
        return jsonify({'success': True, 'result': build_dashboard_stats()}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/examples')
def api_examples():
    """Expose public demo examples for frontend consumers."""
    return jsonify({'success': True, 'examples': DEMO_EXAMPLES}), 200


@app.route('/api/keys/create', methods=['POST'])
def api_create_key():
    """Create a new API key for clients."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400
        if not _is_admin_authorized(data):
            return jsonify({'success': False, 'error': 'Admin authorization failed'}), 403

        name = str(data.get('name', 'default-key')).strip()
        created = create_api_key(name)
        return jsonify({'success': True, 'result': created}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/keys')
def api_list_keys():
    """List API keys and usage metadata."""
    try:
        if not _is_admin_authorized():
            return jsonify({'success': False, 'error': 'Admin authorization failed'}), 403
        return jsonify({'success': True, 'result': list_api_keys()}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history')
def history():
    """View prediction history."""
    predictions = load_history(Config.HISTORY_LIMIT)
    return render_template('history.html', predictions=predictions)


@app.route('/model-info')
def model_info():
    """Get information about our model."""
    try:
        p = get_predictor()
        info = p.get_model_info()
        info['capabilities'] = [
            'sentiment-analysis',
            'emotion-detection',
            'explainability',
            'rule-based-model-comparison',
            'multilingual-auto-translate',
            'text-rewrite',
            'share-card'
        ]
        return jsonify(info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        p = get_predictor()
        return jsonify({
            'status': 'healthy',
            'server': 'Our Own Sentiment Server',
            'model_loaded': p.is_loaded,
            'model_type': 'TF-IDF + Logistic Regression',
            'api_keys_enforced': Config.ENFORCE_API_KEYS,
            'rewrite_available': bool(Config.GOOGLE_AI_STUDIO_API_KEY),
            'rewrite_model_primary': Config.GOOGLE_AI_MODEL,
            'rewrite_model_fallback': Config.GOOGLE_AI_FALLBACK_MODEL,
            'rewrite_retry': {
                'max_retries': Config.GOOGLE_AI_MAX_RETRIES,
                'backoff_seconds': Config.GOOGLE_AI_RETRY_BACKOFF_SECONDS
            },
            'google_max_concurrent_requests': Config.GOOGLE_AI_MAX_CONCURRENT_REQUESTS,
            'rate_limits': {
                'default': Config.RATE_LIMIT_DEFAULT,
                'rewrite': Config.RATE_LIMIT_REWRITE
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/docs')
def api_docs():
    """API documentation."""
    docs = {
        'name': 'Our Own Sentiment Analysis API',
        'description': '100% self-owned server with our own trained model',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'Web interface',
            'GET /examples': 'Public demo examples page',
            'GET /dashboard': 'Live analytics dashboard page',
            'POST /analyze': 'Web form analysis',
            'POST /api/sentiment': 'Single text analysis (JSON)',
            'POST /api/sentiment/batch': 'Batch analysis',
            'POST /api/compare': 'Model comparison mode',
            'POST /api/rewrite': 'Check grammar and rewrite text with Google AI Studio',
            'POST /api/share-card': 'Generate SVG share card from analysis',
            'GET /api/dashboard/stats': 'Live dashboard statistics JSON',
            'GET /api/examples': 'Public example payloads',
            'POST /api/keys/create': 'Create API key (admin)',
            'GET /api/keys': 'List API keys (admin)',
            'GET /history': 'View prediction history',
            'GET /model-info': 'Model information',
            'GET /health': 'Health check',
            'GET /api/docs': 'API documentation'
        },
        'model': {
            'type': 'TF-IDF + Logistic Regression',
            'training': 'Custom trained on our dataset',
            'owner': 'We own this model!'
        }
    }
    return jsonify(docs), 200


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(429)
def too_many_requests(e):
    description = str(getattr(e, 'description', 'Too many requests'))
    retry_after_seconds = None
    raw_retry_after = getattr(e, 'retry_after', None)
    if raw_retry_after is not None:
        try:
            retry_after_seconds = max(1, int(math.ceil(float(raw_retry_after))))
        except (TypeError, ValueError):
            retry_after_seconds = None

    payload = {
        'success': False,
        'error': 'Too Many Requests',
        'message': description
    }
    if retry_after_seconds is not None:
        payload['retry_after_seconds'] = retry_after_seconds

    response = jsonify(payload)
    response.status_code = 429
    if retry_after_seconds is not None:
        response.headers['Retry-After'] = str(retry_after_seconds)
    return response


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("STARTING OUR OWN SENTIMENT ANALYSIS SERVER")
    print("=" * 60)
    print(f"Server running at: http://{Config.HOST}:{Config.PORT}")
    print("This server uses OUR OWN trained model!")
    print("=" * 60)
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
