import glob
import logging
import os
import random
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from urllib.parse import urlencode

import httpx
import jwt
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from httpx import AsyncClient
from openai import OpenAI
from pydantic import BaseModel
from starlette.responses import JSONResponse
from user_agents import parse as parse_ua

from main import logger
from openrouter_key_manager import OpenRouterAPIClient

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_CLIENT_ID = "147419489204-mcv45kv1ndceffp1efnn2925cfet1ocb.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET = "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"
    # GOOGLE_REDIRECT_URI = "http://localhost:8000/auth/google/callback"
    # FRONTEND_URI = "http://localhost:3000"
    GOOGLE_REDIRECT_URI = "https://api.impulsepdr.online/auth/google/callback"
    FRONTEND_URI = "https://impulsepdr.online"
    DATA_DIR = "data"
    THEMES_DIR = os.path.join(DATA_DIR, "themes")
    IMAGES_DIR = os.path.join(DATA_DIR, "output_images")
    DB_FILE = os.path.join(DATA_DIR, "impulse_pdr.db")
    VISITS_FILE = "visit_logs.json"
    PREMIUM_VISITS_FILE = "premium_visit_logs.json"
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "filatova")
    MONOBANK_TOKEN = os.getenv("MONOBANK_TOKEN", "ud2yUaJH_kx4QbbuAmZObvlesfGTTwp1D_PW9lrjuqtg")
    MUTE_DURATION = timedelta(hours=1)


AMOUNT_TO_DAYS = {
    49_00: 7,  # 29 грн
    99_00: 30,  # 99 грн
    199_00: 90  # 199 грн
}

# Create directories if they don't exist
os.makedirs(Config.THEMES_DIR, exist_ok=True)
os.makedirs(Config.IMAGES_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Impulse PDR API", version="1.0.0")
api_client = OpenRouterAPIClient()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database setup
def init_db():
    """Initialize database with required tables"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            username TEXT,
            first_name TEXT,
            photo_url TEXT,
            created_at TEXT,
            last_login TEXT
        )
        ''')

        # Create answers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            question_id TEXT,
            is_correct INTEGER,
            timestamp TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')

        # Create favorites table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            question_id TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id),
            UNIQUE(user_id, question_id)
        )
        ''')

        # Create user credits table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_credits (
            user_id TEXT PRIMARY KEY,
            credits_used INTEGER DEFAULT 0,
            daily_limit INTEGER DEFAULT 20,
            last_reset_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS premium_subscriptions (
            user_id TEXT PRIMARY KEY,
            is_active BOOLEAN DEFAULT FALSE,
            start_date TEXT,
            end_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_time_limits (
            user_id TEXT PRIMARY KEY,
            remaining_time INTEGER DEFAULT 1800,
            last_update TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        ''')

        conn.commit()

# Добавим таблицу для хранения рекомендаций
def init_recommendations_table():
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_recommendations (
            user_id TEXT PRIMARY KEY,
            recommendations TEXT,
            last_updated TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        ''')
        conn.commit()

# Database connection helper
@contextmanager
def db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(Config.DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# Initialize database
init_db()
init_recommendations_table()

class UsageUpdate(BaseModel):
    user_id: str
    remaining_time: int
    is_premium: bool

# Models
class UserRequest(BaseModel):
    user_id: str

class UserAnswerPayload(BaseModel):
    question_id: str
    is_correct: bool
    user_id: str

class FavoritePayload(BaseModel):
    question_id: str
    user_id: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class PDRQuestionRequest(BaseModel):
    question: str
    context: str = ""

class CodeExchangeRequest(BaseModel):
    code: str

class CreditInfoResponse(BaseModel):
    user_id: str
    daily_limit: int
    credits_used: int
    credits_remaining: int
    last_reset_date: str


class PremiumActivationRequest(BaseModel):
    user_id: int
    invoice_id: str


class MonobankWebhookPayload(BaseModel):
    invoiceId: str
    status: str  # example: 'success'
    amount: int
    ccy: int
    reference: Optional[str] = None
    createdDate: Optional[str] = None
    modifiedDate: Optional[str] = None



# Модель для ответа о статусе премиума
class PremiumStatusResponse(BaseModel):
    user_id: str
    is_active: bool
    start_date: Optional[str]
    end_date: Optional[str]
    days_remaining: Optional[int]
    benefits: Dict[str, Any]


MUTED_KEYS: Dict[str, datetime] = {}
# Update these constants at the top of your file
KEY_MUTE_DURATION = timedelta(minutes=15)  # Reduced from 1 hour to 15 minutes
MAX_RETRIES = 3  # Maximum retries with different keys


def load_keys() -> List[str]:
    try:
        raw_keys = os.getenv("API_KEYS", "")
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        valid_keys = [k for k in keys if k.startswith("sk-or-v1-")]
        return valid_keys
    except Exception as e:
        logger.error(f"Failed to load keys: {e}")
        return []


def is_muted(key: str) -> bool:
    """Check if key is temporarily muted"""
    muted_until = MUTED_KEYS.get(key)
    return muted_until and muted_until > datetime.now()


def mute_key(key: str):
    """Temporarily mute a rate-limited key"""
    MUTED_KEYS[key] = datetime.now() + KEY_MUTE_DURATION
    logger.warning(f"Muted key {key[-4:]}... for {KEY_MUTE_DURATION}")


def get_active_key(retry_count: int = 0) -> str:
    """Get first available non-muted key with retry limit"""
    if retry_count >= MAX_RETRIES:
        raise HTTPException(429, "Maximum retries exceeded")

    keys = load_keys()
    for key in keys:
        if not is_muted(key):
            return key

    # If all keys are muted, try the least recently muted one
    if MUTED_KEYS:
        oldest_muted = min(MUTED_KEYS.items(), key=lambda x: x[1])
        if datetime.now() > oldest_muted[1] + timedelta(minutes=5):  # Small buffer
            return oldest_muted[0]

    raise HTTPException(429, "All keys temporarily exhausted")


# Helper functions
def get_openai_client(api_key: str) -> OpenAI:
    """Get OpenAI client configured for OpenRouter"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


async def verify_openrouter_key(key: str) -> bool:
    """Verify if an OpenRouter key is valid"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {key}"}
            )
            return response.status_code == 200
    except Exception:
        return False


def get_openrouter_headers(key: str) -> dict:
    """Return complete headers required by OpenRouter"""
    return {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://impulsepdr.online",  # Your actual domain
        "X-Title": "Impulse PDR",  # Your application name
        "Content-Type": "application/json",
        # OpenRouter now requires additional headers:
        "X-API-Version": "1.0",  # Added requirement
        "Accept": "application/json"  # Explicit accept header
    }


async def call_openrouter(messages: list, model: str = "deepseek/deepseek-r1-0528:free") -> dict:
    """Robust OpenRouter API call implementation"""
    key = get_active_key()
    if not key:
        raise HTTPException(status_code=503, detail="No active API keys available")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=get_openrouter_headers(key),
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,  # Recommended default
                    "max_tokens": 1000  # Prevent excessive responses
                }
            )

            # Handle specific error cases
            if response.status_code == 401:
                error_data = response.json()
                logger.error(f"Auth failed for key {key[-8:]}: {error_data}")
                mute_key(key)
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API credentials - key has been temporarily disabled"
                )

            response.raise_for_status()
            return response.json()

    except httpx.ReadTimeout:
        logger.error("OpenRouter API timeout")
        raise HTTPException(status_code=504, detail="API request timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenRouter API error: {e.response.text}")
        raise HTTPException(status_code=502, detail="AI service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def load_all_themes() -> List[Dict]:
    """Load all theme files from the themes directory"""
    themes = []
    for file in glob.glob(os.path.join(Config.THEMES_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            themes.append(json.load(f))
    return themes


def get_question_by_id(question_id: str) -> Optional[Dict]:
    """Find a question by its ID"""
    try:
        theme_id, q_idx = map(int, question_id.split("_"))
        theme_file = os.path.join(Config.THEMES_DIR, f"theme_{theme_id}.json")

        if not os.path.exists(theme_file):
            return None

        with open(theme_file, "r", encoding="utf-8") as f:
            theme = json.load(f)
            return theme.get("questions", [])[q_idx]
    except (ValueError, IndexError, FileNotFoundError):
        return None

@app.post("/webhooks/monobank", tags=["webhooks"])
async def monobank_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"🔥 Вебхук Monobank: {data}")

        status = data.get("status")
        amount = data.get("amount")
        ccy = data.get("ccy")
        reference = data.get("reference")
        invoice_id = data.get("invoiceId")

        # Проверка на успешную оплату
        if status != "success":
            logger.warning(f"⛔ Платіж неуспішний: статус = {status}")
            raise HTTPException(status_code=400, detail="Платіж ще не завершений")

        # Проверка валюты
        if ccy != 980:
            logger.warning(f"⛔ Валюта не гривня: ccy = {ccy}")
            raise HTTPException(status_code=400, detail="Непідтримувана валюта")

        # Проверка допустимой суммы
        duration_days = AMOUNT_TO_DAYS.get(amount)
        if not duration_days:
            logger.warning(f"⛔ Непідтримувана сума платежу: {amount}")
            raise HTTPException(status_code=400, detail="Невірна сума")

        # Извлекаем user_id из reference
        match = re.match(r"^premium_(\d+)_", reference or "")
        if not match:
            logger.warning(f"⛔ Невірний формат reference: {reference}")
            raise HTTPException(status_code=400, detail="Невірний формат посилання")

        user_id = match.group(1)
        logger.info(f"✅ Отримано user_id з reference: {user_id}")

        # Проверка, был ли уже обработан invoice_id (опционально — нужна таблица processed_invoices)
        # if is_invoice_already_processed(invoice_id):
        #     logger.info(f"⚠️ Інвойс {invoice_id} вже оброблений")
        #     return {"status": "already_processed"}

        # Выдача премиума
        success = grant_premium_to_user(user_id=user_id, duration_days=duration_days)
        if success:
            logger.info(f"✅ Преміум успішно активовано для user_id={user_id} на {duration_days} днів")
        else:
            logger.error(f"❌ Не вдалося активувати преміум для user_id={user_id}")
            raise HTTPException(status_code=500, detail="Не вдалося активувати преміум")

        # Здесь можно сохранить обработанный invoice_id, чтобы избежать повторов

        return {"status": "ok"}

    except Exception as e:
        logger.exception(f"❌ Помилка у вебхуку Monobank: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


def grant_premium_to_user(user_id: str, duration_days: int) -> bool:
    """Активировать или продлить премиум-подписку для пользователя"""
    now = datetime.utcnow()
    try:
        with sqlite3.connect(Config.DB_FILE) as conn:
            cursor = conn.cursor()

            # Получаем текущую подписку, если есть
            cursor.execute('''
                SELECT is_active, end_date FROM premium_subscriptions WHERE user_id = ?
            ''', (user_id,))
            row = cursor.fetchone()

            if row:
                is_active, end_date_str = row
                if is_active and end_date_str:
                    try:
                        end_date = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        end_date = now
                    new_end_date = max(end_date, now) + timedelta(days=duration_days)
                else:
                    new_end_date = now + timedelta(days=duration_days)

                cursor.execute('''
                    UPDATE premium_subscriptions
                    SET is_active = 1,
                        start_date = ?,
                        end_date = ?
                    WHERE user_id = ?
                ''', (now.isoformat(), new_end_date.isoformat(), user_id))
            else:
                new_end_date = now + timedelta(days=duration_days)
                cursor.execute('''
                    INSERT INTO premium_subscriptions (user_id, is_active, start_date, end_date)
                    VALUES (?, 1, ?, ?)
                ''', (user_id, now.isoformat(), new_end_date.isoformat()))

            conn.commit()
        return True
    except Exception as e:
        print(f"❌ Ошибка при продлении подписки: {e}")
        return False


async def _grant_premium(user_id: str, duration_days: int):
    today = datetime.utcnow()
    end_date = today + timedelta(days=duration_days)

    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM premium_subscriptions WHERE user_id = ?", (user_id,))
        has_premium = cursor.fetchone() is not None

        if has_premium:
            cursor.execute('''
                UPDATE premium_subscriptions 
                SET end_date = ?, is_active = TRUE
                WHERE user_id = ?
            ''', (end_date.isoformat(), user_id))
        else:
            cursor.execute('''
                INSERT INTO premium_subscriptions 
                (user_id, is_active, start_date, end_date)
                VALUES (?, ?, ?, ?)
            ''', (user_id, True, today.isoformat(), end_date.isoformat()))

        cursor.execute('''
            INSERT OR REPLACE INTO user_credits 
            (user_id, daily_limit, credits_used, last_reset_date)
            VALUES (?, 200, 0, ?)
        ''', (user_id, today.date().isoformat()))

        conn.commit()

    return {
        "status": "success",
        "message": "Premium activated by payment",
        "end_date": end_date.isoformat(),
        "duration_days": duration_days
    }


@app.get("/premium/status", tags=["premium"])
async def get_premium_status(
        user_id: str = Query(..., description="ID пользователя для проверки статуса")
):
    print("id:" + str(user_id))
    """Получить текущий статус премиум-подписки пользователя"""
    try:
        today = datetime.utcnow()

        with db_connection() as conn:
            cursor = conn.cursor()

            # Получаем информацию о подписке
            cursor.execute(
                "SELECT is_active, start_date, end_date FROM premium_subscriptions WHERE user_id = ?",
                (user_id,)
            )
            subscription = cursor.fetchone()

            if not subscription:
                return PremiumStatusResponse(
                    user_id=user_id,
                    is_active=False,
                    start_date=None,
                    end_date=None,
                    days_remaining=None,
                    benefits={
                        "ai_credits_per_day": 20,
                        "max_tests_per_day": 50,
                        "description": "Базовый доступ (20 кредита ИИ в день, максимум 50 тестов)"
                    }
                )

            is_active = bool(subscription["is_active"])
            start_date = subscription["start_date"]
            end_date = subscription["end_date"]

            # Проверяем, не истекла ли подписка
            if is_active and end_date:
                end_datetime = datetime.fromisoformat(end_date)
                if today > end_datetime:
                    is_active = False
                    # Обновляем статус в БД
                    cursor.execute(
                        "UPDATE premium_subscriptions SET is_active = FALSE WHERE user_id = ?",
                        (user_id,)
                    )
                    conn.commit()

            # Получаем информацию о кредитах
            cursor.execute(
                "SELECT daily_limit FROM user_credits WHERE user_id = ?",
                (user_id,)
            )
            credit_info = cursor.fetchone()
            daily_limit = credit_info["daily_limit"] if credit_info else 20

            # Рассчитываем оставшиеся дни
            days_remaining = None
            if is_active and end_date:
                end_datetime = datetime.fromisoformat(end_date)
                days_remaining = (end_datetime - today).days

            return PremiumStatusResponse(
                user_id=user_id,
                is_active=is_active,
                start_date=start_date,
                end_date=end_date,
                days_remaining=days_remaining,
                benefits={
                    "ai_credits_per_day": daily_limit,
                    "max_tests_per_day": "unlimited" if is_active else 50,
                    "description": "Премиум доступ" if is_active else "Базовый доступ"
                }
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/premium/cancel", tags=["premium"])
async def cancel_premium(
        user_id: str = Query(..., description="ID пользователя для отмены премиума")
):
    """Отменить премиум-подписку пользователя"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()

            # Устанавливаем is_active в FALSE
            cursor.execute(
                "UPDATE premium_subscriptions SET is_active = FALSE WHERE user_id = ?",
                (user_id,)
            )

            # Возвращаем стандартный лимит кредитов (3 в день)
            cursor.execute('''
                UPDATE user_credits 
                SET daily_limit = 20
                WHERE user_id = ?
            ''', (user_id,))

            conn.commit()

            return {
                "status": "success",
                "message": "Premium subscription cancelled",
                "user_id": user_id
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Обновим функцию проверки лимитов тестов
def check_test_limits(user_id: str) -> bool:
    """Проверить, может ли пользователь пройти еще тест"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Проверяем премиум-статус
        cursor.execute(
            "SELECT is_active, end_date FROM premium_subscriptions WHERE user_id = ?",
            (user_id,)
        )
        premium = cursor.fetchone()

        # У премиум-пользователей нет ограничений
        if premium and premium["is_active"]:
            end_date = datetime.fromisoformat(premium["end_date"])
            if datetime.utcnow() <= end_date:
                return True

        # Для обычных пользователей проверяем лимит (50 тестов в день)
        today = datetime.utcnow().date().isoformat()
        cursor.execute('''
            SELECT COUNT(*) as test_count 
            FROM answers 
            WHERE user_id = ? AND date(timestamp) = ?
        ''', (user_id, today))

        test_count = cursor.fetchone()["test_count"]
        return test_count < 50


@app.get("/user/detailed-stats", tags=["stats"])
async def get_detailed_stats(user_id: str):
    """
    Возвращает максимально подробную статистику пользователя.
    Запрашивает рекомендации у ИИ не чаще одного раза в сутки.
    """
    try:
        # Получаем базовую статистику
        progress = get_user_progress(user_id)

        # Получаем информацию о кредитах
        credit_info = get_user_credit_info(user_id)

        # Получаем статус премиума
        premium_status = await get_premium_status(user_id)

        # Получаем топ ошибок
        top_errors = get_top_errors(user_id)

        # Получаем избранные вопросы
        favorites = await get_favorites(user_id)

        # Формируем базовый ответ
        response = {
            "user_id": user_id,
            "progress": progress,
            "credit_info": credit_info,
            "premium_status": premium_status,
            "top_errors": top_errors[:10],  # Берем топ-10 ошибок
            "favorites_count": len(favorites),
            "last_updated": datetime.utcnow().isoformat()
        }

        # Проверяем, нужно ли обновлять рекомендации
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT recommendations, last_updated FROM user_recommendations WHERE user_id = ?",
                (user_id,)
            )
            recommendation_data = cursor.fetchone()

            need_update = True
            if recommendation_data:
                last_updated = datetime.fromisoformat(recommendation_data["last_updated"])
                need_update = (datetime.utcnow() - last_updated) > timedelta(days=1)
                response["recommendations"] = recommendation_data["recommendations"]

            # Если нужно обновить или рекомендаций еще нет
            if need_update or not recommendation_data:
                # Формируем запрос к ИИ
                ai_prompt = (
                    f"Пользователь с ID {user_id} имеет следующую статистику:\n"
                    f"- Всего отвечено вопросов: {progress['total']}\n"
                    f"- Правильных ответов: {progress['correct']} ({progress['accuracy']:.1%})\n"
                    f"- Серия правильных ответов подряд: {progress['streak']}\n"
                    f"- Тем с прогрессом: {progress['themes']}\n"
                    f"\nТоп ошибок пользователя:\n"
                )

                for i, error in enumerate(top_errors[:5], 1):
                    ai_prompt += f"{i}. {error['question']['question']} (ошибок: {error['error_count']})\n"

                ai_prompt += (
                    "\nДайте краткие рекомендации (3-5 пунктов) по улучшению результатов, "
                    "основываясь на этой статистике. Говорите на украинском языке."
                )

                # Запрашиваем рекомендации у ИИ
                try:
                    ai_response = await call_openrouter(
                        messages=[
                            {"role": "system", "content": "Ты помощник для подготовки к экзамену по ПДР."},
                            {"role": "user", "content": ai_prompt}
                        ],
                        model="google/gemma-3-27b-it:free"
                    )

                    recommendations = ai_response["choices"][0]["message"]["content"]

                    # Сохраняем рекомендации
                    now = datetime.utcnow().isoformat()
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO user_recommendations 
                            (user_id, recommendations, last_updated)
                            VALUES (?, ?, ?)
                        ''', (user_id, recommendations, now))
                        conn.commit()

                    response["recommendations"] = recommendations
                    response["recommendations_updated"] = True
                except Exception as e:
                    # Если не удалось получить рекомендации, используем старые или заглушку
                    if recommendation_data:
                        response["recommendations"] = recommendation_data["recommendations"]
                        response["recommendations_updated"] = False
                    else:
                        response["recommendations"] = "Не удалось загрузить рекомендации. Попробуйте позже."
                        response["recommendations_updated"] = False

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoints
@app.get("/", tags=["status"])
async def read_root():
    """Health check endpoint"""
    if os.path.exists(Config.VISITS_FILE):
        try:
            with open(Config.VISITS_FILE, "r", encoding="utf-8") as f:
                visits = json.load(f)
        except json.JSONDecodeError:
            visits = []
    else:
        visits = []

    return {
        "message": "API works",
        "visits_total": len(visits),
        "visits": visits  # Return only last 10 visits
    }

@app.get("/premium/visits", tags=["status"])
async def read_root():
    """Health check endpoint"""
    if os.path.exists(Config.PREMIUM_VISITS_FILE):
        try:
            with open(Config.PREMIUM_VISITS_FILE, "r", encoding="utf-8") as f:
                visits = json.load(f)
        except json.JSONDecodeError:
            visits = []
    else:
        visits = []

    return {
        "message": "API works",
        "visits_total": len(visits),
        "visits": visits  # Return only last 10 visits
    }


@app.post("/api/visit", tags=["analytics"])
async def increment_visit(request: Request):
    """Track a visit to the application"""
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    parsed_ua = parse_ua(user_agent)

    visit_data = {
        "ip": client_ip,
        "device": "mobile" if parsed_ua.is_mobile else "tablet" if parsed_ua.is_tablet else "desktop",
        "browser": parsed_ua.browser.family,
        "os": parsed_ua.os.family,
        "time": datetime.utcnow().isoformat(),
    }

    # Load existing visits
    visits = []
    if os.path.exists(Config.VISITS_FILE):
        try:
            with open(Config.VISITS_FILE, "r", encoding="utf-8") as f:
                visits = json.load(f)
        except json.JSONDecodeError:
            visits = []

    # Add new visit and save
    visits.append(visit_data)
    with open(Config.VISITS_FILE, "w", encoding="utf-8") as f:
        json.dump(visits, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "visit": visit_data}

@app.post("/api/premium/visit", tags=["analytics"])
async def increment_visit(request: Request):
    """Track a visit to the application"""
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    parsed_ua = parse_ua(user_agent)

    visit_data = {
        "ip": client_ip,
        "device": "mobile" if parsed_ua.is_mobile else "tablet" if parsed_ua.is_tablet else "desktop",
        "browser": parsed_ua.browser.family,
        "os": parsed_ua.os.family,
        "time": datetime.utcnow().isoformat(),
    }

    # Load existing visits
    visits = []
    if os.path.exists(Config.PREMIUM_VISITS_FILE):
        try:
            with open(Config.PREMIUM_VISITS_FILE, "r", encoding="utf-8") as f:
                visits = json.load(f)
        except json.JSONDecodeError:
            visits = []

    # Add new visit and save
    visits.append(visit_data)
    with open(Config.PREMIUM_VISITS_FILE, "w", encoding="utf-8") as f:
        json.dump(visits, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "visit": visit_data}


@app.post("/usage/update", tags=["time_limit"])
async def update_usage_time(
        user_id: str = Body(..., embed=True),
        time_used: int = Body(0),
        reset: bool = Body(False)
):
    print("time_used:" + str(time_used))
    """Update user's time usage"""
    try:
        today = datetime.utcnow()

        with db_connection() as conn:
            cursor = conn.cursor()

            # Check premium status first
            cursor.execute(
                "SELECT is_active, end_date FROM premium_subscriptions WHERE user_id = ?",
                (user_id,)
            )
            premium = cursor.fetchone()
            is_premium = False
            if premium and premium["is_active"]:
                end_date = datetime.fromisoformat(premium["end_date"])
                is_premium = today <= end_date

            # Premium users have no time limits
            if is_premium:
                return {
                    "status": "success",
                    "user_id": user_id,
                    "is_premium": True,
                    "message": "Premium user - no time limit applied"
                }

            # Get current time data
            cursor.execute(
                "SELECT remaining_time, last_update FROM user_time_limits WHERE user_id = ?",
                (user_id,)
            )
            time_data = cursor.fetchone()

            if time_data:
                remaining_time = time_data["remaining_time"]
                last_update = datetime.fromisoformat(time_data["last_update"])

                # Reset if requested or if it's a new day
                if reset or last_update.date() < today.date():
                    remaining_time = 10  # 1 minute in seconds
                else:
                    remaining_time = max(0, remaining_time - 1)

                # Update the record
                cursor.execute(
                    "UPDATE user_time_limits SET remaining_time = ?, last_update = ? WHERE user_id = ?",
                    (remaining_time, today.isoformat(), user_id)
                )
            else:
                # Create new record if doesn't exist
                remaining_time = 180 - time_used if not reset else 180
                cursor.execute(
                    "INSERT INTO user_time_limits (user_id, remaining_time, last_update) VALUES (?, ?, ?)",
                    (user_id, remaining_time, today.isoformat())
                )

            conn.commit()

            return {
                "status": "success",
                "user_id": user_id,
                "remaining_time": remaining_time,
                "time_expired": remaining_time <= 0,
                "last_update": today.isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/usage", tags=["time_limit"])
async def get_usage_data(
        user_id: str = Query(..., description="User ID")
):
    """Get current time usage data"""
    try:
        today = datetime.utcnow()

        with db_connection() as conn:
            cursor = conn.cursor()

            # Check premium status
            cursor.execute(
                "SELECT is_active, end_date FROM premium_subscriptions WHERE user_id = ?",
                (user_id,)
            )
            premium = cursor.fetchone()
            is_premium = False
            if premium and premium["is_active"]:
                end_date = datetime.fromisoformat(premium["end_date"])
                is_premium = today <= end_date

            # Get time data
            cursor.execute(
                "SELECT remaining_time, last_update FROM user_time_limits WHERE user_id = ?",
                (user_id,)
            )
            time_data = cursor.fetchone()

            if time_data:
                remaining_time = time_data["remaining_time"]
                last_update = datetime.fromisoformat(time_data["last_update"])

                # Reset if new day
                if not is_premium and last_update.date() < today.date():
                    remaining_time = 180
                    cursor.execute(
                        "UPDATE user_time_limits SET remaining_time = ?, last_update = ? WHERE user_id = ?",
                        (remaining_time, today.isoformat(), user_id)
                    )
                    conn.commit()
            else:
                # Create new record
                remaining_time = 180
                cursor.execute(
                    "INSERT INTO user_time_limits (user_id, remaining_time, last_update) VALUES (?, ?, ?)",
                    (user_id, remaining_time, today.isoformat())
                )
                conn.commit()

            return {
                "user_id": user_id,
                "remaining_time": remaining_time,
                "is_premium": is_premium,
                "time_expired": not is_premium and remaining_time <= 0,
                "last_update": today.isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/usage/update", tags=["time_limit"])
async def update_usage_data(
        user_id: str = Body(..., description="ID пользователя"),
        time_used: int = Body(0, description="Использованное время в секундах (по умолчанию 0)"),
        reset: bool = Body(False, description="Сбросить таймер (по умолчанию False)")
):
    """Обновить данные об использовании времени"""
    try:
        today = datetime.utcnow()

        with db_connection() as conn:
            cursor = conn.cursor()

            # Проверяем премиум-статус
            cursor.execute(
                "SELECT is_active, end_date FROM premium_subscriptions WHERE user_id = ?",
                (user_id,)
            )
            premium = cursor.fetchone()
            is_premium = False
            if premium and premium["is_active"]:
                end_date = datetime.fromisoformat(premium["end_date"])
                is_premium = today <= end_date

            # Если пользователь премиум, игнорируем обновление времени
            if is_premium:
                return {
                    "status": "success",
                    "message": "Premium user - time limit not applied",
                    "user_id": user_id,
                    "is_premium": True
                }

            # Получаем текущие данные
            cursor.execute(
                "SELECT remaining_time, last_update FROM user_time_limits WHERE user_id = ?",
                (user_id,)
            )
            time_data = cursor.fetchone()

            if time_data:
                remaining_time = time_data["remaining_time"]
                last_update = datetime.fromisoformat(time_data["last_update"])

                # Если сброс запрошен или новый день
                if reset or last_update.date() < today.date():
                    remaining_time = 1800
                else:
                    remaining_time = max(0, remaining_time - time_used)

                # Обновляем запись
                cursor.execute(
                    "UPDATE user_time_limits SET remaining_time = ?, last_update = ? WHERE user_id = ?",
                    (remaining_time, today.isoformat(), user_id)
                )
            else:
                # Создаем новую запись
                remaining_time = 1800 - time_used if not reset else 1800
                cursor.execute(
                    "INSERT INTO user_time_limits (user_id, remaining_time, last_update) VALUES (?, ?, ?)",
                    (user_id, remaining_time, today.isoformat())
                )

            conn.commit()

            return {
                "status": "success",
                "user_id": user_id,
                "remaining_time": remaining_time,
                "time_expired": remaining_time <= 0,
                "last_update": today.isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/usage/reset", tags=["time_limit"])
async def reset_usage_data(
        user_id: str = Body(..., description="ID пользователя для сброса времени")
):
    """Сбросить таймер использования времени"""
    try:
        today = datetime.utcnow()

        with db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE user_time_limits SET remaining_time = 1800, last_update = ? WHERE user_id = ?",
                (today.isoformat(), user_id)
            )

            if cursor.rowcount == 0:
                cursor.execute(
                    "INSERT INTO user_time_limits (user_id, remaining_time, last_update) VALUES (?, ?, ?)",
                    (user_id, 1800, today.isoformat())
                )

            conn.commit()

            return {
                "status": "success",
                "user_id": user_id,
                "remaining_time": 1800,
                "last_update": today.isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Authentication endpoints
@app.get("/auth/google/callback", tags=["auth"])
async def google_callback(request: Request):
    """Google OAuth callback handler"""
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        error_description = request.query_params.get("error_description", "")
        return RedirectResponse(
            f"{Config.FRONTEND_URI}?error={error}&description={error_description}"
        )

    if not code:
        return RedirectResponse(f"{Config.FRONTEND_URI}?error=missing_code")

    try:
        # Exchange code for tokens
        token_data = {
            "code": code,
            "client_id": Config.GOOGLE_CLIENT_ID,
            "client_secret": Config.GOOGLE_CLIENT_SECRET,
            "redirect_uri": Config.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        }

        response = requests.post(Config.GOOGLE_TOKEN_URL, data=token_data)
        response.raise_for_status()
        token_json = response.json()

        # Get user info
        user_info = {}
        if "access_token" in token_json:
            user_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            headers = {"Authorization": f"Bearer {token_json['access_token']}"}
            user_response = requests.get(user_url, headers=headers)
            if user_response.status_code == 200:
                user_info = user_response.json()

        # Save/update user in DB
        if user_info.get("sub"):
            with db_connection() as conn:
                cursor = conn.cursor()

                user_exists = cursor.execute(
                    "SELECT 1 FROM users WHERE user_id = ?",
                    (user_info["sub"],)
                ).fetchone() is not None

                if user_exists:
                    cursor.execute('''
                        UPDATE users SET
                            username = ?,
                            first_name = ?,
                            photo_url = ?,
                            last_login = ?
                        WHERE user_id = ?
                    ''', (
                        user_info.get("email"),
                        user_info.get("given_name") or user_info.get("name"),
                        user_info.get("picture"),
                        datetime.utcnow().isoformat(),
                        user_info["sub"]
                    ))
                else:
                    cursor.execute('''
                        INSERT INTO users (
                            user_id,
                            username,
                            first_name,
                            photo_url,
                            created_at,
                            last_login
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        user_info["sub"],
                        user_info.get("email"),
                        user_info.get("given_name") or user_info.get("name"),
                        user_info.get("picture"),
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat()
                    ))

                conn.commit()

        # Prepare response
        frontend_params = {
            "access_token": token_json["access_token"],
            "refresh_token": token_json.get("refresh_token", ""),
            "sub": user_info.get("sub"),
            "email": user_info.get("email"),
            "name": user_info.get("given_name") or user_info.get("name"),
        }

        return RedirectResponse(f"{Config.FRONTEND_URI}/auth/callback?{urlencode(frontend_params)}")

    except requests.exceptions.RequestException as e:
        error_msg = f"OAuth error: {str(e)}"
        if hasattr(e, 'response'):
            error_msg += f" | Response: {e.response.text}"
        return RedirectResponse(
            f"{Config.FRONTEND_URI}?error=server_error&message={error_msg}"
        )
    except Exception as e:
        return RedirectResponse(
            f"{Config.FRONTEND_URI}?error=unknown&message={str(e)}"
        )


@app.post("/auth/google/exchange", tags=["auth"])
async def exchange_code(request: CodeExchangeRequest):
    """Exchange Google auth code for tokens"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            Config.GOOGLE_TOKEN_URL,
            data={
                "code": request.code,
                "client_id": Config.GOOGLE_CLIENT_ID,
                "client_secret": Config.GOOGLE_CLIENT_SECRET,
                "redirect_uri": Config.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange code")

    tokens = response.json()
    id_token = tokens.get("id_token")

    if not id_token:
        raise HTTPException(status_code=400, detail="No ID token in response")

    # Decode id_token (JWT) without verifying for simplicity
    payload = jwt.decode(id_token, options={"verify_signature": False})

    return RedirectResponse(
        url=f"{Config.FRONTEND_URI}/?access_token={tokens.get('access_token')}"
            f"&refresh_token={tokens.get('refresh_token')}"
            f"&expires_in={tokens.get('expires_in')}"
            f"&sub={payload.get('sub')}"
            f"&name={payload.get('name')}"
            f"&email={payload.get('email')}"
            f"&picture={payload.get('picture')}"
    )


async def chat_completion(messages: list, model: str = "google/gemma-3-27b-it:free") -> dict:
    """Robust OpenRouter API call with proper error handling"""
    key = get_active_key()
    if not key:
        raise HTTPException(status_code=503, detail="No available API keys")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=get_openrouter_headers(key),
                json={
                    "model": model,
                    "messages": messages
                }
            )

            # Handle specific OpenRouter errors
            if response.status_code == 401:
                error_data = response.json()
                logger.error(f"OpenRouter auth failed: {error_data}")
                mute_key(key)
                raise HTTPException(
                    status_code=401,
                    detail=f"API authentication failed: {error_data.get('error', {}).get('message')}"
                )

            response.raise_for_status()
            return response.json()

    except httpx.ReadTimeout:
        logger.error("OpenRouter API timeout")
        raise HTTPException(status_code=504, detail="API request timeout")
    except Exception as e:
        logger.error(f"OpenRouter API error: {str(e)}")
        raise HTTPException(status_code=502, detail="AI service unavailable")


# Default system prompt for PDR questions
DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Ти — помічник з правил дорожнього руху України."
        "Твоя мета — відповідати просто, грамотно та зрозумілою українською мовою."
        "Пояснюй ПДР як для людини, яка готується до іспиту."
        "Не використовуй складну лексику, відповідай коротко та чітко."
        "Ти відповідаеш на питання лише один раз."
        "Ти не можешь спілкуватись з користувачем після своєї відповіді."
    )
}


@app.get("/admin/verify-keys", tags=["admin"])
async def verify_keys(admin_token: str = Query(...)):
    """Test all API keys (admin only)"""
    if admin_token != Config.ADMIN_TOKEN:
        raise HTTPException(403, "Forbidden")

    results = []
    for key in load_keys():
        async with AsyncClient(timeout=10.0) as client:
            try:
                # Test with a simple request
                test_response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "HTTP-Referer": "https://impulsepdr.online",
                        "X-Title": "Key Tester"
                    },
                    json={
                        "model": "google/gemma-3-27b-it:free",
                        "messages": [{"role": "user", "content": "Test"}]
                    }
                )
                valid = test_response.status_code == 200
                if valid:
                    try:
                        data = test_response.json()
                        valid = bool(data.get("choices"))
                    except:
                        valid = False
            except Exception as e:
                valid = False

        results.append({
            "key": f"{key[:4]}...{key[-4:]}",
            "valid": valid,
            "muted": is_muted(key),
            "muted_until": MUTED_KEYS.get(key, {}).isoformat() if is_muted(key) else None
        })

    return {"results": results}


def validate_key_format(key: str) -> bool:
    """Validate OpenRouter key format"""
    return key.startswith("sk-or-v1-") and len(key) > 30


def get_openrouter_headers(key: str) -> dict:
    """Proper OpenRouter headers with required fields"""
    return {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://impulsepdr.online",  # Required by OpenRouter
        "X-Title": "Impulse PDR",  # Required by OpenRouter
        "Content-Type": "application/json"
    }


@app.get("/pdr/ai-health", tags=["ai"])
async def check_ai_health():
    try:
        # Create properly formatted test message
        test_messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant. Respond with 'OK' if operational."
            ),
            ChatMessage(
                role="user",
                content="Test connection"
            )
        ]

        test_response = await chat_completion(ChatRequest(messages=test_messages))
        return {"status": "healthy", "response": test_response}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"AI service unavailable: {str(e)}"
        )


@app.post("/pdr/question")
async def ask_pdr_question(
        request: PDRQuestionRequest,
        user_id: str = Query(...)
):
    try:
        messages = [
            {"role": "user",
             "content": f"prompt: {DEFAULT_SYSTEM_PROMPT}, content:{request.context}\n\n{request.question}"}
        ]

        response = await api_client.call_api(messages)

        if "choices" not in response or not response["choices"]:
            logger.error(f"Invalid OpenRouter response: {response}")
            raise HTTPException(status_code=502, detail="Invalid response from language model")

            # Только после успешного ответа используем кредит
        if not use_credit(user_id):
            raise HTTPException(
                status_code=500,
                detail="Failed to update credit count"
            )

        return {
            "answer": response["choices"][0]["message"]["content"],
            "usage": response.get("usage", {})
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Processing error")


@app.get("/admin/keys/status")
async def get_key_status(admin_token: str = Query(...)):
    """Check status of all API keys (admin only)"""
    if admin_token != os.getenv("ADMIN_TOKEN") or "filatova":
        raise HTTPException(status_code=403, detail="Forbidden")

    keys_status = []
    for key in api_client.key_manager.load_keys():
        is_valid = await api_client.key_manager.verify_key(key)
        keys_status.append({
            "key": f"{key[:4]}...{key[-4:]}",
            "valid": is_valid,
            "muted": api_client.key_manager.is_key_muted(key),
            "muted_until": api_client.key_manager.muted_keys.get(key)
        })

    return {"keys": keys_status}


# User endpoints
@app.get("/users", tags=["users"])
async def get_all_users():
    """Get all registered users with stats"""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        SELECT 
            u.*,
            COUNT(DISTINCT a.question_id) as questions_answered,
            COUNT(DISTINCT f.question_id) as favorites_count
        FROM users u
        LEFT JOIN answers a ON u.user_id = a.user_id
        LEFT JOIN favorites f ON u.user_id = f.user_id
        GROUP BY u.user_id
        ORDER BY u.created_at DESC
        ''')

        users = []
        for row in cursor.fetchall():
            user = dict(row)
            users.append({
                "user_id": user["user_id"],
                "username": user["username"],
                "first_name": user["first_name"],
                "photo_url": user["photo_url"],
                "created_at": user["created_at"],
                "last_login": user["last_login"],
                "stats": {
                    "questions_answered": user["questions_answered"],
                    "favorites_count": user["favorites_count"]
                }
            })

        return JSONResponse(content={"users": users}, media_type="application/json; charset=utf-8")


@app.get("/user/credits", tags=["users"])
async def get_user_credits(user_id: str = Query(...)):
    """Get user credit information"""
    info = get_user_credit_info(user_id)
    if not info:
        raise HTTPException(status_code=404, detail="User not found")
    return info


@app.post("/user/credits/add", tags=["users"])
async def add_credits(
        user_id: str = Query(...),
        amount: int = Query(..., gt=0),
        admin_token: str = Query(...)
):
    """Add credits to user (admin only)"""
    if admin_token != Config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        new_credits = update_user_credits(user_id, amount)
        return {
            "user_id": user_id,
            "added": amount,
            "new_balance": new_credits
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/user/credits/reset", tags=["users"])
async def reset_credits(
        user_id: str = Query(...),
        admin_token: str = Query(None)
):
    """Reset user credits (admin only)"""
    if admin_token and admin_token != Config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        today = datetime.utcnow().date().isoformat()
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_credits 
                (user_id, credits_used, last_reset_date)
                VALUES (?, 0, ?)
            ''', (user_id, today))
            conn.commit()

        return {
            "status": "success",
            "user_id": user_id,
            "credits_used": 0,
            "reset_date": today
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/set_credits_limit", tags=["admin"])
async def set_credits_limit(
        user_id: str = Query(...),
        new_limit: int = Query(..., gt=0, le=100),
        admin_token: str = Query(...)
):
    """Set user credits limit (admin only)"""
    if admin_token != Config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE user_credits 
            SET 
                daily_limit = ?,
                credits_used = CASE 
                    WHEN credits_used > ? THEN 0 
                    ELSE credits_used 
                END
            WHERE user_id = ?
        ''', (new_limit, new_limit, user_id))

        if cursor.rowcount == 0:
            cursor.execute('''
                INSERT INTO user_credits 
                (user_id, daily_limit, credits_used, last_reset_date)
                VALUES (?, ?, 0, ?)
            ''', (user_id, new_limit, datetime.utcnow().date().isoformat()))

        conn.commit()
        return {
            "status": "success",
            "new_limit": new_limit,
            "user_id": user_id
        }


# Question endpoints
@app.get("/themes", tags=["questions"])
async def get_themes(user_id: Optional[str] = None):
    """Get all available themes"""
    themes = []
    for file in glob.glob(os.path.join(Config.THEMES_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            theme_data = json.load(f)
            themes.append({
                "index": int(theme_data.get("index", 0)),
                "name": str(theme_data.get("name", "")),
                "question_count": len(theme_data.get("questions", [])),
                "last_answered_index": theme_data.get("last_answered_index"),
                "accuracy": theme_data.get("accuracy"),
            })
    return sorted(themes, key=lambda x: x["index"])

@app.get("/themes/preview", tags=["questions"])
async def get_themes_preview(user_id: Optional[str] = None):
    """Get all available themes without questions field"""
    themes = []
    for file in glob.glob(os.path.join(Config.THEMES_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            theme_data = json.load(f)
            themes.append({
                "index": int(theme_data.get("index", 0)),
                "name": str(theme_data.get("name", "")),
                "question_count": len(theme_data.get("questions", [])),
                "last_answered_index": theme_data.get("last_answered_index"),
                "accuracy": theme_data.get("accuracy"),
            })
    return sorted(themes, key=lambda x: x["index"])

from fastapi import HTTPException, Query
from datetime import datetime
import os
import json

@app.get("/themes/{theme_id}", tags=["questions"])
async def get_theme_by_id(
    theme_id: int,
    user_id: str = None,
    offset: int = 0,
    limit: int = 300
):
    """Get questions for a specific theme"""

    # 🔒 Ограничение на доступ к темам > 15 без премиума
    if theme_id > 15:
        if not user_id:
            raise HTTPException(status_code=403, detail="Потрібна преміум-підписка для доступу до цієї теми")

        with db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT is_active, end_date FROM premium_subscriptions WHERE user_id = ?",
                (user_id,)
            )
            subscription = cursor.fetchone()

            is_premium = False

            if subscription:
                is_active = bool(subscription["is_active"])
                end_date = subscription["end_date"]
                if is_active and end_date:
                    end_datetime = datetime.fromisoformat(end_date)
                    if datetime.utcnow() <= end_datetime:
                        is_premium = True

            if not is_premium:
                raise HTTPException(status_code=403, detail="Потрібна преміум-підписка для доступу до цієї теми")

    # ✅ Открываем файл темы
    theme_file = os.path.join(Config.THEMES_DIR, f"theme_{theme_id}.json")
    if not os.path.exists(theme_file):
        raise HTTPException(status_code=404, detail="Theme not found")

    with open(theme_file, "r", encoding="utf-8") as f:
        theme_data = json.load(f)

    questions = []
    for idx, q in enumerate(theme_data.get("questions", [])[offset:offset + limit]):
        question = q.copy()
        real_index = offset + idx
        question_id = f"{theme_id}_{real_index}"
        question["id"] = question_id

        if user_id:
            with db_connection() as conn:
                cursor = conn.cursor()

                # Check if answered
                cursor.execute(
                    "SELECT is_correct FROM answers WHERE user_id = ? AND question_id = ?",
                    (user_id, question_id)
                )
                answer = cursor.fetchone()
                question["was_answered_correctly"] = bool(answer[0]) if answer else None

                # Check if favorite
                cursor.execute(
                    "SELECT 1 FROM favorites WHERE user_id = ? AND question_id = ?",
                    (user_id, question_id)
                )
                question["is_favorite"] = cursor.fetchone() is not None

        #question.pop("explanation", None)
        questions.append(question)

    return {
        "index": theme_data["index"],
        "name": theme_data["name"],
        "question_count": len(theme_data.get("questions", [])),
        "questions": questions
    }

@app.get("/themes/output_images/{filename}")
async def get_image(filename: str):
    file_path = os.path.join("data", "output_images", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не знайдено")

    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/ticket/random", tags=["questions"])
async def get_random_questions(user_id: Optional[str] = None):
    """Get random set of questions"""
    all_questions = []
    for theme in load_all_themes():
        for idx, q in enumerate(theme.get("questions", [])):
            question = q.copy()
            question_id = f"{theme['index']}_{idx}"
            question["id"] = question_id
            question.pop("explanation", None)
            all_questions.append(question)

    questions = random.sample(all_questions, min(20, len(all_questions)))

    if user_id:
        with db_connection() as conn:
            cursor = conn.cursor()
            for question in questions:
                # Get answer status
                cursor.execute(
                    "SELECT is_correct FROM answers WHERE user_id = ? AND question_id = ?",
                    (user_id, question["id"])
                )
                answer = cursor.fetchone()
                question["was_answered_correctly"] = answer[0] if answer else None

                # Check if favorite
                cursor.execute(
                    "SELECT 1 FROM favorites WHERE user_id = ? AND question_id = ?",
                    (user_id, question["id"])
                )
                question["is_favorite"] = cursor.fetchone() is not None

    return questions


@app.get("/search/questions", tags=["questions"])
async def search_questions(
        query: str = Query(..., min_length=1),
        user_id: Optional[str] = None
):
    """Search questions by text"""
    query = query.lower().strip()
    results = []

    for theme in load_all_themes():
        for idx, q in enumerate(theme.get("questions", [])):
            question_text = (q.get("question", "") or "").lower()
            answers = " ".join(q.get("answers", []) or []).lower()
            explanation = (q.get("explanation", "") or "").lower()
            theme_name = (theme.get("name", "") or "Unknown")

            if (query in question_text or
                    query in answers or
                    query in explanation):
                question = q.copy()
                question_id = f"{theme['index']}_{idx}"
                question["id"] = question_id
                question["theme_name"] = theme_name

                if user_id:
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        # Check answer
                        cursor.execute(
                            "SELECT is_correct FROM answers WHERE user_id = ? AND question_id = ?",
                            (user_id, question_id)
                        )
                        answer = cursor.fetchone()
                        question["was_answered_correctly"] = answer[0] if answer else None

                        # Check favorite
                        cursor.execute(
                            "SELECT 1 FROM favorites WHERE user_id = ? AND question_id = ?",
                            (user_id, question_id)
                        )
                        question["is_favorite"] = cursor.fetchone() is not None

                results.append(question)

    return {"results": results}


@app.get("/image", tags=["questions"])
async def get_image(
        path: str = Query(..., description="Relative path to image")
):
    """Get image by path"""
    safe_path = os.path.normpath(path).lstrip('/')
    image_path = os.path.join(Config.IMAGES_DIR, os.path.basename(safe_path))

    if not os.path.isfile(image_path):
        image_name = os.path.basename(safe_path).lower()
        for file in os.listdir(Config.IMAGES_DIR):
            if file.lower() == image_name:
                image_path = os.path.join(Config.IMAGES_DIR, file)
                break
        else:
            raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/jpeg")


# User progress endpoints
@app.get("/user/progress", tags=["progress"])
async def get_progress(user_id: str = Query(..., description="User ID")):
    return get_user_progress(user_id)

def get_user_progress(user_id: str):
    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM answers WHERE user_id = ?
        ''', (user_id,))
        total = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) FROM answers WHERE user_id = ? AND is_correct = 1
        ''', (user_id,))
        correct = cursor.fetchone()[0]

        wrong = total - correct
        accuracy = correct / total if total > 0 else 0.0

        # Серия подряд правильных
        cursor.execute('''
            SELECT is_correct FROM answers
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        streak = 0
        for (is_correct,) in cursor.fetchall():
            if is_correct == 1:
                streak += 1
            else:
                break

        # Кол-во тем с прогрессом
        cursor.execute('''
            SELECT COUNT(DISTINCT substr(question_id, 0, instr(question_id, ":")))
            FROM answers WHERE user_id = ?
        ''', (user_id,))
        themes = cursor.fetchone()[0]

        return {
            'total': total,
            'correct': correct,
            'wrong': wrong,
            'accuracy': accuracy,
            'streak': streak,
            'themes': themes,
        }

@app.get("/theme/progress", tags=["progress"])
async def get_theme_progress(theme_id: int, user_id: str = Query(...)):
    """Get user's progress for a specific theme"""
    with db_connection() as conn:
        cursor = conn.cursor()

        response = {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy": 0.0,
            "last_question": -1,  # -1 означает, что нет отвеченных вопросов
        }

        try:
            # Получаем последний отвеченный вопрос
            cursor.execute(
                """SELECT question_id 
                FROM answers 
                WHERE user_id = ? AND question_id LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT 1""",
                (user_id, f"{theme_id}_%")
            )
            last_question = cursor.fetchone()
            if last_question and last_question[0]:
                try:
                    response["last_question"] = int(last_question[0].split("_")[1])
                except:
                    response["last_question"] = -1

            # Остальная статистика...

        except Exception as e:
            print(f"Error: {str(e)}")

        return response

@app.get("/user/errors", tags=["progress"])
async def get_error_questions(user_id: str = Query(..., min_length=1)):
    """Get questions user answered incorrectly"""
    if not user_id:
        return {"questions": []}

    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT question_id FROM answers WHERE user_id = ? AND is_correct = 0",
                (user_id,)
            )
            error_ids = [row[0] for row in cursor.fetchall()]

        questions = []
        for theme in load_all_themes():
            for idx, q in enumerate(theme.get("questions", [])):
                qid = f"{theme['index']}_{idx}"
                if qid in error_ids:
                    question = q.copy()
                    question["id"] = qid
                    question["correct_index"] = q.get("correct_index")  # 👈 добавлено
                    question.pop("explanation", None)
                    questions.append(question)

        return {"questions": questions}
    except Exception as e:
        return {"questions": []}

@app.get("/user/top-errors", tags=["progress"])
def get_top_errors(user_id: str):
    """
    Return up to 100 most frequent wrong questions with full data
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT question_id, COUNT(*) as error_count
            FROM answers
            WHERE user_id = ? AND is_correct = 0
            GROUP BY question_id
            ORDER BY error_count DESC
            LIMIT 100
        ''', (user_id,))
        error_counts = cursor.fetchall()

    error_map = {row[0]: row[1] for row in error_counts}
    result = []

    for filename in os.listdir(Config.THEMES_DIR):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(Config.THEMES_DIR, filename), encoding='utf-8') as f:
            theme = json.load(f)
            theme_index = str(theme.get("index"))
            for idx, q in enumerate(theme.get("questions", [])):
                qid = f"{theme_index}:{idx}"
                if qid in error_map:
                    result.append({
                        "question": q,
                        "question_id": qid,
                        "theme_index": int(theme_index),
                        "error_count": error_map[qid]
                    })

    return result

@app.get("/user/favorites", tags=["progress"])
async def get_favorites(user_id: str):
    """Get user's favorite questions"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Get all favorite question IDs for user
        cursor.execute(
            "SELECT question_id FROM favorites WHERE user_id = ?",
            (user_id,)
        )
        favorite_ids = [row[0] for row in cursor.fetchall()]

        favorites = []
        for theme in load_all_themes():
            for idx, q in enumerate(theme.get("questions", [])):
                question_id = f"{theme['index']}_{idx}"
                if question_id in favorite_ids:
                    question = q.copy()
                    question["id"] = question_id
                    question["is_favorite"] = True

                    # Get answer status
                    cursor.execute(
                        "SELECT is_correct FROM answers WHERE user_id = ? AND question_id = ?",
                        (user_id, question_id)
                    )
                    answer = cursor.fetchone()
                    question["was_answered_correctly"] = bool(answer[0]) if answer else None

                    favorites.append(question)

        return favorites


# User action endpoints
@app.post("/user/answer", tags=["actions"])
async def submit_answer(payload: UserAnswerPayload):
    """Submit user's answer to a question"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()

            # Add user if not exists
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
                (payload.user_id, datetime.utcnow().isoformat())
            )

            # Save answer
            cursor.execute(
                """INSERT INTO answers (user_id, question_id, is_correct, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (payload.user_id, payload.question_id, int(payload.is_correct),
                 datetime.utcnow().isoformat())
            )

            conn.commit()
            return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/favorites/toggle", tags=["actions"])
async def toggle_favorite(payload: FavoritePayload):
    """Toggle favorite status for a question"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()

            # Ensure user exists
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, first_name, created_at) VALUES (?, ?, ?)",
                (payload.user_id, "User", datetime.utcnow().isoformat())
            )

            # Check current status
            cursor.execute(
                "SELECT 1 FROM favorites WHERE user_id = ? AND question_id = ?",
                (payload.user_id, payload.question_id)
            )
            is_favorite = cursor.fetchone() is not None

            if is_favorite:
                cursor.execute(
                    "DELETE FROM favorites WHERE user_id = ? AND question_id = ?",
                    (payload.user_id, payload.question_id)
                )
                action = "removed"
            else:
                cursor.execute(
                    "INSERT OR IGNORE INTO favorites (user_id, question_id) VALUES (?, ?)",
                    (payload.user_id, payload.question_id)
                )
                action = "added"

            conn.commit()
            return {"status": "ok", "action": action}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/user/reset", tags=["actions"])
async def reset_progress(user_id: str = Query(...)):
    """Reset user's progress"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()

            # Delete all user answers
            cursor.execute(
                "DELETE FROM answers WHERE user_id = ?",
                (user_id,)
            )

            # Delete all user favorites
            cursor.execute(
                "DELETE FROM favorites WHERE user_id = ?",
                (user_id,)
            )

            conn.commit()
            return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Credit management functions
def init_user_credits(user_id: str) -> bool:
    """Initialize credits for a user if needed"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute('SELECT 1 FROM users WHERE user_id = ?', (user_id,))
        if not cursor.fetchone():
            return False

        # Create credit record if not exists
        today = datetime.utcnow().date().isoformat()
        cursor.execute('''
            INSERT OR IGNORE INTO user_credits 
            (user_id, credits_used, daily_limit, last_reset_date)
            VALUES (?, 0, 20, ?)
        ''', (user_id, today))

        conn.commit()
        return True


# Обновим функцию get_user_credit_info для учета премиума
def get_user_credit_info(user_id: str) -> Optional[dict]:
    """Get user's credit information with auto-reset"""
    if not init_user_credits(user_id):
        return None

    with db_connection() as conn:
        cursor = conn.cursor()

        # Проверяем премиум-статус
        cursor.execute(
            "SELECT is_active, end_date FROM premium_subscriptions WHERE user_id = ?",
            (user_id,)
        )
        premium = cursor.fetchone()

        is_premium = False
        if premium and premium["is_active"]:
            end_date = datetime.fromisoformat(premium["end_date"])
            is_premium = datetime.utcnow() <= end_date
            if not is_premium:
                # Если подписка истекла, обновляем статус
                cursor.execute(
                    "UPDATE premium_subscriptions SET is_active = FALSE WHERE user_id = ?",
                    (user_id,)
                )

        # Get current date in UTC
        today = datetime.utcnow().date().isoformat()

        # Для премиум-пользователей устанавливаем лимит 10, для остальных - 3
        default_limit = 200 if is_premium else 20

        # Reset if new day
        cursor.execute('''
            UPDATE user_credits 
            SET 
                credits_used = CASE 
                    WHEN last_reset_date != ? THEN 0 
                    ELSE credits_used 
                END,
                daily_limit = ?,
                last_reset_date = ?
            WHERE user_id = ?
            RETURNING daily_limit, credits_used, last_reset_date
        ''', (today, default_limit, today, user_id))

        result = cursor.fetchone()
        if not result:
            return None

        daily_limit, credits_used, last_reset = result
        conn.commit()

        return {
            'user_id': user_id,
            'daily_limit': daily_limit,
            'credits_used': credits_used,
            'credits_remaining': max(0, daily_limit - credits_used),
            'last_reset_date': today if last_reset != today else last_reset,
            'is_premium': is_premium,
            'max_tests_per_day': "unlimited" if is_premium else 50
        }


def update_user_credits(user_id: str, change: int) -> Optional[dict]:
    """Update user's credit count and return new state

    Args:
        user_id: ID of the user to update credits for
        change: Positive number to add credits (decrease credits_used),
                Negative number to use credits (increase credits_used)

    Returns:
        Dictionary with updated credit info if successful, None otherwise
        {
            'user_id': str,
            'daily_limit': int,
            'credits_used': int,
            'credits_remaining': int,
            'last_reset_date': str
        }
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        today = datetime.utcnow().date().isoformat()

        try:
            # Reset credits if new day
            cursor.execute('''
                UPDATE user_credits 
                SET credits_used = 0, 
                    last_reset_date = ?
                WHERE user_id = ? AND last_reset_date != ?
            ''', (today, user_id, today))

            # For adding credits (negative change decreases credits_used)
            if change > 0:
                cursor.execute('''
                    UPDATE user_credits 
                    SET credits_used = MAX(0, credits_used - ?)
                    WHERE user_id = ?
                    RETURNING daily_limit, credits_used, last_reset_date
                ''', (change, user_id))
            # For using credits (check limit)
            else:
                cursor.execute('''
                    UPDATE user_credits 
                    SET credits_used = credits_used + ?
                    WHERE user_id = ? 
                      AND credits_used + ? <= daily_limit
                    RETURNING daily_limit, credits_used, last_reset_date
                ''', (abs(change), user_id, abs(change)))

            result = cursor.fetchone()
            if not result:
                conn.rollback()
                return None

            daily_limit, credits_used, last_reset = result
            conn.commit()

            return {
                'user_id': user_id,
                'daily_limit': daily_limit,
                'credits_used': credits_used,
                'credits_remaining': max(0, daily_limit - credits_used),
                'last_reset_date': last_reset
            }

        except sqlite3.Error as e:
            conn.rollback()
            logging.error(f"Error updating credits for user {user_id}: {str(e)}")
            return None


def use_credit(user_id: str) -> bool:
    """Safely use one credit from user's account"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # First check available credits
        info = get_user_credit_info(user_id)
        if not info or info['credits_remaining'] <= 0:
            return False

        # Use credit
        cursor.execute('''
                    UPDATE user_credits 
                    SET credits_used = credits_used + 1 
                    WHERE user_id = ? AND credits_used < daily_limit
                ''', (user_id,))

        conn.commit()
        return cursor.rowcount > 0


def can_use_credit(user_id: str) -> bool:
    """Check if user can use a credit"""
    info = get_user_credit_info(user_id)
    if not info:
        return False
    return info['credits_remaining'] > 0


# Debug endpoints
@app.get("/debug/user", tags=["debug"])
async def debug_user(user_id: str = Query(...)):
    """Debug endpoint to get user data"""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()
        return {"user": dict(user) if user else None}


@app.get("/debug/answers", tags=["debug"])
async def debug_all_answers():
    """Debug endpoint to get all answers"""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM answers")
        columns = [column[0] for column in cursor.description]
        answers = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"answers": answers}


# Feedback endpoint
@app.post("/feedback", tags=["feedback"])
async def send_feedback(
        feedback: str = Body(...),
        rating: int = Body(..., ge=1, le=5),
        user_id: str = Body(...)
):
    """Submit user feedback"""
    try:
        # In a real application, you would store this in a database
        # For now, we'll just log it
        logging.info(f"Feedback from user {user_id}: Rating={rating}, Feedback={feedback}")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Main entry point
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdr_service.log'),
            logging.StreamHandler()
        ]
    )

    # Run the application
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use our logging config
    )
