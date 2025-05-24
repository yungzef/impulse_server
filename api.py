import asyncio
from pathlib import Path
import httpx
import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
import glob
import random
from datetime import datetime, timedelta
import sqlite3
import uvicorn
from urllib.parse import urlencode
import requests
from starlette.responses import JSONResponse
from user_agents import parse as parse_ua
import logging
from contextlib import contextmanager

from main import logger

# Load environment variables
load_dotenv()


# Configuration
class Config:
    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://api.impulsepdr.online/auth/google/callback")
    FRONTEND_URI = os.getenv("FRONTEND_URI", "https://impulsepdr.online")
    DATA_DIR = "data"
    THEMES_DIR = os.path.join(DATA_DIR, "themes")
    IMAGES_DIR = os.path.join(DATA_DIR, "output_images")
    DB_FILE = os.path.join(DATA_DIR, "impulse_pdr.db")
    VISITS_FILE = "visit_logs.json"
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "filatova")
    MUTE_DURATION = timedelta(hours=1)


# Create directories if they don't exist
os.makedirs(Config.THEMES_DIR, exist_ok=True)
os.makedirs(Config.IMAGES_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Impulse PDR API", version="1.0.0")

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
            daily_limit INTEGER DEFAULT 3,
            last_reset_date TEXT,
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


# Models
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
    context: str


class CodeExchangeRequest(BaseModel):
    code: str


class CreditInfoResponse(BaseModel):
    user_id: str
    daily_limit: int
    credits_used: int
    credits_remaining: int
    last_reset_date: str


# Rate limiting
MUTED_KEYS: Dict[str, datetime] = {}


def load_keys(filename: str = "keys.json") -> List[str]:
    """Load API keys from file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)["api_keys"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load API keys: {str(e)}")


def is_muted(key: str) -> bool:
    """Check if API key is muted"""
    mute_time = MUTED_KEYS.get(key)
    return mute_time and datetime.utcnow() < mute_time


def mute_key(key: str):
    """Mute an API key"""
    MUTED_KEYS[key] = datetime.utcnow() + Config.MUTE_DURATION


# Helper functions
def get_openai_client(api_key: str) -> OpenAI:
    """Get OpenAI client configured for OpenRouter"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


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
        "visits": visits[-10:]  # Return only last 10 visits
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

        return RedirectResponse(f"{Config.FRONTEND_URI}?{urlencode(frontend_params)}")

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


# AI endpoints
@app.post("/chat/completions", tags=["ai"])
async def chat_completion(request: ChatRequest):
    """
    Endpoint for handling chat completions using OpenRouter API.
    Requires messages in the format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    """
    keys = load_keys()
    for key in keys:
        if is_muted(key):
            continue

        client = get_openai_client(key)
        try:
            completion = client.chat.completions.create(
                model="google/gemma-3-27b-it:free",
                extra_headers={
                    "HTTP-Referer": Config.FRONTEND_URI,
                    "X-Title": "Impulse PDR",
                },
                messages=[msg.dict() for msg in request.messages]
            )
            return {"response": completion.choices[0].message.content}
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                mute_key(key)
                continue
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")

    raise HTTPException(
        status_code=429,
        detail="All API keys are exhausted or rate limited. Please try again later."
    )


# Default system prompt for PDR questions
DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Ти — помічник з правил дорожнього руху України. "
        "Твоя мета — відповідати просто, грамотно та зрозумілою українською мовою. "
        "Пояснюй ПДР як для людини, яка готується до іспиту. "
        "Не використовуй складну лексику, відповідай коротко та чітко. "
        "Якщо щось неясно — став уточнюючі запитання."
    )
}


@app.post("/pdr/question", tags=["ai"])
async def ask_pdr_question(
        request: PDRQuestionRequest,
        user_id: str = Query(...)
):
    try:
        # Add validation for empty question
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Initialize credits for user with more error handling
        if not init_user_credits(user_id):
            raise HTTPException(status_code=404, detail="User not found or could not be initialized")

        # Check available credits with null check
        credit_info = get_user_credit_info(user_id)
        if credit_info is None:
            raise HTTPException(status_code=500, detail="Could not retrieve credit information")

        if credit_info['credits_remaining'] <= 0:
            raise HTTPException(
                status_code=402,
                detail=f"No available credits for today. Your daily limit: {credit_info.get('daily_limit', 0)}"
            )

        # Prepare AI request with better error handling
        try:
            messages = [
                ChatMessage(**DEFAULT_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"контекст вопроса:{request.context or 'нет контекста'}. вопрос пользователя:{request.question}"
                )
            ]

            # Add timeout for the AI service call
            response = await asyncio.wait_for(
                chat_completion(ChatRequest(messages=messages)),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="AI service timeout")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")

        # Validate AI response
        if not response or not isinstance(response, dict) or "response" not in response:
            raise HTTPException(status_code=502, detail="Invalid response from AI service")

        # Use credit with transaction safety
        try:
            if not use_credit(user_id):
                raise HTTPException(status_code=500, detail="Failed to update credit count")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Credit update failed: {str(e)}")

        return {
            "response": response["response"],
            "credits_remaining": get_user_credit_info(user_id).get('credits_remaining', 0)
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        logger.error(f"Unexpected error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question"
        )


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


@app.get("/themes/{theme_id}", tags=["questions"])
async def get_theme_by_id(
        theme_id: int,
        user_id: str = None,
        offset: int = 0,
        limit: int = 300
):
    """Get questions for a specific theme"""
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

        question.pop("explanation", None)
        questions.append(question)

    return {
        "index": theme_data["index"],
        "name": theme_data["name"],
        "question_count": len(theme_data.get("questions", [])),
        "questions": questions
    }


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
async def get_progress(user_id: str = Query(...)):
    """Get user's overall progress"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Get total answers count
        cursor.execute(
            "SELECT COUNT(*) FROM answers WHERE user_id = ?",
            (user_id,)
        )
        total = cursor.fetchone()[0]

        # Get correct answers count
        cursor.execute(
            "SELECT COUNT(*) FROM answers WHERE user_id = ? AND is_correct = 1",
            (user_id,)
        )
        correct = cursor.fetchone()[0]

        wrong = total - correct
        accuracy = correct / total if total > 0 else 0.0

        return {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": round(accuracy, 2)
        }


@app.get("/theme/progress", tags=["progress"])
async def get_theme_progress(theme_id: int, user_id: str = Query(...)):
    """Get user's progress for a specific theme"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Get theme answers count
        cursor.execute(
            "SELECT COUNT(*) FROM answers WHERE user_id = ? AND question_id LIKE ?",
            (user_id, f"{theme_id}_%")
        )
        total = cursor.fetchone()[0]

        # Get correct answers count
        cursor.execute(
            "SELECT COUNT(*) FROM answers WHERE user_id = ? AND question_id LIKE ? AND is_correct = 1",
            (user_id, f"{theme_id}_%")
        )
        correct = cursor.fetchone()[0]

        wrong = total - correct
        accuracy = correct / total if total > 0 else 0.0

        # Get last answered question index
        last_answered = 0
        cursor.execute(
            "SELECT question_id FROM answers WHERE user_id = ? AND question_id LIKE ? ORDER BY timestamp DESC LIMIT 1",
            (user_id, f"{theme_id}_%")
        )
        last_question = cursor.fetchone()
        if last_question:
            try:
                last_answered = int(last_question[0].split("_")[1])
            except (IndexError, ValueError):
                pass

        return {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": round(accuracy, 2),
            "last_question": last_answered
        }


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
                    question.pop("explanation", None)
                    questions.append(question)

        return {"questions": questions}
    except Exception as e:
        return {"questions": []}


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


@app.post("/user/favorites/add", tags=["actions"])
async def add_favorite(payload: FavoritePayload):
    """Add question to user's favorites"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()

            # Ensure user exists
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, first_name, created_at) VALUES (?, ?, ?)",
                (payload.user_id, "User", datetime.utcnow().isoformat())
            )

            # Add favorite
            cursor.execute(
                "INSERT OR IGNORE INTO favorites (user_id, question_id) VALUES (?, ?)",
                (payload.user_id, payload.question_id)
            )

            conn.commit()
            return {"status": "ok", "message": "Favorite added"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/user/favorites/remove", tags=["actions"])
async def remove_favorite(payload: FavoritePayload):
    """Remove question from user's favorites"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM favorites WHERE user_id = ? AND question_id = ?",
                (payload.user_id, payload.question_id)
            )
            conn.commit()
            return {"status": "ok", "message": "Favorite removed"}
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
            VALUES (?, 0, 3, ?)
        ''', (user_id, today))

        conn.commit()
        return True


def get_user_credit_info(user_id: str) -> Optional[dict]:
    """Get user's credit information with auto-reset"""
    if not init_user_credits(user_id):
        return None

    with db_connection() as conn:
        cursor = conn.cursor()

        # Get current date in UTC
        today = datetime.utcnow().date().isoformat()
        # Check if credit reset is needed
        cursor.execute('''
                    SELECT last_reset_date, daily_limit, credits_used 
                    FROM user_credits 
                    WHERE user_id = ?
                ''', (user_id,))

        row = cursor.fetchone()
        if not row:
            return None

        last_reset, daily_limit, credits_used = row

        # Reset if new day
        if last_reset != today:
            cursor.execute('''
                        UPDATE user_credits 
                        SET credits_used = 0, 
                            last_reset_date = ?
                        WHERE user_id = ?
                    ''', (today, user_id))
            conn.commit()
            credits_used = 0

        return {
            'user_id': user_id,
            'daily_limit': daily_limit,
            'credits_used': credits_used,
            'credits_remaining': max(0, daily_limit - credits_used),
            'last_reset_date': today if last_reset != today else last_reset,
            'next_reset_time': '00:00:00 UTC'  # Reset time
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


# Utility functions for testing
def _create_test_data():
    """Create test data for development"""
    with db_connection() as conn:
        cursor = conn.cursor()

        # Add test user
        cursor.execute('''
                    INSERT OR IGNORE INTO users 
                    (user_id, username, first_name, photo_url, created_at, last_login)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
            "test_user_123",
            "test@example.com",
            "Test User",
            "https://example.com/photo.jpg",
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))

        # Add test credits
        cursor.execute('''
                    INSERT OR IGNORE INTO user_credits 
                    (user_id, credits_used, daily_limit, last_reset_date)
                    VALUES (?, ?, ?, ?)
                ''', (
            "test_user_123",
            1,  # 1 credit used
            3,  # Daily limit 3
            datetime.utcnow().date().isoformat()
        ))

        conn.commit()


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

    # Create test data if in development
    if os.getenv("ENVIRONMENT") == "development":
        _create_test_data()

    # Run the application
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use our logging config
    )
