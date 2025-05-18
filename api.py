from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
import glob
import random
from datetime import datetime
import sqlite3
import uvicorn
from urllib.parse import urlencode
import requests
from starlette.responses import RedirectResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
THEMES_DIR = os.path.join(DATA_DIR, "themes")
IMAGES_DIR = os.path.join(DATA_DIR, "output_images")
DB_FILE = os.path.join(DATA_DIR, "impulse_pdr.db")
VISITS_FILE = Path("visits.json")

os.makedirs(THEMES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_FILE)
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

    conn.commit()
    conn.close()

init_db()

class UserAnswerPayload(BaseModel):
    question_id: str
    is_correct: bool
    user_id: str


class FavoritePayload(BaseModel):
    question_id: str
    user_id: str

@app.get("/")
def read_root():
    return {"message": "API works"}

@app.post("/api/visit")
def increment_visit():
    if not VISITS_FILE.exists():
        with open(VISITS_FILE, "w") as f:
            json.dump({"count": 0}, f)

    with open(VISITS_FILE, "r") as f:
        data = json.load(f)

    data["count"] += 1

    with open(VISITS_FILE, "w") as f:
        json.dump(data, f)

    return {"message": "Visit recorded", "total": data["count"]}

load_dotenv()

# Конфигурация
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = "https://api.impulsepdr.online/auth/google/callback"  # https://api.impulsepdr.online/auth/google/callback - http://localhost:8000/auth/google/callback
FRONTEND_URI = "https://impulsepdr.online" # https://impulsepdr.online - http://localhost:64979

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    # Получаем параметры из URL
    code = request.query_params.get("code")
    error = request.query_params.get("error")
    state = request.query_params.get("state")

    # Обработка ошибок
    if error:
        error_description = request.query_params.get("error_description", "")
        return RedirectResponse(f"{FRONTEND_URI}?error={error}&description={error_description}")

    if not code:
        return RedirectResponse(f"{FRONTEND_URI}?error=missing_code")

    try:
        # 1. Обмен кода на токены
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        }

        response = requests.post(token_url, data=token_data)
        response.raise_for_status()  # Проверка на ошибки HTTP
        token_json = response.json()

        # 2. Получение информации о пользователе (опционально)
        user_info = {}
        if "access_token" in token_json:
            user_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            headers = {"Authorization": f"Bearer {token_json['access_token']}"}
            user_response = requests.get(user_url, headers=headers)
            if user_response.status_code == 200:
                user_info = user_response.json()

        # 3. Подготовка данных для фронтенда
        frontend_params = {
            **token_json,
            **user_info,
            "state": state  # Передаём обратно state для проверки CSRF
        }

        # 4. Редирект во Flutter с токенами и данными пользователя
        return RedirectResponse(f"{FRONTEND_URI}?{urlencode(frontend_params)}")

    except requests.exceptions.RequestException as e:
        # Обработка ошибок сети/API
        error_msg = f"OAuth error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" | Response: {e.response.text}"
        return RedirectResponse(f"{FRONTEND_URI}?error=server_error&message={error_msg}")

    except Exception as e:
        # Обработка других ошибок
        return RedirectResponse(f"{FRONTEND_URI}?error=unknown&message={str(e)}")

@app.get("/debug/user")
async def debug_user(user_id: str = Query(...)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return {"user": user}

@app.get("/users")
async def get_all_users():
    """
    Returns all registered users from the database
    Response format:
    {
        "users": [
            {
                "user_id": "string",
                "username": "string",
                "first_name": "string",
                "photo_url": "string",
                "created_at": "string",
                "last_login": "string"
            },
            ...
        ]
    }
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # To get results as dictionaries
        cursor = conn.cursor()

        # Get all users with additional stats
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
            # Convert SQLite Row to dict and format fields
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

        conn.close()
        return {"users": users}

    except sqlite3.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/search/questions")
async def search_questions(
        query: str = Query(..., min_length=1),
        user_id: Optional[str] = None
):
    """Поиск вопросов по тексту"""
    try:
        query = query.lower().strip()
        results = []

        for theme in load_all_themes():
            for idx, q in enumerate(theme.get("questions", [])):
                # Безопасное получение полей с проверкой на None
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
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()

                        # Проверка ответа
                        cursor.execute(
                            "SELECT is_correct FROM answers WHERE user_id = ? AND question_id = ?",
                            (user_id, question_id)
                        )
                        answer = cursor.fetchone()
                        question["was_answered_correctly"] = answer[0] if answer else None

                        # Проверка избранного
                        cursor.execute(
                            "SELECT 1 FROM favorites WHERE user_id = ? AND question_id = ?",
                            (user_id, question_id)
                        )
                        question["is_favorite"] = cursor.fetchone() is not None

                        conn.close()

                    results.append(question)

        return {"results": results}

    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/image")
async def get_image(
        path: str = Query(..., description="Relative path to image, e.g. 'output_images/01_01.jpeg'")):
    safe_path = os.path.normpath(path).lstrip('/')
    image_path = os.path.join(IMAGES_DIR, os.path.basename(safe_path))

    if not os.path.isfile(image_path):
        image_name = os.path.basename(safe_path).lower()
        for file in os.listdir(IMAGES_DIR):
            if file.lower() == image_name:
                image_path = os.path.join(IMAGES_DIR, file)
                break
        else:
            raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/themes")
async def get_themes(user_id: Optional[str] = None):
    themes = []
    for file in glob.glob(os.path.join(THEMES_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            theme_data = json.load(f)
            themes.append({
                "index": int(theme_data.get("index", 0)),  # Ensure int
                "name": str(theme_data.get("name", "")),   # Ensure string
                "question_count": int(len(theme_data.get("questions", []))),  # Ensure int
                # Add explicit null checks for optional fields
                "last_answered_index": theme_data.get("last_answered_index"),
                "accuracy": theme_data.get("accuracy"),
            })
    return sorted(themes, key=lambda x: x["index"])


@app.get("/themes/{theme_id}")
async def get_theme_by_id(theme_id: int, user_id: str = None):
    theme_file = os.path.join(THEMES_DIR, f"theme_{theme_id}.json")
    if not os.path.exists(theme_file):
        raise HTTPException(status_code=404, detail="Theme not found")

    with open(theme_file, "r", encoding="utf-8") as f:
        theme_data = json.load(f)

        questions = []
        for idx, q in enumerate(theme_data.get("questions", [])):
            question = q.copy()
            question_id = f"{theme_id}_{idx}"
            question["id"] = question_id

            if user_id:
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()

                # Check if question was answered
                cursor.execute(
                    "SELECT is_correct FROM answers WHERE user_id = ? AND question_id = ?",
                    (user_id, question_id)
                )
                answer = cursor.fetchone()
                question["was_answered_correctly"] = bool(answer[0]) if answer else None

                # Check if question is favorite
                cursor.execute(
                    "SELECT 1 FROM favorites WHERE user_id = ? AND question_id = ?",
                    (user_id, question_id)
                )
                question["is_favorite"] = cursor.fetchone() is not None

                conn.close()

            questions.append(question)

        return {
            "index": theme_data["index"],
            "name": theme_data["name"],
            "question_count": len(questions),
            "questions": questions
        }


@app.get("/ticket/random")
async def get_random_questions(user_id: Optional[str] = None):
    all_questions = []
    for file in glob.glob(os.path.join(THEMES_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            theme = json.load(f)
            theme_index = theme["index"]
            for idx, q in enumerate(theme.get("questions", [])):
                question = q.copy()
                question_id = f"{theme_index}_{idx}"
                question["id"] = question_id
                all_questions.append(question)

    questions = random.sample(all_questions, min(20, len(all_questions)))

    if user_id:
        conn = sqlite3.connect(DB_FILE)
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

        conn.close()

    return questions


@app.get("/user/errors")
async def get_error_questions(user_id: str = Query(..., min_length=1)):
    """Получение ошибок пользователя"""
    print(f"Request to /user/errors with user_id: {user_id}")  # Логирование
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Всегда возвращаем dict с полем 'questions'
        if not user_id:
            return {"questions": []}

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
                    questions.append(question)

        return {"questions": questions}  # Гарантированно возвращаем словарь с списком

    except Exception as e:
        print(f"Error in /user/errors: {str(e)}")
        return {"questions": []}  # Возвращаем пустой список при ошибке
    finally:
        conn.close()

@app.get("/debug/answers")
async def debug_all_answers():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM answers")
        columns = [column[0] for column in cursor.description]
        answers = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/user/favorites")
async def get_favorites(user_id: str):
    conn = sqlite3.connect(DB_FILE)
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
                # Convert SQLite's 0/1 to bool or None if not answered
                question["was_answered_correctly"] = bool(answer[0]) if answer else None

                favorites.append(question)

    conn.close()
    return favorites

@app.post("/user/favorites/add")
async def add_favorite(
        request: Request,
        user_id: str = Query(...),
):
    try:
        data = await request.json()
        question_id = data.get('question_id')

        if not question_id:
            raise HTTPException(status_code=400, detail="question_id is required")

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Ensure user exists
        cursor.execute(
            "INSERT OR IGNORE INTO users (user_id, first_name, created_at) VALUES (?, ?, ?)",
            (user_id, "User", datetime.utcnow().isoformat())
        )

        # Add favorite
        cursor.execute(
            "INSERT OR IGNORE INTO favorites (user_id, question_id) VALUES (?, ?)",
            (user_id, question_id)
        )

        conn.commit()
        conn.close()
        return {"status": "ok", "message": "Favorite added"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/user/favorites/remove")
async def remove_favorite(
        request: Request,
        user_id: str = Query(...),
):
    try:
        data = await request.json()
        question_id = data.get('question_id')

        if not question_id:
            raise HTTPException(status_code=400, detail="question_id is required")

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM favorites WHERE user_id = ? AND question_id = ?",
            (user_id, question_id)
        )

        conn.commit()
        conn.close()
        return {"status": "ok", "message": "Favorite removed"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/user/answer")
async def submit_answer(request: Request):
    try:
        data = await request.json()
        print(f"Received answer: {data}")

        # Проверка данных
        if not all(k in data for k in ['user_id', 'question_id', 'is_correct']):
            raise HTTPException(status_code=400, detail="Missing fields")

        # Подключение к БД
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Добавляем пользователя, если его нет
        cursor.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
            (data['user_id'], datetime.utcnow().isoformat())
        )

        # Сохраняем ответ
        cursor.execute(
            """INSERT INTO answers (user_id, question_id, is_correct, timestamp)
               VALUES (?, ?, ?, ?)""",
            (data['user_id'], data['question_id'], int(data['is_correct']), datetime.utcnow().isoformat()))

        conn.commit()
        conn.close()

        return {"status": "success"}
    except Exception as e:
        print(f"Error in submit_answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/progress")
async def get_progress(user_id: str = Query(...)):
    conn = sqlite3.connect(DB_FILE)
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

    conn.close()
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(accuracy, 2)
    }


@app.get("/theme/progress")
async def get_theme_progress(theme_id: int, user_id: str = Query(...)):
    conn = sqlite3.connect(DB_FILE)
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

    conn.close()
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(accuracy, 2),
        "last_question": last_answered
    }


@app.post("/user/reset")
async def reset_progress(user_id: str = Query(...)):
    try:
        conn = sqlite3.connect(DB_FILE)
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
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/feedback")
async def send_feedback(
        feedback: str = Body(...),
        rating: int = Body(...),
        user_id: str = Body(...)
):
    try:
        # Here you would typically store feedback in the database
        # For now we'll just return success
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Helper functions
def load_all_themes() -> List[Dict]:
    themes = []
    for file in glob.glob(os.path.join(THEMES_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            themes.append(json.load(f))
    return themes


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)