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

@app.get("/debug/user")
async def debug_user(user_id: str = Query(...)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return {"user": user}


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
                "index": theme_data["index"],
                "name": theme_data["name"],
                "question_count": len(theme_data.get("questions", []))
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
                question["was_answered_correctly"] = answer[0] if answer else None

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