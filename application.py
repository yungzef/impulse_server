# application.py — исправленная версия

from fastapi import FastAPI, HTTPException, Query, Request, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
import glob
import random
from datetime import datetime
import hashlib, hmac, urllib.parse
import uvicorn

app = FastAPI()

application = app

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
USERS_FILE = os.path.join(DATA_DIR, "users.json")
BOT_TOKEN = "7788226951:AAEQDNKMq-THOs3CrHaGTRS7xZmOrlSZDv0"

os.makedirs(THEMES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump([], f)


class TelegramAuthPayload(BaseModel):
    id: str
    first_name: str
    username: str = ""
    hash: str
    auth_date: str

class AnswerPayload(BaseModel):
    question_id: int
    is_correct: bool
    telegram_id: str

class FavoritePayload(BaseModel):
    question_id: int
    telegram_id: str

def verify_telegram_auth(payload: dict, token: str):
    auth_hash = payload.pop("hash")
    data_check_string = "\n".join([f"{k}={v}" for k, v in sorted(payload.items())])
    secret_key = hmac.new(token.encode(), msg=b"", digestmod=hashlib.sha256).digest()
    calc_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc_hash, auth_hash)


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: str, data: List[Dict]):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_user_exists(telegram_id: str):
    """Создаёт пользователя, если его нет"""
    users = load_json(USERS_FILE)
    user = next((u for u in users if u["telegram_id"] == telegram_id), None)

    if not user:
        users.append({
            "id": str(len(users) + 1),
            "telegram_id": telegram_id,
            "username": "",
            "first_name": "User",
            "created_at": datetime.utcnow().isoformat(),
            "favorites": [],
            "answers": []
        })
        save_json(USERS_FILE, users)

@app.get("/")
def read_root():
    return {"message": "API works"}


@app.get("/auth/telegram")
async def auth_telegram(
        request: Request,
        tgWebAppData: str = Query(...),
        tgWebAppVersion: str = Query(None),
        tgWebAppPlatform: str = Query(None),
):
    try:
        parsed_data = dict(urllib.parse.parse_qsl(tgWebAppData))

        if not verify_telegram_auth(parsed_data.copy(), BOT_TOKEN):
            raise HTTPException(status_code=401, detail="Invalid Telegram auth")

        user_data = json.loads(parsed_data.get("user", "{}"))

        users = load_json(USERS_FILE)
        user_exists = any(u.get("telegram_id") == str(user_data.get("id")) for u in users)

        if not user_exists:
            users.append({
                "id": str(len(users) + 1),
                "telegram_id": str(user_data.get("id")),
                "username": user_data.get("username"),
                "first_name": user_data.get("first_name"),
                "photo_url": user_data.get("photo_url"),
                "created_at": datetime.utcnow().isoformat(),
                "favorites": [],
                "answers": []
            })
            save_json(USERS_FILE, users)

        return {"status": "authenticated", "user_id": user_data.get("id")}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth")
async def auth_user(data: TelegramAuthPayload):
    if not verify_telegram_auth(data.dict().copy(), BOT_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid Telegram login")

    users = load_json(USERS_FILE)
    user_exists = any(user["telegram_id"] == data.id for user in users)

    if not user_exists:
        users.append({
            "id": str(len(users) + 1),
            "telegram_id": data.id,
            "username": data.username,
            "first_name": data.first_name,
            "created_at": datetime.utcnow().isoformat(),
            "favorites": [],
            "answers": []
        })
        save_json(USERS_FILE, users)

    return {"status": "ok"}


@app.get("/image")
async def get_image(
        path: str = Query(..., description="Относительный путь к изображению, например 'output_images/01_01.jpeg'")):
    # Безопасная обработка пути к изображению
    safe_path = os.path.normpath(path).lstrip('/')
    image_path = os.path.join(IMAGES_DIR, os.path.basename(safe_path))

    if not os.path.isfile(image_path):
        # Попробуем найти похожий файл (на случай различий в регистре)
        image_name = os.path.basename(safe_path).lower()
        for file in os.listdir(IMAGES_DIR):
            if file.lower() == image_name:
                image_path = os.path.join(IMAGES_DIR, file)
                break
        else:
            raise HTTPException(status_code=404, detail="Изображение не найдено")

    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/themes")
async def get_themes(telegram_id: Optional[str] = None):
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
async def get_theme_by_id(theme_id: int, telegram_id: str = None):
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

            if telegram_id:
                users = load_json(USERS_FILE)
                user = next((u for u in users if u["telegram_id"] == telegram_id), None)
                if user:
                    user_answers = {a["question_id"]: a["is_correct"]
                                    for a in user.get("answers", [])}
                    question["was_answered_correctly"] = user_answers.get(question_id)
                    question["is_favorite"] = question_id in user.get("favorites", [])

            questions.append(question)

        return {
            "index": theme_data["index"],
            "name": theme_data["name"],
            "question_count": len(questions),
            "questions": questions
        }

@app.get("/ticket/random")
async def get_random_questions(telegram_id: Optional[str] = None):
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

    if telegram_id:
        users = load_json(USERS_FILE)
        user = next((u for u in users if u["telegram_id"] == telegram_id), None)
        if user:
            user_answers = {a["question_id"]: a["is_correct"] for a in user.get("answers", [])}
            user_favorites = user.get("favorites", [])

            for question in questions:
                question["was_answered_correctly"] = user_answers.get(question["id"])
                question["is_favorite"] = question["id"] in user_favorites

    return questions


@app.get("/user/errors")
async def get_error_questions(telegram_id: str):
    users = load_json(USERS_FILE)
    user = next((u for u in users if u["telegram_id"] == telegram_id), None)

    if not user:
        return {"questions": []}

    error_ids = [a["question_id"] for a in user.get("answers", [])
                 if not a["is_correct"]]

    error_questions = []
    for theme in load_all_themes():
        for idx, q in enumerate(theme.get("questions", [])):
            question_id = f"{theme['index']}_{idx}"
            if question_id in error_ids:
                error_questions.append({
                    "id": question_id,
                    "question": q.get("question", ""),
                    "answers": q.get("answers", []),
                    "correct_answer": q.get("correct_answer", ""),
                    "correct_index": q.get("correct_index", 0),
                    "image": q.get("image"),
                    "explanation": q.get("explanation", ""),
                    "is_favorite": question_id in user.get("favorites", []),
                    "was_answered_correctly": False
                })

    return {"questions": error_questions}


@app.get("/user/favorites")
async def get_favorites(telegram_id: str):
    users = load_json(USERS_FILE)
    user = next((u for u in users if u["telegram_id"] == telegram_id), None)

    if not user:
        return []

    favorites = user.get("favorites", [])
    result = []

    # Ищем вопросы во всех темах
    for theme_file in glob.glob(os.path.join(THEMES_DIR, "*.json")):
        with open(theme_file, "r", encoding="utf-8") as f:
            theme = json.load(f)
            theme_index = theme["index"]
            for idx, q in enumerate(theme.get("questions", [])):
                question_id = f"{theme_index}_{idx}"
                if question_id in favorites:
                    question = q.copy()
                    question["id"] = question_id
                    question["is_favorite"] = True
                    # Добавляем информацию о правильности ответа
                    user_answer = next(
                        (a for a in user.get("answers", [])
                         if a["question_id"] == question_id),
                        None
                    )
                    if user_answer:
                        question["was_answered_correctly"] = user_answer["is_correct"]
                    result.append(question)

    return result


@app.post("/user/favorites/add")
async def add_favorite(
        request: Request,
        telegram_id: str = Query(...),
):
    try:
        data = await request.json()
        question_id = data.get('question_id')

        if not question_id:
            raise HTTPException(status_code=400, detail="question_id is required")

        users = load_json(USERS_FILE)
        user = next((u for u in users if u["telegram_id"] == telegram_id), None)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if "favorites" not in user:
            user["favorites"] = []

        if question_id not in user["favorites"]:
            user["favorites"].append(question_id)

        save_json(USERS_FILE, users)
        return {"status": "ok", "message": "Favorite added"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/user/favorites/remove")
async def remove_favorite(
        request: Request,
        telegram_id: str = Query(...),
):
    try:
        data = await request.json()
        question_id = data.get('question_id')

        if not question_id:
            raise HTTPException(status_code=400, detail="question_id is required")

        users = load_json(USERS_FILE)
        user = next((u for u in users if u["telegram_id"] == telegram_id), None)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if "favorites" in user and question_id in user["favorites"]:
            user["favorites"].remove(question_id)

        save_json(USERS_FILE, users)
        return {"status": "ok", "message": "Favorite removed"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Изменяем метод submit_answer
@app.post("/user/answer")
async def submit_answer(
        data: dict = Body(...)  # Принимаем JSON данные
):
    try:
        # Проверяем обязательные поля
        if not all(key in data for key in ['question_id', 'is_correct', 'telegram_id']):
            raise HTTPException(status_code=400, detail="Missing required fields")

        users = load_json(USERS_FILE)
        user = next((u for u in users if u["telegram_id"] == data['telegram_id']), None)

        if not user:
            user = {
                "id": str(len(users) + 1),
                "telegram_id": data['telegram_id'],
                "username": "",
                "first_name": "User",
                "created_at": datetime.utcnow().isoformat(),
                "favorites": [],
                "answers": []
            }
            users.append(user)

        # Обновляем ответы
        user["answers"] = [a for a in user.get("answers", [])
                           if a["question_id"] != data['question_id']]

        user["answers"].append({
            "question_id": data['question_id'],
            "is_correct": data['is_correct'],
            "timestamp": datetime.utcnow().isoformat()
        })

        save_json(USERS_FILE, users)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/user/progress")
async def get_progress(telegram_id: str = Query(...)):
    users = load_json(USERS_FILE)
    user = next((u for u in users if u["telegram_id"] == telegram_id), None)

    if not user:
        return {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy": 0.0
        }

    answers = user.get("answers", [])
    total = len(answers)
    correct = sum(1 for a in answers if a["is_correct"])
    wrong = total - correct
    accuracy = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(accuracy, 2)  # Округляем до 2 знаков
    }

# Старые маршруты для обратной совместимости
@app.get("/ticket/errors")
async def old_get_error_questions(telegram_id: str):
    return await get_error_questions(telegram_id)


@app.get("/favorites")
async def old_get_favorites(telegram_id: str):
    return await get_favorites(telegram_id)


@app.post("/favorites/add")
async def old_add_favorite(data: FavoritePayload):
    return await add_favorite(data)


@app.post("/favorites/remove")
async def old_remove_favorite(data: FavoritePayload):
    return await remove_favorite(data)


@app.post("/answer")
async def old_submit_answer(data: AnswerPayload):
    return await submit_answer(data)


@app.get("/theme/progress")
async def get_theme_progress(theme_id: int, telegram_id: str = Query(...)):
    users = load_json(USERS_FILE)
    user = next((u for u in users if u["telegram_id"] == telegram_id), None)

    if not user:
        return {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy": 0.0,
            "last_question": 0
        }

    theme_answers = [a for a in user.get("answers", [])
                     if a["question_id"].startswith(f"{theme_id}_")]

    total = len(theme_answers)
    correct = sum(1 for a in theme_answers if a["is_correct"])
    wrong = total - correct
    accuracy = correct / total if total > 0 else 0.0

    # Находим последний отвеченный вопрос
    last_answered = 0
    for a in theme_answers:
        q_index = int(a["question_id"].split("_")[1])
        if q_index > last_answered:
            last_answered = q_index

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(accuracy, 2),
        "last_question": last_answered
    }

@app.get("/progress")
async def old_get_progress(telegram_id: str):
    return await get_progress(telegram_id)

# Вспомогательные функции
def load_users() -> List[Dict]:
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users: List[Dict]):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def get_or_create_user(users: List[Dict], telegram_id: str) -> Dict:
    user = next((u for u in users if u["telegram_id"] == telegram_id), None)
    if not user:
        user = {
            "telegram_id": telegram_id,
            "answers": [],
            "favorites": []
        }
        users.append(user)
    return user

def load_all_themes() -> List[Dict]:
    themes = []
    for file in glob.glob(os.path.join(THEMES_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            themes.append(json.load(f))
    return themes

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)