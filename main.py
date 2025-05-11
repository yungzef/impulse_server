# 📁 project_root/main.py

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, types, Router
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

router = Router()

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Config ===
BOT_TOKEN = os.getenv("BOT_TOKEN") or "7788226951:AAEQDNKMq-THOs3CrHaGTRS7xZmOrlSZDv0"
WEBAPP_URL = "https://horrible-izabel-yungzef-75a7e3d6.koyeb.app"

# === Telegram Bot Handler ===
@router.message(Command("start"))
async def handle_start(message: types.Message):
    logger.info(f"Received /start from {message.from_user.id}")

    await message.answer(
        "Привіт! Натисни кнопку нижче, щоб перейти до додатку Імпульс:",
    )

# === Main Entrypoint ===
async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())