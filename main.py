import asyncio
import logging
import os
import random
import json
from datetime import datetime
from pathlib import Path
import glob

from aiogram import Bot, Dispatcher, types, Router
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InputFile
from PIL import Image, ImageDraw, ImageFont, ImageOps

router = Router()

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Config ===
BOT_TOKEN = os.getenv("BOT_TOKEN") or "7651268029:AAEjgGQNJ7WfMVGdBao0hz5MaIlF_j8zAH4"
CHANNEL_ID = "-1002580971858"  # Или ID канала, если это приватный канал
BACKGROUND_IMAGES_PATH = "data/backgrounds"  # Путь к папке с фоновыми изображениями


# === Image Generator Functions ===
def get_random_background():
    try:
        if not os.path.exists(BACKGROUND_IMAGES_PATH):
            logger.warning(f"Background images directory not found: {BACKGROUND_IMAGES_PATH}")
            return None

        bg_images = [f for f in glob.glob(f"{BACKGROUND_IMAGES_PATH}/*")
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not bg_images:
            logger.warning("No background images found")
            return None

        return random.choice(bg_images)
    except Exception as e:
        logger.error(f"Error getting random background: {e}")
        return None


def create_square_canvas(width, background_path=None):
    """Создает квадратное изображение с заданным фоном или черным цветом"""
    if background_path:
        try:
            bg = Image.open(background_path)
            # Обрезаем фон до квадрата
            bg = ImageOps.fit(bg, (width, width), method=Image.LANCZOS)
            return bg.convert("RGBA")
        except Exception as e:
            logger.error(f"Error loading background image: {e}")

    # Если фон не найден или произошла ошибка - черный квадрат
    return Image.new("RGBA", (width, width), (0, 0, 0, 255))


def draw_multiline_text(draw, text, font, x, y, max_width, fill, line_spacing=4):
    lines = []
    current_line = ""
    for word in text.split():
        test_line = current_line + word + " "
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > max_width:
            lines.append(current_line.rstrip())
            current_line = word + " "
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line.rstrip())

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_spacing
    return y, lines


def draw_checkmark(draw, x, y, size, color):
    thickness = size // 10
    x1 = x
    y1 = y + size * 0.5
    x2 = x + size * 0.3
    y2 = y + size * 0.85
    x3 = x + size
    y3 = y
    draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)
    draw.line([(x2, y2), (x3, y3)], fill=color, width=thickness)


def wrap_text_by_pixel(draw, text, font, max_width):
    """Wrap text to fit within specified pixel width."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        # Test if adding the word exceeds max width
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]

        if test_width <= max_width:
            current_line.append(word)
        else:
            if current_line:  # Only add line if it has content
                lines.append(' '.join(current_line))
            current_line = [word]

    # Add the last line
    if current_line:
        lines.append(' '.join(current_line))

    return lines

def generate_quiz_image(quiz_data, output_path, force_incorrect=False):
    try:
        # Configuration parameters
        MIN_WIDTH = 800
        QUESTION_FONT_SIZE = 40
        ANSWER_FONT_SIZE = 36
        PADDING = 40
        CONTAINER_PADDING = 48
        ANSWER_SPACING = 24
        OPTION_MIN_HEIGHT = 60

        # Load fonts
        try:
            font_question = ImageFont.truetype("Montserrat-Bold.ttf", QUESTION_FONT_SIZE)
            font_answer = ImageFont.truetype("Montserrat-Regular.ttf", ANSWER_FONT_SIZE)
        except:
            # Fallback to default fonts if custom fonts not available
            font_question = ImageFont.load_default()
            font_answer = ImageFont.load_default()

        # Create temporary image for text measurements
        dummy_img = Image.new("RGB", (MIN_WIDTH, 100), (255, 255, 255))
        dummy_draw = ImageDraw.Draw(dummy_img)

        def calculate_content_width():
            """Calculate required width based on content."""
            # Calculate question width
            question_lines = wrap_text_by_pixel(
                dummy_draw,
                quiz_data["question"],
                font_question,
                MIN_WIDTH - 2 * CONTAINER_PADDING
            )
            question_width = max(
                [dummy_draw.textbbox((0, 0), line, font=font_question)[2]
                 for line in question_lines]
            ) if question_lines else 0

            # Calculate answers width
            answers_width = 0
            for ans in quiz_data["answers"]:
                ans_lines = wrap_text_by_pixel(
                    dummy_draw,
                    ans,
                    font_answer,
                    MIN_WIDTH - 2 * CONTAINER_PADDING - 64
                )
            if ans_lines:
                ans_width = max(
                    dummy_draw.textbbox((0, 0), line, font=font_answer)[2]
                    for line in ans_lines
                )
            answers_width = max(answers_width, ans_width)

            # Calculate image width if exists
            img_width = 0
            if "image" in quiz_data and quiz_data["image"]:
                try:
                    with Image.open("data/" + quiz_data["image"]) as img:
                        img_width = img.width
                except Exception as e:
                    logger.error(f"Error loading question image: {e}")

            # Determine final width with padding
            content_width = max(question_width, answers_width, img_width, MIN_WIDTH)
            return content_width + 2 * CONTAINER_PADDING

        # Determine final dimensions
        content_width = calculate_content_width()
        current_y = CONTAINER_PADDING

        # Create background
        background_path = get_random_background()
        if background_path:
            try:
                bg = Image.open(background_path)
                bg = bg.resize((content_width, content_width), Image.LANCZOS)
                container = bg.convert("RGBA")
            except Exception as e:
                logger.error(f"Error loading background: {e}")
                container = Image.new("RGBA", (content_width, content_width), (0, 0, 0, 255))
        else:
            container = Image.new("RGBA", (content_width, content_width), (0, 0, 0, 255))

        draw = ImageDraw.Draw(container)

        # Add question image if exists
        if "image" in quiz_data and quiz_data["image"]:
            try:
                question_image = Image.open("data/" + quiz_data["image"]).convert("RGBA")
                # Scale to fit container width (with padding)
                new_width = content_width - 2 * CONTAINER_PADDING
                ratio = new_width / question_image.width
                new_height = int(question_image.height * ratio)
                question_image = question_image.resize((new_width, new_height), Image.LANCZOS)

                # Center the image
                x_offset = (container.width - question_image.width) // 2
                container.paste(question_image, (x_offset, current_y), question_image)
                current_y += question_image.height + 30
            except Exception as e:
                logger.error(f"Error processing question image: {e}")

        # Add question text
        question_lines = wrap_text_by_pixel(
            draw,
            quiz_data["question"],
            font_question,
            content_width - 2 * CONTAINER_PADDING
        )

        for line in question_lines:
            bbox = draw.textbbox((CONTAINER_PADDING, current_y), line, font=font_question)
            draw.text(
                (CONTAINER_PADDING, current_y),
                line,
                font=font_question,
                fill="white"
            )
            current_y += bbox[3] - bbox[1] + 6

        current_y += 20

        # Add answer options
        for idx, ans in enumerate(quiz_data["answers"]):
            is_correct = idx == quiz_data["correct_index"] and not force_incorrect
            answer_lines = wrap_text_by_pixel(
                draw,
                ans,
                font_answer,
                content_width - 2 * CONTAINER_PADDING - 64
            )

            # Calculate answer block height
            line_heights = [
                draw.textbbox((0, 0), line, font=font_answer)[3]
                for line in answer_lines
            ]
            answer_height = max(
                OPTION_MIN_HEIGHT,
                sum(line_heights) + 20 + (len(answer_lines) - 1) * 6
            )

            # Draw answer container
            border_color = "#4CAF50" if is_correct else "#CFCFCF"
            text_color = "#4CAF50" if is_correct else "white"

            draw.rounded_rectangle(
                [(CONTAINER_PADDING, current_y),
                 (content_width - CONTAINER_PADDING, current_y + answer_height)],
                radius=24,
                outline=border_color,
                width=3,
                fill=None
            )

            # Draw answer text
            text_y = current_y + 10
            for line in answer_lines:
                draw.text(
                    (CONTAINER_PADDING + 32, text_y),
                    line,
                    font=font_answer,
                    fill=text_color
                )
                text_y += draw.textbbox((0, 0), line, font=font_answer)[3] + 6

            # Add checkmark for correct answer
            if is_correct:
                check_size = 32
                check_x = content_width - CONTAINER_PADDING - check_size - 40
                check_y = current_y + (answer_height - check_size) // 2
                draw_checkmark(draw, check_x, check_y, check_size, "#4CAF50")

            current_y += answer_height + ANSWER_SPACING

        # Crop excess height
        final_height = current_y + CONTAINER_PADDING
        container = container.crop((0, 0, content_width, final_height))
        container.save(output_path, "PNG")
        return True

    except Exception as e:
        logger.error(f"Error generating quiz image: {e}")
        return False

async def get_random_question_with_image():
    try:
        theme_files = glob.glob("data/themes/*.json")
        logger.info(f"Found {len(theme_files)} theme files")

        if not theme_files:
            logger.warning("No theme files found in data/themes/")
            return None

        theme_path = random.choice(theme_files)
        logger.info(f"Selected theme file: {theme_path}")

        with open(theme_path, encoding="utf-8") as f:
            theme_data = json.load(f)

        questions_with_images = [q for q in theme_data["questions"] if q.get("image")]
        logger.info(f"Found {len(questions_with_images)} questions with images in this theme")

        if not questions_with_images:
            logger.warning("No questions with images found in this theme")
            return None

        question = random.choice(questions_with_images)
        logger.info(f"Selected question: {question['question'][:50]}...")
        return question
    except Exception as e:
        logger.error(f"Error getting random question: {e}")
        return None


async def send_quiz_to_channel(bot: Bot):
    try:
        logger.info("Starting to send quiz to channel")

        question = await get_random_question_with_image()
        if not question:
            logger.warning("No question with image found")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated/quiz_{timestamp}.png"

        os.makedirs("generated", exist_ok=True)

        if not generate_quiz_image(question, output_path, force_incorrect=True):
            logger.error("Failed to generate quiz image")
            return

        logger.info(f"Generated image at {output_path}, sending to channel")

        # Исправленный способ отправки файла
        try:
            # Вариант 1: Используем путь к файлу
            await bot.send_photo(
                chat_id=CHANNEL_ID,
                photo=types.FSInputFile(output_path),
                caption="Нове питання з ПДР 🚦"
            )

            # Или Вариант 2: Читаем файл в бинарном режиме
            # with open(output_path, 'rb') as photo:
            #     await bot.send_photo(
            #         chat_id=CHANNEL_ID,
            #         photo=photo,
            #         caption="Нове питання з ПДР 🚦"
            #     )

            logger.info("Quiz successfully sent to channel")
        except Exception as send_error:
            logger.error(f"Error sending photo: {send_error}")
            raise

        # Удаляем временный файл
        try:
            os.remove(output_path)
        except Exception as remove_error:
            logger.error(f"Error removing temp file: {remove_error}")
    except Exception as e:
        logger.error(f"Error sending quiz to channel: {e}")

async def scheduled_posting(bot: Bot):
    while True:
        try:
            await send_quiz_to_channel(bot)
        except Exception as e:
            logger.error(f"Error in scheduled posting: {e}")

        # Ждем 1 час перед следующей отправкой
        await asyncio.sleep(3600)


# === Telegram Bot Handler ===
@router.message(Command("start"))
async def handle_start(message: types.Message):
    logger.info(f"Received /start from {message.from_user.id}")
    await message.answer("Привіт! Натисни кнопку нижче, щоб перейти до додатку Імпульс:")


@router.channel_post()
async def log_chat_id(message: types.Message):
    chat = message.chat
    logger.info(f"Bot interacted with chat: {chat.id} ({chat.title or chat.username or chat.first_name})")

# === Main Entrypoint ===
async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(bot=bot, storage=MemoryStorage())
    dp.include_router(router)

    # Запускаем фоновую задачу для регулярной отправки вопросов
    asyncio.create_task(scheduled_posting(bot))

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())