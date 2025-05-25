import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import httpx
from fastapi import HTTPException
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()  # Загрузка переменных окружения из .env


class OpenRouterKeyManager:
    def __init__(self):
        self.muted_keys: Dict[str, datetime] = {}
        self.mute_duration = timedelta(minutes=15)
        self.required_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def load_keys(self) -> list:
        try:
            raw_keys = os.getenv("API_KEYS", "")
            keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
            return [k for k in keys if k.startswith("sk-or-v1-")]
        except Exception as e:
            logger.error(f"Failed to load keys from .env: {e}")
            return []

    def is_key_muted(self, key: str) -> bool:
        mute_until = self.muted_keys.get(key)
        return bool(mute_until and mute_until > datetime.now())

    def mute_key(self, key: str):
        self.muted_keys[key] = datetime.now() + self.mute_duration
        logger.warning(f"Muted key {key[-4:]}... until {self.muted_keys[key]}")

    async def verify_key(self, key: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/auth/key",
                    headers={"Authorization": f"Bearer {key}"}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Key verification failed: {e}")
            return False

    def get_active_key(self) -> Optional[str]:
        for key in self.load_keys():
            if not self.is_key_muted(key):
                return key
        return None


class OpenRouterAPIClient:
    def __init__(self):
        self.timeout = 30.0
        self.key_manager = OpenRouterKeyManager()

    async def call_api(self, messages: list, model: str = "deepseek/deepseek-prover-v2:free"):
        key = self.key_manager.get_active_key()
        if not key:
            raise HTTPException(status_code=503, detail="No active API keys available")

        headers = {
            "Authorization": f"Bearer {key}",
        }

        payload = {
            "model": model,
            "messages": messages
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )

        if response.status_code == 401:
            self.key_manager.mute_key(key)
            is_valid = await self.key_manager.verify_key(key)
            if not is_valid:
                logger.error(f"Key {key[-8:]} is permanently invalid")
            raise HTTPException(status_code=401, detail="Invalid API credentials")

        if response.status_code != 200:
            logger.error(f"OpenRouter error: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()