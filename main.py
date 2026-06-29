"""
Искра — бэкенд-прокси для генерации заданий.

Зачем нужен: ключ к LLM живёт ТОЛЬКО здесь, на сервере. Фронт (Mini App)
никогда не видит ключ — он дёргает /generate, а сервер ходит к модели.

Запуск локально:
    pip install -r requirements.txt
    cp .env.example .env   # вписать LLM_API_KEY
    uvicorn main:app --reload

Деплой на Railway: см. README.md в корне проекта.
"""

import json
import os
import re

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Конфиг из переменных окружения ──────────────────────────────────────────
# По умолчанию — OpenAI-совместимый эндпоинт (OpenRouter). Доступен из РФ.
# Чтобы перейти на свой прокси / GigaChat-шлюз — меняешь только эти три строки.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")

# Origin твоего фронта на GitHub Pages, напр. https://username.github.io
# Можно перечислить через запятую. "*" — только для локальной отладки.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# ── Системный промпт (финальная версия: мягкий уровень 3) ───────────────────
SYSTEM_PROMPT = """\
Ты — генератор заданий для приватной игры «Правда или Действие» для ОДНОЙ ПАРЫ
взрослых (18+), играющих вместе по обоюдному согласию. Только русский язык.

РЕЖИМЫ:
- «правда» — личный или откровенный вопрос, на который игрок отвечает вслух.
  Откровенность вопроса растёт вместе с уровнем.
- «действие» — физическое задание, которое игрок выполняет.

УРОВНИ (касаются и «правды», и «действия»):
1 «Лёгкий» — флирт и романтика без откровенности.
2 «Горячий» — чувственно и с намёком: поцелуи, смелые прикосновения,
  лёгкое раздевание, эротичные вопросы. Эротика есть, графики нет.
3 «Очень горячий» — откровенно и чувственно. Подразумеваются любые интимные
  ласки и действия для пары наедине, но формулируй их намёком и через ощущение,
  а не прямым называнием частей тела. Тон — тёплый и обволакивающий, как
  игровая карточка, а не клиническое описание. Коротко, по делу, с лёгкой
  недосказанностью, которая оставляет место воображению.

ЖЁСТКИЕ ПРАВИЛА (не нарушать никогда):
- Только эти два игрока, только по согласию; любое задание можно пропустить.
- В формулировки уровней 2–3 вшивай контроль игрока: «она/он направляет
  словами», «на свой выбор», «если хочешь», «в комфортном темпе».
- На уровне 3 избегай анатомических терминов в лоб — передавай через действие,
  темп и ощущение.
- НИКОГДА: без согласия; боль/унижение сверх явно заданных границ; опасное
  для здоровья; третьи лица; несовершеннолетние; животные; что-либо незаконное.
  Строго соблюдай список исключённых тем.
- Обстановка «на людях» → на уровнях 2–3 без обнажения и явных действий
  на публике, держи в рамках приличия.
- Обращайся по именам, чередуй адресата.

ФОРМАТ ОТВЕТА: только JSON-массив, без markdown и пояснений:
[{"mode":"правда|действие","for":"имя|имя|оба","text":"..."}]
"""

app = FastAPI(title="Искра API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Схема запроса ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    name_a: str = "Игрок 1"
    name_b: str = "Игрок 2"
    level: int = Field(1, ge=1, le=3)
    mode: str = "действие"            # "правда" | "действие"
    setting: str = "дома"            # "дома" | "на людях"
    exclude: str = ""                # темы-исключения через запятую
    used: list[str] = []             # недавние тексты — чтобы не повторяться
    count: int = Field(12, ge=1, le=20)


def build_user_message(req: GenerateRequest) -> str:
    used_block = "\n".join(f"- {t}" for t in req.used[-25:]) or "(пока пусто)"
    return (
        f"Игроки: {req.name_a}, {req.name_b}\n"
        f"Уровень: {req.level}\n"
        f"Режим: {req.mode}\n"
        f"Обстановка: {req.setting}\n"
        f"Исключить темы: {req.exclude or '(нет)'}\n"
        f"Сгенерируй {req.count} заданий режима «{req.mode}» уровня {req.level}.\n\n"
        f"Уже использованные (не повторять и не перефразировать):\n{used_block}"
    )


def parse_tasks(raw: str) -> list[dict]:
    """Достаём JSON-массив, даже если модель обернула его в ```json или текст."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    # Если вокруг массива остался текст — вырезаем от первой [ до последней ]
    if not cleaned.startswith("["):
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    # Оставляем только валидные карточки
    out = []
    for item in data:
        text = (item.get("text") or "").strip()
        if text:
            out.append(
                {
                    "mode": item.get("mode", "действие"),
                    "for": item.get("for", "оба"),
                    "text": text,
                }
            )
    return out


@app.get("/")
def health():
    return {"ok": True, "service": "iskra"}
    @app.get("/debug")
def debug():
    key = os.environ.get("LLM_API_KEY", "")
    return {
        "key_seen": bool(key),
        "key_len": len(key),
        "model": os.environ.get("LLM_MODEL", ""),
        "base_url": os.environ.get("LLM_BASE_URL", ""),
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not LLM_API_KEY:
        raise HTTPException(500, "LLM_API_KEY не задан на сервере")

    payload = {
        "model": LLM_MODEL,
        "temperature": 1.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(req)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Модель недоступна: {e}")

    try:
        tasks = parse_tasks(raw)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(502, "Модель вернула не-JSON, попробуй ещё раз")

    if not tasks:
        raise HTTPException(502, "Пустой ответ модели")

    return {"tasks": tasks}
