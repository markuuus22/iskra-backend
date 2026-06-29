"""
Искра — бэкенд-прокси для генерации заданий.

Ключ к LLM живёт ТОЛЬКО здесь. Фронт дёргает /generate, сервер ходит к модели.
"""

import json
import os
import re

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Конфиг из переменных окружения ──────────────────────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

LEVEL_NAMES = {1: "Light", 2: "Medium", 3: "Hard", 4: "SO HOT!"}

# ── Системный промпт ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Ты — генератор заданий для приватной игры «Правда или Действие» для ОДНОЙ ПАРЫ
взрослых (18+), играющих вместе по обоюдному согласию. Только русский язык.

ЯЗЫК И ТОН: пиши живым, грамотным русским языком — без орфографических и
грамматических ошибок, без канцелярита и корявых оборотов. Каждое задание —
одно-два естественных предложения, как сказал бы живой человек. Звучи тепло,
игриво и уверенно. Обращайся к игрокам по именам и чередуй адресата.

РЕЖИМЫ:
- «правда» — личный или откровенный вопрос, на который игрок отвечает вслух.
  Откровенность вопроса растёт вместе с уровнем.
- «действие» — физическое задание, которое игрок выполняет.

УРОВНИ (касаются и «правды», и «действия»):
1 «Light» — флирт и романтика без откровенности: комплименты, лёгкие
  прикосновения, личные вопросы, игривые мини-задания.
2 «Medium» — чувственно и с намёком: поцелуи, смелые прикосновения, лёгкое
  раздевание, эротичные вопросы. Эротика есть, графики нет.
3 «Hard» — откровенно и чувственно. Подразумеваются интимные ласки для пары
  наедине, но формулируй намёком и через ощущение, а не прямым называнием.
  Тон тёплый и обволакивающий, с лёгкой недосказанностью.
4 «SO HOT!» — максимально откровенно и без эвфемизмов. Здесь можно прямо
  называть интимные действия и описывать смелые, пошлые сексуальные задания
  для пары наедине. Самый горячий уровень — не сдерживайся в откровенности.
  Но это по-прежнему игровая карточка-ЗАДАНИЕ: коротко и по делу, без длинного
  эротического повествования.

ЖЁСТКИЕ ПРАВИЛА (не нарушать никогда, на любом уровне):
- Только эти два игрока, только по обоюдному согласию; любое задание можно
  пропустить. Даже на уровне 4 оставляй пространство для согласия и
  комфортного темпа («если хочется», «в своём темпе», партнёр направляет).
- НИКОГДА: что-либо без согласия; боль или унижение сверх явно заданных
  границ; опасное для здоровья; третьи лица; несовершеннолетние; животные;
  что-либо незаконное. Строго соблюдай список исключённых тем.
- Обстановка «на людях» → на уровнях 2–4 без обнажения и откровенных действий
  на публике, держи в рамках приличия.

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


class GenerateRequest(BaseModel):
    name_a: str = "Игрок 1"
    name_b: str = "Игрок 2"
    level: int = Field(1, ge=1, le=4)
    mode: str = "действие"
    setting: str = "дома"
    exclude: str = ""
    used: list[str] = []
    count: int = Field(12, ge=1, le=20)


def build_user_message(req: GenerateRequest) -> str:
    used_block = "\n".join(f"- {t}" for t in req.used[-25:]) or "(пока пусто)"
    level_name = LEVEL_NAMES.get(req.level, "")
    return (
        f"Игроки: {req.name_a}, {req.name_b}\n"
        f"Уровень: {req.level} ({level_name})\n"
        f"Режим: {req.mode}\n"
        f"Обстановка: {req.setting}\n"
        f"Исключить темы: {req.exclude or '(нет)'}\n"
        f"Сгенерируй {req.count} заданий режима «{req.mode}» уровня {req.level}.\n\n"
        f"Уже использованные (не повторять и не перефразировать):\n{used_block}"
    )


def parse_tasks(raw: str) -> list[dict]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    if not cleaned.startswith("["):
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
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


async def call_model(req: GenerateRequest) -> list[dict]:
    payload = {
        "model": LLM_MODEL,
        "temperature": 0.9,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(req)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    return parse_tasks(raw)


@app.get("/")
def health():
    return {"ok": True, "service": "iskra"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not LLM_API_KEY:
        raise HTTPException(500, "LLM_API_KEY не задан на сервере")

    last_err = "неизвестно"
    # До 2 попыток: гасим разовые сбои модели и кривой не-JSON.
    for _ in range(2):
        try:
            tasks = await call_model(req)
            if tasks:
                return {"tasks": tasks}
            last_err = "пустой ответ"
        except httpx.HTTPError as e:
            last_err = f"модель недоступна: {e}"
        except (json.JSONDecodeError, ValueError):
            last_err = "модель вернула не-JSON"

    raise HTTPException(502, f"Не удалось сгенерировать ({last_err})")
