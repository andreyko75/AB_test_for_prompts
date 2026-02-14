# -*- coding: utf-8 -*-
"""
Шаг 1. Генерация ответов A/B.

Для каждого описания товара из датасета запрашивает ответ модели
с промптом А и с промптом Б (модель и параметры одинаковые).
Сохраняет output/results.json.

Дальше: откройте results.json, проставьте в каждом элементе items:
  relevance_a, relevance_b — число от 1 до 5 (релевантность поста);
  hallucination_a, hallucination_b — 0 или 1 (есть ли выдуманные факты о здоровье).
После заполнения запустите: python evaluate_and_report.py
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "product_descriptions.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RESULTS_JSON = OUTPUT_DIR / "results.json"


def load_dataset():
    """Загружает список описаний товаров из JSON."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm(client: OpenAI, system_prompt: str, user_message: str) -> tuple[str, float]:
    """Вызов модели. Возвращает (текст ответа, латентность в секундах)."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config.MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    latency = time.perf_counter() - start
    text = response.choices[0].message.content or ""
    return text.strip(), latency


def main():
    client = OpenAI()
    dataset = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    items = []
    for i, item in enumerate(dataset):
        idx = item["id"]
        description = item["description"]
        print(f"[{i+1}/{len(dataset)}] id={idx} ...")

        response_a, latency_a = call_llm(client, config.PROMPT_A, description)
        response_b, latency_b = call_llm(client, config.PROMPT_B, description)

        items.append({
            "id": idx,
            "input": description,
            "response_a": response_a,
            "response_b": response_b,
            "latency_a": round(latency_a, 2),
            "latency_b": round(latency_b, 2),
            "relevance_a": None,
            "relevance_b": None,
            "hallucination_a": None,
            "hallucination_b": None,
        })

    instructions = (
        "Заполните в каждом элементе items четыре поля (замените null на числа), затем сохраните файл и запустите: python evaluate_and_report.py\n\n"
        "Параметры:\n\n"
        "• relevance_a (число от 1 до 5) — релевантность ответа A: насколько пост соответствует описанию товара и насколько подходит как продающий пост для Telegram. 1 = совсем не подходит, 5 = отлично.\n\n"
        "• relevance_b (число от 1 до 5) — то же для ответа B.\n\n"
        "• hallucination_a (0 или 1) — есть ли в ответе A выдуманные факты о здоровье или эффектах продукта, которых нет в исходном описании. 0 = только факты из описания или общедоступные проверяемые данные; 1 = есть придуманные утверждения.\n\n"
        "• hallucination_b (0 или 1) — то же для ответа B.\n\n"
        "Рекомендуется оценивать слепо: смотрите на текст поста, не опираясь на то, какой вариант (A или B) вы считаете «новым»."
    )
    payload = {
        "meta": {
            "model": config.MODEL,
            "created_by": "run_ab_test.py",
            "instructions": instructions,
        },
        "items": items,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Готово. Результаты: {RESULTS_JSON}")
    print("Проставьте метрики (relevance_a/b, hallucination_a/b) и запустите: python evaluate_and_report.py")


if __name__ == "__main__":
    main()
