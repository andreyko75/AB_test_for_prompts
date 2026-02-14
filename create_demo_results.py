# -*- coding: utf-8 -*-
"""
Демо: создаёт output/results.json без вызова API (несколько записей с заглушками).

Нужен только чтобы проверить шаг 2: откройте results.json, проставьте
relevance_a/b и hallucination_a/b вручную, затем запустите evaluate_and_report.py.
Для полного A/B-теста используйте run_ab_test.py (нужен OPENAI_API_KEY в .env).
"""

import json
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "product_descriptions.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RESULTS_JSON = OUTPUT_DIR / "results.json"


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    # Берём только первые 5 записей для демо
    subset = dataset[:5]
    items = []
    for item in subset:
        desc = item["description"]
        short = (desc[:60] + "...") if len(desc) > 60 else desc
        items.append({
            "id": item["id"],
            "input": desc,
            "response_a": f"[Демо-ответ A] Пост по описанию: {short}",
            "response_b": f"[Демо-ответ B] Пост с мед. терминами: {short}",
            "latency_a": 1.0,
            "latency_b": 1.0,
            "relevance_a": None,
            "relevance_b": None,
            "hallucination_a": None,
            "hallucination_b": None,
        })
    instructions = (
        "Заполните в каждом элементе items четыре поля (замените null на числа), затем сохраните файл и запустите: python evaluate_and_report.py\n\n"
        "Параметры:\n\n"
        "• relevance_a (1–5) — релевантность ответа A: соответствие описанию товара и пригодность как продающий пост для Telegram. 1 = не подходит, 5 = отлично.\n\n"
        "• relevance_b (1–5) — то же для ответа B.\n\n"
        "• hallucination_a (0 или 1) — выдуманные факты о здоровье в ответе A. 0 = только факты из описания/проверяемые данные, 1 = есть придуманное.\n\n"
        "• hallucination_b (0 или 1) — то же для ответа B.\n\n"
        "Оценивайте по смыслу текста, не по метке A/B."
    )
    payload = {
        "meta": {
            "model": "gpt-4o-mini",
            "created_by": "create_demo_results.py",
            "instructions": instructions,
        },
        "items": items,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Демо results.json записан: {RESULTS_JSON}")
    print("Проставьте метрики и запустите: python evaluate_and_report.py")


if __name__ == "__main__":
    main()
