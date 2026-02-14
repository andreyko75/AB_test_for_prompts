# -*- coding: utf-8 -*-
"""
Шаг 2. Обработка ваших оценок и формирование отчёта.

Читает output/results.json. Ожидает, что вы уже проставили в каждом элементе items:
  relevance_a, relevance_b — число от 1 до 5;
  hallucination_a, hallucination_b — 0 или 1.

Считает метрики по вашим оценкам, при наличии OPENAI_API_KEY запрашивает оценку AI,
сравнивает совпадение и пишет output/report.txt и output/dataset.txt.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RESULTS_JSON = OUTPUT_DIR / "results.json"
REPORT_TXT = OUTPUT_DIR / "report.txt"
DATASET_TXT = OUTPUT_DIR / "dataset.txt"

# Промпт для AI-оценки (LLM-as-a-Judge)
EVAL_SYSTEM = """Ты — эксперт по оценке текстов. Оцени ответы по двум критериям.
Релевантность (1–5): насколько пост соответствует описанию товара и насколько он продающий для Telegram.
Галлюцинация (0 или 1): 0 — только факты из описания или общедоступные проверяемые данные; 1 — есть выдуманные утверждения о здоровье/эффектах.
Верни строго один JSON без markdown и комментариев, с ключами: relevance_a, relevance_b, hallucination_a, hallucination_b."""


def load_results() -> dict:
    """Загружает results.json."""
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def check_scores_filled(items: list[dict]) -> list[str]:
    """Проверяет, что во всех элементах проставлены оценки. Возвращает список id с пропусками."""
    missing = []
    for it in items:
        need = ("relevance_a", "relevance_b", "hallucination_a", "hallucination_b")
        for key in need:
            val = it.get(key)
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                missing.append(it["id"])
                break
    return list(dict.fromkeys(missing))


def compute_metrics(items: list[dict], prefix: str = "") -> dict:
    """Считает среднюю релевантность и долю галлюцинаций для A и B.
    prefix: '' для ваших оценок (relevance_a), 'ai_' для AI (ai_relevance_a).
    """
    n = len(items)
    ra, rb = f"{prefix}relevance_a", f"{prefix}relevance_b"
    ha, hb = f"{prefix}hallucination_a", f"{prefix}hallucination_b"
    rel_a = sum(float(it[ra]) for it in items) / n
    rel_b = sum(float(it[rb]) for it in items) / n
    hall_a = sum(int(it[ha]) for it in items) / n
    hall_b = sum(int(it[hb]) for it in items) / n
    return {
        "n": n,
        "relevance_a": round(rel_a, 2),
        "relevance_b": round(rel_b, 2),
        "relevance_diff": round(rel_b - rel_a, 2),
        "hallucination_rate_a": round(hall_a, 2),
        "hallucination_rate_b": round(hall_b, 2),
        "hallucination_diff": round(hall_b - hall_a, 2),
    }


def run_ai_evaluation(items: list[dict]) -> list[dict] | None:
    """Запрашивает у LLM оценку каждого примера. Возвращает items с полями ai_relevance_a/b, ai_hallucination_a/b или None при ошибке/отсутствии ключа."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    client = OpenAI()
    out = []
    for i, it in enumerate(items):
        inp = (it.get("input") or "")[:2000]
        resp_a = (it.get("response_a") or "")[:1500]
        resp_b = (it.get("response_b") or "")[:1500]
        user_msg = f"""Описание товара:\n{inp}\n\nОтвет A:\n{resp_a}\n\nОтвет B:\n{resp_b}\n\nОцени оба ответа и верни JSON: relevance_a, relevance_b (1-5), hallucination_a, hallucination_b (0 или 1)."""
        try:
            r = client.chat.completions.create(
                model=config.MODEL,
                temperature=0,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = (r.choices[0].message.content or "").strip()
            # Вытаскиваем JSON из ответа (могут быть markdown-блоки)
            m = re.search(r"\{[^{}]*\}", text)
            if m:
                data = json.loads(m.group())
                it_copy = dict(it)
                it_copy["ai_relevance_a"] = _clamp(int(data.get("relevance_a", 3)), 1, 5)
                it_copy["ai_relevance_b"] = _clamp(int(data.get("relevance_b", 3)), 1, 5)
                it_copy["ai_hallucination_a"] = 1 if data.get("hallucination_a") in (1, True, "1") else 0
                it_copy["ai_hallucination_b"] = 1 if data.get("hallucination_b") in (1, True, "1") else 0
                out.append(it_copy)
            else:
                out.append(it)
        except Exception:
            out.append(it)
        print(f"  AI оценка: {i+1}/{len(items)}")
    return out if out else None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def compute_agreement(items: list[dict]) -> dict:
    """Совпадение ваших оценок и AI по каждому полю: доля точного совпадения и средняя разница по релевантности."""
    n = len(items)
    exact = {"relevance_a": 0, "relevance_b": 0, "hallucination_a": 0, "hallucination_b": 0}
    diff_sum = {"relevance_a": 0, "relevance_b": 0}
    for it in items:
        for k in ("relevance_a", "relevance_b"):
            u, a = it.get(k), it.get(f"ai_{k}")
            if u is not None and a is not None:
                exact[k] += 1 if u == a else 0
                diff_sum[k] += abs(float(u) - float(a))
        for k in ("hallucination_a", "hallucination_b"):
            u, a = it.get(k), it.get(f"ai_{k}")
            if u is not None and a is not None:
                exact[k] += 1 if int(u) == int(a) else 0
    return {
        "relevance_a_match_pct": round(100 * exact["relevance_a"] / n, 1) if n else 0,
        "relevance_b_match_pct": round(100 * exact["relevance_b"] / n, 1) if n else 0,
        "hallucination_a_match_pct": round(100 * exact["hallucination_a"] / n, 1) if n else 0,
        "hallucination_b_match_pct": round(100 * exact["hallucination_b"] / n, 1) if n else 0,
        "relevance_a_mae": round(diff_sum["relevance_a"] / n, 2) if n else 0,
        "relevance_b_mae": round(diff_sum["relevance_b"] / n, 2) if n else 0,
    }


def write_report(metrics: dict, ai_metrics: dict | None = None, agreement: dict | None = None) -> None:
    """Пишет короткий текстовый отчёт (ваши оценки, при наличии — оценка AI и сравнение)."""
    mde = config.MDE_RELEVANCE
    guard_ok = metrics["hallucination_diff"] <= config.GUARD_HALLUCINATION_MAX_INCREASE
    primary_ok = metrics["relevance_diff"] >= mde

    lines = [
        "ОТЧЁТ ПО A/B-ТЕСТИРОВАНИЮ ПРОМПТОВ",
        "Посты LR для Telegram (на основе описаний товаров)",
        "",
        "1. ГИПОТЕЗА (SMART)",
        "Если добавить в промпт требование использовать медицинскую и нутрициологическую "
        "терминологию (действующие вещества, механизмы действия), то релевантность постов "
        f"для аудитории, интересующейся здоровьем, вырастет не менее чем на {mde} по шкале 1–5, "
        "при этом доля ответов с галлюцинациями не увеличится.",
        "",
        "2. МЕТРИКИ",
        f"Primary: релевантность (1–5). MDE = {mde}.",
        "Guard: доля ответов с галлюцинациями (утверждения о здоровье, которых нет в описании).",
        "",
        "3. РЕЗУЛЬТАТЫ (ваши субъективные оценки)",
        f"Размер выборки: {metrics['n']} запросов.",
        f"Средняя релевантность A: {metrics['relevance_a']}, B: {metrics['relevance_b']}, разница (B − A): {metrics['relevance_diff']}.",
        f"Доля галлюцинаций A: {metrics['hallucination_rate_a']}, B: {metrics['hallucination_rate_b']}.",
        "",
    ]
    if ai_metrics:
        lines.extend([
            "3b. РЕЗУЛЬТАТЫ (оценка AI, LLM-as-a-Judge)",
            f"Средняя релевантность A: {ai_metrics['relevance_a']}, B: {ai_metrics['relevance_b']}, разница (B − A): {ai_metrics['relevance_diff']}.",
            f"Доля галлюцинаций A: {ai_metrics['hallucination_rate_a']}, B: {ai_metrics['hallucination_rate_b']}.",
            "",
        ])
    lines.extend([
        "4. ВЫВОДЫ (по вашим оценкам)",
    ])
    if primary_ok:
        lines.append(f"Релевантность: улучшение не менее чем на MDE ({mde}) достигнуто. Вариант Б выигрывает по Primary-метрике.")
    else:
        lines.append(f"Релевантность: улучшение меньше MDE ({mde}). По Primary-метрике значимого улучшения нет.")
    if guard_ok:
        lines.append("Guard-метрика: доля галлюцинаций у Б не превышает долю у А. Вариант Б прошёл по guard-метрике.")
    else:
        lines.append("Guard-метрика: доля галлюцинаций у Б выросла. Вариант Б не прошёл по guard-метрике.")
    lines.append("")
    if primary_ok and guard_ok:
        lines.append("Решение: принимаю промпт Б в продакшен — есть измеримое улучшение релевантности без ухудшения безопасности (галлюцинации).")
    elif guard_ok:
        lines.append("Решение: по текущим критериям недостаточно оснований для перехода на Б; можно доработать промпт и повторить тест.")
    else:
        lines.append("Решение: вариант Б не принимаю — нарушение guard-метрики недопустимо.")
    lines.append("")

    if agreement:
        lines.extend([
            "5. СРАВНЕНИЕ: ваши оценки vs оценка AI",
            f"Точное совпадение по релевантности A: {agreement['relevance_a_match_pct']}%, по релевантности B: {agreement['relevance_b_match_pct']}%.",
            f"Точное совпадение по галлюцинациям A: {agreement['hallucination_a_match_pct']}%, по галлюцинациям B: {agreement['hallucination_b_match_pct']}%.",
            f"Средняя абсолютная разница по релевантности A: {agreement['relevance_a_mae']}, по B: {agreement['relevance_b_mae']} (чем меньше — тем ближе мнения).",
            "",
        ])

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_dataset_txt() -> None:
    """Формирует текстовый файл датасета для приложения к отчёту (скриншоты к ДЗ)."""
    path = Path(__file__).resolve().parent / "dataset" / "product_descriptions.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = ["ДАТАСЕТ ДЛЯ A/B-ТЕСТА (описания товаров LR)", ""]
    for item in data:
        lines.append(f"--- id: {item['id']} | {item.get('note', '')} ---")
        lines.append(item["description"])
        lines.append("")
    with open(DATASET_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    if not RESULTS_JSON.exists():
        print(f"Файл {RESULTS_JSON} не найден. Сначала выполните: python run_ab_test.py")
        return

    data = load_results()
    items = data.get("items", [])

    if not items:
        print("В results.json нет элементов items.")
        return

    missing = check_scores_filled(items)
    if missing:
        print(f"Не проставлены метрики для id: {', '.join(missing)}")
        print("Заполните в results.json: relevance_a, relevance_b (1–5), hallucination_a, hallucination_b (0 или 1).")
        return

    write_dataset_txt()
    metrics = compute_metrics(items)

    ai_metrics = None
    agreement = None
    if os.getenv("OPENAI_API_KEY"):
        print("Запрашиваю оценку AI (LLM-as-a-Judge)...")
        items_with_ai = run_ai_evaluation(items)
        valid_ai = [it for it in (items_with_ai or []) if it.get("ai_relevance_a") is not None]
        if valid_ai:
            ai_metrics = compute_metrics(valid_ai, prefix="ai_")
            agreement = compute_agreement(valid_ai)
        if not valid_ai:
            print("Оценка AI не получена (ошибки API или разбора ответов).")
    else:
        print("OPENAI_API_KEY не задан — сравнение с оценкой AI пропущено.")

    write_report(metrics, ai_metrics=ai_metrics, agreement=agreement)
    print(f"Отчёт: {REPORT_TXT}")
    print(f"Датасет (текст): {DATASET_TXT}")


if __name__ == "__main__":
    main()
