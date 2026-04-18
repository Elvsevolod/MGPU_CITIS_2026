"""
Шаг 1: Ручное обнаружение аномалий в датасете тестирования детей по истории.

Задание: выявить нарушения рекомендации о частоте прохождения тестирования
(не чаще 1 раза в 3 месяца), а также технические ошибки, логические противоречия
и некорректные/неполные записи.
"""

import csv
import re
import os
from collections import defaultdict, Counter
from datetime import datetime, date

# ─────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────
DATA_FILE = "hakaton.csv"
OUT_DIR = "results_manual"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("РУЧНОЕ ОБНАРУЖЕНИЕ АНОМАЛИЙ")
print("=" * 70)

with open(DATA_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    rows = list(reader)

print(f"Загружено записей: {len(rows)}\n")

TODAY = date(2026, 4, 17)  # дата актуальности анализа

# Накопители для всех аномалий
all_anomalies = []  # список словарей: {index, our_number, category, description}


def flag(row_idx, row, category, description):
    """Добавляет запись об аномалии в общий список."""
    all_anomalies.append({
        "index": row_idx + 2,           # номер строки в CSV (с учётом заголовка)
        "our_number": row["our_number"],
        "child": f"{row['last_name']} {row['first_name']} {row['middle_name']}".strip(),
        "category": category,
        "description": description,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 1. НАРУШЕНИЕ ЧАСТОТЫ ТЕСТИРОВАНИЯ (прямое требование задания)
#    Рекомендация: не чаще 1 раза в 3 месяца (90 дней).
#    Идентификатор ребёнка — комбинация фамилии, имени и даты рождения,
#    так как поле id_doc может содержать ошибки.
# ═══════════════════════════════════════════════════════════════════════════
print("1. Проверка частоты тестирования (правило ≥ 90 дней между тестами)...")

# Группируем все тесты по ребёнку
child_tests = defaultdict(list)  # ключ → список (дата, индекс строки)
for i, row in enumerate(rows):
    try:
        test_date = datetime.strptime(row["test_date"], "%Y-%m-%d").date()
        # Ключ: фамилия + имя + дата рождения — уникальный идентификатор ребёнка
        key = (row["last_name"].upper(), row["first_name"].upper(), row["bdate"])
        child_tests[key].append((test_date, i))
    except ValueError:
        pass  # Некорректная дата — обработается в другой проверке

freq_violations = 0
for key, tests in child_tests.items():
    tests_sorted = sorted(tests, key=lambda x: x[0])
    for j in range(1, len(tests_sorted)):
        prev_date, prev_idx = tests_sorted[j - 1]
        curr_date, curr_idx = tests_sorted[j]
        diff_days = (curr_date - prev_date).days
        if diff_days < 90:
            # Нарушение: интервал меньше 90 дней
            flag(curr_idx, rows[curr_idx], "ЧАСТОТА",
                 f"Тест через {diff_days} дн. после предыдущего "
                 f"({prev_date} → {curr_date}), норма ≥ 90 дн.")
            freq_violations += 1

print(f"   Найдено нарушений частоты: {freq_violations}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ОТРИЦАТЕЛЬНЫЕ / НЕЧИСЛОВЫЕ НОМЕРА ДОКУМЕНТОВ
#    Номер документа (id_doc) не может быть отрицательным числом — это
#    признак технической ошибки при вводе или выгрузке данных.
# ═══════════════════════════════════════════════════════════════════════════
print("2. Проверка номеров документов ребёнка и законного представителя...")

neg_child = 0
neg_guard = 0
for i, row in enumerate(rows):
    doc = row["id_doc"].strip()
    # Отрицательный id_doc у ребёнка
    if re.match(r"^-\d+$", doc):
        flag(i, row, "ДОКУМЕНТ_РЕБЁНОК",
             f"Отрицательный номер документа ребёнка: {doc}")
        neg_child += 1

    gdoc = row["guard_id_doc"].strip()
    # Отрицательный id_doc у законного представителя
    if re.match(r"^-\d+$", gdoc):
        flag(i, row, "ДОКУМЕНТ_ПРЕДСТАВИТЕЛЬ",
             f"Отрицательный номер документа представителя: {gdoc}")
        neg_guard += 1

print(f"   Отрицательный id_doc (ребёнок): {neg_child}")
print(f"   Отрицательный id_doc (представитель): {neg_guard}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 3. НЕСООТВЕТСТВИЕ КОДА ВАРИАНТА КЛАССУ РЕБЁНКА
#    Нормальный код варианта — числовая строка длиной 5–7 знаков.
#    По структуре кода: средние две цифры кодируют класс
#    (например, 80109 → класс 01 = 1, 90406 → класс 04 = 4).
#    Явно некорректные варианты: '0', '1', '10', '727.040501',
#    'УЧ 120309,ПЧ 120309', '08.09.2025' и т.д.
# ═══════════════════════════════════════════════════════════════════════════
print("3. Проверка кода варианта тестирования...")

bad_variant_fmt = 0
variant_class_mismatch = 0
for i, row in enumerate(rows):
    v = row["variant"].strip()
    cls = row["class"].strip()

    # Проверка: вариант должен быть числовой строкой из 5–7 цифр
    if not re.match(r"^\d{5,7}$", v):
        flag(i, row, "ВАРИАНТ_ФОРМАТ",
             f"Некорректный формат варианта: '{v}' (ожидается 5–7 цифр)")
        bad_variant_fmt += 1
        continue  # Дальнейшую проверку класса не делаем

    # Проверка соответствия варианта и класса.
    # Структура кода варианта: {год/серия}{класс(2 цифры)}{последовательность(2 цифры)}
    # Класс всегда закодирован в позициях [-4:-2] (4-я и 3-я с конца).
    # Примеры: 80109 → "01" → 1; 100507 → "05" → 5; 90406 → "04" → 4
    try:
        embedded_class = int(v[-4:-2])
        actual_class = int(cls)
        if embedded_class != actual_class:
            flag(i, row, "ВАРИАНТ_КЛАСС",
                 f"Вариант '{v}' кодирует класс {embedded_class}, "
                 f"но в записи класс {actual_class}")
            variant_class_mismatch += 1
    except (ValueError, IndexError):
        pass

print(f"   Некорректный формат варианта: {bad_variant_fmt}")
print(f"   Несоответствие варианта и класса: {variant_class_mismatch}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 4. НЕСООТВЕТСТВИЕ ВОЗРАСТА РЕБЁНКА И КЛАССА
#    Стандарт: класс N соответствует детям в возрасте N+5 … N+8 лет
#    (с учётом вариантов школьного возраста и возможного переноса).
#    Например, класс 1 → возраст 6–9 лет, класс 11 → возраст 16–19 лет.
# ═══════════════════════════════════════════════════════════════════════════
print("4. Проверка соответствия возраста ребёнка и класса...")

age_class_issues = 0
for i, row in enumerate(rows):
    try:
        bdate = datetime.strptime(row["bdate"], "%Y-%m-%d").date()
        test_date = datetime.strptime(row["test_date"], "%Y-%m-%d").date()
        cls = int(row["class"])

        # Возраст на момент тестирования в годах
        age = (test_date - bdate).days / 365.25

        # Ожидаемый диапазон возраста для класса
        age_min = cls + 5
        age_max = cls + 8

        if not (age_min <= age <= age_max):
            flag(i, row, "ВОЗРАСТ_КЛАСС",
                 f"Возраст {age:.1f} лет не соответствует классу {cls} "
                 f"(ожидается {age_min}–{age_max} лет)")
            age_class_issues += 1
    except (ValueError, TypeError):
        pass

print(f"   Несоответствий возраста и класса: {age_class_issues}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 5. СЛИШКОМ МОЛОДОЙ ЗАКОННЫЙ ПРЕДСТАВИТЕЛЬ
#    Законный представитель должен быть старше ребёнка не менее чем на 14 лет
#    (минимальный юридический возраст родителя по российскому законодательству).
# ═══════════════════════════════════════════════════════════════════════════
print("5. Проверка возраста законного представителя...")

young_guard_issues = 0
for i, row in enumerate(rows):
    try:
        child_bdate = datetime.strptime(row["bdate"], "%Y-%m-%d").date()
        guard_bdate = datetime.strptime(row["guard_bdate"], "%Y-%m-%d").date()

        # Разница в летах между ребёнком и представителем
        age_diff = (child_bdate - guard_bdate).days / 365.25

        if age_diff < 14:
            flag(i, row, "ПРЕДСТАВИТЕЛЬ_ВОЗРАСТ",
                 f"Представитель ({row['guard_last_name']}) старше ребёнка "
                 f"всего на {age_diff:.1f} лет (норма ≥ 14 лет)")
            young_guard_issues += 1
    except (ValueError, TypeError):
        pass

print(f"   Слишком молодых представителей: {young_guard_issues}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 6. НЕКОРРЕКТНЫЙ КЛАСС (вне диапазона 1–11)
#    Класс обучения в российской школе — от 1 до 11.
# ═══════════════════════════════════════════════════════════════════════════
print("6. Проверка допустимых значений класса (1–11)...")

bad_class = 0
for i, row in enumerate(rows):
    cls_str = row["class"].strip()
    try:
        cls = int(cls_str)
        if not (1 <= cls <= 11):
            flag(i, row, "КЛАСС_ДИАПАЗОН",
                 f"Класс вне диапазона 1–11: {cls}")
            bad_class += 1
    except ValueError:
        flag(i, row, "КЛАСС_ФОРМАТ",
             f"Нечисловое значение класса: '{cls_str}'")
        bad_class += 1

print(f"   Записей с некорректным классом: {bad_class}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 7. НЕКОРРЕКТНЫЙ РЕГИСТР ПОЛЯ RESULT
#    Поле result должно иметь единообразный вид.
#    Обнаружены варианты: 'Достаточный', 'ДОСТАТОЧНЫЙ', 'НЕдостаточный' и т.д.
# ═══════════════════════════════════════════════════════════════════════════
print("7. Проверка единообразия значения поля 'result'...")

VALID_RESULTS = {"достаточный", "недостаточный"}
bad_result = 0
result_counts = Counter()
for i, row in enumerate(rows):
    r = row["result"].strip()
    result_counts[r] += 1
    if r.lower() not in VALID_RESULTS:
        flag(i, row, "РЕЗУЛЬТАТ_ЗНАЧЕНИЕ",
             f"Неизвестное значение результата: '{r}'")
        bad_result += 1
    elif r != r.capitalize() and r.lower() in VALID_RESULTS:
        # Значение корректное по смыслу, но написано в неправильном регистре
        flag(i, row, "РЕЗУЛЬТАТ_РЕГИСТР",
             f"Некорректный регистр результата: '{r}' (ожидается 'Достаточный'/'Недостаточный')")
        bad_result += 1

print(f"   Записей с проблемой в поле result: {bad_result}")
print(f"   Распределение значений: {dict(result_counts)}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 8. ПРОВЕРКА КОРРЕКТНОСТИ ОГРН
#    ОГРН (основной государственный регистрационный номер) состоит из 13 цифр.
#    Если ОГРН содержит нецифровые символы или имеет другую длину — это ошибка.
# ═══════════════════════════════════════════════════════════════════════════
print("8. Проверка корректности ОГРН (напр. и площадки)...")

bad_ogrn = 0
for i, row in enumerate(rows):
    ogrn_n = row["ogrn_naprav"].strip()
    ogrn_a = row["ogrn_area"].strip()

    if not re.match(r"^\d{13}$", ogrn_n):
        flag(i, row, "ОГРН_НАПРАВИВШАЯ",
             f"Некорректный ОГРН направившей школы: '{ogrn_n}'")
        bad_ogrn += 1

    if not re.match(r"^\d{13}$", ogrn_a):
        flag(i, row, "ОГРН_ПЛОЩАДКА",
             f"Некорректный ОГРН площадки тестирования: '{ogrn_a}'")
        bad_ogrn += 1

print(f"   Записей с некорректным ОГРН: {bad_ogrn}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 9. ПУСТЫЕ ОБЯЗАТЕЛЬНЫЕ ПОЛЯ
#    ФИО ребёнка и представителя, даты рождения, номер результата — обязательны.
#    Отчество может быть пустым (иностранные граждане и т.д.).
# ═══════════════════════════════════════════════════════════════════════════
print("9. Проверка обязательных полей на пустые значения...")

REQUIRED_FIELDS = [
    "last_name", "first_name", "bdate", "gender", "id_doc",
    "guard_last_name", "guard_first_name", "guard_bdate",
    "our_number", "ogrn_naprav", "name_naprav",
    "ogrn_area", "name_area", "variant", "class", "test_date", "result",
]
empty_required = 0
for i, row in enumerate(rows):
    for field in REQUIRED_FIELDS:
        if not row[field].strip():
            flag(i, row, "ПУСТОЕ_ПОЛЕ",
                 f"Пустое обязательное поле: '{field}'")
            empty_required += 1

print(f"   Всего пустых обязательных полей: {empty_required}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 10. ДАТА РОЖДЕНИЯ РЕБЁНКА В БУДУЩЕМ ИЛИ СЛИШКОМ ДАВНО
#     Дата рождения не может быть в будущем.
#     Ребёнок не может быть старше 20 лет (нет смысла тестировать взрослых).
#     Ребёнок не может быть моложе 5 лет на дату тестирования.
# ═══════════════════════════════════════════════════════════════════════════
print("10. Проверка дат рождения ребёнка...")

bad_bdate = 0
for i, row in enumerate(rows):
    try:
        bdate = datetime.strptime(row["bdate"], "%Y-%m-%d").date()
        test_dt = datetime.strptime(row["test_date"], "%Y-%m-%d").date()
        age_at_test = (test_dt - bdate).days / 365.25

        if bdate > TODAY:
            flag(i, row, "ДАТА_РОЖДЕНИЯ",
                 f"Дата рождения в будущем: {bdate}")
            bad_bdate += 1
        elif age_at_test < 5:
            flag(i, row, "ДАТА_РОЖДЕНИЯ",
                 f"Ребёнок слишком мал на дату теста: {age_at_test:.1f} лет")
            bad_bdate += 1
        elif age_at_test > 20:
            flag(i, row, "ДАТА_РОЖДЕНИЯ",
                 f"Возраст на дату теста слишком большой: {age_at_test:.1f} лет")
            bad_bdate += 1
    except ValueError:
        flag(i, row, "ДАТА_РОЖДЕНИЯ",
             f"Некорректный формат даты рождения: '{row['bdate']}'")
        bad_bdate += 1

print(f"   Записей с аномальной датой рождения: {bad_bdate}\n")


# ═══════════════════════════════════════════════════════════════════════════
# ИТОГОВАЯ СТАТИСТИКА
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("ИТОГОВАЯ СТАТИСТИКА АНОМАЛИЙ")
print("=" * 70)

category_counts = Counter(a["category"] for a in all_anomalies)
# Уникальные записи (одна запись может иметь несколько аномалий)
unique_records = len(set(a["our_number"] for a in all_anomalies))

print(f"Всего флагов аномалий: {len(all_anomalies)}")
print(f"Уникальных записей с хотя бы одной аномалией: {unique_records}")
print(f"Доля аномальных записей: {unique_records / len(rows) * 100:.1f}%\n")
print("По категориям:")
for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<30} {cnt:>6}")


# ═══════════════════════════════════════════════════════════════════════════
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ═══════════════════════════════════════════════════════════════════════════
out_path = os.path.join(OUT_DIR, "manual_anomalies.csv")
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "our_number", "child", "category", "description"])
    writer.writeheader()
    writer.writerows(all_anomalies)

print(f"\nРезультаты сохранены: {out_path}")

# Также сохраняем сводку по категориям
summary_path = os.path.join(OUT_DIR, "manual_anomalies_summary.csv")
with open(summary_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "count"])
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        writer.writerow([cat, cnt])

print(f"Сводка по категориям: {summary_path}")
