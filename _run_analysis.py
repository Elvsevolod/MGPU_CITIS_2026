"""
Проверочный скрипт — та же логика, что в ноутбуке full_analysis.ipynb.
Запускается отдельно для проверки корректности и сбора статистики.
"""
import pandas as pd
import numpy as np
import re, os
from datetime import date, datetime
from collections import defaultdict, Counter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

TODAY = date(2026, 4, 17)
OUT_DIR = 'results_notebook'
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv('hakaton.csv', sep=';', dtype=str, keep_default_na=False).fillna('')
print(f'Загружено: {len(df)} записей\n')

df['anomaly_flags'] = ''
df['fixes_applied'] = ''
anomalies = []
fixes_log = []

def flag(idx, category, description):
    existing = df.at[idx, 'anomaly_flags']
    df.at[idx, 'anomaly_flags'] = (existing + '|' + category).lstrip('|')
    anomalies.append({'row_idx': idx, 'our_number': df.at[idx, 'our_number'],
                      'child': f"{df.at[idx,'last_name']} {df.at[idx,'first_name']}",
                      'category': category, 'description': description})

def fix_log(idx, field, old_val, new_val, reason):
    existing = df.at[idx, 'fixes_applied']
    df.at[idx, 'fixes_applied'] = (existing + '|' + field).lstrip('|')
    fixes_log.append({'row_idx': idx, 'our_number': df.at[idx, 'our_number'],
                      'field': field, 'old_value': old_val, 'new_value': new_val, 'reason': reason})

# --- R1: result normalization ---
fixed_result = 0
for idx, row in df.iterrows():
    r = row['result'].strip()
    r_up = r.upper()
    if r_up == 'ДОСТАТОЧНЫЙ' or r_up == 'НЕДОСТАТОЧНЫЙ':
        continue
    if 'ДОСТАТОЧН' in r_up:
        normalized = 'Недостаточный' if ('НЕ' in r_up or r_up.startswith('НЕД')) else 'Достаточный'
        if r != normalized:
            fix_log(idx, 'result', r, normalized, f'Регистр: "{r}"→"{normalized}"')
            df.at[idx, 'result'] = normalized
            fixed_result += 1
    else:
        flag(idx, 'РЕЗУЛЬТАТ_ЗНАЧЕНИЕ', f'Неизвестное result: "{r}"')
print(f'R1 result fixes: {fixed_result}')

# --- R2: id_doc normalization ---
fixed_doc = 0; invalid_doc = 0
for idx, row in df.iterrows():
    orig = row['id_doc'].strip()
    if not orig:
        flag(idx, 'ДОКУМЕНТ_РЕБЁНОК', 'Пустое id_doc'); invalid_doc += 1; continue
    cleaned = re.sub(r'[^0-9-]', '', orig)
    cleaned = re.sub(r'(?<!^)-', '', cleaned)
    try:
        final = str(abs(int(cleaned)))
    except (ValueError, OverflowError):
        flag(idx, 'ДОКУМЕНТ_РЕБЁНОК', f'Нечисловой id_doc: "{orig}"'); invalid_doc += 1; continue
    if final != orig:
        fix_log(idx, 'id_doc', orig, final, 'Очистка/abs')
        df.at[idx, 'id_doc'] = final; fixed_doc += 1
print(f'R2 id_doc fixes: {fixed_doc}, invalid: {invalid_doc}')

# --- R3: guard_id_doc normalization ---
fixed_gdoc = 0; invalid_gdoc = 0
for idx, row in df.iterrows():
    orig = row['guard_id_doc'].strip()
    if not orig:
        flag(idx, 'ДОКУМЕНТ_ПРЕДСТАВИТЕЛЬ', 'Пустое guard_id_doc'); invalid_gdoc += 1; continue
    cleaned = re.sub(r'[^0-9-]', '', orig)
    cleaned = re.sub(r'(?<!^)-', '', cleaned)
    try:
        final = str(abs(int(cleaned)))
    except (ValueError, OverflowError):
        flag(idx, 'ДОКУМЕНТ_ПРЕДСТАВИТЕЛЬ', f'Нечисловой guard_id_doc: "{orig}"'); invalid_gdoc += 1; continue
    if final != orig:
        fix_log(idx, 'guard_id_doc', orig, final, 'Очистка/abs')
        df.at[idx, 'guard_id_doc'] = final; fixed_gdoc += 1
print(f'R3 guard_id_doc fixes: {fixed_gdoc}, invalid: {invalid_gdoc}')

# --- R4: id_doc == guard_id_doc ---
same_doc = 0
for idx, row in df.iterrows():
    doc, gdoc = row['id_doc'].strip(), row['guard_id_doc'].strip()
    if doc and gdoc and doc == gdoc:
        flag(idx, 'ДОКУМЕНТ_СОВПАДЕНИЕ', f'id_doc==guard_id_doc={doc}'); same_doc += 1
print(f'R4 same_doc: {same_doc}')

# --- R5: empty required fields ---
REQUIRED = ['last_name','first_name','bdate','gender','id_doc','guard_last_name',
            'guard_first_name','guard_bdate','our_number','ogrn_naprav','name_naprav',
            'ogrn_area','name_area','variant','class','test_date','result']
empty_count = 0; field_stats = Counter()
for idx, row in df.iterrows():
    for field in REQUIRED:
        if not row[field].strip():
            flag(idx, 'ПУСТОЕ_ПОЛЕ', f'Пустое: "{field}"')
            field_stats[field] += 1; empty_count += 1
print(f'R5 empty fields: {empty_count}, fields: {dict(field_stats.most_common(5))}')

# --- R6: OGRN ---
fixed_ogrn = 0; bad_ogrn = 0
for idx, row in df.iterrows():
    for field, cat in [('ogrn_naprav','ОГРН_НАПРАВИВШАЯ'),('ogrn_area','ОГРН_ПЛОЩАДКА')]:
        orig = row[field].strip()
        cleaned = re.sub(r'\D', '', orig)
        if re.match(r'^\d{13}$', cleaned):
            if cleaned != orig:
                fix_log(idx, field, orig, cleaned, 'Убраны нецифровые символы ОГРН')
                df.at[idx, field] = cleaned; fixed_ogrn += 1
        else:
            flag(idx, cat, f'Некорректный ОГРН ({field}): "{orig}" len={len(cleaned)}'); bad_ogrn += 1
print(f'R6 OGRN fixes: {fixed_ogrn}, bad: {bad_ogrn}')

# --- R7: class format/range ---
bad_class = 0
for idx, row in df.iterrows():
    cls_str = row['class'].strip()
    try:
        cls = int(cls_str)
        if not (1 <= cls <= 11):
            flag(idx, 'КЛАСС_ДИАПАЗОН', f'Класс вне 1–11: {cls}'); bad_class += 1
    except ValueError:
        flag(idx, 'КЛАСС_ФОРМАТ', f'Нечисловой класс: "{cls_str}"'); bad_class += 1
print(f'R7 bad class: {bad_class}')

# --- R8: variant format/class match ---
bad_fmt = 0; bad_cls_match = 0
for idx, row in df.iterrows():
    v, cls = row['variant'].strip(), row['class'].strip()
    if not re.match(r'^\d{5,7}$', v):
        flag(idx, 'ВАРИАНТ_ФОРМАТ', f'Некорректный вариант: "{v}"'); bad_fmt += 1; continue
    try:
        embedded, actual = int(v[-4:-2]), int(cls)
        if embedded != actual:
            flag(idx, 'ВАРИАНТ_КЛАСС', f'Вариант "{v}" кодирует {embedded}, class={actual}'); bad_cls_match += 1
    except (ValueError, IndexError):
        pass
print(f'R8 variant fmt: {bad_fmt}, class mismatch: {bad_cls_match}')

# --- R9: bdate checks ---
bad_bdate = 0; too_young = 0
for idx, row in df.iterrows():
    try:
        bdate = datetime.strptime(row['bdate'], '%Y-%m-%d').date()
        tdate = datetime.strptime(row['test_date'], '%Y-%m-%d').date()
    except ValueError:
        flag(idx, 'ДАТА_РОЖДЕНИЯ', f'Некорректная дата'); bad_bdate += 1; continue
    age = (tdate - bdate).days / 365.25
    if bdate > TODAY:
        flag(idx, 'ДАТА_РОЖДЕНИЯ', f'bdate в будущем: {bdate}'); bad_bdate += 1
    elif bdate > tdate:
        flag(idx, 'ДАТА_РОЖДЕНИЯ', f'bdate > test_date'); bad_bdate += 1
    elif age < 5:
        flag(idx, 'ДАТА_РОЖДЕНИЯ', f'Возраст {age:.1f} < 5'); bad_bdate += 1
    elif age > 20:
        flag(idx, 'ДАТА_РОЖДЕНИЯ', f'Возраст {age:.1f} > 20'); bad_bdate += 1
    else:
        sep1 = date(tdate.year, 9, 1)
        if (sep1 - bdate).days / 365.25 < 6.5:
            flag(idx, 'ДАТА_РОЖДЕНИЯ', f'Возраст на 1 сентября < 6.5'); too_young += 1
print(f'R9 bad bdate: {bad_bdate}, too young sep1: {too_young}')

# --- R10: age-class ---
age_cls = 0
for idx, row in df.iterrows():
    try:
        bdate = datetime.strptime(row['bdate'], '%Y-%m-%d').date()
        tdate = datetime.strptime(row['test_date'], '%Y-%m-%d').date()
        cls   = int(row['class'])
        age   = (tdate - bdate).days / 365.25
        if not (cls + 5 <= age <= cls + 8):
            flag(idx, 'ВОЗРАСТ_КЛАСС', f'Возраст {age:.1f} не соответствует классу {cls}'); age_cls += 1
    except (ValueError, TypeError):
        pass
print(f'R10 age-class: {age_cls}')

# --- R11: guardian age ---
young_guard = 0
for idx, row in df.iterrows():
    try:
        cb = datetime.strptime(row['bdate'], '%Y-%m-%d').date()
        gb = datetime.strptime(row['guard_bdate'], '%Y-%m-%d').date()
        if (cb - gb).days / 365.25 < 14:
            flag(idx, 'ПРЕДСТАВИТЕЛЬ_ВОЗРАСТ', f'Представитель старше ребёнка менее чем на 14 лет'); young_guard += 1
    except ValueError:
        pass
print(f'R11 young guard: {young_guard}')

# --- R12: frequency ---
child_tests = defaultdict(list)
for idx, row in df.iterrows():
    try:
        td = datetime.strptime(row['test_date'], '%Y-%m-%d').date()
        child_tests[(row['last_name'].upper(), row['first_name'].upper(), row['bdate'])].append((td, idx))
    except ValueError:
        pass
freq_viol = 0
for key, tests in child_tests.items():
    for j, (d2, i2) in enumerate(sorted(tests)):
        if j == 0: d1 = d2; continue
        diff = (d2 - d1).days
        if diff < 90:
            flag(i2, 'ЧАСТОТА', f'Тест через {diff} дн. (< 90)'); freq_viol += 1
        d1 = d2
print(f'R12 freq: {freq_viol}')

# --- R13: gender-name (skip if pymorphy2 broken) ---
gender_errors = 0
try:
    import pymorphy2
    FEMALE_EX = {'антонина','евгения','серафима','валерия','эльмира','анжела','варвара',
                 'владислава','яна','алефтина','ольга','елена','камия','таисия','александра','римма'}
    MALE_EX   = {'михаил','ивлиане','илтизир'}
    morph = pymorphy2.MorphAnalyzer()
    for idx, row in df.iterrows():
        gender_val, name = row['gender'].strip(), row['first_name'].strip()
        if not gender_val or not name:
            continue
        expected = 'femn' if gender_val == 'Ж' else 'masc'
        nl = name.lower()
        if nl in FEMALE_EX: actual = 'femn'
        elif nl in MALE_EX:  actual = 'masc'
        else:
            parsed = morph.parse(name)[0]
            g = parsed.tag.grammemes
            actual = 'femn' if 'femn' in g else ('masc' if 'masc' in g else 'unknown')
        if actual != 'unknown' and actual != expected:
            flag(idx, 'ПОЛ_ИМЯ', f'Имя "{name}" — пол не совпадает с gender={gender_val}')
            gender_errors += 1
    print(f'R13 gender-name: {gender_errors}')
except Exception as e:
    print(f'R13 skipped: {e}')

# ─── Summary ────────────────────────────────────────────────────────
n_total   = len(df)
n_fixed   = (df['fixes_applied'] != '').sum()
n_flagged = (df['anomaly_flags'] != '').sum()
n_clean   = n_total - n_flagged

print('\n' + '='*55)
print('ИТОГИ РУЧНОЙ ПРОВЕРКИ')
print('='*55)
print(f'Всего записей:            {n_total}')
print(f'Исправлено (FIX):         {n_fixed} ({n_fixed/n_total*100:.1f}%)')
print(f'Помечено как аномалия:    {n_flagged} ({n_flagged/n_total*100:.1f}%)')
print(f'Чистых записей:           {n_clean} ({n_clean/n_total*100:.1f}%)')
print()
cat_counts = Counter(a['category'] for a in anomalies)
print('По категориям:')
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f'  {cat:<35} {cnt:>6}')

# Save reports
pd.DataFrame(anomalies).to_csv(f'{OUT_DIR}/manual_anomalies.csv', index=False, encoding='utf-8')
pd.DataFrame(fixes_log).to_csv(f'{OUT_DIR}/fixes_log.csv', index=False, encoding='utf-8')

# ─── Isolation Forest ────────────────────────────────────────────────
print('\n' + '='*55)
print('ISOLATION FOREST')
print('='*55)

df_clean = df[df['anomaly_flags'] == ''].copy().reset_index(drop=True)
print(f'Чистых записей для обучения: {len(df_clean)}')

df_clean['bdate_dt']       = pd.to_datetime(df_clean['bdate'], errors='coerce')
df_clean['test_date_dt']   = pd.to_datetime(df_clean['test_date'], errors='coerce')
df_clean['guard_bdate_dt'] = pd.to_datetime(df_clean['guard_bdate'], errors='coerce')
df_clean['class_num']      = pd.to_numeric(df_clean['class'], errors='coerce')
df_clean['age_at_test']    = ((df_clean['test_date_dt'] - df_clean['bdate_dt']).dt.days / 365.25)
df_clean['age_class_dev']  = df_clean['age_at_test'] - (df_clean['class_num'] + 6.5)
df_clean['guard_age_diff'] = ((df_clean['bdate_dt'] - df_clean['guard_bdate_dt']).dt.days / 365.25)
df_clean['result_code']    = (~df_clean['result'].str.upper().str.strip().str.contains('НЕДОСТ', na=False)).astype(int)
df_clean['gender_code']    = (df_clean['gender'].str.strip() == 'М').astype(int)
df_clean['test_month']     = df_clean['test_date_dt'].dt.month
df_clean['test_year']      = df_clean['test_date_dt'].dt.year

key_col = df_clean['last_name'].str.upper() + '|' + df_clean['first_name'].str.upper() + '|' + df_clean['bdate']
df_clean['tests_per_child'] = key_col.map(key_col.value_counts())
df_clean = df_clean.sort_values(['last_name','first_name','bdate','test_date_dt'])
df_clean['days_since_prev'] = df_clean.groupby([df_clean['last_name'].str.upper(),
                                                  df_clean['first_name'].str.upper(),
                                                  'bdate'])['test_date_dt'].diff().dt.days.fillna(-1)
df_clean = df_clean.reset_index(drop=True)

FEATURES = ['age_at_test','class_num','age_class_dev','guard_age_diff',
            'result_code','gender_code','test_month','test_year',
            'tests_per_child','days_since_prev']

df_model = df_clean.dropna(subset=FEATURES).copy()
print(f'Записей с полными признаками: {len(df_model)}')

X_scaled = StandardScaler().fit_transform(df_model[FEATURES].values)
iforest = IsolationForest(n_estimators=200, contamination=0.05, n_jobs=-1, random_state=42)
iforest.fit(X_scaled)
preds  = iforest.predict(X_scaled)
scores = iforest.decision_function(X_scaled)
df_model['iforest_pred']  = preds
df_model['anomaly_score'] = scores
df_model['is_ml_anomaly'] = (preds == -1).astype(int)

n_ml = (preds == -1).sum()
print(f'ML-аномалий: {n_ml} ({n_ml/len(df_model)*100:.1f}%)')

# Compare features for norm vs anomaly
df_ma = df_model[df_model['is_ml_anomaly']==1]
df_mn = df_model[df_model['is_ml_anomaly']==0]
print('\nСредние значения признаков:')
print(f"{'Признак':<20} {'Норма':>10} {'Аномалия':>10} {'Разница':>10}")
for f in FEATURES:
    n_mean = df_mn[f].mean()
    a_mean = df_ma[f].mean()
    print(f'{f:<20} {n_mean:>10.2f} {a_mean:>10.2f} {a_mean-n_mean:>10.2f}')

# Top anomalies
print('\nТоп-10 ML аномалий:')
print(df_ma.sort_values('anomaly_score').head(10)[
    ['our_number','last_name','first_name','class','test_date','anomaly_score','age_at_test','tests_per_child']
].to_string())

# Save
df_model.to_csv(f'{OUT_DIR}/full_scored.csv', sep=';', index=False, encoding='utf-8')
df_ma.to_csv(f'{OUT_DIR}/ml_anomalies.csv', sep=';', index=False, encoding='utf-8')

summary = dict(total=n_total, fixed=int(n_fixed), manual_anom=int(n_flagged),
               clean_for_ml=len(df_model), ml_anom=int(n_ml),
               truly_clean=int(len(df_model)-n_ml))
pd.DataFrame([summary]).to_csv(f'{OUT_DIR}/summary.csv', index=False, encoding='utf-8')

print('\n' + '='*55)
print('ФИНАЛЬНАЯ СВОДКА')
print('='*55)
print(f"Всего записей в датасете:        {summary['total']}")
print(f"Исправлено (FIX-правила):        {summary['fixed']}")
print(f"Ручных аномалий (FLAG):          {summary['manual_anom']}")
print(f"Подано на Isolation Forest:      {summary['clean_for_ml']}")
print(f"Новых ML-аномалий:               {summary['ml_anom']}")
print(f"Полностью чистых записей:        {summary['truly_clean']}")
