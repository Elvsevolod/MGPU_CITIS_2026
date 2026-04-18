"""
Полный прогон: все 25 628 строк hakaton.csv через GigaChat Embeddings + t-SNE.

Кэш переиспользуется из results_embeddings/ — уже посчитанные тексты не пересчитываются.
Результаты сохраняются в results_embeddings_full/.

Запуск: python3 step3_full_run.py
"""

import os, json, time, uuid, warnings, requests
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "results_embeddings_full", "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ── Конфигурация ──
GIGACHAT_KEY  = "ZTY5NWJiYTAtN2RlMC00YWZhLWI5YzUtZDcxNGNhOTUzMTkyOjlhMWVkYjcyLTdmMTctNDhjNS1iNjY1LTBjMzcwNDE1YjE4NQ=="
DATA_FILE     = "hakaton.csv"
MANUAL_FLAGS  = "results_manual/manual_anomalies.csv"
IFOREST_FILE  = "results_iforest/iforest_anomalies.csv"
CACHE_FILE    = "results_embeddings/embeddings_cache.json"   # общий кэш
OUTPUT_DIR    = "results_embeddings_full"
BATCH_SIZE    = 8
TSNE_PERP     = 50    # для большой выборки perplexity побольше
TSNE_ITER     = 1000
PCA_DIM       = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Auth ──
class GigaChatAuth:
    def __init__(self, key):
        self.key = key
        self._token = None
        self._exp = 0

    def get_token(self):
        if self._token and time.time() < self._exp - 120:
            return self._token
        r = requests.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Accept": "application/json",
                     "RqUID": str(uuid.uuid4()),
                     "Authorization": f"Basic {self.key}"},
            data={"scope": "GIGACHAT_API_PERS"},
            verify=False, timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        self._token = d["access_token"]
        self._exp = d.get("expires_at", (time.time() + 1800) * 1000) / 1000.0
        return self._token

auth = GigaChatAuth(GIGACHAT_KEY)

def embed_batch(texts):
    r = requests.post(
        "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {auth.get_token()}"},
        json={"model": "Embeddings", "input": texts},
        verify=False, timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    return [item["embedding"] for item in r.json()["data"]]

def row_to_text(row):
    child = " ".join(filter(None, [row.get("last_name",""), row.get("first_name",""), row.get("middle_name","")])).strip()
    guard = " ".join(filter(None, [row.get("guard_last_name",""), row.get("guard_first_name",""), row.get("guard_middle_name","")])).strip()
    return (
        f"Ребёнок: {child}, дата рождения: {row.get('bdate','')}, пол: {row.get('gender','')}, "
        f"документ: {row.get('id_doc','')}. "
        f"Представитель: {guard}, дата рождения: {row.get('guard_bdate','')}, пол: {row.get('guard_gender','')}. "
        f"Номер результата: {row.get('our_number','')}. "
        f"Школа направляющая: {row.get('name_naprav','')}. "
        f"Площадка тестирования: {row.get('name_area','')}. "
        f"Класс: {row.get('class','')}, вариант: {row.get('variant','')}, "
        f"дата теста: {row.get('test_date','')}, результат: {row.get('result','')}."
    )

CAT_COLOR = {
    "Норма":                  "#54A24B",
    "ML аномалия":            "#9467BD",
    "ЧАСТОТА":                "#D62728",
    "ДОКУМЕНТ_РЕБЁНОК":       "#FF7F0E",
    "ВАРИАНТ_ФОРМАТ":         "#BCBD22",
    "ВОЗРАСТ_КЛАСС":          "#1F77B4",
    "РЕЗУЛЬТАТ_РЕГИСТР":      "#17BECF",
    "ВАРИАНТ_КЛАСС":          "#8C564B",
    "Прочие ручные":          "#E377C2",
}

MONTH_RU = {1:"Янв",2:"Фев",3:"Мар",4:"Апр",5:"Май",6:"Июн",
            7:"Июл",8:"Авг",9:"Сен",10:"Окт",11:"Ноя",12:"Дек"}

# ════════════════════════════════════════════════════════════
print("=" * 70)
print("ПОЛНЫЙ ПРОГОН: ВСЕ СТРОКИ ДАТАСЕТА")
print("=" * 70)

# ── 1. Загрузка ──
print("\n[1/6] Загрузка датасета...")
df = pd.read_csv(DATA_FILE, sep=";", dtype=str, keep_default_na=False)
print(f"  Записей: {len(df)}")

# ── 2. Тексты ──
print("[2/6] Преобразование строк в тексты...")
texts = [row_to_text(row) for _, row in df.iterrows()]
print(f"  Готово: {len(texts)} текстов")

# ── 3. Эмбеддинги с кэшем ──
print("[3/6] Эмбеддинги через GigaChat...")
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"  Загружен кэш: {len(cache)} текстов")

missing = [i for i, t in enumerate(texts) if t not in cache]
print(f"  В кэше: {len(texts) - len(missing)} / {len(texts)}")
print(f"  Нужно API-запросов: {(len(missing) + BATCH_SIZE - 1) // BATCH_SIZE} батчей × {BATCH_SIZE}")

t0 = time.time()
n_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE

for b, start in enumerate(range(0, len(missing), BATCH_SIZE)):
    batch_idx = missing[start:start + BATCH_SIZE]
    batch_texts = [texts[i] for i in batch_idx]

    for attempt in range(3):
        try:
            embs = embed_batch(batch_texts)
            for i, emb in zip(batch_idx, embs):
                cache[texts[i]] = emb
            break
        except Exception as e:
            print(f"  ⚠ батч {b+1} попытка {attempt+1}: {e}")
            time.sleep(2 ** (attempt + 1))

    # Кэш: сохраняем каждые 50 батчей
    if b % 50 == 0:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    # Прогресс каждые 100 батчей
    if (b + 1) % 100 == 0 or (b + 1) == n_batches:
        elapsed = time.time() - t0
        speed = (b + 1) / elapsed if elapsed > 0 else 0
        eta = (n_batches - b - 1) / speed if speed > 0 else 0
        print(f"  [{b+1}/{n_batches}] {(b+1)/n_batches*100:.0f}%  "
              f"elapsed={elapsed/60:.1f} мин  ETA={eta/60:.1f} мин  "
              f"кэш={len(cache)}")

# Финальное сохранение кэша
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)
print(f"  Кэш сохранён: {CACHE_FILE}  ({len(cache)} записей)")

embeddings = np.array([cache[t] for t in texts], dtype=np.float32)
print(f"  Матрица: {embeddings.shape}")

# Сохраняем
np.save(os.path.join(OUTPUT_DIR, "embeddings_full.npy"), embeddings)
df.to_csv(os.path.join(OUTPUT_DIR, "rows_full.csv"), index=False, sep=";")
print(f"  Эмбеддинги сохранены: {OUTPUT_DIR}/embeddings_full.npy")

# ── 4. Метки ──
print("[4/6] Метки аномальности...")
manual = pd.read_csv(MANUAL_FLAGS, dtype=str)
iforest = pd.read_csv(IFOREST_FILE, sep=";", dtype=str)
manual_set  = set(manual["our_number"])
iforest_set = set(iforest["our_number"])
manual_cat  = manual.groupby("our_number")["category"].apply(
    lambda x: x.value_counts().index[0]).to_dict()

labels = []
for _, row in df.iterrows():
    our = row["our_number"]
    if our in manual_set:
        cat = manual_cat.get(our, "Прочие ручные")
        labels.append(cat if cat in CAT_COLOR else "Прочие ручные")
    elif our in iforest_set:
        labels.append("ML аномалия")
    else:
        labels.append("Норма")

labels = np.array(labels)
cats = [c for c in CAT_COLOR if (labels == c).any()]
for cat in cats:
    print(f"  {cat}: {(labels==cat).sum()}")

# ── 5. Снижение размерности ──
print(f"[5/6] PCA({PCA_DIM}D) → t-SNE 2D...")
n_pca = min(PCA_DIM, embeddings.shape[0]-1, embeddings.shape[1])
pca = PCA(n_components=n_pca, random_state=42)
emb_pca = pca.fit_transform(embeddings)
print(f"  PCA объяснённая дисперсия: {pca.explained_variance_ratio_.sum()*100:.1f}%")

perp = min(TSNE_PERP, len(df) // 4)
print(f"  t-SNE 2D (perplexity={perp}, iter={TSNE_ITER})  —  может занять 5–10 мин...")
tsne2 = TSNE(n_components=2, perplexity=perp, max_iter=TSNE_ITER,
             random_state=42, init="pca", learning_rate="auto", verbose=1)
coords_2d = tsne2.fit_transform(emb_pca)
np.save(os.path.join(OUTPUT_DIR, "tsne_2d_full.npy"), coords_2d)

print("  t-SNE 3D...")
tsne3 = TSNE(n_components=3, perplexity=perp, max_iter=TSNE_ITER,
             random_state=42, init="pca", learning_rate="auto", verbose=0)
coords_3d = tsne3.fit_transform(emb_pca)
np.save(os.path.join(OUTPUT_DIR, "tsne_3d_full.npy"), coords_3d)

# ── 6. Графики ──
print("[6/6] Визуализация...")

def plot2d(coords, labels, cats, title, fname, figsize=(16, 11)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")
    draw_order = [c for c in ["Норма","ML аномалия"] if c in cats]
    draw_order += [c for c in cats if c not in draw_order]
    for cat in draw_order:
        mask = labels == cat
        if not mask.any(): continue
        alpha = 0.15 if cat == "Норма" else 0.75
        size  = 4    if cat == "Норма" else 16
        zord  = 1    if cat == "Норма" else 3
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=CAT_COLOR[cat], s=size, alpha=alpha,
                   linewidths=0, zorder=zord)
    handles = [mpatches.Patch(color=CAT_COLOR[c],
               label=f"{c} ({(labels==c).sum()})") for c in draw_order if (labels==c).any()]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Компонента 1"); ax.set_ylabel("Компонента 2")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {p}")

def plot3d(coords, labels, cats, fname):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(16, 11))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#F8F9FA")
    draw_order = [c for c in ["Норма","ML аномалия"] if c in cats]
    draw_order += [c for c in cats if c not in draw_order]
    for cat in draw_order:
        mask = labels == cat
        if not mask.any(): continue
        ax.scatter(coords[mask,0], coords[mask,1], coords[mask,2],
                   c=CAT_COLOR[cat], s=(3 if cat=="Норма" else 14),
                   alpha=(0.12 if cat=="Норма" else 0.70), linewidths=0,
                   label=f"{cat} ({mask.sum()})")
    ax.set_title("GigaChat Embeddings — t-SNE 3D (полный датасет)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_xlabel("Ось 1"); ax.set_ylabel("Ось 2"); ax.set_zlabel("Ось 3")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {p}")

# t-SNE 2D — полный
plot2d(coords_2d, labels, cats,
       f"GigaChat Embeddings — t-SNE 2D (полный датасет, n={len(df)})",
       "viz_tsne_2d_full.png")

# t-SNE 3D
plot3d(coords_3d, labels, cats, "viz_tsne_3d_full.png")

# t-SNE 2D — только аномалии крупнее
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_facecolor("#F8F9FA")
# сначала норма очень мелко
mask_n = labels == "Норма"
ax.scatter(coords_2d[mask_n,0], coords_2d[mask_n,1],
           c="#CCCCCC", s=2, alpha=0.08, linewidths=0, zorder=1, label=f"Норма ({mask_n.sum()})")
# потом аномалии крупнее
for cat in [c for c in cats if c != "Норма"]:
    mask = labels == cat
    if not mask.any(): continue
    ax.scatter(coords_2d[mask,0], coords_2d[mask,1],
               c=CAT_COLOR[cat], s=25, alpha=0.85, linewidths=0.3,
               edgecolors="white", zorder=3)
handles = [mpatches.Patch(color="#CCCCCC", label=f"Норма ({mask_n.sum()})")]
handles += [mpatches.Patch(color=CAT_COLOR[c], label=f"{c} ({(labels==c).sum()})")
            for c in cats if c != "Норма" and (labels==c).any()]
ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
ax.set_title(f"GigaChat Embeddings — t-SNE 2D, аномалии выделены\n(n={len(df)}, полный датасет)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Компонента 1"); ax.set_ylabel("Компонента 2")
ax.grid(True, alpha=0.25, linewidth=0.5)
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, "viz_tsne_2d_anomalies_highlight.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Сохранено: {p}")

# ── Итог ──
print("\n" + "="*70)
print("ИТОГ")
print("="*70)
print(f"  Записей обработано: {len(df)}")
print(f"  Вектор: {embeddings.shape[1]}D  (GigaChat Embeddings)")
print(f"  Файлы в {OUTPUT_DIR}/:")
for fn in ["viz_tsne_2d_full.png", "viz_tsne_3d_full.png",
           "viz_tsne_2d_anomalies_highlight.png",
           "embeddings_full.npy", "tsne_2d_full.npy"]:
    p = os.path.join(OUTPUT_DIR, fn)
    if os.path.exists(p):
        size = os.path.getsize(p) // 1024
        print(f"    {fn}  ({size} КБ)")
