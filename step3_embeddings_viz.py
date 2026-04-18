"""
Шаг 3: Эмбеддинги строк датасета через GigaChat + визуализация t-SNE/PCA.

Каждая строка hakaton.csv преобразуется в текстовое описание,
отправляется в GigaChat Embeddings (модель Sber), получается вектор
размерностью 1024. Затем PCA → t-SNE снижает размерность до 2D/3D
для визуализации. Точки раскрашиваются по типу аномальности.

Запуск:
    python3 step3_embeddings_viz.py

Параметры:
    SAMPLE_SIZE   - сколько строк брать (по умолчанию 2000)
    BATCH_SIZE    - строк за один API-запрос (макс. 8 для GigaChat)
    CACHE_FILE    - куда кэшировать эмбеддинги (не платить API дважды)
    OUTPUT_DIR    - куда сохранять графики
"""

import os
import json
import time
import uuid
import warnings
import requests
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache_citis")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")   # без GUI — просто сохраняем файлы
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ─────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────
GIGACHAT_KEY = os.getenv(
    "GIGACHAT_AUTHORIZATION_KEY",
    "ZTY5NWJiYTAtN2RlMC00YWZhLWI5YzUtZDcxNGNhOTUzMTkyOjlhMWVkYjcyLTdmMTctNDhjNS1iNjY1LTBjMzcwNDE1YjE4NQ=="
)

DATA_FILE       = "hakaton.csv"
MANUAL_FLAGS    = "results_manual/manual_anomalies.csv"
IFOREST_FILE    = "results_iforest/iforest_anomalies.csv"
CACHE_FILE      = "results_embeddings/embeddings_cache.json"
OUTPUT_DIR      = "results_embeddings"

SAMPLE_SIZE     = 2000   # строк для визуализации (полный прогон ~26 мин)
BATCH_SIZE      = 8      # строк за один запрос к GigaChat
TSNE_PERPLEXITY = 40     # перплексия t-SNE (обычно 5–50)
TSNE_ITER       = 1000   # итераций t-SNE
PCA_DIM         = 50     # сначала снижаем PCA до 50D, потом t-SNE

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# АУТЕНТИФИКАЦИЯ GIGACHAT
# ─────────────────────────────────────────────────────────────
class GigaChatAuth:
    """Получает и кэширует access token для GigaChat API."""

    def __init__(self, client_secret: str):
        self.auth_url    = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.client_secret = client_secret
        self._token: str | None = None
        self._expires_at: float = 0

    def get_token(self) -> str:
        # Обновляем токен за 2 минуты до истечения
        if self._token and time.time() < self._expires_at - 120:
            return self._token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.client_secret.strip()}",
        }
        resp = requests.post(
            self.auth_url,
            headers=headers,
            data={"scope": "GIGACHAT_API_PERS"},
            verify=False,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        # expires_at приходит в миллисекундах
        self._expires_at = data.get("expires_at", (time.time() + 1800) * 1000) / 1000.0
        return self._token


# ─────────────────────────────────────────────────────────────
# ЭМБЕДДИНГИ ЧЕРЕЗ GIGACHAT
# ─────────────────────────────────────────────────────────────
class GigaChatEmbeddings:
    """Получает векторные представления текстов через GigaChat Embeddings."""

    API_URL = "https://gigachat.devices.sberbank.ru/api/v1/embeddings"

    def __init__(self, auth: GigaChatAuth):
        self.auth = auth

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth.get_token()}",
        }

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Эмбеддинг одного батча (до 8 текстов)."""
        payload = {"model": "Embeddings", "input": texts}
        resp = requests.post(
            self.API_URL,
            headers=self._headers(),
            json=payload,
            verify=False,
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"GigaChat Embeddings error {resp.status_code}: {resp.text[:200]}")
        return [item["embedding"] for item in resp.json()["data"]]

    def embed_all(
        self,
        texts: list[str],
        batch_size: int = 8,
        cache_path: str | None = None,
    ) -> np.ndarray:
        """
        Эмбеддит все тексты батчами. Поддерживает кэширование:
        - Уже посчитанные тексты не пересчитываются
        - Кэш сохраняется после каждого батча (защита от обрыва)
        """
        # Загружаем кэш
        cache: dict[str, list[float]] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            print(f"  Загружен кэш: {len(cache)} текстов")

        # Определяем, что ещё не посчитано
        missing_idx = [i for i, t in enumerate(texts) if t not in cache]
        print(f"  Всего текстов: {len(texts)}")
        print(f"  Уже в кэше: {len(texts) - len(missing_idx)}")
        print(f"  Нужно посчитать: {len(missing_idx)}")

        if missing_idx:
            n_batches = (len(missing_idx) + batch_size - 1) // batch_size
            print(f"  Батчей: {n_batches} (по {batch_size} текстов)")

            for b_num, batch_start in enumerate(range(0, len(missing_idx), batch_size)):
                batch_idx = missing_idx[batch_start:batch_start + batch_size]
                batch_texts = [texts[i] for i in batch_idx]

                retry = 0
                while retry < 3:
                    try:
                        embeddings = self.embed_batch(batch_texts)
                        for i, emb in zip(batch_idx, embeddings):
                            cache[texts[i]] = emb
                        break
                    except Exception as e:
                        retry += 1
                        print(f"    ⚠️  Батч {b_num+1} ошибка (попытка {retry}/3): {e}")
                        time.sleep(2 ** retry)

                # Сохраняем кэш после каждого батча
                if cache_path and b_num % 10 == 0:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)

                if (b_num + 1) % 20 == 0 or (b_num + 1) == n_batches:
                    pct = (b_num + 1) / n_batches * 100
                    print(f"  [{b_num+1}/{n_batches}] {pct:.0f}%  ({len(cache)} текстов в кэше)")

            # Финальное сохранение кэша
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
                print(f"  Кэш сохранён: {cache_path}")

        # Собираем финальный массив
        return np.array([cache[t] for t in texts], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# ПРЕОБРАЗОВАНИЕ СТРОКИ ДАТАСЕТА В ТЕКСТ
# ─────────────────────────────────────────────────────────────
def row_to_text(row: pd.Series) -> str:
    """
    Превращает строку датасета в читаемое текстовое описание для эмбеддинга.
    Включает все смысловые поля: ребёнок, представитель, школы, тест, результат.
    """
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


# ─────────────────────────────────────────────────────────────
# РАЗМЕТКА: тип аномальности для каждой строки
# ─────────────────────────────────────────────────────────────
def build_labels(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Возвращает:
      labels    — массив строк с типом для каждой записи
      categories — уникальные категории (для легенды)
      colors     — цвет каждой категории
    """
    # Загружаем аномалии
    manual = pd.read_csv(MANUAL_FLAGS, dtype=str)
    iforest = pd.read_csv(IFOREST_FILE, sep=";", dtype=str)

    manual_our   = set(manual["our_number"])
    iforest_our  = set(iforest["our_number"])

    # Главные категории ручных аномалий для каждой нашей записи
    manual_cat = (
        manual.groupby("our_number")["category"]
        .apply(lambda x: x.value_counts().index[0])  # самая частая
        .to_dict()
    )

    # Категория → цвет
    CAT_COLOR = {
        "Норма":                   "#54A24B",  # зелёный
        "ML аномалия":             "#9467BD",  # фиолетовый
        "ЧАСТОТА":                 "#D62728",  # красный
        "ДОКУМЕНТ_РЕБЁНОК":        "#FF7F0E",  # оранжевый
        "ВАРИАНТ_ФОРМАТ":          "#BCBD22",  # жёлто-зелёный
        "ВОЗРАСТ_КЛАСС":           "#1F77B4",  # синий
        "РЕЗУЛЬТАТ_РЕГИСТР":       "#17BECF",  # голубой
        "ВАРИАНТ_КЛАСС":           "#8C564B",  # коричневый
        "Прочие ручные":           "#E377C2",  # розовый
    }

    labels = []
    for _, row in df.iterrows():
        our = row["our_number"]
        if our in manual_our:
            cat = manual_cat.get(our, "Прочие ручные")
            if cat not in CAT_COLOR:
                cat = "Прочие ручные"
            labels.append(cat)
        elif our in iforest_our:
            labels.append("ML аномалия")
        else:
            labels.append("Норма")

    # Уникальные категории в порядке убывания важности
    order = list(CAT_COLOR.keys())
    present = sorted(set(labels), key=lambda x: order.index(x) if x in order else 99)
    colors = [CAT_COLOR.get(c, "#AAAAAA") for c in present]

    return np.array(labels), present, colors


# ─────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────
def plot_embeddings(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    colors: list[str],
    title: str,
    filename: str,
):
    """Строит scatter plot с раскраской по категориям."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")

    cat_color_map = dict(zip(categories, colors))

    # Рисуем категории снизу вверх: сначала фон (Норма), потом аномалии
    draw_order = [c for c in ["Норма", "ML аномалия"] if c in categories]
    draw_order += [c for c in categories if c not in draw_order]

    for cat in draw_order:
        mask = labels == cat
        if not mask.any():
            continue
        alpha = 0.25 if cat == "Норма" else 0.80
        size  = 6  if cat == "Норма" else 18
        zord  = 1  if cat == "Норма" else 3
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=cat_color_map[cat],
            s=size,
            alpha=alpha,
            linewidths=0,
            zorder=zord,
            label=f"{cat} ({mask.sum()})",
        )

    # Легенда
    handles = [
        mpatches.Patch(color=cat_color_map[c], label=f"{c} ({(labels==c).sum()})")
        for c in draw_order if (labels == c).any()
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Компонента 1", fontsize=10)
    ax.set_ylabel("Компонента 2", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")


def plot_3d_embeddings(
    coords_3d: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    colors: list[str],
    filename: str,
):
    """3D scatter plot."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#F8F9FA")

    cat_color_map = dict(zip(categories, colors))
    draw_order = [c for c in ["Норма", "ML аномалия"] if c in categories]
    draw_order += [c for c in categories if c not in draw_order]

    for cat in draw_order:
        mask = labels == cat
        if not mask.any():
            continue
        alpha = 0.15 if cat == "Норма" else 0.75
        size  = 4    if cat == "Норма" else 14
        ax.scatter(
            coords_3d[mask, 0],
            coords_3d[mask, 1],
            coords_3d[mask, 2],
            c=cat_color_map[cat],
            s=size,
            alpha=alpha,
            linewidths=0,
            label=f"{cat} ({mask.sum()})",
        )

    ax.set_title("Эмбеддинги GigaChat — 3D t-SNE\n(раскраска по типу аномалии)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Ось 1")
    ax.set_ylabel("Ось 2")
    ax.set_zlabel("Ось 3")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")


# ─────────────────────────────────────────────────────────────
# ГЛАВНЫЙ ПАЙПЛАЙН
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ЭМБЕДДИНГИ GIGACHAT + ВИЗУАЛИЗАЦИЯ t-SNE")
    print("=" * 70)

    # ── 1. Загрузка и выборка ──
    print(f"\n[1/6] Загрузка данных (выборка: {SAMPLE_SIZE} строк)...")
    df_full = pd.read_csv(DATA_FILE, sep=";", dtype=str, keep_default_na=False)

    # Стратифицированная выборка: сохраняем пропорцию аномалий
    manual_set  = set(pd.read_csv(MANUAL_FLAGS, dtype=str)["our_number"])
    iforest_set = set(pd.read_csv(IFOREST_FILE, sep=";", dtype=str)["our_number"])

    df_full["_label"] = df_full["our_number"].apply(
        lambda x: "manual" if x in manual_set
        else ("ml" if x in iforest_set else "normal")
    )
    label_counts = df_full["_label"].value_counts()
    print(f"  Распределение в датасете: {dict(label_counts)}")

    # Берём пропорционально или все, если меньше квоты
    frames = []
    for lbl, total in label_counts.items():
        quota = int(SAMPLE_SIZE * total / len(df_full))
        quota = max(quota, min(50, total))  # минимум 50 от каждого класса
        sample = df_full[df_full["_label"] == lbl].sample(
            n=min(quota, total), random_state=42
        )
        frames.append(sample)

    df = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    # Ограничиваем до SAMPLE_SIZE
    df = df.head(SAMPLE_SIZE).reset_index(drop=True)
    print(f"  Выбрано строк: {len(df)}  (normal={( df['_label']=='normal').sum()}, "
          f"manual={(df['_label']=='manual').sum()}, ml={(df['_label']=='ml').sum()})")

    # ── 2. Преобразование строк в тексты ──
    print("\n[2/6] Преобразование строк в тексты...")
    texts = [row_to_text(row) for _, row in df.iterrows()]
    print(f"  Примерная длина текста: {len(texts[0])} символов")
    print(f"  Пример: {texts[0][:120]}...")

    # ── 3. Получение эмбеддингов ──
    print("\n[3/6] Получение эмбеддингов через GigaChat Embeddings...")
    auth = GigaChatAuth(GIGACHAT_KEY)
    embedder = GigaChatEmbeddings(auth)

    t_start = time.time()
    embeddings = embedder.embed_all(texts, batch_size=BATCH_SIZE, cache_path=CACHE_FILE)
    t_elapsed = time.time() - t_start

    print(f"  Форма матрицы эмбеддингов: {embeddings.shape}")
    print(f"  Время: {t_elapsed:.1f} сек.")
    print(f"  Размерность вектора: {embeddings.shape[1]}D")

    # Сохраняем сырые эмбеддинги
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    df.drop(columns=["_label"], errors="ignore").to_csv(
        os.path.join(OUTPUT_DIR, "embeddings_rows.csv"), index=False, sep=";"
    )
    print(f"  Эмбеддинги сохранены: {OUTPUT_DIR}/embeddings.npy")

    # ── 4. Разметка по типам аномалий ──
    print("\n[4/6] Построение меток аномальности...")
    labels, categories, colors = build_labels(df)
    for cat in categories:
        print(f"  {cat}: {(labels==cat).sum()}")

    # ── 5. Снижение размерности ──
    print(f"\n[5/6] Снижение размерности: {embeddings.shape[1]}D → PCA({PCA_DIM}D) → t-SNE(2D/3D)...")

    # PCA до 50D (ускоряет t-SNE, убирает шум)
    n_pca = min(PCA_DIM, embeddings.shape[0] - 1, embeddings.shape[1])
    print(f"  PCA: {embeddings.shape[1]}D → {n_pca}D...")
    pca = PCA(n_components=n_pca, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Объяснённая дисперсия PCA: {var_explained:.1f}%")

    # t-SNE 2D
    print(f"  t-SNE 2D (perplexity={TSNE_PERPLEXITY}, iter={TSNE_ITER})...")
    tsne2 = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, len(df) // 4),
        max_iter=TSNE_ITER,
        random_state=42,
        init="pca",
        learning_rate="auto",
        verbose=0,
    )
    coords_2d = tsne2.fit_transform(emb_pca)
    np.save(os.path.join(OUTPUT_DIR, "tsne_2d.npy"), coords_2d)

    # t-SNE 3D
    print("  t-SNE 3D...")
    tsne3 = TSNE(
        n_components=3,
        perplexity=min(TSNE_PERPLEXITY, len(df) // 4),
        max_iter=TSNE_ITER,
        random_state=42,
        init="pca",
        learning_rate="auto",
        verbose=0,
    )
    coords_3d = tsne3.fit_transform(emb_pca)
    np.save(os.path.join(OUTPUT_DIR, "tsne_3d.npy"), coords_3d)

    # PCA 2D (быстрый альтернативный вид)
    print("  PCA 2D (для сравнения)...")
    pca2 = PCA(n_components=2, random_state=42)
    coords_pca2d = pca2.fit_transform(embeddings)
    np.save(os.path.join(OUTPUT_DIR, "pca_2d.npy"), coords_pca2d)

    # ── 6. Визуализация ──
    print("\n[6/6] Визуализация...")

    plot_embeddings(
        coords_2d, labels, categories, colors,
        title=f"GigaChat Embeddings — t-SNE 2D\n"
              f"(n={len(df)}, perplexity={TSNE_PERPLEXITY}, раскраска по типу аномалии)",
        filename="viz_tsne_2d.png",
    )

    plot_3d_embeddings(
        coords_3d, labels, categories, colors,
        filename="viz_tsne_3d.png",
    )

    plot_embeddings(
        coords_pca2d, labels, categories, colors,
        title=f"GigaChat Embeddings — PCA 2D\n"
              f"(n={len(df)}, объяснено: {pca2.explained_variance_ratio_.sum()*100:.1f}%)",
        filename="viz_pca_2d.png",
    )

    # ── Итог ──
    print("\n" + "=" * 70)
    print("ИТОГ")
    print("=" * 70)
    print(f"  Строк обработано:   {len(df)}")
    print(f"  Размерность модели: {embeddings.shape[1]}D  (GigaChat Embeddings)")
    print(f"  Файлы графиков:")
    print(f"    {OUTPUT_DIR}/viz_tsne_2d.png   ← основной вид")
    print(f"    {OUTPUT_DIR}/viz_tsne_3d.png")
    print(f"    {OUTPUT_DIR}/viz_pca_2d.png    ← быстрый линейный вид")
    print(f"  Кэш эмбеддингов: {CACHE_FILE}")
    print(f"  Сырые векторы: {OUTPUT_DIR}/embeddings.npy")


if __name__ == "__main__":
    main()
