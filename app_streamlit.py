import os
import gc
import json
import requests
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import streamlit as st
import folium
from streamlit_folium import st_folium
from huggingface_hub import hf_hub_url

# ==============================================================================
# 1. PAGE CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="EcoWarden Croatia | Detekcija Odlagališta",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

META_REPO_ID = "SilvioS/metafornow"
META_FILENAME = "metadata_index.parquet"
PROD_REPO_ID = "SilvioS/eloo_dgu_scanner_output"
TEST_REPO_ID = "SilvioS/eloo_dgu_triage_results"

try:
    HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

SEMANTIC_CLASS_COLORS = {
    "Clean Nature":        "#2ca02c",
    "Urban / Roofs":       "#1f77b4",
    "Natural Karst Rock":  "#7f7f7f",
    "Construction Rubble": "#ff7f0e",
    "Municipal Garbage":   "#d62728",
    "Tires / Bulky Waste": "#9467bd",
    "Unknown":             "#17becf"
}

TRIAGE_UI_MAP = {
    "AUTO_VERIFIED_WASTE": {
        "label": "🔴 Visoka vjerojatnost otpada (Potvrđeno)",
        "short": "Potvrđeni otpad",
        "color": "#d62728",
        "explanation": "Svi AI skeneri snažno se slažu da se na ovoj lokaciji nalazi odlagalište otpada."
    },
    "ROUTE_TO_CROWD_DISAGREEMENT": {
        "label": "🟠 Potrebna provjera (Neslaganje modela)",
        "short": "Potrebna provjera",
        "color": "#ff7f0e",
        "explanation": (
            "Skeneri terena ili toplinska mapa vide sumnjivu promjenu na tlu, "
            "dok detektori pojedinačnih objekata nisu sigurni. Provjerite lokaciju pomoću Geo Filtera!"
        )
    },
    "ROUTE_TO_CROWD_AMBIGUOUS": {
        "label": "🟡 Granični slučaj (Srednja sigurnost)",
        "short": "Granični slučaj",
        "color": "#e6b800",
        "explanation": "Modeli su umjereno nesigurni. Preporučuje se vizualni pregled."
    },
    "AUTO_CLEAN": {
        "label": "🟢 Čisto područje (Nizak rizik)",
        "short": "Čisto područje",
        "color": "#2ca02c",
        "explanation": "Svi modeli se slažu da snimka prikazuje čistu prirodu ili normalne objekte."
    }
}

ZUPANIJE_BBOX = {
    "Sve županije (Cijela Hrvatska)": [13.4, 42.3, 19.5, 46.6],
    "Grad Zagreb i Zagrebačka": [15.3, 45.4, 16.7, 46.1],
    "Splitsko-dalmatinska": [15.9, 42.9, 17.5, 43.9],
    "Istarska": [13.4, 44.7, 14.3, 45.6],
    "Primorsko-goranska": [14.1, 44.4, 15.2, 45.7],
    "Zadarska": [14.5, 43.8, 16.2, 44.6],
    "Osječko-baranjska": [17.8, 45.2, 19.1, 46.0],
    "Sisačko-moslavačka": [15.7, 44.9, 17.2, 45.7],
    "Karlovačka": [15.0, 44.8, 15.9, 45.7],
    "Varaždinska i Međimurska": [15.8, 46.1, 16.9, 46.6],
    "Dubrovačko-neretvanska": [16.5, 42.3, 18.6, 43.2],
}

SCALAR_META_COLS = [
    "chunk_id", "file_name", "cluster_id", "image_year",
    "corine_code", "corine_class", "centroid_lon", "centroid_lat",
    "img_width_px", "img_height_px", "meters_per_pixel", "repo_image_path",
    "aw_out1_global_waste_prob", "aw_out2_cam_global_mean",
    "aw_out3_cam_global_max", "aw_out4_cam_area_pct_gt_05",
    "siglip_out6_total_waste_max", "siglip_out10_suspect_area_m2",
    "siglip_out11_primary_waste_type", "skyclip_out6_total_waste_max",
    "owlv2_out9_total_waste_max_score", "owlv2_out10_total_waste_box_count",
    "owlv2_out11_detected_boxes_area_m2",
    "yoloworld_out9_total_waste_max_score", "yoloworld_out10_total_waste_box_count",
    "yoloworld_out11_detected_boxes_area_m2",
    "ens_mean_waste_prob", "ens_max_min_diff",
    "ens_active_learning_score", "ens_triage_bucket",
    "aw_hotspot_area_m2", "max_detected_area_m2", "_source_repo"
]

JSON_META_COLS = [
    "cluster_id", "file_name", "chunk_id", "image_year", "repo_image_path",
    "aw_cam_grid_32x32_json", "all_tile_records_json",
    "owlv2_detected_boxes_json", "yoloworld_detected_boxes_json"
]


# ==============================================================================
# 2. ZERO-DISK-WRITE IN-MEMORY LOADER (@st.cache_resource)
# ==============================================================================
def assign_approx_zupanija(lon: float, lat: float) -> str:
    for zup, (lon_min, lat_min, lon_max, lat_max) in ZUPANIJE_BBOX.items():
        if zup == "Sve županije (Cijela Hrvatska)":
            continue
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return zup
    return "Ostale županije"


@st.cache_resource(ttl=3600, show_spinner="Učitavanje indeksa u radnu memoriju (0 MB na disku)...")
def load_shared_metadata_resource() -> tuple[pd.DataFrame, pa.Table, list]:
    """
    Uses @st.cache_resource so Streamlit NEVER duplicates the DataFrame across reruns,
    and streams directly into BytesIO so ZERO files are written to the server disk!
    """
    logs = []
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    try:
        url = hf_hub_url(repo_id=META_REPO_ID, filename=META_FILENAME, repo_type="dataset")
        logs.append(f"Streaming u RAM: {META_REPO_ID}/{META_FILENAME}")
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        pq_buffer = BytesIO(resp.content)
        full_arrow_table = pq.read_table(pq_buffer)
        schema_names = full_arrow_table.schema.names

        # 1. Scalar DataFrame for fast filtering & map display
        scalar_cols = [c for c in SCALAR_META_COLS if c in schema_names]
        df = full_arrow_table.select(scalar_cols).to_pandas()

        # 2. Compact Arrow Table for JSON overlays (uses 5x less RAM than Pandas strings)
        json_cols = [c for c in JSON_META_COLS if c in schema_names]
        json_arrow_table = full_arrow_table.select(json_cols)

        del full_arrow_table
        gc.collect()
        logs.append(f"Uspješno učitano {len(df)} snimaka u dijeljenu memoriju (bez zapisivanja na disk).")

    except Exception as e:
        logs.append(f"GREŠKA: {type(e).__name__}: {str(e)}")
        return pd.DataFrame(), None, logs

    if not df.empty:
        if "aw_hotspot_area_m2" not in df.columns:
            total_area = df["img_width_px"] * df["img_height_px"] * (df["meters_per_pixel"] ** 2)
            df["aw_hotspot_area_m2"] = (total_area * (df["aw_out4_cam_area_pct_gt_05"] / 100.0)).round(1)

        if "max_detected_area_m2" not in df.columns:
            df["max_detected_area_m2"] = df[[
                "aw_hotspot_area_m2",
                "siglip_out10_suspect_area_m2",
                "owlv2_out11_detected_boxes_area_m2",
                "yoloworld_out11_detected_boxes_area_m2"
            ]].max(axis=1).round(1)

        if "aw_out3_cam_global_max" not in df.columns:
            df["aw_out3_cam_global_max"] = df["aw_out1_global_waste_prob"]

        if "zupanija" not in df.columns:
            df["zupanija"] = [
                assign_approx_zupanija(lon, lat)
                for lon, lat in zip(df["centroid_lon"], df["centroid_lat"])
            ]

        df["status_label"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["label"])
        df["status_short"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["short"])
        df["color_hex"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["color"])
        df["warden_confidence_pct"] = (df["ens_mean_waste_prob"] * 100).round(0).astype(int)
        df["disagreement_pct"] = (df["ens_max_min_diff"] * 100).round(0).astype(int)

    return df, json_arrow_table, logs


@st.cache_data(max_entries=10, show_spinner="Preuzimanje slika za odabranu lokaciju u RAM...")
def fetch_cluster_cache_bucket(cluster_id: int, _json_arrow_table: pa.Table) -> dict:
    """
    10-LOCATION CACHE BUCKET (Zero Disk Writes!):
    - Filters the in-memory PyArrow table in <1ms.
    - Downloads ONLY raw compressed JPEG bytes (~150KB) into RAM via HTTP (never touches disk!).
    """
    if _json_arrow_table is None:
        return {}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    # Fast in-memory Arrow filter for this cluster_id
    mask = pc.equal(_json_arrow_table["cluster_id"], int(cluster_id))
    sub_table = _json_arrow_table.filter(mask)
    rows_list = sub_table.to_pylist()

    cluster_payloads = {}
    for row in rows_list:
        fname = str(row["file_name"])
        chunk_id = str(row["chunk_id"])
        repo_img_path = str(row.get("repo_image_path") or f"images/{chunk_id}/{fname}")

        jpg_bytes = None
        for candidate_repo in [PROD_REPO_ID, TEST_REPO_ID]:
            try:
                img_url = hf_hub_url(repo_id=candidate_repo, filename=repo_img_path, repo_type="dataset")
                resp = requests.get(img_url, headers=headers, timeout=12)
                if resp.status_code == 200 and len(resp.content) > 100:
                    jpg_bytes = resp.content
                    break
            except Exception:
                continue

        cluster_payloads[fname] = {
            "jpg_bytes": jpg_bytes,  # Stored as compact compressed bytes (~150KB instead of 7MB PIL object!)
            "aw_cam_32": np.array(json.loads(row.get("aw_cam_grid_32x32_json") or "[]"), dtype=np.float32),
            "tiles": json.loads(row.get("all_tile_records_json") or "[]"),
            "owlv2_boxes": json.loads(row.get("owlv2_detected_boxes_json") or "[]"),
            "yolo_boxes": json.loads(row.get("yoloworld_detected_boxes_json") or "[]"),
        }

    return cluster_payloads


# ==============================================================================
# 3. GEO FILTER RENDERER
# ==============================================================================
def render_geo_filter_view(
    payload: dict,
    meta_row: pd.Series,
    active_filter: str,
    pyramid_scale: str = "L0_384px",
    tile_threshold: float = 0.45,
    box_min_score: float = 0.05
) -> plt.Figure:
    plt.close("all")  # Prevent any orphaned Matplotlib figures from leaking RAM

    jpg_bytes = payload.get("jpg_bytes")
    if not jpg_bytes:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Slika nije dostupna u repozitoriju", ha="center", va="center")
        ax.axis("off")
        return fig

    img = Image.open(BytesIO(jpg_bytes)).convert("RGB")
    w, h = img.size
    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=130)
    ax.imshow(img)
    ax.axis("off")

    if active_filter == "📸 Originalna snimka":
        ax.set_title(
            f"Originalni DGU DOF5 Ortofoto ({meta_row['meters_per_pixel']} m/px) — Godina: {meta_row['image_year']}",
            fontsize=11, fontweight="bold", pad=10
        )

    elif active_filter == "🔥 Toplinska mapa (Heatmap)":
        cam_32 = payload["aw_cam_32"]
        if cam_32.size > 0:
            cam_pil = Image.fromarray(cam_32, mode="F").resize((w, h), resample=Image.BICUBIC)
            cam_upsampled = np.clip(np.array(cam_pil, dtype=np.float32), 0.0, 1.0)
            im = ax.imshow(cam_upsampled, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03, label="Intenzitet sumnje (AerialWaste CAM)")
        ax.set_title(
            f"Geo Filter: Toplinska mapa (Max intenzitet: {meta_row['aw_out3_cam_global_max']:.0%} | Površina: {meta_row['aw_hotspot_area_m2']:.0f} m²)",
            fontsize=11, fontweight="bold", pad=10
        )

    elif active_filter == "🟩 Skener terena (Pločice)":
        tiles = payload["tiles"]
        if pyramid_scale != "Sve razine":
            tiles = [t for t in tiles if t.get("level_id") == pyramid_scale]

        for t in tiles:
            x1, y1, x2, y2 = t["bbox_px"]
            w_prob = float(t.get("w_prob", max(
                float(t.get("siglip_out6_total_waste_prob", 0.0)),
                float(t.get("skyclip_out6_total_waste_prob", 0.0))
            )))
            dom_cls = t.get("dom_cls") or t.get("siglip_dominant_class", "Unknown")
            is_flag = w_prob >= tile_threshold
            col = SEMANTIC_CLASS_COLORS.get(dom_cls, "#17becf")

            if is_flag:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.2, edgecolor=col, facecolor=col, alpha=0.25))
                ax.text(
                    x1 + 8, y1 + 26, f"{dom_cls}\n({w_prob:.0%})",
                    color="white", fontsize=7.5, fontweight="bold",
                    bbox=dict(facecolor=col, alpha=0.8, pad=1.5, edgecolor="none")
                )
            else:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.6, edgecolor="white", facecolor="none", alpha=0.35))
        ax.set_title(
            f"Geo Filter: Skener terena — Sumnjiva površina: {meta_row['siglip_out10_suspect_area_m2']:.0f} m²",
            fontsize=11, fontweight="bold", pad=10
        )

    elif active_filter == "🎯 Detekcija hrpa (Okviri)":
        all_boxes = [("OWLv2", b) for b in payload["owlv2_boxes"]] + [("YOLO", b) for b in payload["yolo_boxes"]]
        drawn_count = 0
        for model_tag, b in all_boxes:
            score = float(b.get("score", 0.0))
            if score < box_min_score:
                continue
            drawn_count += 1
            bx1, by1, bx2, by2 = b["bbox_px"]
            col = b.get("color", "#ff7f0e" if model_tag == "OWLv2" else "#d62728")
            ax.add_patch(patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2.4, edgecolor=col, facecolor=col, alpha=0.22))
            ax.text(
                bx1 + 4, max(16, by1 - 6), f"{b.get('label', 'Otpad')} ({score:.0%})",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(facecolor=col, alpha=0.85, pad=1.5, edgecolor="none")
            )
        ax.set_title(
            f"Geo Filter: Detekcija pojedinačnih hrpa ({drawn_count} detektiranih objekata)",
            fontsize=11, fontweight="bold", pad=10
        )

    elif active_filter == "🎛️ Kombinirani prikaz":
        cam_32 = payload["aw_cam_32"]
        if cam_32.size > 0:
            cam_pil = Image.fromarray(cam_32, mode="F").resize((w, h), resample=Image.BICUBIC)
            cam_upsampled = np.clip(np.array(cam_pil, dtype=np.float32), 0.0, 1.0)
            ax.imshow(cam_upsampled, cmap="jet", alpha=0.30, vmin=0, vmax=1)

        all_boxes = payload["owlv2_boxes"] + payload["yolo_boxes"]
        for b in all_boxes:
            if float(b.get("score", 0.0)) >= box_min_score:
                bx1, by1, bx2, by2 = b["bbox_px"]
                col = b.get("color", "#d62728")
                ax.add_patch(patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2.2, edgecolor=col, facecolor="none"))
        ax.set_title("Geo Filter: Kombinirani prikaz (Toplinska mapa + Okviri hrpa)", fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


# ==============================================================================
# 4. MAIN APPLICATION LAYOUT
# ==============================================================================
def main():
    st.title("🛰️ EcoWarden Croatia — Sustav za nadzor divljih odlagališta")
    st.caption("Interaktivni pregled DGU DOF5 ortofoto snimaka (50 cm/px) pomoću 5-modelnog AI sustava i Geo Filtera.")

    df_meta, json_arrow_table, boot_logs = load_shared_metadata_resource()

    with st.expander("🛠️ Dijagnostika sustava i upravljanje memorijom", expanded=df_meta.empty):
        for log_line in boot_logs:
            st.write(f"- `{log_line}`")
        st.write("- Način rada: **100% u radnoj memoriji (RAM)** — aplikacija ne zapisuje privremene datoteke na disk servera.")

    if df_meta.empty:
        st.error("Nema dostupnih podataka. Provjerite dijagnostiku iznad.")
        return

    if "selected_cluster_id" not in st.session_state:
        st.session_state["selected_cluster_id"] = int(df_meta["cluster_id"].iloc[0])

    # --------------------------------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("📍 1. Lokacija i Vrijeme")
        available_zup = ["Sve županije (Cijela Hrvatska)"] + sorted(df_meta["zupanija"].unique().tolist())
        selected_zup = st.selectbox("Županija / Regija:", available_zup)

        st.divider()
        st.subheader("📅 Godina snimanja")
        year_counts = df_meta["image_year"].value_counts().sort_index()
        selected_years = []
        for yr, count in year_counts.items():
            if st.checkbox(f"Godina {yr} ({count} snimaka)", value=True, key=f"chk_yr_{yr}"):
                selected_years.append(yr)

        st.divider()
        st.subheader("🌲 Tip zemljišta (CORINE)")
        available_corine = sorted(df_meta["corine_class"].dropna().unique().tolist())
        selected_corine = st.multiselect("Filtriraj po tipu okoliša:", options=available_corine, default=available_corine)

        st.divider()
        st.subheader("🤖 Filtriranje po AI skeneru")
        selected_scanner = st.selectbox(
            "Prikaz snimaka prema detekciji skenera:",
            options=[
                "Svi skeneri zajedno (Komitet 5 modela)",
                "Samo AerialWaste (Toplinska mapa / CAM >= 50%)",
                "Samo SigLIP / SkyCLIP (Tekstura tla / Rubble)",
                "Samo Detektori objekata (OWLv2 / YOLO-World hrpe)"
            ]
        )

        st.divider()
        st.subheader("📐 Veličina sumnjive zone")
        size_metric_col = st.radio(
            "Mjera površine:",
            options=["max_detected_area_m2", "aw_hotspot_area_m2", "siglip_out10_suspect_area_m2"],
            format_func=lambda c: (
                "Maksimalna detektirana površina (bilo koji model)" if c == "max_detected_area_m2"
                else ("AerialWaste CAM žarište (m²)" if c == "aw_hotspot_area_m2" else "SigLIP sumnjiva površina pločica (m²)")
            )
        )
        max_possible_area = int(min(15000, df_meta[size_metric_col].max())) if not df_meta.empty else 5000
        min_size_m2 = st.slider("Minimalna površina (m²):", 0, max(500, max_possible_area), 0, 50)

        st.divider()
        st.header("🚦 Status AI provjere")
        all_buckets = list(TRIAGE_UI_MAP.keys())
        selected_buckets = st.multiselect(
            "Filtriraj po prioritetu:",
            options=all_buckets,
            default=all_buckets,
            format_func=lambda b: TRIAGE_UI_MAP[b]["label"]
        )
