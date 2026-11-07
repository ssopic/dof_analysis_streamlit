import os
import json
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyarrow.parquet as pq
import streamlit as st
import folium
from streamlit_folium import st_folium
from huggingface_hub import hf_hub_download

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

CACHE_DIR = "./hf_streamlit_cache"

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


# ==============================================================================
# 2. ULTRA-LOW-RAM METADATA LOADER & 10-CLUSTER CACHE BUCKET
# ==============================================================================
def assign_approx_zupanija(lon: float, lat: float) -> str:
    for zup, (lon_min, lat_min, lon_max, lat_max) in ZUPANIJE_BBOX.items():
        if zup == "Sve županije (Cijela Hrvatska)":
            continue
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return zup
    return "Ostale županije"


@st.cache_data(ttl=3600, show_spinner="Učitavanje indeksa lokacija s Hugging Face-a...")
def load_lightweight_metadata() -> tuple[pd.DataFrame, str, list]:
    logs = []
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        logs.append(f"Preuzimanje {META_REPO_ID}/{META_FILENAME}...")
        local_path = hf_hub_download(
            repo_id=META_REPO_ID,
            filename=META_FILENAME,
            repo_type="dataset",
            local_dir=CACHE_DIR,
            token=HF_TOKEN
        )
        logs.append(f"Datoteka preuzeta: {local_path}")

        schema_cols = pq.read_schema(local_path).names
        cols_to_read = [c for c in SCALAR_META_COLS if c in schema_cols]
        df = pd.read_parquet(local_path, columns=cols_to_read)
        logs.append(f"Učitano {len(df)} redaka (samo skalarni stupci, bez teških JSON-a u RAM-u).")

    except Exception as e:
        logs.append(f"GREŠKA: {type(e).__name__}: {str(e)}")
        return pd.DataFrame(), "", logs

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

    return df, local_path, logs


@st.cache_data(max_entries=10, show_spinner="Preuzimanje i predmemoriranje svih godina za odabranu lokaciju...")
def fetch_cluster_cache_bucket(cluster_id: int, meta_parquet_path: str) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    json_cols = [
        "file_name", "chunk_id", "image_year", "repo_image_path",
        "aw_cam_grid_32x32_json", "all_tile_records_json",
        "owlv2_detected_boxes_json", "yoloworld_detected_boxes_json"
    ]
    schema_cols = pq.read_schema(meta_parquet_path).names
    cols_to_load = [c for c in json_cols if c in schema_cols]

    df_cluster_jsons = pd.read_parquet(
        meta_parquet_path,
        columns=cols_to_load,
        filters=[("cluster_id", "==", int(cluster_id))]
    )

    cluster_payloads = {}
    for _, row in df_cluster_jsons.iterrows():
        fname = str(row["file_name"])
        chunk_id = str(row["chunk_id"])
        repo_img_path = str(row.get("repo_image_path", f"images/{chunk_id}/{fname}"))

        pil_img = None
        for candidate_repo in [PROD_REPO_ID, TEST_REPO_ID]:
            try:
                local_jpg = hf_hub_download(
                    repo_id=candidate_repo,
                    filename=repo_img_path,
                    repo_type="dataset",
                    local_dir=CACHE_DIR,
                    token=HF_TOKEN
                )
                pil_img = Image.open(local_jpg).convert("RGB")
                break
            except Exception:
                continue

        if pil_img is None:
            remote_subpath = f"triage_results/{chunk_id}_triage.parquet"
            for candidate_repo in [PROD_REPO_ID, TEST_REPO_ID]:
                try:
                    local_pq = hf_hub_download(
                        repo_id=candidate_repo,
                        filename=remote_subpath,
                        repo_type="dataset",
                        local_dir=CACHE_DIR,
                        token=HF_TOKEN
                    )
                    df_img = pd.read_parquet(
                        local_pq,
                        columns=["file_name", "image_jpeg_bytes"],
                        filters=[("file_name", "==", fname)]
                    )
                    if not df_img.empty and df_img.iloc[0]["image_jpeg_bytes"]:
                        pil_img = Image.open(BytesIO(df_img.iloc[0]["image_jpeg_bytes"])).convert("RGB")
                        break
                except Exception:
                    continue

        cluster_payloads[fname] = {
            "pil_img": pil_img,
            "aw_cam_32": np.array(json.loads(row.get("aw_cam_grid_32x32_json", "[]") or "[]"), dtype=np.float32),
            "tiles": json.loads(row.get("all_tile_records_json", "[]") or "[]"),
            "owlv2_boxes": json.loads(row.get("owlv2_detected_boxes_json", "[]") or "[]"),
            "yolo_boxes": json.loads(row.get("yoloworld_detected_boxes_json", "[]") or "[]"),
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
    img = payload.get("pil_img")
    if img is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Slika nije dostupna u repozitoriju", ha="center", va="center")
        ax.axis("off")
        return fig

    w, h = img.size
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=140)
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

    df_meta, meta_parquet_path, boot_logs = load_lightweight_metadata()

    with st.expander("🛠️ Dijagnostika sustava i status učitavanja", expanded=df_meta.empty):
        for log_line in boot_logs:
            st.write(f"- `{log_line}`")
        st.write("- Predmemorija slika: aktiviran spremnik za zadnjih **10 lokacija** (sve godine odjednom).")

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

        max_map_pins = st.slider("Maksimalan broj točaka na karti (za brzinu):", 100, 1500, 500, 100)

    # Apply Sidebar Filters
    df_filtered = df_meta[
        (df_meta["image_year"].isin(selected_years)) &
        (df_meta["corine_class"].isin(selected_corine)) &
        (df_meta[size_metric_col] >= min_size_m2) &
        (df_meta["ens_triage_bucket"].isin(selected_buckets))
    ].copy()

    if selected_zup != "Sve županije (Cijela Hrvatska)":
        df_filtered = df_filtered[df_filtered["zupanija"] == selected_zup]

    if selected_scanner == "Samo AerialWaste (Toplinska mapa / CAM >= 50%)":
        df_filtered = df_filtered[
            (df_filtered["aw_out3_cam_global_max"] >= 0.50) |
            (df_filtered["aw_out1_global_waste_prob"] >= 0.40)
        ]
    elif selected_scanner == "Samo SigLIP / SkyCLIP (Tekstura tla / Rubble)":
        df_filtered = df_filtered[
            (df_filtered["siglip_out6_total_waste_max"] >= 0.45) |
            (df_filtered["skyclip_out6_total_waste_max"] >= 0.45)
        ]
    elif selected_scanner == "Samo Detektori objekata (OWLv2 / YOLO-World hrpe)":
        df_filtered = df_filtered[
            (df_filtered["owlv2_out9_total_waste_max_score"] >= 0.20) |
            (df_filtered["yoloworld_out9_total_waste_max_score"] >= 0.20)
        ]

    df_filtered = df_filtered.sort_values(by="ens_active_learning_score", ascending=False).reset_index(drop=True)

    # Detect if sidebar filters were changed by the user; if so, pan to the first filtered cluster
    current_filter_sig = (
        selected_zup,
        tuple(selected_years),
        tuple(selected_corine),
        selected_scanner,
        size_metric_col,
        min_size_m2,
        tuple(selected_buckets)
    )
    if st.session_state.get("last_filter_sig") != current_filter_sig:
        st.session_state["last_filter_sig"] = current_filter_sig
        if not df_filtered.empty and st.session_state["selected_cluster_id"] not in df_filtered["cluster_id"].values:
            st.session_state["selected_cluster_id"] = int(df_filtered["cluster_id"].iloc[0])

    # --------------------------------------------------------------------------
    # TOP SECTION: SUMMARY KPIS & FOLIUM MAP (SAFE CLICK-TO-LOAD + DIRECT JUMP)
    # --------------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Ukupno snimaka u filteru", len(df_filtered))
    kpi2.metric("Unikatnih ELOO lokacija", df_filtered["cluster_id"].nunique())
    kpi3.metric("🟠 Zahtijeva provjeru", int((df_filtered["ens_triage_bucket"] == "ROUTE_TO_CROWD_DISAGREEMENT").sum()))
    kpi4.metric("🔴 Potvrđeni otpad", int((df_filtered["ens_triage_bucket"] == "AUTO_VERIFIED_WASTE").sum()))

    # Deduplicate by cluster_id for the map & dropdown (1 pin per physical ELOO location)
    cluster_summary = (
        df_filtered.drop_duplicates(subset=["cluster_id"], keep="first")
        .copy()
        .reset_index(drop=True)
    )

    col_map, col_selector = st.columns([2.2, 1])

    with col_selector:
        st.subheader("🎯 Odaberi ili upiši lokaciju")

        # NEW: Direct Cluster ID Input (Evades all active sidebar filters!)
        with st.form("direct_cluster_jump_form", clear_on_submit=False):
            st.caption("🔎 **Izravan skok na lokaciju** (zaobilazi sve filtere):")
            j_col1, j_col2 = st.columns([1.8, 1])
            raw_jump_id = j_col1.text_input(
                "Upiši broj lokacije (Cluster ID):",
                placeholder="npr. 1868",
                label_visibility="collapsed"
            )
            jump_submitted = j_col2.form_submit_button("Učitaj", use_container_width=True, type="primary")

            if jump_submitted and raw_jump_id.strip():
                clean_str = raw_jump_id.strip().lstrip("#")
                if clean_str.lstrip("-").isdigit():
                    target_cid = int(clean_str)
                    if target_cid in df_meta["cluster_id"].values:
                        st.session_state["selected_cluster_id"] = target_cid
                    else:
                        st.warning(f"⚠️ Lokacija #{target_cid} ne postoji u bazi podataka.")
                else:
                    st.warning("⚠️ Unesite samo broj lokacije (npr. 1868).")

        # Build dropdown list; if the active cluster was chosen via direct jump and is outside filters, prepend it!
        filtered_cids = cluster_summary["cluster_id"].astype(int).tolist()
        active_cid = int(st.session_state["selected_cluster_id"])

        if active_cid not in filtered_cids:
            cid_list = [active_cid] + filtered_cids
        else:
            cid_list = filtered_cids

        selected_cluster_id = st.selectbox(
            "Filtrirane lokacije (ili aktivni izravni unos):",
            options=cid_list,
            index=cid_list.index(active_cid),
            format_func=lambda cid: (
                f"Lokacija #{cid} 📌 (Izravni unos — izvan filtera)"
                if cid not in filtered_cids else f"Lokacija #{cid}"
            )
        )
        st.session_state["selected_cluster_id"] = int(selected_cluster_id)

        # Always fetch full history from df_meta so all years are shown regardless of filters
        df_cluster_history = df_meta[df_meta["cluster_id"] == selected_cluster_id].sort_values("image_year")

        st.markdown("**📅 Povijest sumnjive površine kroz godine (m²):**")
        history_chart_df = df_cluster_history[["image_year", "max_detected_area_m2"]].set_index("image_year")
        history_chart_df.columns = ["Maks. detektirana površina (m²)"]
        st.bar_chart(history_chart_df, height=165, color="#ff7f0e")

    with col_map:
        st.caption(f"🗺️ Interaktivna karta (prikazano top {min(len(cluster_summary), max_map_pins)} filtriranih lokacija — **kliknite na točku za učitavanje**):")
        active_lat = float(df_cluster_history.iloc[-1]["centroid_lat"])
        active_lon = float(df_cluster_history.iloc[-1]["centroid_lon"])

        m = folium.Map(location=[active_lat, active_lon], zoom_start=12, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satelitska snimka",
            max_zoom=19,
        ).add_to(m)

        # Render filtered pins
        map_subset = cluster_summary.head(max_map_pins)
        active_drawn = False
        for row in map_subset.itertuples():
            cid = int(row.cluster_id)
            is_active = (cid == selected_cluster_id)
            if is_active:
                active_drawn = True
            folium.CircleMarker(
                location=[row.centroid_lat, row.centroid_lon],
                radius=10 if is_active else 6,
                color="#ffffff" if is_active else row.color_hex,
                weight=3 if is_active else 1,
                fill=True,
                fill_color=row.color_hex,
                fill_opacity=0.9,
                tooltip=f"Lokacija #{cid} | {row.status_short} | Max površina: {row.max_detected_area_m2:.0f} m²",
            ).add_to(m)

        # If the active cluster was loaded via Direct Jump (outside filters), still draw its pin on the map!
        if not active_drawn:
            latest_active = df_cluster_history.iloc[-1]
            folium.CircleMarker(
                location=[float(latest_active["centroid_lat"]), float(latest_active["centroid_lon"])],
                radius=11,
                color="#ffffff",
                weight=3,
                fill=True,
                fill_color=latest_active["color_hex"],
                fill_opacity=0.95,
                tooltip=f"📌 Lokacija #{selected_cluster_id} (Izravni unos) | {latest_active['status_short']}",
            ).add_to(m)

        map_data = st_folium(
            m,
            height=420,
            use_container_width=True,
            returned_objects=["last_object_clicked"]
        )

        if map_data and map_data.get("last_object_clicked") and not cluster_summary.empty:
            click_lat = map_data["last_object_clicked"]["lat"]
            click_lon = map_data["last_object_clicked"]["lng"]
            dists = (cluster_summary["centroid_lat"] - click_lat) ** 2 + (cluster_summary["centroid_lon"] - click_lon) ** 2
            nearest_cid = int(cluster_summary.loc[dists.idxmin(), "cluster_id"])
            if nearest_cid != st.session_state["selected_cluster_id"]:
                st.session_state["selected_cluster_id"] = nearest_cid
                st.rerun()

    st.divider()

    # --------------------------------------------------------------------------
    # BOTTOM SECTION: INSPECTION DESK & 10-LOCATION CACHED GEO FILTERS
    # --------------------------------------------------------------------------
    st.header(f"🔍 Inspekcijski stol: Lokacija #{selected_cluster_id}")

    cluster_bucket = fetch_cluster_cache_bucket(selected_cluster_id, meta_parquet_path)

    cluster_years = df_cluster_history["image_year"].tolist()
    if len(cluster_years) > 1:
        selected_img_year = st.select_slider(
            "⏳ Vremenski klizač (Trenutni prijelaz između već učitanih godina):",
            options=cluster_years,
            value=cluster_years[-1]
        )
    else:
        selected_img_year = cluster_years[0]
        st.info(f"📅 Dostupna snimka iz godine: **{selected_img_year}**")

    active_row = df_cluster_history[df_cluster_history["image_year"] == selected_img_year].iloc[0]
    active_payload = cluster_bucket.get(active_row["file_name"], {})
    bucket_info = TRIAGE_UI_MAP.get(active_row["ens_triage_bucket"], TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])

    desk_left, desk_right = st.columns([1.6, 1])

    with desk_left:
        active_filter = st.radio(
            "🎨 Odaberi Geo Filter (Trenutni prijelaz bez ponovnog učitavanja):",
            options=[
                "📸 Originalna snimka",
                "🔥 Toplinska mapa (Heatmap)",
                "🟩 Skener terena (Pločice)",
                "🎯 Detekcija hrpa (Okviri)",
                "🎛️ Kombinirani prikaz"
            ],
            horizontal=True
        )

        pyramid_scale = "L0_384px"
        tile_thresh = 0.45
        if active_filter == "🟩 Skener terena (Pločice)":
            sub_c1, sub_c2 = st.columns(2)
            pyramid_scale = sub_c1.selectbox("Veličina mreže:", options=["L0_384px", "L1_768px", "Sve razine"])
            tile_thresh = sub_c2.slider("Prag osjetljivosti:", 0.20, 0.90, 0.45, 0.05)

        fig = render_geo_filter_view(
            payload=active_payload,
            meta_row=active_row,
            active_filter=active_filter,
            pyramid_scale=pyramid_scale,
            tile_threshold=tile_thresh
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with desk_right:
        st.subheader("🧠 Što AI modeli vide na ovoj snimci?")
        st.markdown(
            f"<div style='padding:14px; border-radius:8px; border-left: 6px solid {bucket_info['color']}; background-color: rgba(128,128,128,0.1);'>"
            f"<h4 style='margin:0;'>{bucket_info['label']}</h4>"
            f"<p style='margin-top:8px; margin-bottom:0; font-size:14px;'>{bucket_info['explanation']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Ukupni AI indeks sumnje", f"{active_row['warden_confidence_pct']}%")
        m2.metric("Neslaganje između modela", f"{active_row['disagreement_pct']}%")

        m3, m4 = st.columns(2)
        m3.metric("Dominantni tip (AI)", str(active_row["siglip_out11_primary_waste_type"]))
        m4.metric("Maks. detektirana površina", f"{active_row['max_detected_area_m2']:.0f} m²")

        with st.expander("📊 Detaljni rezultati po vrsti AI skenera", expanded=True):
            st.caption("Vršni (lokalni) rezultati po pojedinom skeneru na ovoj snimci:")
            st.progress(
                float(np.clip(active_row["siglip_out6_total_waste_max"], 0.0, 1.0)),
                text=f"Skener teksture tla (SigLIP): {active_row['siglip_out6_total_waste_max']:.0%} ({active_row['siglip_out10_suspect_area_m2']:.0f} m²)"
            )
            st.progress(
                float(np.clip(active_row["skyclip_out6_total_waste_max"], 0.0, 1.0)),
                text=f"Satelitski skener (SkyCLIP): {active_row['skyclip_out6_total_waste_max']:.0%}"
            )
            aw_peak = float(np.clip(active_row.get("aw_out3_cam_global_max", active_row["aw_out1_global_waste_prob"]), 0.0, 1.0))
            st.progress(
                aw_peak,
                text=f"Toplinska mapa (AerialWaste CAM vrh): {aw_peak:.0%} (Žarište: {active_row['aw_hotspot_area_m2']:.0f} m² | Globalno: {active_row['aw_out1_global_waste_prob']:.0%})"
            )
            st.progress(
                float(np.clip(active_row["owlv2_out9_total_waste_max_score"], 0.0, 1.0)),
                text=f"Detektor hrpa (OWLv2): {active_row['owlv2_out9_total_waste_max_score']:.0%} ({active_row['owlv2_out10_total_waste_box_count']} okvira)"
            )
            st.progress(
                float(np.clip(active_row["yoloworld_out9_total_waste_max_score"], 0.0, 1.0)),
                text=f"Detektor objekata (YOLO-World): {active_row['yoloworld_out9_total_waste_max_score']:.0%} ({active_row['yoloworld_out10_total_waste_box_count']} okvira)"
            )

        gmaps_url = f"https://www.google.com/maps/@{active_row['centroid_lat']},{active_row['centroid_lon']},18z/data=!3m1!1e3"
        st.markdown(
            f"**CORINE Tip:** {active_row['corine_class']}  \n"
            f"**GPS:** `{active_row['centroid_lat']:.6f}, {active_row['centroid_lon']:.6f}`  \n"
            f"[🌍 Otvori u Google Maps (Satelitski prikaz)]({gmaps_url})"
        )

        st.divider()
        st.subheader("✍️ Odluka komunalnog redara")
        verdict = None
        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("✅ Potvrdi otpad", use_container_width=True, type="primary"):
            verdict = "CONFIRMED_WASTE"
        if col_b2.button("🚗 Lažni alarm", use_container_width=True):
            verdict = "FALSE_ALARM"
        if col_b3.button("🌲 Čisto", use_container_width=True):
            verdict = "CLEAN"

        if verdict:
            st.success(f"Zabilježeno: {verdict}")


if __name__ == "__main__":
    main()
