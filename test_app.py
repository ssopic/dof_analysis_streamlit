import os
import json
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label as connected_components
import pyarrow.parquet as pq
import streamlit as st
import pydeck as pdk
from huggingface_hub import HfApi, hf_hub_download

# ==============================================================================
# 1. PAGE CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="EcoWarden Croatia | Detekcija Odlagališta",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROD_REPO_ID = "SilvioS/eloo_dgu_scanner_output"
TEST_REPO_ID = "SilvioS/eloo_dgu_triage_results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
CACHE_DIR = "./hf_streamlit_cache"

# Standardized semantic colors from WardenViz
SEMANTIC_CLASS_COLORS = {
    "Clean Nature":        "#2ca02c",  # Green
    "Urban / Roofs":       "#1f77b4",  # Blue
    "Natural Karst Rock":  "#7f7f7f",  # Gray
    "Construction Rubble": "#ff7f0e",  # Orange
    "Municipal Garbage":   "#d62728",  # Red
    "Tires / Bulky Waste": "#9467bd",  # Purple
    "Unknown":             "#17becf"   # Cyan
}

# User-friendly translations for non-technical Wardens
TRIAGE_UI_MAP = {
    "AUTO_VERIFIED_WASTE": {
        "label": "🔴 Visoka vjerojatnost otpada (Potvrđeno)",
        "short": "Potvrđeni otpad",
        "color": [214, 39, 40, 210],
        "hex": "#d62728",
        "explanation": "Svi AI skeneri (i teksturni i objektni) snažno se slažu da se na ovoj lokaciji nalazi odlagalište otpada."
    },
    "ROUTE_TO_CROWD_DISAGREEMENT": {
        "label": "🟠 Potrebna provjera (Neslaganje modela)",
        "short": "Potrebna provjera",
        "color": [255, 127, 14, 210],
        "hex": "#ff7f0e",
        "explanation": (
            "Skeneri terena vide sumnjivu teksturu (npr. građevinski šut ili iskop), "
            "dok detektori objekata ne vide jasne hrpe. **Napomena:** Ovo se često događa "
            "u urbanim zonama, zbog žutih automobila, krovova ili prirodnog krša. Provjerite pomoću Geo Filtera!"
        )
    },
    "ROUTE_TO_CROWD_AMBIGUOUS": {
        "label": "🟡 Granični slučaj (Srednja sigurnost)",
        "short": "Granični slučaj",
        "color": [230, 184, 0, 210],
        "hex": "#e6b800",
        "explanation": "Modeli su umjereno nesigurni jer lokacija izgleda granično. Preporučuje se vizualni pregled."
    },
    "AUTO_CLEAN": {
        "label": "🟢 Čisto područje (Nizak rizik)",
        "short": "Čisto područje",
        "color": [44, 160, 44, 180],
        "hex": "#2ca02c",
        "explanation": "Svi modeli se slažu da snimka prikazuje čistu prirodu, uobičajene krovove ili prirodni krš."
    }
}

# Approximate bounding boxes [lon_min, lat_min, lon_max, lat_max] for Croatian Županije
# (Used as a fallback if your metadata JSONL does not yet have a 'zupanija' column)
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

# Columns to load into RAM for the macro map (excludes image_jpeg_bytes & heavy JSONs!)
LIGHTWEIGHT_META_COLS = [
    "chunk_id", "file_name", "cluster_id", "image_year",
    "corine_class", "centroid_lon", "centroid_lat",
    "img_width_px", "img_height_px", "meters_per_pixel",
    "aw_out1_global_waste_prob", "aw_out4_cam_area_pct_gt_05",
    "siglip_out6_total_waste_max", "siglip_out10_suspect_area_m2",
    "siglip_out11_primary_waste_type", "skyclip_out6_total_waste_max",
    "owlv2_out9_total_waste_max_score", "owlv2_out10_total_waste_box_count",
    "owlv2_out11_detected_boxes_area_m2",
    "yoloworld_out9_total_waste_max_score", "yoloworld_out10_total_waste_box_count",
    "yoloworld_out11_detected_boxes_area_m2",
    "ens_mean_waste_prob", "ens_max_min_diff",
    "ens_active_learning_score", "ens_triage_bucket"
]


# ==============================================================================
# 2. DATA LOADING & LAZY IMAGE EXTRACTION (MEMORY SAFE)
# ==============================================================================
def assign_approx_zupanija(lon: float, lat: float) -> str:
    """Fallback spatial helper if metadata doesn't explicitly contain a 'zupanija' column."""
    for zup, (lon_min, lat_min, lon_max, lat_max) in ZUPANIJE_BBOX.items():
        if zup == "Sve županije (Cijela Hrvatska)":
            continue
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return zup
    return "Ostale županije"


@st.cache_data(ttl=3600, show_spinner="Učitavanje indeksa lokacija (bez slika)...")
def load_lightweight_metadata(max_chunks_to_index: int = 5, metadata_jsonl_path: str = None) -> pd.DataFrame:
    """
    Loads ONLY lightweight metadata into memory.
    If a pre-compiled metadata.jsonl exists, it loads that directly.
    Otherwise, it reads only the non-binary metadata columns from the Hugging Face Parquet chunks.
    """
    if metadata_jsonl_path and os.path.exists(metadata_jsonl_path):
        df = pd.read_json(metadata_jsonl_path, lines=True)
    else:
        os.makedirs(CACHE_DIR, exist_ok=True)
        hf_api = HfApi(token=HF_TOKEN)
        repo_to_use = PROD_REPO_ID

        try:
            all_files = hf_api.list_repo_files(repo_id=PROD_REPO_ID, repo_type="dataset")
            pq_files = [f for f in all_files if f.startswith("triage_results/") and f.endswith("_triage.parquet")]
            if not pq_files:
                raise FileNotFoundError("No parquet chunks in PROD_REPO_ID")
        except Exception:
            repo_to_use = TEST_REPO_ID
            all_files = hf_api.list_repo_files(repo_id=TEST_REPO_ID, repo_type="dataset")
            pq_files = [f for f in all_files if f.startswith("triage_results/") and f.endswith("_triage.parquet")]

        def _chunk_sort_key(path_str: str):
            base = os.path.basename(path_str).replace("_triage.parquet", "")
            parts = base.split("_")
            return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 999999

        pq_files = sorted(pq_files, key=_chunk_sort_key)[:max_chunks_to_index]

        dfs = []
        for pq_subpath in pq_files:
            local_pq = hf_hub_download(
                repo_id=repo_to_use,
                filename=pq_subpath,
                repo_type="dataset",
                local_dir=CACHE_DIR,
                token=HF_TOKEN
            )
            # PyArrow column projection: never loads image_jpeg_bytes into RAM during indexing
            schema_cols = pq.read_schema(local_pq).names
            cols_to_read = [c for c in LIGHTWEIGHT_META_COLS if c in schema_cols]
            df_part = pd.read_parquet(local_pq, columns=cols_to_read)
            df_part["_source_repo"] = repo_to_use
            dfs.append(df_part)

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not df.empty:
        if "zupanija" not in df.columns:
            df["zupanija"] = [
                assign_approx_zupanija(lon, lat)
                for lon, lat in zip(df["centroid_lon"], df["centroid_lat"])
            ]
        # Map colors and friendly labels for PyDeck
        df["status_label"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["label"])
        df["status_short"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["short"])
        df["color_rgba"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["color"])
        # Friendly 0-100% confidence score for Wardens
        df["warden_confidence_pct"] = (df["ens_mean_waste_prob"] * 100).round(0).astype(int)
        df["disagreement_pct"] = (df["ens_max_min_diff"] * 100).round(0).astype(int)

    return df


def fetch_single_image_payload(chunk_id: str, file_name: str, repo_id: str = PROD_REPO_ID) -> dict:
    """
    On-demand loader: Fetches the heavy JPEG bytes and bounding box JSON strings
    ONLY for the single selected orthophoto.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    remote_subpath = f"triage_results/{chunk_id}_triage.parquet"

    try:
        local_pq = hf_hub_download(
            repo_id=repo_id,
            filename=remote_subpath,
            repo_type="dataset",
            local_dir=CACHE_DIR,
            token=HF_TOKEN
        )
    except Exception:
        local_pq = hf_hub_download(
            repo_id=TEST_REPO_ID,
            filename=remote_subpath,
            repo_type="dataset",
            local_dir=CACHE_DIR,
            token=HF_TOKEN
        )

    heavy_cols = [
        "file_name", "image_jpeg_bytes", "repo_image_path",
        "aw_cam_grid_32x32_json", "all_tile_records_json",
        "owlv2_detected_boxes_json", "yoloworld_detected_boxes_json"
    ]
    # Filter directly via PyArrow so we only pull the matching file_name row into memory
    df_row = pd.read_parquet(
        local_pq,
        columns=heavy_cols,
        filters=[("file_name", "==", file_name)]
    )

    if df_row.empty:
        raise ValueError(f"Slika {file_name} nije pronađena u {chunk_id}.")

    row = df_row.iloc[0]

    # Decode RGB PIL image from embedded lossless JPEG bytes
    if row["image_jpeg_bytes"] is not None and len(row["image_jpeg_bytes"]) > 0:
        pil_img = Image.open(BytesIO(row["image_jpeg_bytes"])).convert("RGB")
    else:
        img_path = hf_hub_download(
            repo_id=repo_id,
            filename=str(row["repo_image_path"]),
            repo_type="dataset",
            local_dir=CACHE_DIR,
            token=HF_TOKEN
        )
        pil_img = Image.open(img_path).convert("RGB")

    return {
        "pil_img": pil_img,
        "aw_cam_32": np.array(json.loads(row.get("aw_cam_grid_32x32_json", "[]")), dtype=np.float32),
        "tiles": json.loads(row.get("all_tile_records_json", "[]")),
        "owlv2_boxes": json.loads(row.get("owlv2_detected_boxes_json", "[]")),
        "yolo_boxes": json.loads(row.get("yoloworld_detected_boxes_json", "[]")),
    }


# ==============================================================================
# 3. INSTAGRAM-STYLE "GEO FILTER" RENDERER (ZERO-GPU PIL + MATPLOTLIB)
# ==============================================================================
def render_geo_filter_view(
    payload: dict,
    meta_row: pd.Series,
    active_filter: str,
    pyramid_scale: str = "L0_384px",
    tile_threshold: float = 0.45,
    box_min_score: float = 0.05
) -> plt.Figure:
    """
    Renders the selected DGU DOF5 orthophoto with the chosen "Geo Filter" overlay.
    """
    img = payload["pil_img"]
    w, h = img.size

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    ax.imshow(img)
    ax.axis("off")

    # FILTER 1: Raw Orthophoto
    if active_filter == "📸 Originalna snimka":
        ax.set_title(
            f"Originalni DGU DOF5 Ortofoto ({meta_row['meters_per_pixel']} m/px) — Godina: {meta_row['image_year']}",
            fontsize=11, fontweight="bold", pad=10
        )

    # FILTER 2: AerialWaste CAM Heatmap Filter
    elif active_filter == "🔥 Toplinska mapa (Heatmap)":
        cam_32 = payload["aw_cam_32"]
        if cam_32.size > 0:
            cam_pil = Image.fromarray(cam_32, mode="F").resize((w, h), resample=Image.BICUBIC)
            cam_upsampled = np.clip(np.array(cam_pil, dtype=np.float32), 0.0, 1.0)
            im = ax.imshow(cam_upsampled, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03, label="Intenzitet sumnje (AerialWaste CAM)")
        ax.set_title(
            f"Geo Filter: Toplinska mapa kontaminacije (Površina >50%: {meta_row['aw_out4_cam_area_pct_gt_05']:.1f}%)",
            fontsize=11, fontweight="bold", pad=10
        )

    # FILTER 3: Terrain Scanner (SigLIP / SkyCLIP Semantic Tiles)
    elif active_filter == "🟩 Skener terena (Pločice)":
        tiles = payload["tiles"]
        if pyramid_scale != "Sve razine":
            tiles = [t for t in tiles if t.get("level_id") == pyramid_scale]

        for t in tiles:
            x1, y1, x2, y2 = t["bbox_px"]
            w_prob = max(
                float(t.get("siglip_out6_total_waste_prob", 0.0)),
                float(t.get("skyclip_out6_total_waste_prob", 0.0))
            )
            dom_cls = t.get("siglip_dominant_class", "Unknown")
            is_flag = w_prob >= tile_threshold
            col = SEMANTIC_CLASS_COLORS.get(dom_cls, "#17becf")

            if is_flag:
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2.2, edgecolor=col, facecolor=col, alpha=0.25
                ))
                ax.text(
                    x1 + 8, y1 + 26,
                    f"{dom_cls}\n({w_prob:.0%})",
                    color="white", fontsize=7.5, fontweight="bold",
                    bbox=dict(facecolor=col, alpha=0.8, pad=1.5, edgecolor="none")
                )
            else:
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=0.6, edgecolor="white", facecolor="none", alpha=0.35
                ))

        ax.set_title(
            f"Geo Filter: Skener terena — Sumnjiva površina: {meta_row['siglip_out10_suspect_area_m2']:.0f} m²",
            fontsize=11, fontweight="bold", pad=10
        )

    # FILTER 4: Debris Outlines (OWLv2 + YOLO-Worldv2 Object Boxes)
    elif active_filter == "🎯 Detekcija hrpa (Okviri)":
        all_boxes = [
            ("OWLv2", b) for b in payload["owlv2_boxes"]
        ] + [
            ("YOLO", b) for b in payload["yolo_boxes"]
        ]
        drawn_count = 0
        for model_tag, b in all_boxes:
            score = float(b.get("score", 0.0))
            if score < box_min_score:
                continue
            drawn_count += 1
            bx1, by1, bx2, by2 = b["bbox_px"]
            col = b.get("color", "#ff7f0e" if model_tag == "OWLv2" else "#d62728")
            ax.add_patch(patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=2.4, edgecolor=col, facecolor=col, alpha=0.22
            ))
            ax.text(
                bx1 + 4, max(16, by1 - 6),
                f"{b.get('label', 'Otpad')} ({score:.0%})",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(facecolor=col, alpha=0.85, pad=1.5, edgecolor="none")
            )
        ax.set_title(
            f"Geo Filter: Detekcija pojedinačnih hrpa ({drawn_count} detektiranih objekata)",
            fontsize=11, fontweight="bold", pad=10
        )

    # FILTER 5: Combined All-in-One Overlay
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
                ax.add_patch(patches.Rectangle(
                    (bx1, by1), bx2 - bx1, by2 - by1,
                    linewidth=2.2, edgecolor=col, facecolor="none"
                ))
        ax.set_title("Geo Filter: Kombinirani prikaz (Toplinska mapa + Okviri hrpa)", fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


# ==============================================================================
# 4. MAIN APPLICATION LAYOUT
# ==============================================================================
def main():
    st.title("🛰️ EcoWarden Croatia — Sustav za nadzor divljih odlagališta")
    st.caption("Interaktivni pregled DGU DOF5 ortofoto snimaka (50 cm/px) pomoću 5-modelnog AI sustava i Geo Filtera.")

    # Load lightweight metadata into cached memory
    with st.sidebar:
        st.header("⚙️ Postavke podataka")
        chunks_to_load = st.number_input("Broj paketa (chunkova) za indeksiranje", min_value=1, max_value=193, value=3)

    df_meta = load_lightweight_metadata(max_chunks_to_index=int(chunks_to_load))

    if df_meta.empty:
        st.error("Nema dostupnih podataka u Hugging Face repozitoriju.")
        return

    # --------------------------------------------------------------------------
    # SIDEBAR: GLOBAL LOCATION, YEAR & STATUS FILTERS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.divider()
        st.header("📍 1. Odabir lokacije i vremena")

        available_zup = ["Sve županije (Cijela Hrvatska)"] + sorted(df_meta["zupanija"].unique().tolist())
        selected_zup = st.selectbox("Županija / Regija:", available_zup)

        available_years = sorted(df_meta["image_year"].unique().tolist())
        selected_years = st.multiselect(
            "Godine snimanja na karti:",
            options=available_years,
            default=available_years
        )

        st.header("🚦 2. Status AI provjere")
        all_buckets = list(TRIAGE_UI_MAP.keys())
        selected_buckets = st.multiselect(
            "Filtriraj po prioritetu:",
            options=all_buckets,
            default=all_buckets,
            format_func=lambda b: TRIAGE_UI_MAP[b]["label"]
        )

    # Filter metadata for the Macro Map
    df_filtered = df_meta[
        (df_meta["image_year"].isin(selected_years)) &
        (df_meta["ens_triage_bucket"].isin(selected_buckets))
    ].copy()

    if selected_zup != "Sve županije (Cijela Hrvatska)":
        df_filtered = df_filtered[df_filtered["zupanija"] == selected_zup]

    # Sort by Active Learning Priority Score so highest priority sites are first
    df_filtered = df_filtered.sort_values(by="ens_active_learning_score", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------------------------
    # TOP SECTION: MACRO MAP & SUMMARY KPIs
    # --------------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Ukupno snimaka u prikazu", len(df_filtered))
    kpi2.metric("Unikatnih ELOO lokacija (Clustera)", df_filtered["cluster_id"].nunique())
    kpi3.metric(
        "🟠 Zahtijeva ljudsku provjeru",
        int((df_filtered["ens_triage_bucket"] == "ROUTE_TO_CROWD_DISAGREEMENT").sum())
    )
    kpi4.metric(
        "🔴 Automatski potvrđen otpad",
        int((df_filtered["ens_triage_bucket"] == "AUTO_VERIFIED_WASTE").sum())
    )

    col_map, col_selector = st.columns([2.2, 1])

    with col_selector:
        st.subheader("🎯 Odaberi lokaciju za inspekciju")
        if df_filtered.empty:
            st.warning("Nema lokacija za odabrane filtere.")
            return

        # Group unique clusters with their top priority status
        cluster_summary = (
            df_filtered.groupby("cluster_id", as_index=False)
            .first()[["cluster_id", "corine_class", "status_short", "ens_active_learning_score", "zupanija"]]
            .sort_values(by="ens_active_learning_score", ascending=False)
        )

        selected_cluster_id = st.selectbox(
            "Odaberi ELOO Lokaciju (Cluster ID):",
            options=cluster_summary["cluster_id"].tolist(),
            format_func=lambda cid: (
                f"Lokacija #{cid} | "
                f"{cluster_summary.loc[cluster_summary['cluster_id']==cid, 'status_short'].values[0]} "
                f"({cluster_summary.loc[cluster_summary['cluster_id']==cid, 'corine_class'].values[0][:22]}...)"
            )
        )

        # Subset all time-series rows for the selected cluster_id across the entire dataset
        df_cluster_history = df_meta[df_meta["cluster_id"] == selected_cluster_id].sort_values("image_year")

        st.markdown("**📅 Povijest širenja na ovoj lokaciji:**")
        history_chart_df = df_cluster_history[["image_year", "siglip_out10_suspect_area_m2"]].set_index("image_year")
        history_chart_df.columns = ["Sumnjiva površina (m²)"]
        st.bar_chart(history_chart_df, height=180, color="#ff7f0e")

    with col_map:
        # Center map on the currently selected cluster
        active_lat = float(df_cluster_history.iloc[-1]["centroid_lat"])
        active_lon = float(df_cluster_history.iloc[-1]["centroid_lon"])

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position=["centroid_lon", "centroid_lat"],
            get_fill_color="color_rgba",
            get_radius=180,
            radius_min_pixels=6,
            radius_max_pixels=18,
            pickable=True,
            auto_highlight=True,
        )

        # Highlight ring around the currently selected cluster
        highlight_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_cluster_history.tail(1),
            get_position=["centroid_lon", "centroid_lat"],
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[255, 255, 255, 255],
            get_radius=320,
            radius_min_pixels=12,
            radius_max_pixels=26,
            stroked=True,
            line_width_min_pixels=3,
        )

        view_state = pdk.ViewState(latitude=active_lat, longitude=active_lon, zoom=11.5, pitch=0)
        tooltip = {
            "html": (
                "<b>Lokacija (Cluster):</b> #{cluster_id}<br/>"
                "<b>Godina snimke:</b> {image_year}<br/>"
                "<b>Status:</b> {status_short}<br/>"
                "<b>Sumnjiva površina:</b> {siglip_out10_suspect_area_m2} m²<br/>"
                "<b>Zemljište (CORINE):</b> {corine_class}"
            ),
            "style": {"backgroundColor": "#1e1e1e", "color": "white", "fontSize": "12px"}
        }

        st.pydeck_chart(pdk.Deck(
            layers=[scatter_layer, highlight_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/satellite-streets-v12"
        ))

    st.divider()

    # --------------------------------------------------------------------------
    # BOTTOM SECTION: WARDEN INSPECTION DESK & "GEO FILTERS"
    # --------------------------------------------------------------------------
    st.header(f"🔍 Inspekcijski stol: Lokacija #{selected_cluster_id}")

    # Time-series year selector for this specific cluster
    cluster_years = df_cluster_history["image_year"].tolist()
    if len(cluster_years) > 1:
        selected_img_year = st.select_slider(
            "⏳ Vremenski klizač (Odaberite godinu ortofoto snimke za usporedbu):",
            options=cluster_years,
            value=cluster_years[-1]
        )
    else:
        selected_img_year = cluster_years[0]
        st.info(f"📅 Za ovu lokaciju dostupna je snimka iz godine: **{selected_img_year}**")

    # Get the exact metadata row for (selected_cluster_id, selected_img_year)
    active_row = df_cluster_history[df_cluster_history["image_year"] == selected_img_year].iloc[0]
    bucket_info = TRIAGE_UI_MAP.get(active_row["ens_triage_bucket"], TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])

    desk_left, desk_right = st.columns([1.6, 1])

    with desk_left:
        # Instagram-style "Geo Filter" Bar
        active_filter = st.radio(
            "🎨 Odaberi Geo Filter (Pogled kroz oči AI skenera):",
            options=[
                "📸 Originalna snimka",
                "🔥 Toplinska mapa (Heatmap)",
                "🟩 Skener terena (Pločice)",
                "🎯 Detekcija hrpa (Okviri)",
                "🎛️ Kombinirani prikaz"
            ],
            horizontal=True
        )

        # Optional sub-controls if Terrain Scanner is selected
        pyramid_scale = "L0_384px"
        tile_thresh = 0.45
        if active_filter == "🟩 Skener terena (Pločice)":
            sub_c1, sub_c2 = st.columns(2)
            pyramid_scale = sub_c1.selectbox(
                "Veličina mreže skenera:",
                options=["L0_384px", "L1_768px", "Sve razine"],
                format_func=lambda x: "Detaljna mreža (L0 - 384px)" if x == "L0_384px" else ("Širi kontekst (L1 - 768px)" if x == "L1_768px" else "Sve razine")
            )
            tile_thresh = sub_c2.slider("Prag osjetljivosti skenera:", 0.20, 0.90, 0.45, 0.05)

        # Lazy-load ONLY the currently selected image into Session State (clears previous image from RAM!)
        state_key = f"{active_row['chunk_id']}_{active_row['file_name']}"
        if st.session_state.get("loaded_image_key") != state_key:
            with st.spinner(f"Preuzimanje ortofoto snimke {active_row['file_name']}..."):
                st.session_state["loaded_payload"] = fetch_single_image_payload(
                    chunk_id=active_row["chunk_id"],
                    file_name=active_row["file_name"],
                    repo_id=active_row.get("_source_repo", PROD_REPO_ID)
                )
                st.session_state["loaded_image_key"] = state_key

        fig = render_geo_filter_view(
            payload=st.session_state["loaded_payload"],
            meta_row=active_row,
            active_filter=active_filter,
            pyramid_scale=pyramid_scale,
            tile_threshold=tile_thresh
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)  # Immediately release Matplotlib figure memory

    with desk_right:
        st.subheader("🧠 Što AI modeli vide na ovoj snimci?")

        # Plain-language status banner
        st.markdown(
            f"<div style='padding:14px; border-radius:8px; border-left: 6px solid {bucket_info['hex']}; background-color: rgba(128,128,128,0.1);'>"
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
        m3.metric("Dominantni tip (AI procjena)", str(active_row["siglip_out11_primary_waste_type"]))
        m4.metric("Sumnjiva površina terena", f"{active_row['siglip_out10_suspect_area_m2']:.0f} m²")

        with st.expander("📊 Detaljni rezultati po vrsti AI skenera (za dublju analizu)", expanded=True):
            st.caption("Usporedba kako različite vrste umjetne inteligencije tumače ovu snimku:")
            st.progress(
                float(np.clip(active_row["siglip_out6_total_waste_max"], 0.0, 1.0)),
                text=f"Skener teksture tla (SigLIP): {active_row['siglip_out6_total_waste_max']:.0%}"
            )
            st.progress(
                float(np.clip(active_row["skyclip_out6_total_waste_max"], 0.0, 1.0)),
                text=f"Satelitski skener terena (SkyCLIP): {active_row['skyclip_out6_total_waste_max']:.0%}"
            )
            st.progress(
                float(np.clip(active_row["aw_out1_global_waste_prob"], 0.0, 1.0)),
                text=f"Detektor globalne deponije (AerialWaste): {active_row['aw_out1_global_waste_prob']:.0%}"
            )
            st.progress(
                float(np.clip(active_row["owlv2_out9_total_waste_max_score"], 0.0, 1.0)),
                text=f"Detektor pojedinačnih hrpa (OWLv2): {active_row['owlv2_out9_total_waste_max_score']:.0%} ({active_row['owlv2_out10_total_waste_box_count']} okvira)"
            )
            st.progress(
                float(np.clip(active_row["yoloworld_out9_total_waste_max_score"], 0.0, 1.0)),
                text=f"Brzi detektor objekata (YOLO-World): {active_row['yoloworld_out9_total_waste_max_score']:.0%} ({active_row['yoloworld_out10_total_waste_box_count']} okvira)"
            )

        st.markdown(
            f"**Kategorija zemljišta (CORINE):** {active_row['corine_class']}  \n"
            f"**GPS Koordinate (WGS84):** `{active_row['centroid_lat']:.6f}, {active_row['centroid_lon']:.6f}`  \n"
            f"[🌍 Otvori točnu lokaciju u Google Maps](https://www.google.com/maps/search/?api=1&query={active_row['centroid_lat']},{active_row['centroid_lon']})"
        )

        st.divider()
        st.subheader("✍️ Odluka komunalnog redara / Wardena")
        st.caption("Pomozite sustavu da nauči razlikovati stvarni otpad od lažnih alarma (automobila, krovova, krša):")

        warden_note = st.text_input("Kratka napomena (opcionalno, npr. 'Žuti auto na parkiralištu' ili 'Građevinski šut uz šumu'):")

        b_col1, b_col2, b_col3 = st.columns(3)
        verdict = None
        if b_col1.button("✅ Potvrdi otpad", use_container_width=True, type="primary"):
            verdict = "CONFIRMED_WASTE"
        if b_col2.button("🚗 Lažni alarm", use_container_width=True):
            verdict = "FALSE_ALARM_URBAN_OR_CAR"
        if b_col3.button("🌲 Čisto", use_container_width=True):
            verdict = "CLEAN_NATURE"

        if verdict:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "file_name": active_row["file_name"],
                "cluster_id": int(active_row["cluster_id"]),
                "image_year": str(active_row["image_year"]),
                "ai_triage_bucket": active_row["ens_triage_bucket"],
                "warden_verdict": verdict,
                "note": warden_note
            }
            if "warden_logs" not in st.session_state:
                st.session_state["warden_logs"] = []
            st.session_state["warden_logs"].append(log_entry)
            st.success(f"Zabilježena odluka: **{verdict}** za `{active_row['file_name']}`!")

        if st.session_state.get("warden_logs"):
            df_logs = pd.DataFrame(st.session_state["warden_logs"])
            st.download_button(
                "📥 Preuzmi zapisnik provjera (CSV)",
                data=df_logs.to_csv(index=False).encode("utf-8"),
                file_name="ecowarden_verifikacije.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
