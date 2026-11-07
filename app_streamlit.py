import os
import json
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import folium
from streamlit_folium import st_folium
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
HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
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
        "explanation": "Skeneri terena vide sumnjivu teksturu, dok detektori objekata ne vide jasne hrpe. Provjerite Geo Filtere!"
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


# ==============================================================================
# 2. DATA LOADING & CACHED IMAGE PAYLOAD FETCHING
# ==============================================================================
def assign_approx_zupanija(lon: float, lat: float) -> str:
    for zup, (lon_min, lat_min, lon_max, lat_max) in ZUPANIJE_BBOX.items():
        if zup == "Sve županije (Cijela Hrvatska)":
            continue
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return zup
    return "Ostale županije"


@st.cache_data(ttl=3600)
def load_lightweight_metadata() -> pd.DataFrame:
    """Loads metadata_index.parquet directly from Hugging Face with error reporting."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        local_path = hf_hub_download(
            repo_id="SilvioS/metafornow",
            filename="metadata_index.parquet",
            repo_type="dataset",
            local_dir=CACHE_DIR,
            token=HF_TOKEN
        )
        df = pd.read_parquet(local_path)
    except Exception as e:
        st.error(f"⚠️ Detektirana greška pri učitavanju metatačaka: {str(e)}")
        return pd.DataFrame()

    if not df.empty:
        if "zupanija" not in df.columns:
            df["zupanija"] = [assign_approx_zupanija(lon, lat) for lon, lat in zip(df["centroid_lon"], df["centroid_lat"])]
        df["status_label"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["label"])
        df["status_short"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["short"])
        df["color_hex"] = df["ens_triage_bucket"].map(lambda b: TRIAGE_UI_MAP.get(b, TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])["color"])
        df["warden_confidence_pct"] = (df["ens_mean_waste_prob"] * 100).round(0).astype(int)
        df["disagreement_pct"] = (df["ens_max_min_diff"] * 100).round(0).astype(int)

    return df


@st.cache_data(show_spinner="Preuzimanje ortofoto snimke...")
def fetch_single_image_payload(chunk_id: str, file_name: str, repo_id: str = PROD_REPO_ID) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    remote_subpath = f"triage_results/{chunk_id}_triage.parquet"

    try:
        local_pq = hf_hub_download(repo_id=repo_id, filename=remote_subpath, repo_type="dataset", local_dir=CACHE_DIR, token=HF_TOKEN)
    except Exception:
        local_pq = hf_hub_download(repo_id=TEST_REPO_ID, filename=remote_subpath, repo_type="dataset", local_dir=CACHE_DIR, token=HF_TOKEN)

    heavy_cols = [
        "file_name", "image_jpeg_bytes", "repo_image_path",
        "aw_cam_grid_32x32_json", "all_tile_records_json",
        "owlv2_detected_boxes_json", "yoloworld_detected_boxes_json"
    ]
    df_row = pd.read_parquet(local_pq, columns=heavy_cols, filters=[("file_name", "==", file_name)])

    if df_row.empty:
        raise ValueError(f"Slika {file_name} nije pronađena.")

    row = df_row.iloc[0]
    if row["image_jpeg_bytes"] is not None and len(row["image_jpeg_bytes"]) > 0:
        pil_img = Image.open(BytesIO(row["image_jpeg_bytes"])).convert("RGB")
    else:
        img_path = hf_hub_download(repo_id=repo_id, filename=str(row["repo_image_path"]), repo_type="dataset", local_dir=CACHE_DIR, token=HF_TOKEN)
        pil_img = Image.open(img_path).convert("RGB")

    return {
        "pil_img": pil_img,
        "aw_cam_32": np.array(json.loads(row.get("aw_cam_grid_32x32_json", "[]")), dtype=np.float32),
        "tiles": json.loads(row.get("all_tile_records_json", "[]")),
        "owlv2_boxes": json.loads(row.get("owlv2_detected_boxes_json", "[]")),
        "yolo_boxes": json.loads(row.get("yoloworld_detected_boxes_json", "[]")),
    }


# ==============================================================================
# 3. GEO FILTER RENDERER
# ==============================================================================
def render_geo_filter_view(payload: dict, meta_row: pd.Series, active_filter: str, pyramid_scale: str = "L0_384px", tile_threshold: float = 0.45, box_min_score: float = 0.05) -> plt.Figure:
    img = payload["pil_img"]
    w, h = img.size

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    ax.imshow(img)
    ax.axis("off")

    if active_filter == "📸 Originalna snimka":
        ax.set_title(f"Originalni DGU DOF5 Ortofoto ({meta_row['meters_per_pixel']} m/px) — Godina: {meta_row['image_year']}", fontsize=11, fontweight="bold", pad=10)

    elif active_filter == "🔥 Toplinska mapa (Heatmap)":
        cam_32 = payload["aw_cam_32"]
        if cam_32.size > 0:
            cam_pil = Image.fromarray(cam_32, mode="F").resize((w, h), resample=Image.BICUBIC)
            cam_upsampled = np.clip(np.array(cam_pil, dtype=np.float32), 0.0, 1.0)
            im = ax.imshow(cam_upsampled, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03, label="Intenzitet sumnje (AerialWaste CAM)")
        ax.set_title(f"Geo Filter: Toplinska mapa kontaminacije (Površina >50%: {meta_row['aw_out4_cam_area_pct_gt_05']:.1f}%)", fontsize=11, fontweight="bold", pad=10)

    elif active_filter == "🟩 Skener terena (Pločice)":
        tiles = payload["tiles"]
        if pyramid_scale != "Sve razine":
            tiles = [t for t in tiles if t.get("level_id") == pyramid_scale]

        for t in tiles:
            x1, y1, x2, y2 = t["bbox_px"]
            w_prob = t.get("w_prob", 0.0)
            dom_cls = t.get("dom_cls", "Unknown")
            is_flag = w_prob >= tile_threshold
            col = SEMANTIC_CLASS_COLORS.get(dom_cls, "#17becf")

            if is_flag:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.2, edgecolor=col, facecolor=col, alpha=0.25))
                ax.text(x1 + 8, y1 + 26, f"{dom_cls}\n({w_prob:.0%})", color="white", fontsize=7.5, fontweight="bold", bbox=dict(facecolor=col, alpha=0.8, pad=1.5, edgecolor="none"))
            else:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.6, edgecolor="white", facecolor="none", alpha=0.35))
        ax.set_title(f"Geo Filter: Skener terena — Sumnjiva površina: {meta_row['siglip_out10_suspect_area_m2']:.0f} m²", fontsize=11, fontweight="bold", pad=10)

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
            ax.text(bx1 + 4, max(16, by1 - 6), f"{b.get('label', 'Otpad')} ({score:.0%})", color="white", fontsize=8, fontweight="bold", bbox=dict(facecolor=col, alpha=0.85, pad=1.5, edgecolor="none"))
        ax.set_title(f"Geo Filter: Detekcija pojedinačnih hrpa ({drawn_count} detektiranih objekata)", fontsize=11, fontweight="bold", pad=10)

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

    df_meta = load_lightweight_metadata()
    if df_meta.empty:
        st.error("Nema dostupnih podataka.")
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
                "Samo AerialWaste (Toplinska deponija)",
                "Samo SigLIP / SkyCLIP (Tekstura tla / Rubble)",
                "Samo Detektori objekata (OWLv2 / YOLO-World hrpe)"
            ]
        )

        st.divider()
        st.subheader("📐 Minimalna veličina sumnjive zone")
        min_size_m2 = st.slider("Minimalna površina (m²):", 0, 5000, 0, 100)

        st.divider()
        st.header("🚦 Status AI provjere")
        all_buckets = list(TRIAGE_UI_MAP.keys())
        selected_buckets = st.multiselect(
            "Filtriraj po prioritetu:",
            options=all_buckets,
            default=all_buckets,
            format_func=lambda b: TRIAGE_UI_MAP[b]["label"]
        )

    # Apply Sidebar Filters
    df_filtered = df_meta[
        (df_meta["image_year"].isin(selected_years)) &
        (df_meta["corine_class"].isin(selected_corine)) &
        (df_meta["max_detected_area_m2"] >= min_size_m2) &
        (df_meta["ens_triage_bucket"].isin(selected_buckets))
    ].copy()

    if selected_zup != "Sve županije (Cijela Hrvatska)":
        df_filtered = df_filtered[df_filtered["zupanija"] == selected_zup]

    if selected_scanner == "Samo AerialWaste (Toplinska deponija)":
        df_filtered = df_filtered[df_filtered["aw_out1_global_waste_prob"] >= 0.40]
    elif selected_scanner == "Samo SigLIP / SkyCLIP (Tekstura tla / Rubble)":
        df_filtered = df_filtered[(df_filtered["siglip_out6_total_waste_max"] >= 0.45) | (df_filtered["skyclip_out6_total_waste_max"] >= 0.45)]
    elif selected_scanner == "Samo Detektori objekata (OWLv2 / YOLO-World hrpe)":
        df_filtered = df_filtered[(df_filtered["owlv2_out9_total_waste_max_score"] >= 0.20) | (df_filtered["yoloworld_out9_total_waste_max_score"] >= 0.20)]

    df_filtered = df_filtered.sort_values(by="ens_active_learning_score", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------------------------
    # TOP SECTION: SUMMARY KPIS & FOLIUM MAP (CLICK TO LOAD)
    # --------------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Ukupno snimaka u prikazu", len(df_filtered))
    kpi2.metric("Unikatnih ELOO lokacija", df_filtered["cluster_id"].nunique())
    kpi3.metric("🟠 Zahtijeva provjeru", int((df_filtered["ens_triage_bucket"] == "ROUTE_TO_CROWD_DISAGREEMENT").sum()))
    kpi4.metric("🔴 Potvrđeni otpad", int((df_filtered["ens_triage_bucket"] == "AUTO_VERIFIED_WASTE").sum()))

    col_map, col_selector = st.columns([2.2, 1])

    with col_selector:
        st.subheader("🎯 Odaberi lokaciju")
        if df_filtered.empty:
            st.warning("Nema lokacija za odabrane filtere.")
            return

        cluster_summary = df_filtered.groupby("cluster_id", as_index=False).first()[["cluster_id", "corine_class", "status_short", "ens_active_learning_score"]].sort_values(by="ens_active_learning_score", ascending=False)

        if st.session_state["selected_cluster_id"] not in cluster_summary["cluster_id"].tolist():
            st.session_state["selected_cluster_id"] = int(cluster_summary["cluster_id"].iloc[0])

        selected_cluster_id = st.selectbox(
            "Odaberi ELOO Lokaciju (Cluster ID):",
            options=cluster_summary["cluster_id"].tolist(),
            index=cluster_summary["cluster_id"].tolist().index(st.session_state["selected_cluster_id"]),
            format_func=lambda cid: f"Lokacija #{cid} | {cluster_summary.loc[cluster_summary['cluster_id']==cid, 'status_short'].values[0]}"
        )
        st.session_state["selected_cluster_id"] = selected_cluster_id

        df_cluster_history = df_meta[df_meta["cluster_id"] == selected_cluster_id].sort_values("image_year")

        st.markdown("**📅 Povijest širenja na ovoj lokaciji:**")
        history_chart_df = df_cluster_history[["image_year", "siglip_out10_suspect_area_m2"]].set_index("image_year")
        history_chart_df.columns = ["Sumnjiva površina (m²)"]
        st.bar_chart(history_chart_df, height=180, color="#ff7f0e")

    with col_map:
        st.caption("🗺️ Interaktivna karta: **Kliknite na bilo koju točku** da biste odmah učitali tu lokaciju!")
        active_lat = float(df_cluster_history.iloc[-1]["centroid_lat"])
        active_lon = float(df_cluster_history.iloc[-1]["centroid_lon"])

        m = folium.Map(location=[active_lat, active_lon], zoom_start=12, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satelitska snimka",
            max_zoom=19,
        ).add_to(m)

        for _, row in df_filtered.iterrows():
            cid = int(row["cluster_id"])
            is_active = (cid == selected_cluster_id)
            radius = 9 if is_active else 6
            color = "#ffffff" if is_active else row["color_hex"]

            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=radius,
                color=color,
                weight=2 if is_active else 1,
                fill=True,
                fill_color=row["color_hex"],
                fill_opacity=0.85,
                tooltip=f"Cluster #{cid} | Godina: {row['image_year']} | {row['status_short']}",
                popup=f"Cluster #{cid}"
            ).add_to(m)

        map_data = st_folium(m, height=420, use_container_width=True)

        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            df_filtered["dist"] = (df_filtered["centroid_lat"] - click_lat)**2 + (df_filtered["centroid_lon"] - click_lon)**2
            nearest_row = df_filtered.loc[df_filtered["dist"].idxmin()]
            new_cid = int(nearest_row["cluster_id"])
            if new_cid != st.session_state["selected_cluster_id"]:
                st.session_state["selected_cluster_id"] = new_cid
                st.rerun()

    st.divider()

    # --------------------------------------------------------------------------
    # BOTTOM SECTION: INSPECTION DESK & GEO FILTERS
    # --------------------------------------------------------------------------
    st.header(f"🔍 Inspekcijski stol: Lokacija #{selected_cluster_id}")

    cluster_years = df_cluster_history["image_year"].tolist()
    if len(cluster_years) > 1:
        selected_img_year = st.select_slider("⏳ Vremenski klizač (Usporedba po godinama):", options=cluster_years, value=cluster_years[-1])
    else:
        selected_img_year = cluster_years[0]
        st.info(f"📅 Godina snimke: **{selected_img_year}**")

    active_row = df_cluster_history[df_cluster_history["image_year"] == selected_img_year].iloc[0]
    bucket_info = TRIAGE_UI_MAP.get(active_row["ens_triage_bucket"], TRIAGE_UI_MAP["ROUTE_TO_CROWD_AMBIGUOUS"])

    desk_left, desk_right = st.columns([1.6, 1])

    with desk_left:
        active_filter = st.radio(
            "🎨 Odaberi Geo Filter (Pogled kroz oči AI skenera):",
            options=["📸 Originalna snimka", "🔥 Toplinska mapa (Heatmap)", "🟩 Skener terena (Pločice)", "🎯 Detekcija hrpa (Okviri)", "🎛️ Kombinirani prikaz"],
            horizontal=True
        )

        pyramid_scale = "L0_384px"
        tile_thresh = 0.45
        if active_filter == "🟩 Skener terena (Pločice)":
            sub_c1, sub_c2 = st.columns(2)
            pyramid_scale = sub_c1.selectbox("Veličina mreže:", options=["L0_384px", "L1_768px", "Sve razine"])
            tile_thresh = sub_c2.slider("Prag osjetljivosti:", 0.20, 0.90, 0.45, 0.05)

        payload = fetch_single_image_payload(chunk_id=active_row["chunk_id"], file_name=active_row["file_name"], repo_id=active_row.get("_source_repo", PROD_REPO_ID))

        fig = render_geo_filter_view(payload=payload, meta_row=active_row, active_filter=active_filter, pyramid_scale=pyramid_scale, tile_threshold=tile_thresh)
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
        m4.metric("Sumnjiva površina", f"{active_row['siglip_out10_suspect_area_m2']:.0f} m²")

        with st.expander("📊 Detaljni rezultati po vrsti AI skenera", expanded=True):
            st.caption("Usporedba globalnih i lokalnih skenera:")
            st.progress(float(np.clip(active_row["siglip_out6_total_waste_max"], 0.0, 1.0)), text=f"Skener teksture tla (SigLIP VLM): {active_row['siglip_out6_total_waste_max']:.0%}")
            st.progress(float(np.clip(active_row["skyclip_out6_total_waste_max"], 0.0, 1.0)), text=f"Satelitski skener (SkyCLIP VLM): {active_row['skyclip_out6_total_waste_max']:.0%}")
            st.progress(float(np.clip(active_row["aw_out1_global_waste_prob"], 0.0, 1.0)), text=f"Globalna deponija (AerialWaste): {active_row['aw_out1_global_waste_prob']:.0%} (Globalna vjerojatnost)")
            st.progress(float(np.clip(active_row["owlv2_out9_total_waste_max_score"], 0.0, 1.0)), text=f"Detektor hrpa (OWLv2): {active_row['owlv2_out9_total_waste_max_score']:.0%} ({active_row['owlv2_out10_total_waste_box_count']} okvira)")
            st.progress(float(np.clip(active_row["yoloworld_out9_total_waste_max_score"], 0.0, 1.0)), text=f"Detektor objekata (YOLO-World): {active_row['yoloworld_out9_total_waste_max_score']:.0%} ({active_row['yoloworld_out10_total_waste_box_count']} okvira)")

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
        if col_b1.button("✅ Potvrdi otpad", use_container_width=True, type="primary"): verdict = "CONFIRMED_WASTE"
        if col_b2.button("🚗 Lažni alarm", use_container_width=True): verdict = "FALSE_ALARM"
        if col_b3.button("🌲 Čisto", use_container_width=True): verdict = "CLEAN"

        if verdict:
            st.success(f"Zabilježeno: {verdict}")


if __name__ == "__main__":
    main()
