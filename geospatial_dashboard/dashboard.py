import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import numpy as np

# ─── Config ──────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="EVolvAI Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ─── Header ──────────────────────────────────────────────────
st.title("⚡ EVolvAI — EV Infrastructure Dashboard")
st.caption("IEEE IES Hackathon | Physics-Constrained Generative AI for Equitable EV Planning")

st.divider()

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Control")
    
    scenario = st.selectbox(
        "Select Counterfactual Scenario",
        options=["baseline", "winter_storm", "fleet_2x"],
        format_func=lambda x: {
            "baseline": "Baseline (Current)",
            "winter_storm": "Winter Storm (+1.8x demand)",
            "fleet_2x": "Fleet Electrification (2.5x demand)"
        }[x]
    )
    
    st.divider()
    st.markdown("**Color Legend**")
    st.markdown("🟢 Green = Node operating normally")
    st.markdown("🔴 Red = Transformer overloaded")
    st.markdown("⚪ Circle size = Charger count")
    
    st.divider()
    st.markdown("**About**")
    st.caption("This dashboard visualizes the IEEE 33-bus system mapped to Hyderabad, India. "
               "Node colors reflect transformer overload status under selected scenario conditions.")


# ─── Fetch data from API ─────────────────────────────────────
@st.cache_data(ttl=5)  # cache for 5 seconds
def fetch_nodes(scenario_key):
    try:
        r = requests.get(f"{API_URL}/api/nodes/{scenario_key}", timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"Cannot connect to API. Is it running? Error: {e}")
        return None

@st.cache_data(ttl=5)
def fetch_gini(scenario_key):
    try:
        r = requests.get(f"{API_URL}/api/gini/{scenario_key}", timeout=5)
        return r.json()
    except:
        return None


data = fetch_nodes(scenario)
gini_data = fetch_gini(scenario)

if data is None:
    st.error("Start the API first: `uvicorn api:app --reload`")
    st.stop()

nodes = data["nodes"]

# ─── Top Metrics Row ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    gini_val = gini_data["gini_index"] if gini_data else "N/A"
    st.metric(
        label="Gini Accessibility Index",
        value=f"{gini_val:.3f}" if isinstance(gini_val, float) else gini_val,
        help="0 = equal access everywhere. 1 = all chargers in one area."
    )

with col2:
    overloaded = sum(1 for n in nodes if n["transformer_overload"])
    st.metric(
        label="Overloaded Transformers",
        value=f"{overloaded} / {len(nodes)}",
        delta=f"{overloaded} at risk",
        delta_color="inverse"
    )

with col3:
    total_chargers = sum(n["charger_count"] for n in nodes)
    st.metric(
        label="Total Chargers",
        value=total_chargers
    )

with col4:
    zero_nodes = sum(1 for n in nodes if n["charger_count"] == 0)
    st.metric(
        label="Nodes Without Chargers",
        value=zero_nodes,
        delta=f"{zero_nodes} underserved zones",
        delta_color="inverse"
    )

st.divider()

# ─── Map + Table Layout ──────────────────────────────────────
map_col, table_col = st.columns([2, 1])

with map_col:
    st.subheader("IEEE 33-Bus System — Hyderabad Grid Map")
    
    # Build Folium map
    center_lat = np.mean([n["lat"] for n in nodes])
    center_lng = np.mean([n["lng"] for n in nodes])
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    for node in nodes:
        color = "red" if node["transformer_overload"] else "green"
        radius = max(5, node["charger_count"] * 2.5)
        
        # Tooltip text
        tooltip_text = (
            f"Node {node['node_id']} | Zone: {node['zone'].upper()}\n"
            f"Chargers: {node['charger_count']}\n"
            f"Gini Score: {node['gini_score']}\n"
            f"Status: {'OVERLOADED' if node['transformer_overload'] else 'Normal'}"
        )
        
        # Popup with more detail
        popup_html = f"""
        <div style='font-family:sans-serif; font-size:13px; min-width:160px'>
            <b>Node {node['node_id']}</b><br>
            Zone: {node['zone'].upper()}<br>
            Chargers: {node['charger_count']}<br>
            Gini Score: {node['gini_score']}<br>
            <span style='color:{"red" if node["transformer_overload"] else "green"}'>
                {"⚠ TRANSFORMER OVERLOADED" if node["transformer_overload"] else "✓ Normal operation"}
            </span>
        </div>
        """
        
        folium.CircleMarker(
            location=[node["lat"], node["lng"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1.5,
            tooltip=tooltip_text,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)
        
        # Node ID label
        folium.Marker(
            location=[node["lat"], node["lng"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px;color:#333;font-weight:bold">{node["node_id"]}</div>',
                icon_size=(20, 10),
                icon_anchor=(10, 0)
            )
        ).add_to(m)
    
    st_folium(m, width=700, height=500)

with table_col:
    st.subheader("Node Status Table")
    
    # Filter options
    filter_opt = st.radio(
        "Show",
        ["All nodes", "Overloaded only", "No chargers"],
        horizontal=False
    )
    
    if filter_opt == "Overloaded only":
        display_nodes = [n for n in nodes if n["transformer_overload"]]
    elif filter_opt == "No chargers":
        display_nodes = [n for n in nodes if n["charger_count"] == 0]
    else:
        display_nodes = nodes
    
    # Build display rows
    import pandas as pd
    rows = []
    for n in display_nodes:
        rows.append({
            "Node": n["node_id"],
            "Zone": n["zone"],
            "Chargers": n["charger_count"],
            "Gini": n["gini_score"],
            "Status": "🔴 Overloaded" if n["transformer_overload"] else "🟢 Normal"
        })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, height=420)
    else:
        st.info("No nodes match this filter.")

# ─── Gini Chart ──────────────────────────────────────────────
st.divider()
st.subheader("Gini Score Distribution Across Nodes")

import pandas as pd

gini_df = pd.DataFrame([
    {"Node": f"N{n['node_id']}", "Gini Score": n["gini_score"], "Zone": n["zone"]}
    for n in sorted(nodes, key=lambda x: x["gini_score"])
])

st.bar_chart(gini_df.set_index("Node")["Gini Score"])
st.caption("Higher Gini score = more underserved / inequitable access. "
           "Nodes above 0.7 are critically underserved.")

# ─── Footer ──────────────────────────────────────────────────
st.divider()
st.caption("EVolvAI | IEEE IES GenAI Hackathon | Dashboard by Krishna (Geospatial & UI Module)")