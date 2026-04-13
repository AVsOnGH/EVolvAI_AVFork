from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from geospatial_dashboard.gini import calculate_gini, get_accessibility_scores

app = FastAPI(
    title="EVolvAI API",
    description="REST API for EV Infrastructure Dashboard — IEEE IES Hackathon",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mock data once at startup
with open("mock_data.json") as f:
    RAW_DATA = json.load(f)

NODES = RAW_DATA["nodes"]

# Scenario configs — multipliers for charger demand
SCENARIOS = {
    "baseline": {
        "label": "Baseline",
        "demand_multiplier": 1.0,
        "temp_drop": 0,
        "overload_threshold": 0.70
    },
    "winter_storm": {
        "label": "Winter Storm",
        "demand_multiplier": 1.8,
        "temp_drop": 15,
        "overload_threshold": 0.50   # more nodes overload under stress
    },
    "fleet_2x": {
        "label": "Fleet 2.5x Electrification",
        "demand_multiplier": 2.5,
        "temp_drop": 0,
        "overload_threshold": 0.40
    }
}


def apply_scenario(nodes: list, scenario_key: str) -> list:
    """Apply scenario multipliers to node data."""
    config = SCENARIOS[scenario_key]
    result = []
    for node in nodes:
        adjusted = dict(node)
        # Under stress, nodes with high gini (underserved) are most likely to overload
        adjusted["transformer_overload"] = node["gini_score"] > config["overload_threshold"]
        # Effective demand increases but charger count stays same (that's the problem)
        adjusted["effective_demand_kw"] = round(
            node["charger_count"] * 7.2 * config["demand_multiplier"], 2
        )
        result.append(adjusted)
    return result


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "EVolvAI API is running", "docs": "/docs"}


@app.get("/api/nodes")
def get_all_nodes():
    """Return all 33 IEEE bus nodes with baseline data."""
    return {
        "scenario": "baseline",
        "node_count": len(NODES),
        "nodes": NODES
    }


@app.get("/api/nodes/{scenario}")
def get_nodes_by_scenario(scenario: str):
    """
    Return nodes adjusted for a specific scenario.
    Options: baseline | winter_storm | fleet_2x
    """
    if scenario not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIOS.keys())}"
        )
    
    adjusted_nodes = apply_scenario(NODES, scenario)
    
    return {
        "scenario": scenario,
        "scenario_label": SCENARIOS[scenario]["label"],
        "demand_multiplier": SCENARIOS[scenario]["demand_multiplier"],
        "node_count": len(adjusted_nodes),
        "nodes": adjusted_nodes
    }


@app.get("/api/gini")
def get_gini_score():
    """Return the Gini accessibility index for baseline charger distribution."""
    scores = get_accessibility_scores(NODES)
    gini = calculate_gini(scores)
    
    overloaded = [n for n in NODES if n["transformer_overload"]]
    
    return {
        "gini_index": gini,
        "interpretation": "0 = perfectly equal access, 1 = fully unequal",
        "total_nodes": len(NODES),
        "overloaded_nodes": len(overloaded),
        "total_chargers": sum(n["charger_count"] for n in NODES),
        "nodes_with_zero_chargers": sum(1 for n in NODES if n["charger_count"] == 0)
    }


@app.get("/api/gini/{scenario}")
def get_gini_by_scenario(scenario: str):
    """Return Gini score for a specific scenario."""
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")
    
    adjusted_nodes = apply_scenario(NODES, scenario)
    scores = get_accessibility_scores(adjusted_nodes)
    gini = calculate_gini(scores)
    overloaded = [n for n in adjusted_nodes if n["transformer_overload"]]
    
    return {
        "scenario": scenario,
        "gini_index": gini,
        "overloaded_nodes": len(overloaded),
        "demand_multiplier": SCENARIOS[scenario]["demand_multiplier"]
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Return list of available scenarios."""
    return {"scenarios": list(SCENARIOS.keys()), "details": SCENARIOS}