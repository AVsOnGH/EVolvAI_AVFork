import numpy as np

def calculate_gini(values: list) -> float:
    """
    Calculate the Gini coefficient for a list of accessibility scores.
    0 = perfect equality (chargers evenly spread)
    1 = perfect inequality (all chargers in one place)
    """
    arr = np.array(values, dtype=float)
    
    if arr.sum() == 0:
        return 0.0
    
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    
    gini = (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))
    return round(float(gini), 4)


def get_accessibility_scores(nodes: list) -> list:
    """
    Convert charger counts per node into accessibility scores.
    Nodes with 0 chargers get score 0.01 (not zero, to avoid div by zero).
    """
    scores = []
    for node in nodes:
        score = node["charger_count"] if node["charger_count"] > 0 else 0.01
        scores.append(score)
    return scores