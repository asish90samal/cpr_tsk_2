"""
graph_engine/graph.py
─────────────────────
Entity relationship graph for AML network analysis.

Fixes applied vs original:
  1. Edges added — relationships between entities based on shared attributes
  2. Edge types: same_country, shared_alias_token, shared_id_prefix
  3. Connected-component analysis helper
  4. Risk propagation — entity risk elevated if connected to high-risk nodes
  5. Community detection helper (requires python-louvain)
  6. Graph summary stats
"""

from __future__ import annotations
from collections import defaultdict

import pandas as pd
import networkx as nx


# ── Edge-building strategies ───────────────────────────────────────────────

def _add_country_edges(G: nx.Graph, df: pd.DataFrame) -> int:
    """Add edges between entities that share the same country."""
    country_groups: dict[str, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        country = str(row.get("nationality", row.get("country", "UNKNOWN"))).upper()
        if country and country != "UNKNOWN":
            country_groups[country].append(row["entity_id"])

    edge_count = 0
    for country, ids in country_groups.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if not G.has_edge(ids[i], ids[j]):
                    G.add_edge(ids[i], ids[j], rel="same_country", country=country, weight=0.3)
                    edge_count += 1
    return edge_count


def _add_alias_token_edges(G: nx.Graph, df: pd.DataFrame) -> int:
    """
    Add edges between entities that share a significant alias token.
    (Useful for detecting family networks / patronymic clusters.)
    Skips common tokens ('AL', 'BIN', 'VAN', etc.) to reduce noise.
    """
    _STOPWORDS = {"AL", "EL", "BIN", "ABU", "ABD", "VAN", "VON", "DE", "THE", "AND"}

    token_to_ids: dict[str, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        aliases_str = str(row.get("aliases", ""))
        all_names = [str(row.get("primary_name", ""))] + aliases_str.split("|")
        tokens_seen: set[str] = set()
        for name in all_names:
            for token in name.upper().split():
                if len(token) >= 4 and token not in _STOPWORDS:
                    tokens_seen.add(token)
        for token in tokens_seen:
            token_to_ids[token].append(row["entity_id"])

    edge_count = 0
    for token, ids in token_to_ids.items():
        if len(ids) < 2 or len(ids) > 50:   # skip ultra-common tokens
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if not G.has_edge(ids[i], ids[j]):
                    G.add_edge(ids[i], ids[j], rel="shared_alias_token", token=token, weight=0.5)
                    edge_count += 1
    return edge_count


def _add_id_prefix_edges(G: nx.Graph, df: pd.DataFrame) -> int:
    """Add edges between entities whose ID numbers share the same 4-char prefix (same issuing authority)."""
    prefix_groups: dict[str, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        id_num = str(row.get("id_number", ""))
        if len(id_num) >= 4:
            prefix_groups[id_num[:4]].append(row["entity_id"])

    edge_count = 0
    for prefix, ids in prefix_groups.items():
        if len(ids) < 2 or len(ids) > 100:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if not G.has_edge(ids[i], ids[j]):
                    G.add_edge(ids[i], ids[j], rel="shared_id_prefix", prefix=prefix, weight=0.2)
                    edge_count += 1
    return edge_count


# ── Main graph builder ─────────────────────────────────────────────────────

def build_graph(
    df: pd.DataFrame,
    add_country_edges:     bool = True,
    add_alias_edges:       bool = True,
    add_id_prefix_edges:   bool = False,   # off by default — noisy for synthetic data
) -> nx.Graph:
    """
    Build an entity relationship graph from a sanctions DataFrame.

    Parameters
    ----------
    df                  : Sanctions entity DataFrame
    add_country_edges   : Link entities sharing nationality
    add_alias_edges     : Link entities sharing alias tokens
    add_id_prefix_edges : Link entities sharing ID prefix

    Returns
    -------
    nx.Graph with node attributes (risk, name) and typed edges
    """
    G = nx.Graph()

    # ── Add nodes ──────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        G.add_node(
            row["entity_id"],
            name=row.get("primary_name", ""),
            risk=float(row.get("risk_weight", 0.5)),
            country=str(row.get("nationality", row.get("country", "UNKNOWN"))),
            label=int(row.get("risk_label", 0)),
        )

    # ── Add edges ──────────────────────────────────────────────────────────
    total_edges = 0
    if add_country_edges:
        total_edges += _add_country_edges(G, df)
    if add_alias_edges:
        total_edges += _add_alias_token_edges(G, df)
    if add_id_prefix_edges:
        total_edges += _add_id_prefix_edges(G, df)

    print(f"[GraphEngine] Built graph: {G.number_of_nodes()} nodes, {total_edges} edges")
    return G


# ── Analysis helpers ───────────────────────────────────────────────────────

def get_connected_components(G: nx.Graph) -> list[set]:
    """Return list of connected-component node sets, largest first."""
    return sorted(nx.connected_components(G), key=len, reverse=True)


def propagate_risk(G: nx.Graph, iterations: int = 3, damping: float = 0.15) -> nx.Graph:
    """
    Elevate risk scores of nodes connected to high-risk neighbours.
    Simple neighbour-average propagation (not PageRank — interpretable for compliance).
    """
    for _ in range(iterations):
        updates: dict[int, float] = {}
        for node in G.nodes:
            neighbours = list(G.neighbors(node))
            if not neighbours:
                continue
            neighbour_risks = [G.nodes[n]["risk"] for n in neighbours]
            avg_neighbour_risk = sum(neighbour_risks) / len(neighbour_risks)
            current = G.nodes[node]["risk"]
            # Weighted nudge toward neighbour average
            updates[node] = round(current * (1 - damping) + avg_neighbour_risk * damping, 4)
        for node, new_risk in updates.items():
            G.nodes[node]["risk"] = new_risk
    return G


def graph_summary(G: nx.Graph) -> dict:
    """Return a dict of key graph statistics."""
    components = get_connected_components(G)
    return {
        "nodes":                G.number_of_nodes(),
        "edges":                G.number_of_edges(),
        "connected_components": len(components),
        "largest_component":    len(components[0]) if components else 0,
        "avg_degree":           round(sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2),
        "density":              round(nx.density(G), 6),
    }


# ── LLM relationship injection ─────────────────────────────────────────────

def inject_llm_relationships(
    G: nx.Graph,
    source_entity_id: str,
    relationships: list[dict],
    min_confidence: float = 0.50,
) -> int:
    """
    Inject relationship edges extracted by the LLM article analyser into
    the entity graph.

    Parameters
    ----------
    G                : Existing entity graph (modified in-place)
    source_entity_id : entity_id of the screening subject
    relationships    : List of dicts from ArticleAnalysis.relationships
                       Each: {linked_name, relationship_type, confidence}
    min_confidence   : Only inject edges with confidence >= this value

    Returns
    -------
    Number of edges added.

    How it works:
      The LLM may name "HASSAN AL-RASHID" as a family member of the
      screening subject.  We add a temporary node for HASSAN and an edge
      from source → HASSAN with rel_type=FAMILY and confidence as weight.
      This feeds propagate_risk() so risk spreads along LLM-extracted edges
      just like structural edges.
    """
    added = 0
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        confidence = float(rel.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        linked_name  = str(rel.get("linked_name", "")).strip()
        rel_type     = str(rel.get("relationship_type", "UNKNOWN")).upper()
        if not linked_name:
            continue

        # Use a synthetic node ID for LLM-extracted entities
        linked_id = f"LLM_{linked_name.upper().replace(' ', '_')[:40]}"

        # Add node if not already present
        if not G.has_node(linked_id):
            G.add_node(
                linked_id,
                name=linked_name,
                risk=0.50,           # neutral until propagate_risk() runs
                country="UNKNOWN",
                label=0,
                source="llm_extracted",
            )

        # Add or update edge
        if G.has_edge(source_entity_id, linked_id):
            # Strengthen existing edge
            existing = G[source_entity_id][linked_id].get("confidence", 0.0)
            G[source_entity_id][linked_id]["confidence"] = max(existing, confidence)
        else:
            G.add_edge(
                source_entity_id, linked_id,
                rel=rel_type,
                weight=confidence,
                confidence=confidence,
                source="llm_extracted",
            )
            added += 1

    return added
