"""
Boston gas pipe triage: network units (run once from your home folder:  python network_units.py)

Uses the full Boston street network to add these columns to data_processed/segments_scored.geojson
(and to the copy in boston-gas-triage/data_processed/, if it exists):

  dead_end           segment is on a dead-end branch of the street network
  branch_id          id of the whole dead-end branch (a tree of streets hanging off the network at one point);
                     -1 for streets on the connected network (through streets)
  possible_loop      a dead end of this branch comes within LOOP_M meters of another street it doesn't
                     connect to, so a gas main could continue through an easement, path, or private way
  edge_of_city       a dead end of this branch is within EDGE_M meters of the edge of Boston's street network,
                     where the street (and its gas main) may continue into a neighboring town
  branch_isolatable  the branch could come off gas as a unit: no possible loop and not at the city edge
  project_id         segments of the same street, in the same neighborhood, that connect end to end:
                     scheduled together as one construction project
"""
import os
import re
from collections import defaultdict, deque

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree

RAW_NET = os.path.join("data_raw", "boston_street_segments.geojson")
TARGETS = ["data_processed/segments_scored.geojson",
           "boston-gas-triage/data_processed/segments_scored.geojson"]
LOOP_M = 30     # a dead end this close to another street may connect through an easement or path
EDGE_M = 60     # a dead end this close to the edge of the network may continue into a neighboring town
SNAP_M = 1      # end points within 1 m are treated as the same intersection

SUFFIX = {"AVENUE": "AV", "AVE": "AV", "AV": "AV", "STREET": "ST", "ST": "ST", "ROAD": "RD", "RD": "RD",
          "BOULEVARD": "BL", "BLVD": "BL", "BL": "BL", "PLACE": "PL", "PL": "PL", "TERRACE": "TER",
          "TER": "TER", "TE": "TER", "COURT": "CT", "CT": "CT", "DRIVE": "DR", "DR": "DR", "LANE": "LA",
          "LN": "LA", "LA": "LA", "SQUARE": "SQ", "SQ": "SQ", "PARKWAY": "PKWY", "PKWY": "PKWY", "PW": "PKWY",
          "HIGHWAY": "HWY", "HWY": "HWY", "WAY": "WAY", "WY": "WAY", "CIRCLE": "CIR", "CIR": "CIR",
          "CI": "CIR", "PARK": "PARK", "PK": "PARK", "ROW": "ROW", "WHARF": "WHF", "WHF": "WHF",
          "ALLEY": "AL", "AL": "AL", "HILL": "HL", "HL": "HL"}
DIRS = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "N": "N", "S": "S", "E": "E", "W": "W"}


def street_key(*parts):
    text = " ".join(str(p) for p in parts if pd.notna(p) and str(p).strip())
    tokens = re.sub(r"[^A-Z0-9 ]", " ", text.upper()).split()
    if not tokens:
        return None
    tokens = [DIRS.get(t, t) if i in (0, len(tokens) - 1) else t for i, t in enumerate(tokens)]
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] in SUFFIX:
            tokens[i] = SUFFIX[tokens[i]]
            break
        if tokens[i] not in DIRS.values():
            break
    return " ".join(tokens)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def node_of(pt):
    return (round(pt[0] / SNAP_M) * SNAP_M, round(pt[1] / SNAP_M) * SNAP_M)


# ---------------------------------------------------------------- 1) street network as a graph
net = gpd.read_file(RAW_NET).to_crs(26986)
id_col = "SEGMENT_ID" if "SEGMENT_ID" in net and net["SEGMENT_ID"].is_unique else "OBJECTID"
net = net[net.geometry.notna() & ~net.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)
get = lambda c: net[c] if c in net else pd.Series(None, index=net.index)
net["st_key"] = [street_key(a, b, c, d) for a, b, c, d in zip(get("PRE_DIR"), get("ST_NAME"), get("ST_TYPE"), get("SUF_DIR"))]
net["nbhd"] = get("NBHD_L").fillna("")

edges = []        # (u, v, seg_id, row)
for row, (sid, geom) in enumerate(zip(net[id_col], net.geometry)):
    coords = list(geom.coords)
    u, v = node_of(coords[0]), node_of(coords[-1])
    if u != v:
        edges.append((u, v, sid, row))
print(f"Street network: {net[id_col].nunique():,} segments, {len(edges):,} edges")

degree, touching = defaultdict(int), defaultdict(list)
for i, (u, v, _, _) in enumerate(edges):
    degree[u] += 1; degree[v] += 1
    touching[u].append(i); touching[v].append(i)
orig_degree = dict(degree)

# ---------------------------------------------------------------- 2) peel off dead-end trees
removed = np.zeros(len(edges), dtype=bool)
queue = deque(n for n, d in degree.items() if d == 1)
while queue:
    n = queue.popleft()
    if degree[n] != 1:
        continue
    for i in touching[n]:
        if not removed[i]:
            removed[i] = True
            u, v, _, _ = edges[i]
            other = v if u == n else u
            degree[n] -= 1; degree[other] -= 1
            if degree[other] == 1:
                queue.append(other)
            break
core_node = {n for n, d in degree.items() if d > 0}   # nodes still on the connected network

# ---------------------------------------------------------------- 3) group dead-end edges into whole branches
uf = UnionFind(len(edges))
by_node = defaultdict(list)
for i in np.flatnonzero(removed):
    u, v, _, _ = edges[i]
    for n in (u, v):
        if n not in core_node:          # join edges only through nodes off the network, so two trees
            by_node[n].append(i)        # attached at the same intersection stay separate branches
for idxs in by_node.values():
    for j in idxs[1:]:
        uf.union(idxs[0], j)
branch_of_edge = {i: uf.find(i) for i in np.flatnonzero(removed)}

# ---------------------------------------------------------------- 4) check each branch's dead ends
tree = STRtree([net.geometry.iloc[e[3]] for e in edges])
# Outline of the street network: buffer every street by BUF_M, merge, and keep only the outer edges, so parks
# and other gaps inside the city are filled in. A dead end within EDGE_M of the real network edge is then
# within BUF_M + EDGE_M of this outline.
BUF_M = 150
area = shapely.union_all(net.geometry.buffer(BUF_M, quad_segs=2).values)
polys = list(area.geoms) if area.geom_type == "MultiPolygon" else [area]
outline = shapely.MultiLineString([p.exterior for p in polys])

branch_edges = defaultdict(list)
for i, b in branch_of_edge.items():
    branch_edges[b].append(i)

branch_loop, branch_edge = {}, {}
for b, idxs in branch_edges.items():
    members = set(idxs)
    branch_nodes = {n for i in idxs for n in edges[i][:2]}          # includes where it joins the network
    leaves = {n for n in branch_nodes if orig_degree[n] == 1}
    loop = at_edge = False
    for leaf in leaves:
        p = Point(leaf)
        for j in tree.query(p.buffer(LOOP_M)):
            if j in members or edges[j][0] in branch_nodes or edges[j][1] in branch_nodes:
                continue                # the branch itself, or the street it branches off
            if net.geometry.iloc[edges[j][3]].distance(p) <= LOOP_M:
                loop = True
                break
        if outline.distance(p) <= BUF_M + EDGE_M:
            at_edge = True
    branch_loop[b], branch_edge[b] = loop, at_edge

# ---------------------------------------------------------------- 5) street projects: same street, same neighborhood, connected
uf2 = UnionFind(len(edges))
by_street_node = defaultdict(list)
for i, (u, v, _, row) in enumerate(edges):
    key = net.at[row, "st_key"]
    if key:
        for n in (u, v):
            by_street_node[(key, net.at[row, "nbhd"], n)].append(i)
for idxs in by_street_node.values():
    for j in idxs[1:]:
        uf2.union(idxs[0], j)

# ---------------------------------------------------------------- 6) one row per segment
rows = []
for i, (u, v, sid, row) in enumerate(edges):
    b = branch_of_edge.get(i)
    rows.append({"seg_id": sid, "dead_end": b is not None,
                 "branch_key": b if b is not None else -1,
                 "possible_loop": branch_loop.get(b, False), "edge_of_city": branch_edge.get(b, False),
                 "project_key": uf2.find(i)})
per_seg = (pd.DataFrame(rows)
             .sort_values("dead_end", ascending=False)          # a segment counts as dead end if any part is
             .drop_duplicates("seg_id"))
per_seg["branch_id"] = np.where(per_seg["branch_key"] >= 0,
                                per_seg["branch_key"].rank(method="dense").astype(int), -1)
per_seg["project_id"] = per_seg["project_key"].rank(method="dense").astype(int)
per_seg["branch_isolatable"] = per_seg["dead_end"] & ~per_seg["possible_loop"] & ~per_seg["edge_of_city"]
per_seg = per_seg.set_index("seg_id")

n_branch = per_seg.loc[per_seg["dead_end"], "branch_id"].nunique()
n_iso = per_seg.loc[per_seg["branch_isolatable"], "branch_id"].nunique()
n_loop = per_seg.loc[per_seg["dead_end"] & per_seg["possible_loop"], "branch_id"].nunique()
n_edge = per_seg.loc[per_seg["dead_end"] & per_seg["edge_of_city"], "branch_id"].nunique()
print(f"Dead-end branches: {n_branch:,} | may connect through an easement or path: {n_loop:,} | "
      f"at the city edge: {n_edge:,} | could come off gas as a unit: {n_iso:,}")

# ---------------------------------------------------------------- 7) save into the app's data file(s)
cols = ["dead_end", "branch_id", "possible_loop", "edge_of_city", "branch_isolatable", "project_id"]
for path in TARGETS:
    if not os.path.exists(path):
        continue
    s = gpd.read_file(path)
    s = s.drop(columns=[c for c in cols if c in s.columns])
    joined = per_seg[cols].reindex(s["seg_id"].values)
    for c in cols:
        s[c] = joined[c].values
    for c in ["dead_end", "possible_loop", "edge_of_city", "branch_isolatable"]:
        s[c] = s[c].fillna(False).astype(bool)
    s["branch_id"] = s["branch_id"].fillna(-1).astype(int)
    missing = s["project_id"].isna()
    if missing.any():                     # a segment not in the raw network becomes its own project
        start = int(s["project_id"].max(skipna=True) or 0) + 1
        s.loc[missing, "project_id"] = np.arange(start, start + missing.sum())
    s["project_id"] = s["project_id"].astype(int)
    s.to_file(path, driver="GeoJSON")
    print(f"Saved {path}: {int(s['dead_end'].sum()):,} dead-end segments, "
          f"{int(s['branch_isolatable'].sum()):,} on branches that could come off gas as a unit, "
          f"{s['project_id'].nunique():,} street projects")
