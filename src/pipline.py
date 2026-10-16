#!/usr/bin/env python3
"""
Ward-level solid-waste logistics pipeline (v3) -- Mohan Garden, MCD Ward 113, Delhi
==================================================================================
One script, one run -> every number the paper needs.

Pipeline
  1. Ward polygon + population from the ward KML (Census-based ward population)
  2. Demand  : W = population x per-capita rate x collection interval
  3. Bins    : N = ceil( W*(1+sigma) / (V*rho*eps) )                 (paper Eq. b)
  4. Blocks  : building centroids -> weighted k-means demand blocks
  5. Sites   : capacitated p-median on the WALKING network, weighted-sum objective
                 J(S) = alpha * (Jw(S) - Jw*) / (Jw_nadir - Jw*)
                      + beta  * (Jt(S) - Jt*) / (Jt_nadir - Jt*)
               Jw = demand-weighted mean walk to the bin (m); Jt = mean truck distance
               bin -> depot (m). Each term is min-max normalised between its ideal (*) and
               nadir value, so alpha/beta are true trade-off weights in [0,1].
               Capacity-aware search: start from the demand-balanced k-means layout, then
               alternate (a) capacitated transportation LP and (b) relocating each bin to
               the candidate node that minimises its own flow-weighted objective; a move
               is kept only if the capacitated objective improves.
               Final assignment is a capacity-constrained transportation LP
               (no bin ever exceeds V*rho*eps kg per collection interval).
  6. Routing : CVRP on the DIRECTED, strongly-connected truck network
               (OR-Tools guided local search) vs capacity-aware nearest neighbour
  7. Scenarios (2 x 2): placement {kmeans+snap, p-median} x routing {NN, OR-Tools GLS}
  8. Outputs : results.csv, bins_*.csv, ward113_map.html

Usage
  pip install osmnx geopandas scikit-learn scipy pandas networkx ortools folium
  python ward113_pipeline_v3.py --kml delhi_wards_under_5mb.kml --ward 113
  python ward113_pipeline_v3.py --selftest          # offline logic check, synthetic data

Each trip (vehicle) = depot -> bins -> depot. The depot is the transfer point where the
truck unloads; the onward leg to the processing site is outside this model.
"""
import argparse
import math
import os
import pickle
import time
import warnings
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------------------------------------------------------------------
# PARAMETERS  (every assumption lives here -> copy this table into the paper)
# --------------------------------------------------------------------------------------
CFG = dict(
    ward_no="113",
    per_capita_kg=0.5,          # DPCC Form V: MCD 0.5 kg/capita/day
    interval_days=2,            # max collection interval
    bin_volume_m3=1.1,          # DPCC Form V: MCD bins are 1.1 m3
    density=610.0,              # kg/m3 (paper ref [5])
    fill_coeff=0.80,            # epsilon
    safety=0.10,                # sigma
    alpha=0.7,                  # citizen convenience weight
    beta=0.3,                   # truck convenience weight
    n_blocks=80,               # demand blocks (aggregation of buildings)
    weight_by="area",           # "area" (footprint m2) or "count" (buildings)
    truck_capacity_kg=5000,     # 5-tonne compactor tipper
    fixed_trip_cost_m=2000,     # solver penalty per extra trip (metre-equivalent)
    speed_kmh=18.0,             # ASSUMPTION
    service_min_per_bin=3.0,    # ASSUMPTION
    offload_min_per_trip=12.0,  # ASSUMPTION
    depot=(28.6190, 77.0300),   # (lat, lon) PLACEHOLDER -> replace with a citable facility
    buffer_m=1500,              # road network buffer around the ward
    walk_cutoff_m=2500.0,       # max walking search radius
    walk_threshold_m=300.0,     # reporting threshold: % of waste within this walk
    pmedian_passes=15,
    solver_seconds=45,
    seed=42,
)

CUSTOM_FILTER = '["highway"~"primary|secondary|tertiary|residential"]'
BAD_HIGHWAYS = {"primary", "primary_link", "trunk", "trunk_link", "motorway", "motorway_link"}
NON_RESIDENTIAL = {
    "commercial", "industrial", "retail", "school", "kindergarten", "university", "college",
    "hospital", "government", "civic", "public", "temple", "church", "mosque", "religious",
    "warehouse", "garage", "garages", "service", "roof", "shed", "transportation",
    "train_station", "construction", "fire_station", "sports_hall", "hotel",
}


# --------------------------------------------------------------------------------------
# GEOMETRY / DATA LOADING
# --------------------------------------------------------------------------------------
def parse_ward_kml(path, ward_no):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    root = ET.parse(path).getroot()
    for pm in root.findall(".//k:Placemark", ns):
        d = {sd.get("name"): sd.text for sd in pm.findall(".//k:SimpleData", ns)}
        if str(d.get("Ward_No")).strip() == str(ward_no):
            rings = []
            for po in pm.findall(".//k:Polygon", ns):
                oc = po.find(".//k:outerBoundaryIs//k:coordinates", ns)
                pts = [tuple(map(float, p.split(",")[:2])) for p in oc.text.split()]
                rings.append(np.array(pts))
            return dict(name=d.get("WardName"), pop=int(d["TotalPop"]), rings_ll=rings)
    raise ValueError(f"Ward {ward_no} not found in {path}")


def points_in_rings(x, y, rings):
    """Vectorised ray casting; True if the point is inside ANY ring."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = np.zeros(len(x), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for ring in rings:
            inside = np.zeros(len(x), bool)
            xr, yr = ring[:, 0], ring[:, 1]
            j = len(ring) - 1
            for i in range(len(ring)):
                xi, yi, xj, yj = xr[i], yr[i], xr[j], yr[j]
                cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
                inside ^= cond
                j = i
            out |= inside
    return out


def load_real_data(cfg, kml_path, refresh=False):
    cache = f"cache_ward{cfg['ward_no']}.pkl"
    if os.path.exists(cache) and not refresh:
        print(f"[data] using cache {cache} (use --refresh to re-download)")
        with open(cache, "rb") as f:
            return pickle.load(f)

    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    ward = parse_ward_kml(kml_path, cfg["ward_no"])
    ward_ll = unary_union([Polygon(r) for r in ward["rings_ll"]])

    for attr in ("overpass_url", "overpass_endpoint"):      # osmnx 2.x / 1.x
        if hasattr(ox.settings, attr):
            setattr(ox.settings, attr, "https://lz4.overpass-api.de/api")
    ox.settings.timeout = 300
    ox.settings.use_cache = True

    depot_lat, depot_lon = cfg["depot"]
    area = unary_union([ward_ll, Point(depot_lon, depot_lat).buffer(0.005)])
    gs = gpd.GeoSeries([area], crs=4326)
    graph_poly = gs.to_crs(gs.estimate_utm_crs()).buffer(cfg["buffer_m"]).to_crs(4326).iloc[0]

    print("[data] downloading road network ...")
    G_ll = ox.graph_from_polygon(graph_poly, network_type="drive",
                                 custom_filter=CUSTOM_FILTER, simplify=True)
    G_ll = ox.truncate.largest_component(G_ll, strongly=True)   # directed, strongly connected
    G = ox.project_graph(G_ll)
    crs = G.graph["crs"]

    time.sleep(5)  # be polite to the Overpass API
    print("[data] downloading building footprints ...")
    b = ox.features_from_polygon(ward_ll, tags={"building": True})
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    bp = b.to_crs(crs)
    cx, cy = bp.geometry.centroid.x.values, bp.geometry.centroid.y.values
    barea = bp.geometry.area.values
    btype = b["building"].astype(str).values

    rings_proj = []
    for poly in gpd.GeoSeries([Polygon(r) for r in ward["rings_ll"]], crs=4326).to_crs(crs):
        rings_proj.append(np.array(poly.exterior.coords))
    inside = points_in_rings(cx, cy, rings_proj)
    resid = ~np.isin(btype, list(NON_RESIDENTIAL))
    keep = inside & resid
    print(f"[data] buildings in ward: {inside.sum()}, excluded non-residential: "
          f"{(inside & ~resid).sum()}, kept: {keep.sum()}")

    depot_xy = gpd.GeoSeries([Point(depot_lon, depot_lat)], crs=4326).to_crs(crs).iloc[0]
    data = dict(
        name=ward["name"], pop=ward["pop"], G=G, G_ll=G_ll, rings=rings_proj, ward_ll=ward_ll,
        bx=cx[keep], by=cy[keep], barea=barea[keep], depot_xy=(depot_xy.x, depot_xy.y),
        depot_ll=(depot_lat, depot_lon),
    )
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def make_synthetic(cfg, seed=42):
    """Offline test bed: grid city with one-way streets, a primary road and clustered buildings."""
    rng = np.random.default_rng(seed)
    step = 100
    coords = np.arange(-300, 1701, step)
    n = len(coords)
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32643"
    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            G.add_node((i, j), x=float(x), y=float(y))

    def add(u, v, hw):
        G.add_edge(u, v, length=float(step), highway=hw)

    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                add((i, j), (i, j + 1), "residential")
                add((i, j + 1), (i, j), "residential")
            if i + 1 < n:
                y = coords[j]
                if y == 700:
                    add((i, j), (i + 1, j), "primary")
                    add((i + 1, j), (i, j), "primary")
                elif j % 2 == 0:
                    add((i, j), (i + 1, j), "residential")
                    add((i + 1, j), (i, j), "residential")
                elif (j // 2) % 2 == 0:
                    add((i, j), (i + 1, j), "residential")      # eastbound only
                else:
                    add((i + 1, j), (i, j), "residential")      # westbound only
    centers = rng.uniform(100, 1300, size=(12, 2))
    pts = np.vstack([c + rng.normal(0, 120, size=(350, 2)) for c in centers])
    pts = pts[(pts[:, 0] > 0) & (pts[:, 0] < 1400) & (pts[:, 1] > 0) & (pts[:, 1] < 1400)]
    return dict(
        name="SYNTHETIC", pop=60029, G=G, G_ll=None,
        rings=[np.array([[0, 0], [1400, 0], [1400, 1400], [0, 1400]], float)],
        ward_ll=None, bx=pts[:, 0], by=pts[:, 1],
        barea=rng.lognormal(4.5, 0.5, size=len(pts)),
        depot_xy=(700.0, -200.0), depot_ll=None,
    )


# --------------------------------------------------------------------------------------
# MODEL COMPONENTS
# --------------------------------------------------------------------------------------
def ceil_int(x):
    return int(math.ceil(x - 1e-9))


def node_xy(G):
    nodes = list(G.nodes)
    xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])
    return nodes, xy


def candidate_nodes(G, rings):
    """Network nodes inside the ward that touch at least one non-arterial street."""
    nodes, xy = node_xy(G)
    inside = points_in_rings(xy[:, 0], xy[:, 1], rings)
    ok = set()
    for u, v, d in G.edges(data=True):
        hw = d.get("highway")
        hws = hw if isinstance(hw, list) else [hw]
        if any(h not in BAD_HIGHWAYS for h in hws):
            ok.add(u)
            ok.add(v)
    return [n for n, ins in zip(nodes, inside) if ins and n in ok]


def simple_undirected(G):
    Gu = nx.Graph()
    for u, v, d in G.edges(data=True):
        L = d["length"]
        if Gu.has_edge(u, v):
            Gu[u][v]["length"] = min(Gu[u][v]["length"], L)
        else:
            Gu.add_edge(u, v, length=L)
    for n, d in G.nodes(data=True):
        Gu.add_node(n, x=d["x"], y=d["y"])
    return Gu


def walking_matrix(G, block_xy, cand_nodes, cutoff, big):
    """D[b, c] = walk (m) from block centroid to candidate node c (centroid->node + network)."""
    Gu = simple_undirected(G)
    nodes, xy = node_xy(Gu)
    acc, idx = cKDTree(xy).query(block_xy)
    cand_index = {n: j for j, n in enumerate(cand_nodes)}
    nb, nc = len(block_xy), len(cand_nodes)
    D = np.full((nb, nc), big)
    cache = {}
    for b in range(nb):
        n = nodes[idx[b]]
        if n not in cache:
            row = np.full(nc, np.inf)
            for m, L in nx.single_source_dijkstra_path_length(Gu, n, cutoff=cutoff, weight="length").items():
                j = cand_index.get(m)
                if j is not None:
                    row[j] = L
            cache[n] = row
        D[b] = np.minimum(cache[n] + acc[b], big)
    return D


def truck_to_depot(G, cand_nodes, depot_node, big):
    """Directed truck distance candidate -> depot (Dijkstra on the reversed graph)."""
    lengths = nx.single_source_dijkstra_path_length(G.reverse(copy=False), depot_node, weight="length")
    return np.array([min(lengths.get(n, big), big) for n in cand_nodes])


def solve_pmedian(D, w, dep, p, alpha, beta, passes, rng):
    """Greedy-add + Teitz-Bart interchange for J(S) = alpha*mean_walk + beta*mean_depot."""
    nb, nc = D.shape
    wn = w / w.sum()

    def objective(S):
        return alpha * (wn * D[:, S].min(1)).sum() + beta * dep[S].mean()

    cur = np.full(nb, np.inf)
    S, dep_sum = [], 0.0
    for k in range(p):
        walk = (wn[:, None] * np.minimum(cur[:, None], D)).sum(0)
        total = alpha * walk + beta * (dep_sum + dep) / (k + 1)
        total[S] = np.inf
        c = int(np.argmin(total))
        S.append(c)
        cur = np.minimum(cur, D[:, c])
        dep_sum += dep[c]
    best_obj = objective(S)
    for _ in range(passes):
        improved = False
        for pos in rng.permutation(p):
            others = [S[i] for i in range(p) if i != pos]
            base = D[:, others].min(1)
            walk = (wn[:, None] * np.minimum(base[:, None], D)).sum(0)
            total = alpha * walk + beta * (dep[others].sum() + dep) / p
            total[others] = np.inf
            c = int(np.argmin(total))
            if total[c] < best_obj - 1e-9:
                S[pos] = c
                best_obj = total[c]
                improved = True
        if not improved:
            break
    return S, best_obj


def solve_pmedian_scalarised(D, w, dep, p, alpha, beta, passes, rng):
    """Min-max normalised weighted sum of mean walk and mean truck-to-depot distance."""
    wn = w / w.sum()
    S_w, _ = solve_pmedian(D, w, dep, p, 1.0, 0.0, passes, rng)      # walk-only optimum
    S_t = list(np.argsort(dep)[:p])                                   # truck-only optimum
    walk = lambda S: float((wn * D[:, S].min(1)).sum())
    jw_star, jt_nadir = walk(S_w), float(dep[S_w].mean())
    jt_star, jw_nadir = float(dep[S_t].mean()), walk(S_t)
    rw = max(jw_nadir - jw_star, 1e-9)
    rt = max(jt_nadir - jt_star, 1e-9)
    S, _ = solve_pmedian(D, w, dep, p, alpha / rw, beta / rt, passes, rng)
    jw, jt = walk(S), float(dep[S].mean())
    J = alpha * (jw - jw_star) / rw + beta * (jt - jt_star) / rt
    info = dict(jw_ideal=jw_star, jw_nadir=jw_nadir, jt_ideal=jt_star, jt_nadir=jt_nadir,
                jw=jw, jt=jt, J=J, a2=alpha / rw, b2=beta / rt)
    return S, info


def capacitated_location_allocation(D, demand, dep, S0, cap, a2, b2, jw_star, jt_star, iters=12):
    """Cooper-style alternate location/allocation with a capacity-constrained allocation step.
    Returns (sites, flow matrix x, capacitated mean walk, normalised objective history)."""
    S = list(S0)
    p, total = len(S), demand.sum()

    def evaluate(S_):
        x_ = capacitated_assignment(D[:, S_], demand, cap)
        if x_ is None:
            return None, None, np.inf
        mw = float((x_ * D[:, S_]).sum() / total)
        J = a2 * (mw - jw_star) + b2 * (float(dep[S_].mean()) - jt_star)
        return x_, mw, J

    x, mw, J = evaluate(S)
    if x is None:
        raise RuntimeError("initial layout cannot satisfy bin capacities")
    history = [J]
    for _ in range(iters):
        used = set(S)
        newS = list(S)
        per_kg = (x * D[:, S]).sum(1) / np.maximum(demand, 1e-9)       # walk per kg, by block
        stress = demand * per_kg
        for j in range(p):
            f = x[:, j]
            if f.sum() > 1.0:
                cost = a2 * (f @ D) / total + b2 * dep / p             # flow-weighted 1-median
            else:
                cost = stress @ D                                      # idle bin -> go where walking is worst
            cost = cost.copy()
            blocked = list(used - {newS[j]})
            cost[blocked] = np.inf
            c = int(np.argmin(cost))
            used.discard(newS[j])
            used.add(c)
            newS[j] = c
        x2, mw2, J2 = evaluate(newS)
        if x2 is None or J2 >= J - 1e-9:
            break
        S, x, mw, J = newS, x2, mw2, J2
        history.append(J)
    return S, x, mw, history


def snap_unique(cand_xy, pts):
    """Snap each point to the nearest still-unused candidate node."""
    tree = cKDTree(cand_xy)
    used, out = set(), []
    for pt in pts:
        _, idxs = tree.query(pt, k=min(len(cand_xy), len(pts) + 5))
        for j in np.atleast_1d(idxs):
            if int(j) not in used:
                used.add(int(j))
                out.append(int(j))
                break
    return out


def capacitated_assignment(Dsel, demand, cap):
    """Transportation LP: min sum x*d  s.t. each block fully served, each bin <= cap kg."""
    nb, ns = Dsel.shape
    nvar = nb * ns
    cols = np.arange(nvar)
    A_eq = coo_matrix((np.ones(nvar), (np.repeat(np.arange(nb), ns), cols)), shape=(nb, nvar)).tocsr()
    A_ub = coo_matrix((np.ones(nvar), (np.tile(np.arange(ns), nb), cols)), shape=(ns, nvar)).tocsr()
    res = linprog(Dsel.ravel(), A_ub=A_ub, b_ub=np.full(ns, cap), A_eq=A_eq, b_eq=demand,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    return res.x.reshape(nb, ns)


def wpercentile(values, weights, q):
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, q / 100.0)])


def assignment_metrics(Dsel, x, cap, thresh):
    mask = x > 1e-6
    dv, wv = Dsel[mask], x[mask]
    loads = x.sum(0)
    return dict(
        mean_walk_m=float((dv * wv).sum() / wv.sum()),
        p90_walk_m=wpercentile(dv, wv, 90),
        pct_within_thresh=float(wv[dv <= thresh].sum() / wv.sum() * 100),
        max_util_pct=float(loads.max() / cap * 100),
        mean_util_pct=float(loads.mean() / cap * 100),
        bins_used=int((loads > 0.5).sum()),
    ), loads


# --------------------------------------------------------------------------------------
# ROUTING
# --------------------------------------------------------------------------------------
def truck_matrix(G, nodes_list, big=1e5):
    n = len(nodes_list)
    M = np.zeros((n, n))
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, nodes_list[i], weight="length")
        for j in range(n):
            M[i, j] = lengths.get(nodes_list[j], big)
    return M


def route_nn(M, demands, cap):
    """Capacity-aware nearest neighbour. Index 0 = depot; bins are 1..n."""
    unvisited = set(range(1, len(demands) + 1))
    trips = []
    while unvisited:
        cur, load, trip = 0, 0.0, []
        while True:
            feas = [j for j in unvisited if load + demands[j - 1] <= cap]
            if not feas:
                break
            j = min(feas, key=lambda k: M[cur, k])
            trip.append(j)
            unvisited.remove(j)
            load += demands[j - 1]
            cur = j
        if not trip:
            raise RuntimeError("a bin exceeds truck capacity")
        trips.append(trip)
    return trips


def route_ortools(M, demands, cap, n_veh, fixed_cost, seconds):
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        print("   [skip] OR-Tools not installed")
        return None
    Mi = np.rint(M).astype(int).tolist()
    dem = [0] + [int(math.ceil(d)) for d in demands]
    manager = pywrapcp.RoutingIndexManager(len(Mi), n_veh, 0)
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(lambda a, b: Mi[manager.IndexToNode(a)][manager.IndexToNode(b)])
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.SetFixedCostOfAllVehicles(int(fixed_cost))
    dcb = routing.RegisterUnaryTransitCallback(lambda a: dem[manager.IndexToNode(a)])
    routing.AddDimensionWithVehicleCapacity(dcb, 0, [int(cap)] * n_veh, True, "Capacity")
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.seconds = int(seconds)
    sol = routing.SolveWithParameters(sp)
    if not sol:
        print("   [warn] OR-Tools found no feasible solution (increase vehicles / time)")
        return None
    trips = []
    for v in range(n_veh):
        idx, trip = routing.Start(v), []
        while not routing.IsEnd(idx):
            n = manager.IndexToNode(idx)
            if n != 0:
                trip.append(n)
            idx = sol.Value(routing.NextVar(idx))
        if trip:
            trips.append(trip)
    return trips


def trip_metrics(trips, M, demands, cfg):
    km = 0.0
    for t in trips:
        path = [0] + t + [0]
        km += sum(M[a, b] for a, b in zip(path[:-1], path[1:])) / 1000.0
    n_bins = sum(len(t) for t in trips)
    minutes = km / cfg["speed_kmh"] * 60 + n_bins * cfg["service_min_per_bin"] \
        + len(trips) * cfg["offload_min_per_trip"]
    return dict(trips=len(trips), total_km=km, vehicle_time_min=minutes,
                max_trip_load_kg=max(sum(demands[j - 1] for j in t) for t in trips))


# --------------------------------------------------------------------------------------
# MAP
# --------------------------------------------------------------------------------------
def draw_map(data, sites_nodes, loads, trips_nodes, out_html, title):
    import folium
    G_ll = data["G_ll"]
    lat0, lon0 = data["depot_ll"]
    m = folium.Map(location=[lat0, lon0], zoom_start=15)
    ring = data["ward_ll"]
    polys = [ring] if ring.geom_type == "Polygon" else list(ring.geoms)
    for p in polys:
        folium.PolyLine([[y, x] for x, y in p.exterior.coords], color="black", weight=2,
                        tooltip=f"Ward {data['name']}").add_to(m)
    for n, load in zip(sites_nodes, loads):
        folium.Marker([G_ll.nodes[n]["y"], G_ll.nodes[n]["x"]],
                      icon=folium.Icon(color="blue", icon="trash"),
                      popup=f"Bin {n}: {load:.0f} kg").add_to(m)
    folium.Marker([lat0, lon0], icon=folium.Icon(color="red", icon="info-sign"),
                  popup="Depot / transfer point").add_to(m)
    palette = ["green", "purple", "orange", "darkblue", "red", "cadetblue", "darkgreen",
               "pink", "black", "gray", "lightred", "beige"]
    for k, trip in enumerate(trips_nodes):
        coords = []
        for u, w in zip(trip[:-1], trip[1:]):
            try:
                sp = nx.shortest_path(G_ll, u, w, weight="length")   # directed: respects one-ways
            except nx.NetworkXNoPath:
                continue
            for a, b in zip(sp[:-1], sp[1:]):
                e = min(G_ll.get_edge_data(a, b).values(), key=lambda d: d.get("length", 1e9))
                if "geometry" in e:
                    xs, ys = e["geometry"].xy
                    coords.extend([[y, x] for x, y in zip(xs, ys)])
                else:
                    coords.extend([[G_ll.nodes[a]["y"], G_ll.nodes[a]["x"]],
                                   [G_ll.nodes[b]["y"], G_ll.nodes[b]["x"]]])
        if coords:
            folium.PolyLine(coords, color=palette[k % len(palette)], weight=5, opacity=0.8,
                            tooltip=f"Trip {k + 1}").add_to(m)
    m.save(out_html)
    print(f"[map] {title} -> {out_html}")


# --------------------------------------------------------------------------------------
# EXPERIMENT
# --------------------------------------------------------------------------------------
def run(cfg, data, make_map=True, seconds=None):
    t_start = time.time()
    rng = np.random.default_rng(cfg["seed"])
    G = data["G"]
    seconds = seconds or cfg["solver_seconds"]

    # ---- 1. demand & bin count --------------------------------------------------------
    pop = 15000
    w_day = pop * cfg["per_capita_kg"]
    w_interval = w_day * cfg["interval_days"]
    q_eff = cfg["bin_volume_m3"] * cfg["density"] * cfg["fill_coeff"]
    p = ceil_int(w_interval * (1 + cfg["safety"]) / q_eff)
    print("\n=== DEMAND & BIN SIZING ===")
    print(f"ward {data['name']}: population {pop:,}  x {cfg['per_capita_kg']} kg/day = {w_day:,.0f} kg/day")
    print(f"collection interval {cfg['interval_days']} d -> {w_interval:,.0f} kg per interval")
    print(f"effective bin capacity V*rho*eps = {q_eff:.1f} kg -> N = {p} bins (safety {cfg['safety']:.0%})")
    print(f"minimum truckloads = {w_interval / cfg['truck_capacity_kg']:.2f} -> "
          f">= {ceil_int(w_interval / cfg['truck_capacity_kg'])} trips of {cfg['truck_capacity_kg']} kg")

    # ---- 2. demand blocks -------------------------------------------------------------
    weights = data["barea"] if cfg["weight_by"] == "area" else np.ones(len(data["bx"]))
    weights = weights / weights.mean()
    pts = np.column_stack([data["bx"], data["by"]])
    km = KMeans(n_clusters=cfg["n_blocks"], n_init=10, random_state=cfg["seed"]).fit(pts, sample_weight=weights)
    block_xy = km.cluster_centers_
    block_w = np.bincount(km.labels_, weights=weights, minlength=cfg["n_blocks"])
    block_demand = block_w / block_w.sum() * w_interval          # kg per interval
    print(f"{len(pts):,} buildings -> {cfg['n_blocks']} demand blocks "
          f"(mean {block_demand.mean():.0f} kg, max {block_demand.max():.0f} kg per interval)")

    # ---- 3. candidates, distances -----------------------------------------------------
    cands = candidate_nodes(G, data["rings"])
    if len(cands) < p:
        raise RuntimeError(f"only {len(cands)} candidate nodes for {p} bins")
    cand_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in cands])
    nodes_all, xy_all = node_xy(G)
    d_depot, depot_idx = cKDTree(xy_all).query(data["depot_xy"])
    depot_node = nodes_all[depot_idx]
    print(f"{len(cands)} candidate bin nodes (arterial-only nodes excluded); "
          f"depot snapped {d_depot:.0f} m to network node")
    if d_depot > 100:
        print("   [warn] depot is >100 m from the road network - check the coordinates")
    BIG = cfg["walk_cutoff_m"] * 2
    print("computing walking distances ...")
    D = walking_matrix(G, block_xy, cands, cfg["walk_cutoff_m"], BIG)
    dep = truck_to_depot(G, cands, depot_node, 1e5)

    # ---- 4. placements ----------------------------------------------------------------
    placements = {}
    t0 = time.time()
    _, info = solve_pmedian_scalarised(D, block_demand, dep, p, cfg["alpha"], cfg["beta"],
                                       cfg["pmedian_passes"], rng)   # only used to get ideal/nadir scales
    print(f"normalisation scales (uncapacitated bounds), alpha={cfg['alpha']}, beta={cfg['beta']}:")
    print(f"   mean walk  : ideal {info['jw_ideal']:.0f} m | nadir {info['jw_nadir']:.0f} m")
    print(f"   mean depot : ideal {info['jt_ideal']:.0f} m | nadir {info['jt_nadir']:.0f} m")
    kb = KMeans(n_clusters=p, n_init=10, random_state=cfg["seed"]).fit(block_xy, sample_weight=block_demand)
    S_km = snap_unique(cand_xy, kb.cluster_centers_)
    placements["kmeans_snap"] = S_km
    S_pm, x_pm, mw_pm, hist = capacitated_location_allocation(
        D, block_demand, dep, S_km, q_eff, info["a2"], info["b2"], info["jw_ideal"], info["jt_ideal"])
    placements["pmedian_cap"] = S_pm
    print(f"capacitated p-median: {len(hist)} accepted iterations in {time.time() - t0:.1f}s, "
          f"normalised J {hist[0]:.3f} -> {hist[-1]:.3f}  (0 = both ideals, 1 = both nadirs)")

    # ---- 5. assignment + routing per scenario -------------------------------------------
    rows, bin_tables, best_routes = [], {}, {}
    for pname, S in placements.items():
        Dsel = D[:, S]
        near = np.zeros(len(S))
        np.add.at(near, Dsel.argmin(1), block_demand)
        overflow_nearest = int((near > q_eff + 1e-6).sum())
        x = capacitated_assignment(Dsel, block_demand, q_eff)
        if x is None:
            print(f"[{pname}] capacitated assignment infeasible - skipping")
            continue
        am, loads = assignment_metrics(Dsel, x, q_eff, cfg["walk_threshold_m"])
        used = np.where(loads > 0.5)[0]
        site_nodes = [cands[S[i]] for i in used]
        used_loads = loads[used]
        bin_tables[pname] = pd.DataFrame(dict(node=site_nodes, load_kg=used_loads,
                                              util_pct=used_loads / q_eff * 100))
        M = truck_matrix(G, [depot_node] + site_nodes)
        n_veh = ceil_int(used_loads.sum() / cfg["truck_capacity_kg"] * 1.25) + 2
        routers = {"NN": lambda: route_nn(M, used_loads, cfg["truck_capacity_kg"])}
        routers["ORTOOLS_GLS"] = lambda: route_ortools(M, used_loads, cfg["truck_capacity_kg"],
                                                      n_veh, cfg["fixed_trip_cost_m"], seconds)
        for rname, fn in routers.items():
            t0 = time.time()
            trips = fn()
            if trips is None:
                continue
            tm = trip_metrics(trips, M, used_loads, cfg)
            rows.append(dict(placement=pname, router=rname, bins=len(S), **am,
                             nearest_assign_overflow_bins=overflow_nearest, **tm,
                             runtime_s=round(time.time() - t0, 1)))
            best_routes[(pname, rname)] = ([[depot_node] + [site_nodes[j - 1] for j in t] + [depot_node]
                                            for t in trips], site_nodes, used_loads)
            print(f"[{pname:11s} | {rname:11s}] trips={tm['trips']:2d}  km={tm['total_km']:.2f}  "
                  f"walk={am['mean_walk_m']:.0f} m  maxutil={am['max_util_pct']:.0f}%")

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("no scenario produced results")
    base = res[(res.placement == "kmeans_snap") & (res.router == "NN")]
    if len(base):
        b = float(base.total_km.iloc[0])
        res["km_reduction_vs_kmeansNN_pct"] = (1 - res.total_km / b) * 100
    for pname in res.placement.unique():
        nn = res[(res.placement == pname) & (res.router == "NN")]
        if len(nn):
            mask = res.placement == pname
            res.loc[mask, "km_reduction_vs_same_placement_NN_pct"] = (1 - res.loc[mask, "total_km"]
                                                                       / float(nn.total_km.iloc[0])) * 100
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== RESULTS ===")
    print(res.round(2).to_string(index=False))
    res.to_csv("results.csv", index=False)
    for pname, t in bin_tables.items():
        t.to_csv(f"bins_{pname}.csv", index=False)
    print("saved results.csv and bins_<placement>.csv")

    if make_map and data.get("G_ll") is not None:
        for key in [("pmedian_cap", "ORTOOLS_GLS"), ("pmedian_cap", "NN")]:
            if key in best_routes:
                trips_nodes, site_nodes, ld = best_routes[key]
                draw_map(data, site_nodes, ld, trips_nodes, "ward113_map.html", f"{key[0]} + {key[1]}")
                break
    print(f"\ntotal runtime {time.time() - t_start:.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kml", default="delhi_wards_under_5mb.kml")
    ap.add_argument("--ward", default=CFG["ward_no"])
    ap.add_argument("--refresh", action="store_true", help="re-download OSM data")
    ap.add_argument("--seconds", type=int, default=None, help="OR-Tools time limit per scenario")
    ap.add_argument("--selftest", action="store_true", help="offline run on synthetic data")
    args = ap.parse_args()
    cfg = dict(CFG, ward_no=str(args.ward))
    if args.selftest:
        data = make_synthetic(cfg)
        run(cfg, data, make_map=False, seconds=args.seconds or 5)
    else:
        data = load_real_data(cfg, args.kml, args.refresh)
        run(cfg, data, make_map=True, seconds=args.seconds)


if __name__ == "__main__":
    main()