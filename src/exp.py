import numpy as np
import networkx as nx
import pickle
import math
import matplotlib.pyplot as plt
import warnings
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

warnings.filterwarnings("ignore", category=DeprecationWarning) 


def setup_environment():
    with open("cache_ward113.pkl", "rb") as f:
        data = pickle.load(f)
    
    G = data["G"]
    pop = 15000  # Pilot Zone
    TOTAL_WASTE = pop * 0.5 * 2
    Q_BIN = 1.1 * 610.0 * 0.80
    NUM_BINS = int(np.ceil((TOTAL_WASTE * 1.1) / Q_BIN))
    
    
    building_coords = np.column_stack([data["bx"], data["by"]])
    kmeans = KMeans(n_clusters=80, n_init=10, random_state=42).fit(building_coords)
    demand_per_block = (np.bincount(kmeans.labels_) / len(building_coords)) * TOTAL_WASTE
    demand_coords = kmeans.cluster_centers_
    
    
    nodes = list(G.nodes)
    candidate_coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])
    depot_coord = np.array([data["depot_xy"]])
    depot_node_id = min(nodes, key=lambda n: (G.nodes[n]['x']-depot_coord[0][0])**2 + (G.nodes[n]['y']-depot_coord[0][1])**2)
    
    dist_dem_cand = distance_matrix(demand_coords, candidate_coords)
    dist_cand_depot = distance_matrix(candidate_coords, depot_coord).flatten()
    
    return G, demand_per_block, dist_dem_cand, dist_cand_depot, nodes, NUM_BINS, depot_node_id


def solve_routing(G, depot_id, selected_nodes, demands, strategy="GLS"):
    routing_nodes = [depot_id] + selected_nodes
    n = len(routing_nodes)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, routing_nodes[i], weight='length')
        for j in range(n): 
            matrix[i, j] = lengths.get(routing_nodes[j], 100000)
            
    manager = pywrapcp.RoutingIndexManager(n, 4, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    routing.SetArcCostEvaluatorOfAllVehicles(
        routing.RegisterTransitCallback(lambda f, t: int(matrix[manager.IndexToNode(f)][manager.IndexToNode(t)]))
    )
    
    demands_list = [0] + list(demands)
    # Increased to 8000 to prevent OR-Tools crash on random un-optimized assignment
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(lambda f: int(demands_list[manager.IndexToNode(f)])), 
        0, [8000]*4, True, 'Cap'
    )
    
    params = pywrapcp.DefaultRoutingSearchParameters()
    if strategy == "NN":
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT 
    elif strategy == "CW":
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS 
    else:
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 3
        
    sol = routing.SolveWithParameters(params)
    return sol.ObjectiveValue() / 1000.0 if sol else None

def evaluate_scenario(alpha, beta, random_placement, strategy, env):
    G, d_per_block, dist_dc, dist_cd, nodes, NUM_BINS, depot_id = env
    
    if random_placement:
        np.random.seed(42)
        selected_idx = np.random.choice(len(nodes), NUM_BINS, replace=False)
    else:
        M_cit = np.max(dist_dc) * np.max(d_per_block)
        M_trk = np.max(dist_cd)
        scores = [alpha * np.sum(d_per_block * dist_dc[:, c]/M_cit) + beta * (dist_cd[c]/M_trk) for c in range(len(nodes))]
        selected_idx = np.argsort(scores)[:NUM_BINS]
        
    dist_dem_selected = dist_dc[:, selected_idx]
    assignments = np.argmin(dist_dem_selected, axis=1)
    
    bin_demands = [0] * NUM_BINS
    for b_idx, s_idx in enumerate(assignments): 
        bin_demands[s_idx] += d_per_block[b_idx]
        
    mean_walk = np.mean(dist_dem_selected[np.arange(len(d_per_block)), assignments])
    
    truck_dist = solve_routing(G, depot_id, [nodes[i] for i in selected_idx], bin_demands, strategy)
    return mean_walk, truck_dist


if __name__ == "__main__":
    env = setup_environment()
    
    print("\n" + "="*50)
    print("EXPERIMENT E2: Scenarios S1-S5")
    print("="*50)
    scenarios = [
        ("S1: Random Bins + Nearest Neighbour", True, 1.0, 0.0, "NN"),
        ("S2: p-Median + Nearest Neighbour", False, 1.0, 0.0, "NN"),
        ("S3: p-Median + Clarke-Wright", False, 1.0, 0.0, "CW"),
        ("S4: p-Median + OR-Tools GLS", False, 1.0, 0.0, "GLS"),
        ("S5: Joint Proposed (α=0.7, β=0.3) + GLS", False, 0.7, 0.3, "GLS")
    ]
    
    for name, rand, a, b, strat in scenarios:
        walk, route = evaluate_scenario(a, b, rand, strat, env)
        if route is None:
            print(f"{name.ljust(40)} | Walk: {walk:5.1f}m | Route: FAILED (Capacity Exceeded)")
        else:
            print(f"{name.ljust(40)} | Walk: {walk:5.1f}m | Route: {route:5.2f}km")
        
    print("\n" + "="*50)
    print("EXPERIMENT E3: Pareto Sweep (α / β)")
    print("="*50)
    alphas = np.linspace(0, 1, 11)
    walk_res, route_res = [], []
    
    for a in alphas:
        b = 1.0 - a
        walk, route = evaluate_scenario(a, b, False, "GLS", env)
        if route is not None:
            walk_res.append(walk)
            route_res.append(route)
            print(f"Alpha: {a:.1f}, Beta: {b:.1f} => Walk: {walk:.1f}m, Route: {route:.2f}km")
        else:
            print(f"Alpha: {a:.1f}, Beta: {b:.1f} => Walk: {walk:.1f}m, Route: FAILED")
        

    plt.figure(figsize=(8,5))
    plt.plot(walk_res, route_res, marker='o', linestyle='-', color='b')
    for i, a in enumerate(alphas): 
        if i < len(walk_res):
            plt.annotate(f"α={a:.1f}", (walk_res[i], route_res[i]+0.1))
    plt.title("Pareto Curve: Citizen Walking Distance vs Fleet Travel Distance")
    plt.xlabel("Mean Citizen Walking Distance (m)")
    plt.ylabel("Total Fleet Haul Distance (km)")
    plt.grid(True)
    plt.savefig("pareto_curve.png")
    print("\n-> Saved 'pareto_curve.png'")
    
    print("\n" + "="*50)
    print("EXPERIMENT E4: Sensitivity Sweep (Bin Volume)")
    print("="*50)
    volumes = [0.8, 1.1, 1.5, 2.0]
    for v in volumes:
        Q = v * 610.0 * 0.80
        N = int(np.ceil((15000 * 1.1) / Q))
        env_temp = (env[0], env[1], env[2], env[3], env[4], N, env[6])
        walk, route = evaluate_scenario(0.7, 0.3, False, "GLS", env_temp)
        if route is None:
            print(f"Volume: {v}m³ | Bins Needed: {N:2d} | Route Dist: FAILED")
        else:
            print(f"Volume: {v}m³ | Bins Needed: {N:2d} | Route Dist: {route:.2f}km")