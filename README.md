# Joint bin placement and capacitated routing for municipal solid waste — MCD Ward 113 (Mohan Garden), Delhi

Code accompanying the paper *"A Solid Waste Logistics Approach for National Capital Using bin modification and CVRP"*
(Raj, Raj, Pal, Singh — Greater Noida Institute of Technology, AKTU).

The pipeline takes a ward boundary and population, places community waste bins on the real street network,
assigns every household's waste to a bin without exceeding bin capacity, and routes collection trucks
with a capacitated vehicle routing solver — then compares the result against simple baselines.

> **Status:** research code. Read [Limitations](#limitations) before reusing any number from this repository.

---

## What it does

```
ward KML + population ──► road network (OSM, directed)  ──► candidate bin sites
                     └──► demand surface (buildings or street length) ──► 500 demand blocks
                                                                │
                      bin count N = ceil( W(1+σ) / (V·ρ·ε) )   │
                                                                ▼
                  capacitated p-median (walking distance vs. truck distance to depot)
                                                                ▼
                       capacity-constrained assignment (transportation LP)
                                                                ▼
                     CVRP routing on the directed truck network (OR-Tools)
                                                                ▼
              results.csv · bins_<placement>.csv · ward113_map.html
```

### Model

**Demand and bin count.** With ward population *P*, per-capita rate *r*, collection interval *D*:

$$W = P \cdot r \cdot D, \qquad N = \left\lceil \frac{W\,(1+\sigma)}{V\,\rho\,\varepsilon} \right\rceil$$

**Bin placement (capacitated p-median, weighted-sum objective).** For a set *S* of *N* candidate nodes:

$$J(S) = \alpha\,\frac{J_w(S)-J_w^{*}}{J_w^{nadir}-J_w^{*}} \;+\; \beta\,\frac{J_t(S)-J_t^{*}}{J_t^{nadir}-J_t^{*}}$$

- *J<sub>w</sub>* — demand-weighted mean walking distance from a demand block to its assigned bin (network distance, metres)
- *J<sub>t</sub>* — mean truck distance from a bin to the depot on the directed street network (metres)
- *\** / *nadir* — ideal and worst values of each term, so α and β are true trade-off weights in [0, 1]

Search: start from the demand-balanced weighted k-means layout, then alternate (a) a capacity-constrained
transportation LP and (b) relocating each bin to the candidate node that minimises its own flow-weighted objective.
A move is kept only if the capacitated objective improves.

**Assignment.** Transportation LP: every demand block is fully served, no bin exceeds *V·ρ·ε* kg per collection
interval. A block's waste may be split across nearby bins (households inside a block choose among nearby bins).

**Routing.** CVRP on the *directed*, strongly connected street graph (one-way streets are respected).
Each trip is depot → bins → depot; the depot is the transfer point where trucks unload. The solver is OR-Tools
with `PATH_CHEAPEST_ARC` + `GUIDED_LOCAL_SEARCH`, with a fixed penalty per trip to discourage extra trucks.

**Scenarios.** A 2 × 2 comparison: placement {k-means + snap to street node, capacitated p-median} × routing
{capacity-aware nearest neighbour, OR-Tools guided local search}.

---

## Repository layout

```
.
├── pipeline.py                     # the whole pipeline (single script)
├── data/
│   ├── delhi_wards_under_5mb.kml   # ward boundaries + population (see Data sources)
│   └── SEC_WW_POP_2022.csv         # ward-wise population table
├── results/                        # outputs of a run (results.csv, bins_*.csv, map)
├── requirements.txt
└── README.md
```

## Installation

Tested with Python 3.14 on macOS (Apple silicon); the package versions below are the ones the results were produced with.

| Package | Version |
|---|---|
| osmnx | 2.1.1 |
| geopandas | 1.1.4 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| pandas | 3.0.5 |
| numpy | 2.5.3 |
| networkx | 3.6.1 |
| ortools | 9.15.6755 |
| folium | 0.20.0 |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install osmnx geopandas scikit-learn scipy pandas networkx ortools folium
```

## Usage

```bash
# full run on Ward 113 (downloads OSM data the first time, then uses cache_ward113.pkl)
python3 -u pipeline.py --kml data/delhi_wards_under_5mb.kml --ward 113

# force a demand model: auto (default) | area | count | street
python3 -u pipeline.py --kml data/delhi_wards_under_5mb.kml --ward 113 --demand street

# longer OR-Tools search per scenario (seconds)
python3 -u pipeline.py --kml data/delhi_wards_under_5mb.kml --ward 113 --seconds 120

# re-download OSM data
python3 -u pipeline.py --kml data/delhi_wards_under_5mb.kml --ward 113 --refresh

# offline logic check on a synthetic grid city (no internet, no OSM)
python3 pipeline.py --selftest
```

The first online run takes a few minutes (Overpass download, walking-distance matrix, two 45 s OR-Tools solves).
Later runs reuse the cache.

### Outputs

| File | Content |
|---|---|
| `results.csv` | one row per scenario: bins, mean/P90 walking distance, share of waste within 300 m, bin utilisation, trips, total km, vehicle time, km reduction vs. baselines |
| `bins_<placement>.csv` | bin node id, assigned load (kg), utilisation (%) |
| `ward113_map.html` | interactive map: ward boundary, bins with loads, depot, one colour per trip (routes follow one-way streets) |
| `cache_ward113.pkl` | cached ward data (road graph, buildings) so results are reproducible without re-downloading |

---

## Parameters

All assumptions live in the `CFG` dictionary at the top of `pipeline.py`.

| Parameter | Value | Source / status |
|---|---|---|
| Per-capita waste, MCD | 0.5 kg/day | DPCC annual report (Form V), MCD |
| Collection interval | 2 days | study assumption |
| Bin volume *V* | 1.1 m³ | DPCC Form V: MCD bins are 1.1 m³ |
| Bulk density *ρ* | 610 kg/m³ | literature value used in the paper |
| Fill coefficient *ε* | 0.80 | study assumption |
| Safety margin *σ* | 10 % | study assumption |
| Weights α, β | 0.7, 0.3 | study assumption (sensitivity sweep recommended) |
| Truck capacity | 5,000 kg | study assumption (5-tonne compactor tipper) |
| Truck speed | 18 km/h | **assumption** |
| Service time per bin | 3 min | **assumption** |
| Offload time per trip | 12 min | **assumption** |
| Depot | (28.6190, 77.0300) | **placeholder — replace with a citable facility** |

## Data sources

- **Ward boundaries and population:** Delhi ward delimitation file (`delhi_wards_under_5mb.kml`, field `TotalPop`) and
  `SEC_WW_POP_2022.csv`. *TODO: add the exact publisher and URL.*
- **Road network and buildings:** © OpenStreetMap contributors (ODbL), downloaded with OSMnx. Record the download
  date next to your results; OSM changes over time.
- **Waste parameters:** Delhi Pollution Control Committee, annual reports on SWM Rules 2016 (Form V), 2021-22 to 2023-24.

---

## Results

> *Replace this section with the output of the full-ward run (`results.csv`) before publishing.*

The table below is from an earlier **pilot configuration** (15,000 people, 80 demand points, 31 bins) and is kept only to
show the output format. **Do not cite it as a Ward 113 result.**

| Placement | Router | Bins | Mean walk (m) | Trips | Total km | Reduction vs k-means + NN |
|---|---|---|---|---|---|---|
| k-means + snap | NN | 31 | 88.3 | 4 | 18.29 | – |
| k-means + snap | OR-Tools GLS | 31 | 88.3 | 4 | 13.09 | 28.4 % |
| capacitated p-median | NN | 31 | 73.8 | 4 | 18.51 | −1.2 % |
| capacitated p-median | OR-Tools GLS | 31 | 73.8 | 4 | 12.71 | 30.5 % |

Observations from the pilot that the full run should re-check: naive nearest-bin assignment overloads 12 of 31 bins
(the capacity-constrained LP exists to prevent this); most of the distance saving comes from routing, not placement.

---

## Limitations

- **OpenStreetMap building footprints are sparse in Ward 113** (the download returned about 80 buildings), far too few
  to represent a ward of about 60,000 people. With `--demand auto` the pipeline falls back to spreading the ward population
  along non-arterial street length. This is a proxy, not measured population density; state it wherever results are reported.
- Waste is modelled as a single stream; Delhi's source-segregation bins (green / blue / black) are not modelled.
- Trips are not scheduled onto a physical fleet (no shift length, no multiple trips per truck).
- The depot is the unloading point; the onward leg to a processing site is out of scope.
- Time figures use assumed speed and service times, not measured ones.
- One ward, no field validation. The p-median search is a heuristic, not an exact optimum.
- Walking distance is measured from block centroids along the street network, so intra-block distances are approximated.

## Reproducibility

- Random seeds are fixed (`seed=42`); OR-Tools search is time-limited, so results can vary slightly between machines.
- Commit `cache_ward113.pkl` (or note the OSM extraction date) so others can rerun without re-downloading.

## Citation

```
Raj, A., Raj, P., Pal, A., Singh, M. (2026). A Solid Waste Logistics Approach for National Capital Using bin
modification and CVRP. [journal / preprint details to be added]
```

## License

Code: MIT (add a `LICENSE` file). Map data: ODbL, © OpenStreetMap contributors. Ward and report data: see the
original publishers' terms.
