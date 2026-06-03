# Plan: Unify RAG across NutriBench, Robot Assistant, and HealthyFoodSubs via Score-Fusion Hybrid

## Context

Today the three retrieval pipelines in `mimir` are inconsistent:

| Pipeline | What it solves | Modes implemented | What's missing |
|---|---|---|---|
| NutriBench (`scripts/run_bench.py`) | Food → nutrition | v0 BM25, v1 text, v2 text+GAT-expand-then-text-rerank, v3 multi-cand variant of v2 | pure GAT, score-fusion hybrid |
| Robot assistant (`nutri_rag/assistant/pipeline.py`) | Food → nutrition → Food → Similar food | text top-1 (eaten side), SQL+blind GAT expansion (recommend side) | hybrid retrieval everywhere; availability filter |
| HealthyFoodSubs (`nutri_graph/scripts/eval_food_subs.py`) | Food → Similar food | text-only, GAT-only, **score-fusion hybrid** `α·cos_gat + (1-α)·cos_text` | (reference — already correct) |

HealthyFoodSubs has demonstrated that **score-fusion hybrid** is the strongest mode for food→similar-food. The goal is to bring the same hybrid scoring into the other two pipelines so:
- All three sub-problems (`Food→nutrition`, `nutrition→Food`, `Food→similar Food`) use a single unified retrieval primitive.
- NutriBench evaluation gains parallel `gat`/`hybrid` modes (via text-bootstrapped pseudo-anchor), letting us measure where fusion helps.
- The robot assistant's food→similar-food step respects what foods are actually available (from existing ZMQ detected-objects store).
- The recommendation step uses nutrient-target embeddings (already present in `node_embeddings.npy`) instead of arbitrary SQL ordering.

The work is split into three phases by user priority. Each phase is independently shippable and verifiable.

**Regret-safe principle (user-requested).** Every change to the robot assistant is **additive**: new code paths are added alongside the existing ones, gated by an env var / constructor argument with the **current behavior as default**. This means each phase ships with the current production behavior preserved, and the new behavior is opt-in for A/B comparison. Toggle env vars are:

| Stage | Env var | Default | Other options | Notes |
|---|---|---|---|---|
| Eaten-side ID (A6) | `EATEN_RETRIEVAL_MODE` | **`hybrid`** (new) | `text_top1` (current) | Hybrid wins per Phase-A; legacy kept as fallback |
| Food→similar food expansion (B2/B5) | `FOOD_NEIGHBOR_MODE` | `gat_only` (current) | `hybrid` | Default unchanged until Phase-B comparison data exists |
| nutrition→Food recommender (C3) | `RECOMMEND_MODE` | `v1` (current SQL seeds) | `v2` (target embedding) | Default unchanged until Phase-C comparison data exists |
| Availability source (B3) | `AVAILABILITY_SOURCE` | `none` (no filter) | `json`, `zmq` | Or set via `nutrition_tool.py` constructor / CLI flag (`--availability-source`) for easy switching in scripts |

---

## Unified retrieval primitive (the shared abstraction)

All three pipelines will call one function:

```python
# nutri_rag/nutri_rag/search.py  (new)
def hybrid_rank(
    q_text: np.ndarray | None,           # 1024-d text query vec or None
    q_gat:  np.ndarray | None,           # 64-d GAT query vec or None
    candidate_fdc_ids: list[int] | None, # None = all foods
    alpha:  float = 0.5,                 # GAT weight; 1-alpha = text weight
    structured_filter: callable | None = None,  # e.g. category, availability, macro≥τ
    structured_score:  callable | None = None,  # additive structured term, e.g. macro_match
    structured_weight: float = 0.0,
    k: int = 5,
) -> list[dict]:
    """
    Score every candidate by:
        s(x) = α·cos(q_gat, x_gat) + (1-α)·cos(q_text, x_text)
             + structured_weight · structured_score(x)
    Apply structured_filter as a hard mask.
    Return top-k as [{fdc_id, description, text_sim, gat_sim, struct, total}, ...].
    """
```

This mirrors HealthyFoodSubs `evaluate_hybrid()` at [eval_food_subs.py:176-226](nutri_graph/scripts/eval_food_subs.py#L176-L226) but generalized: caller chooses which vectors exist, what filter to apply, and whether a structured term contributes.

Modes are just configurations:
- **text-only**: `α=0` → fallback to `cos(q_text, x_text)` (auto-handled when `q_gat is None`)
- **gat-only**: `α=1`
- **hybrid**: `0 < α < 1` (default 0.5)

---

## Phase A — Priority 1: Food → nutrition (NutriBench + robot eaten-side)

### A1. Add `hybrid_rank()` to `nutri_rag/nutri_rag/search.py`

- Reuse the existing singletons `_get_index()`, `_get_gat_index()`, `_get_kb()` at [search.py:30-55](nutri_rag/nutri_rag/search.py#L30-L55).
- Reuse `FoodVectorIndex.embeddings` for `x_text` and `GATIndex.embeddings` for `x_gat`.
- Both index types already use the same `fdc_id ↔ arr_idx` mapping (verified in `_gat_expand` at [search.py:154-214](nutri_rag/nutri_rag/search.py#L154-L214) which uses `index.fdc_ids` to align with GAT rows).

### A2. Add `search_food_v2()` wrapper for NutriBench Food→nutrition

For NutriBench the query is **free text** (no `q_gat`), so we use a **text-bootstrapped pseudo-anchor**:

```python
def search_food_v2(query: str, mode: str, k: int, alpha: float = 0.5) -> pd.DataFrame:
    q_text = embedder.encode([query], task_instruction=FOOD_SEARCH_INSTRUCTION)[0]
    if mode == "text":
        return hybrid_rank(q_text=q_text, q_gat=None, alpha=0.0, k=k)
    seed = top1_text(q_text)                   # pseudo-anchor
    q_gat_star = GATIndex().embeddings[seed.arr_idx]
    if mode == "gat":
        return hybrid_rank(q_text=None, q_gat=q_gat_star, alpha=1.0, k=k)
    if mode == "hybrid":
        return hybrid_rank(q_text=q_text, q_gat=q_gat_star, alpha=alpha, k=k)
```

### A3. New NutriBench task directories

Clone the structure of [nutribench_v2_rag_gat/](nutri_rag/tasks/nutribench_v2_rag_gat/) for two new tasks:

| New task dir | task_name | mode | What its `utils.py` calls |
|---|---|---|---|
| `nutri_rag/tasks/nutribench_v2_rag_gat_pure/` | `nutribench_v2_rag_gat_pure` | gat-only | `search_food_v2(query, mode="gat")` |
| `nutri_rag/tasks/nutribench_v2_rag_hybrid/` | `nutribench_v2_rag_hybrid` | hybrid | `search_food_v2(query, mode="hybrid", alpha=0.5)` |

Each task dir contains:
- `utils.py` — copy of [nutribench_v2_rag_gat/utils.py](nutri_rag/tasks/nutribench_v2_rag_gat/utils.py), only the `BenchRetriever` call changes to pass `mode=` + `alpha=`.
- A YAML config file pointing at the template name (written by `generate_template_yaml` in run_bench.py).
- The USDA reference block formatter is reused as-is from [bench/prompt.py](nutri_rag/bench/prompt.py) (per-item format for gat_pure; multi-candidate format for hybrid since hybrid benefits from showing multiple candidates).

### A4. Extend `BenchRetriever` to accept `mode` + `alpha`

[nutri_rag/bench/retriever.py](nutri_rag/bench/retriever.py) — change the constructor signature from `BenchRetriever(use_gat: bool)` to:

```python
class BenchRetriever:
    def __init__(self, mode: str = "text", alpha: float = 0.5, top_k: int = 1):
        # mode ∈ {"text", "gat_expand", "gat_pure", "hybrid"}
        # gat_expand is the legacy v2/v3 behavior; gat_pure and hybrid are new
```

Backward compatibility: map old `use_gat=True` → `mode="gat_expand"`.

### A5. Register new modes in `run_bench.py` and `run_all_bench.py`

[scripts/run_bench.py:27-52](nutri_rag/scripts/run_bench.py#L27-L52) — add to `TASK_DIRS`:

```python
"v4": {  # gat-pure (pseudo-anchor)
    "src": "nutri_rag/tasks/nutribench_v2_rag_gat_pure",
    "dst": ".../lm_eval/tasks/nutribench/v2/rag_gat_pure",
    "task_name": "nutribench_v2_rag_gat_pure",
    "template_name": "_rag_gat_pure_default_template_yaml",
},
"v5": {  # hybrid score fusion (pseudo-anchor)
    "src": "nutri_rag/tasks/nutribench_v2_rag_hybrid",
    "dst": ".../lm_eval/tasks/nutribench/v2/rag_hybrid",
    "task_name": "nutribench_v2_rag_hybrid",
    "template_name": "_rag_hybrid_default_template_yaml",
},
```

- Add `--mode` choices `v4, v5` at [run_bench.py:120](nutri_rag/scripts/run_bench.py#L120).
- Add CLI `--alpha` flag for hybrid (default 0.5).
- Update mode labels dict at [run_bench.py:157](nutri_rag/scripts/run_bench.py#L157).
- [scripts/run_all_bench.py:28-29](nutri_rag/scripts/run_all_bench.py#L28-L29) — add `v4, v5` to `ALL_MODES` and `RAG_MODES`.

### A6. Make hybrid the default for the robot's eaten-side lookup; keep text top-1 reachable

Add a mode dispatch at [nutri_rag/nutri_rag/assistant/pipeline.py:51](nutri_rag/nutri_rag/assistant/pipeline.py#L51). **Hybrid becomes the new default** because Phase-A NutriBench results will give us evidence that it's the strongest mode. Text top-1 is kept as an opt-in fallback so we can A/B compare and revert instantly if needed.

```python
# nutri_rag/nutri_rag/assistant/pipeline.py
class NutriAssistant:
    def __init__(self, ..., eaten_retrieval_mode: str = None):
        # mode ∈ {"hybrid" (new default), "text_top1" (current behavior, kept as fallback)}
        self._eaten_mode = eaten_retrieval_mode or os.environ.get(
            "EATEN_RETRIEVAL_MODE", "hybrid"  # hybrid is the new default
        )

    def _lookup_eaten_foods(self, meal_description, meal_type="breakfast"):
        items = parse_meal(meal_description)
        eaten = []
        for item in items:
            if self._eaten_mode == "text_top1":
                df = search_food(self.con, item.food_term, k=1)  # legacy path, kept
            else:  # "hybrid" — new default
                df = search_food_v2(item.food_term, mode="hybrid", k=1, alpha=0.5)
            ...
```

Selection at the call site:
- **Default = hybrid** (new behavior backed by NutriBench v5 evidence).
- Fallback to current behavior via `EATEN_RETRIEVAL_MODE=text_top1` env var, or `NutriAssistant(eaten_retrieval_mode="text_top1")` from the tool wrapper.
- Both paths return the same dataframe schema, so downstream code is unchanged.

This means every Phase-A run can A/B both modes on the same meal description by toggling one env var, and if hybrid regresses on a specific case we can revert with a single environment variable, no code change required.

### A7. Verification for Phase A

```bash
# 1. New modes parse and run on a small sample
cd /home/boxun/work/atlas/mimir
python nutri_rag/scripts/run_bench.py --mode v4 --nutrient protein --limit 20
python nutri_rag/scripts/run_bench.py --mode v5 --nutrient protein --limit 20 --alpha 0.5

# 2. Full sweep to compare modes
python nutri_rag/scripts/run_all_bench.py \
    --modes baseline v0 v1 v2 v3 v4 v5 \
    --nutrients carb protein fat energy \
    --limit 200
# Compare acc/MAE columns in nutri_rag/results/

# 3. Robot eaten-side regression test — same eaten input, before vs after
#    Verify "an apple" still resolves to "Apples, raw, with skin"
#    Verify "a bowl of maize flour" now resolves better (ambiguous case)
```

---

## Phase B — Priority 2: Food → similar food with availability (robot Step 4)

### B1. Load text embeddings into `FoodRecommender`

[nutri_rag/nutri_rag/assistant/food_recommender.py:34-46](nutri_rag/nutri_rag/assistant/food_recommender.py#L34-L46) — currently loads only GAT. Add text:

```python
self._gat_embeddings  = np.load(gat_embeddings_path)              # (N_food, 64)
self._text_embeddings = np.load(text_embeddings_path)             # (N_food, 1024)
self._text_fdc_ids    = np.load(text_fdc_ids_path)                # alignment
# Build fdc_id → text_row mapping (text_emb may have different ordering than GAT)
self._fdc_to_text_idx = {int(fid): i for i, fid in enumerate(self._text_fdc_ids)}
```

Defaults from [nutri_rag/config.py](nutri_rag/nutri_rag/config.py): point to existing files `food_text_embeddings.npy` and `food_fdc_ids.npy`.

### B2. Add `_hybrid_neighbors` alongside (not replacing) `_gat_neighbors`

[food_recommender.py:63-97](nutri_rag/nutri_rag/assistant/food_recommender.py#L63-L97) — keep the existing `_gat_neighbors` method as-is so the current behavior remains reachable. Add a new `_hybrid_neighbors`:

```python
def _hybrid_neighbors(self, seed_fdc_id: int, k: int, alpha: float = 0.5,
                       available_fdc_ids: set[int] | None = None) -> list[tuple[int, float]]:
    q_gat  = self._gat_embeddings[self._food_id_to_idx[seed_fdc_id]]
    q_text = self._text_embeddings[self._fdc_to_text_idx[seed_fdc_id]]
    return hybrid_rank(
        q_text=q_text, q_gat=q_gat, alpha=alpha,
        structured_filter=(lambda fid: fid in available_fdc_ids) if available_fdc_ids else None,
        k=k,
    )
```

This is structurally identical to HealthyFoodSubs `evaluate_hybrid()` — both query and candidate are graph nodes, both `q_gat` and `q_text` exist trivially.

**Dispatch at the call site** in `recommend()` (Step 5 of the pipeline) is controlled by an env var / constructor arg:

```python
# food_recommender.py — inside recommend()
mode = self._neighbor_mode  # from FOOD_NEIGHBOR_MODE env var, default "gat_only" (current)
for seed in seeds:
    if mode == "hybrid":
        neighbors = self._hybrid_neighbors(seed.fdc_id, k, alpha=0.5,
                                            available_fdc_ids=available_fdc_ids)
    else:  # "gat_only" — unchanged from today
        neighbors = self._gat_neighbors(seed.fdc_id, k=n_neighbors)
        # availability still applied as a post-filter for fair comparison
        if available_fdc_ids is not None:
            neighbors = [(fid, s) for fid, s in neighbors if fid in available_fdc_ids]
    ...
```

Default = current GAT-only. Opt-in via `FOOD_NEIGHBOR_MODE=hybrid`. Note: availability filter is still applied to both modes (it's an orthogonal concern from the scoring method), enabling fair A/B comparison.

### B3. Add an availability provider (server-side testable, not robot-dependent)

**Key insight from exploring the robot side:**

[robot_side/zmq_bridge_simulation/zmq_object_server.py](nutri-atlas/robot_control/robot_side/zmq_bridge_simulation/zmq_object_server.py) is a **thin reader** over a JSON file at `~/Go2/autonomy_stack_go2/detected_objects.json`. The actual storage is just JSON of the form:

```json
{
  "detected_apple_0":  {"px": 3.2,  "py": -1.1, ...},
  "detected_bottle_1": {"px": 2.8,  "py":  0.4, ...}
}
```

ZMQ adds nothing on top — it just delivers this dict to remote clients. So **moving availability to the server side is just reading the same JSON file directly** (or a server-side mirror of it). This was the user's intuition: "It should be some simple modification."

#### Module layout

```python
# nutri_rag/nutri_rag/assistant/availability.py  (new, ~80 lines)

def get_available_fdc_ids(
    source: str = "json",                  # "json" | "zmq" | "none"
    json_path: str | None = None,          # default: nutri_rag/data/detected_objects.json
    zmq_addr: str | None = None,           # default: tcp://ROBOT_IP:5556
) -> set[int] | None:
    """
    Returns set of fdc_ids representing currently-available foods, or None for no filter.
    """
    if source == "none":
        return None
    if source == "json":
        labels = _read_object_json(json_path or DEFAULT_AVAILABILITY_JSON)
    elif source == "zmq":
        labels = _query_zmq_objects(zmq_addr)        # only when robot is live
    return _labels_to_fdc_ids(labels)

def _read_object_json(path: str) -> set[str]:
    """Parse the same detected_objects.json format the robot's object server reads."""
    with open(path) as f: data = json.load(f)
    # frame names look like "detected_apple_0" → extract "apple"
    return {_label_from_frame_name(k) for k in data.keys()}

def _labels_to_fdc_ids(labels: set[str]) -> set[int]:
    """Map each label to an fdc_id via text top-1 search (cached per process)."""
    # Reuses hybrid_rank(q_text=..., q_gat=None, k=1)
    # Filters out non-food labels (top-1 text_score below threshold).
```

#### Server-side default file location

Add a small **server-side mirror** of `detected_objects.json` at:

```
nutri_rag/data/detected_objects.json
```

Tracked in repo with a sample for testing. The robot's persistent JSON can be `scp`'d / `rsync`'d to this location for any test (one-line cron or manual copy), or hand-edited to simulate scenarios. The `zmq` source is kept as an opt-in for live-robot use.

#### Why this is "simple modification" (user's words)

- **No new infrastructure**: we read the exact same JSON the robot's object server already reads.
- **No robot dependency for testing**: `source="json"` with a local sample file works fully offline.
- **One env var controls source**: `AVAILABILITY_SOURCE=json|zmq|none`, `AVAILABILITY_PATH=...`.
- **The frame-name → label parser** is the only new logic (~5 lines, regex).
- **The label → fdc_id mapping** reuses `hybrid_rank` from Phase A — no new retrieval code.

### B4. Plumb `available_fdc_ids` through the recommend chain

- [food_recommender.py:99-186](nutri_rag/nutri_rag/assistant/food_recommender.py#L99-L186) `recommend(...)` — add `available_fdc_ids: set[int] | None = None` parameter.
- [pipeline.py:69-77](nutri_rag/nutri_rag/assistant/pipeline.py#L69-L77) `NutriAssistant.recommend()` — add same parameter, pass through.
- [nutri-atlas/robot_control/tools/nutrition_tool.py:73-97](nutri-atlas/robot_control/tools/nutrition_tool.py#L73-L97) `GetMealRecommendation.call()` — at call time, fetch `available_fdc_ids` via `get_available_fdc_ids(source="detected_objects")` (configurable via env var `AVAILABILITY_SOURCE`), pass into `assistant.recommend(...)`.

Default `source` is configurable; for cold-start (no detections yet) the call returns `None` → no filter → graceful degradation to current behavior.

### B5. Apply availability + hybrid in the actual GAT expansion site

[food_recommender.py:161-184](nutri_rag/nutri_rag/assistant/food_recommender.py#L161-L184) — replace the `_gat_neighbors` loop with `_hybrid_neighbors(seed, k, alpha, available_fdc_ids)`. The `FoodOption.gat_similarity` field becomes a hybrid score; rename downstream usage to `similarity` for clarity, or keep the name and just document the change.

### B6. Verification for Phase B

```bash
# 1. No availability (regression): recommendation should match current behavior
AVAILABILITY_SOURCE=none python nutri_rag/scripts/demo_assistant.py \
    --eaten "an apple" --next-meal lunch

# 2. Server-side JSON (no robot needed): write a synthetic detected_objects.json
cat > nutri_rag/data/detected_objects.json <<'EOF'
{
  "detected_chicken_0": {"px": 1.0, "py": 0.0},
  "detected_rice_1":    {"px": 2.0, "py": 1.0},
  "detected_spinach_2": {"px": 3.0, "py": 0.5}
}
EOF
AVAILABILITY_SOURCE=json \
    python nutri_rag/scripts/demo_assistant.py --eaten "an apple"
# Verify all recommended fdc_ids map to chicken / rice / spinach foods

# 3. Live ZMQ: with the robot detected-objects server running
AVAILABILITY_SOURCE=zmq \
    python nutri-atlas/robot_control/robot_assistant.py
# Same recommendations should appear as in (2) if the robot has scanned the same items.

# 4. Compare quality: hybrid vs current GAT-only on 20 hand-picked meals
#    (toggle via FOOD_NEIGHBOR_MODE=hybrid vs FOOD_NEIGHBOR_MODE=gat_only)
#    Recommendation diversity (unique food categories in top-5) should not collapse.
```

---

## Phase C — Priority 3: nutrition → Food (using nutrient-node + text target embeddings)

### C1. Verify nutrient-node GAT embeddings exist (one-time check)

Established facts from exploration:
- `nutri_graph/outputs/embeddings/node_embeddings.npy` has shape **(93093, 64)** — all nodes in the trained graph.
- `food_embeddings.npy` has shape **(9991, 64)** — the first `NUM_FOODS` rows of `node_embeddings.npy` (see [train_GAT.py:99-100](nutri_graph/scripts/train_GAT.py#L99-L100)).
- Rows `[NUM_FOODS:]` correspond to nutrient + other node types. The DB has a `nodes_nutrient` table with stable `nutrient_id`.

One small data prep task (one-off): determine the offset and ordering of nutrient nodes in `node_embeddings.npy`. The mapping is whatever `nutri_graph/dataset.py` (referenced from `train_GAT.py`) used to construct `data.node_type`. Read it once, write a tiny script that emits `nutrient_id → row_index` as JSON in `nutri_graph/data/nutrient_emb_index.json`.

If this mapping cannot be recovered cleanly, fall back to the **centroid alternative**: q_gat = mean of GAT vectors for the top-K SQL-matched foods. The structured macro_match term recovers most of the signal.

### C2. Build a structured-target query encoder

```python
# nutri_rag/nutri_rag/assistant/target_encoder.py  (new)
def encode_target(
    targets: dict[str, float],   # {"protein_g": 30, "fat_g": 20, "carb_g": 60, "energy_kcal": 500}
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_text, q_gat) for the structured nutritional target."""
    # q_text: prose descriptor
    desc = (
        f"A balanced meal with approximately {targets['protein_g']:.0f}g protein, "
        f"{targets['fat_g']:.0f}g fat, {targets['carb_g']:.0f}g carbohydrate, "
        f"and {targets['energy_kcal']:.0f} kcal per serving."
    )
    q_text = _get_embedder().encode([desc], task_instruction=FOOD_SEARCH_INSTRUCTION)[0]

    # q_gat: weighted blend of nutrient node embeddings, weighted by target relative magnitudes
    nutrient_node_emb = _load_nutrient_node_embeddings()    # via index from C1
    weights = _normalize({
        "Protein":                        targets["protein_g"],
        "Total lipid (fat)":              targets["fat_g"],
        "Carbohydrate, by difference":    targets["carb_g"],
        "Energy":                         targets["energy_kcal"] / 100.0,  # rescale
    })
    q_gat = sum(w * nutrient_node_emb[name] for name, w in weights.items())
    q_gat /= np.linalg.norm(q_gat) + 1e-10
    return q_text, q_gat
```

This is the "use embedding! And not only GAT embedding, but also text embedding like what we did in healthy food sub" the user asked for. Like HealthyFoodSubs, **both** `q_text` and `q_gat` exist before fusion — only here they're constructed from the structured target rather than read from a food-node dict.

### C3. Add `recommend_v2` alongside (not replacing) the current `recommend`

[food_recommender.py:99-186](nutri_rag/nutri_rag/assistant/food_recommender.py#L99-L186) — keep the current `recommend()` (SQL seeds + GAT/hybrid expansion from Phase B) intact. Add a new `recommend_v2()` that uses target embeddings:

```python
def recommend_v2(self, targets, exclude_fdc_ids=None, available_fdc_ids=None,
                  alpha=0.5, structured_weight=0.5, n_results=10):
    """Target-as-query recommendation (no anchor on eaten foods)."""
    q_text, q_gat = encode_target(targets)

    # Structured candidate pool — SQL as filter, not ranker
    pool = self.con.execute("""
        SELECT fdc_id FROM nodes_food
        WHERE food_category_id IN (SELECT id FROM meal_categories)
    """).df()["fdc_id"].tolist()
    if exclude_fdc_ids:
        pool = [f for f in pool if f not in exclude_fdc_ids]
    if available_fdc_ids is not None:
        pool = [f for f in pool if f in available_fdc_ids]

    return hybrid_rank(
        q_text=q_text, q_gat=q_gat,
        candidate_fdc_ids=pool,
        alpha=alpha,
        structured_score=lambda x: macro_match(targets, get_nutrients(self.con, x)),
        structured_weight=structured_weight,
        k=n_results,
    )
```

This is exactly **HealthyFoodSubs hybrid scoring** applied to the recommend step — same precondition (both query vectors exist), same scoring shape, plus a structured nutrient-fit term that keeps the SQL signal alive.

Critically, the query is the **nutritional target itself** — not the eaten food. This is what the user explicitly asked for: "Note, here should not be find me foods similar to what the user eats!!!"

**Dispatch** at the pipeline level ([pipeline.py:69-160](nutri_rag/nutri_rag/assistant/pipeline.py#L69-L160)) is controlled by env var / constructor arg:

```python
recommend_mode = self._recommend_mode  # from RECOMMEND_MODE env var, default "v1" (current)
if recommend_mode == "v2":
    options = self._recommender.recommend_v2(targets, ...)
else:  # "v1" — unchanged from today
    options = self._recommender.recommend(targets, ...)
```

Default = current `recommend()` behavior. Opt-in via `RECOMMEND_MODE=v2`. Both paths emit the same `FoodOption` list shape, so the downstream LLM prompt formatter is unchanged.

### C4. The "meal_categories" filter

[nutri_rag/data/meal_categories.json](nutri_rag/data/meal_categories.json) (new, one-line list) — list of `food_category_id` values that count as meal-appropriate (excludes "Spices and Herbs", raw bulk ingredients like "Cereal Grains and Pasta — raw", supplement powders, etc.). Manual curation, ~15-20 entries from the ~25 USDA categories. Used as a hard filter to kill nonsense recommendations like "Whey protein isolate, dry powder".

### C5. Verification for Phase C

```bash
# 1. The same eaten food + same gap should now produce meal-shaped recommendations
python nutri_rag/scripts/demo_assistant.py --eaten "an apple" --next-meal lunch
# Before: top recommendations included "Whey protein isolate, dry"
# After: should be actual meal items (chicken, fish, beans, etc.)

# 2. Alpha sweep on a small held-out set to pick a default
for a in 0.0 0.25 0.5 0.75 1.0; do
    python nutri_rag/scripts/demo_assistant.py \
        --eaten "an apple" --next-meal lunch --alpha $a
done

# 3. Combined with availability (Phase B): recommendations should be both
#    nutrient-target-fitting AND in the pantry
AVAILABILITY_SOURCE=json AVAILABILITY_PATH=/tmp/pantry.json \
    python nutri_rag/scripts/demo_assistant.py --eaten "an apple"
```

---

## Files to modify (full inventory)

| Phase | File | Action |
|---|---|---|
| A1 | `nutri_rag/nutri_rag/search.py` | Add `hybrid_rank()`, `search_food_v2()` |
| A2-A3 | `nutri_rag/tasks/nutribench_v2_rag_gat_pure/` (new dir) | New: `utils.py` + yaml |
| A2-A3 | `nutri_rag/tasks/nutribench_v2_rag_hybrid/` (new dir) | New: `utils.py` + yaml |
| A4 | `nutri_rag/bench/retriever.py` | Accept `mode` + `alpha` (back-compat with `use_gat`) |
| A5 | `nutri_rag/scripts/run_bench.py` | Add v4, v5 to `TASK_DIRS`, argparse, labels |
| A5 | `nutri_rag/scripts/run_all_bench.py` | Add v4, v5 to `ALL_MODES`, `RAG_MODES` |
| A6 | `nutri_rag/nutri_rag/assistant/pipeline.py` | Add `eaten_retrieval_mode` dispatch; **default = hybrid (new)**, text top-1 kept as opt-in fallback |
| B1 | `nutri_rag/nutri_rag/config.py` | Add `TEXT_EMBEDDINGS_PATH`, `TEXT_FDC_IDS_PATH`, `DEFAULT_ALPHA`, `DEFAULT_AVAILABILITY_JSON` |
| B1-B2 | `nutri_rag/nutri_rag/assistant/food_recommender.py` | Load text emb, add `_hybrid_neighbors` **alongside** `_gat_neighbors` |
| B3 | `nutri_rag/nutri_rag/assistant/availability.py` (new) | JSON reader (server-side) + ZMQ reader (live robot) + label→fdc_id mapper |
| B3 | `nutri_rag/data/detected_objects.json` (new sample) | Server-side default availability mirror; same JSON format as the robot's |
| B4 | `nutri_rag/nutri_rag/assistant/pipeline.py` | Add `available_fdc_ids` param; add `neighbor_mode` dispatch |
| B4 | `nutri-atlas/robot_control/tools/nutrition_tool.py` | Fetch availability via env-configured source, pass through |
| B5 | `nutri_rag/nutri_rag/assistant/food_recommender.py` | Dispatch GAT-only vs hybrid in expansion; apply availability filter to both |
| C1 | `nutri_graph/scripts/build_nutrient_emb_index.py` (new, one-off) | Emit `nutrient_id → row_idx` JSON |
| C1 | `nutri_graph/data/nutrient_emb_index.json` (new, generated) | Output of above |
| C2 | `nutri_rag/nutri_rag/assistant/target_encoder.py` (new) | `encode_target()` for nutritional target |
| C3 | `nutri_rag/nutri_rag/assistant/food_recommender.py` | Add `recommend_v2()` **alongside** `recommend()` |
| C3 | `nutri_rag/nutri_rag/assistant/pipeline.py` | Dispatch v1 vs v2 recommend via `RECOMMEND_MODE` env var |
| C4 | `nutri_rag/data/meal_categories.json` (new) | Curated list of meal-shaped category IDs |

No other files touched. The existing `evaluate_hybrid()` in `nutri_graph/scripts/eval_food_subs.py` is unchanged — it stays as the reference implementation and one-shot evaluator.

---

## What is intentionally NOT in this plan

- No MMR diversification or preference re-weighting beyond what already exists. Both are useful but outside the user's stated priorities.
- No NutriBench v0 BM25 rework — kept as keyword baseline.
- No retraining of GAT or text embeddings.
- No new dataset construction. We reuse all existing artifacts (`food_text_embeddings.npy`, `food_embeddings.npy`, `node_embeddings.npy`, `nutri_kb.duckdb`).
- No HealthyFoodSubs changes — it's already correct and serves as the reference.

---

## End-to-end verification

After all three phases are merged:

```bash
# A. NutriBench — full sweep including new modes
python nutri_rag/scripts/run_all_bench.py \
    --modes baseline v0 v1 v2 v3 v4 v5 \
    --nutrients carb protein fat energy \
    --limit 500
# Expected: v5 (hybrid) ≥ v3 (text+GAT-expand multi-cand) on most cells.

# B. HealthyFoodSubs — sanity check we didn't break it (it doesn't depend on our changes, but check imports)
python nutri_graph/scripts/eval_food_subs.py --alpha 0.5

# C. Robot assistant — manual smoke test of the full pipeline
python nutri-atlas/robot_control/robot_assistant.py
# Ask: "I ate an apple, suggest a lunch"
# Verify:
#   - Eaten side: "apple" resolves cleanly via hybrid (Phase A6)
#   - Recommendation: items are meal-shaped (Phase C4) and within available foods (Phase B3)
#   - LLM still gets readable, ranked candidate list

# D. Per-phase regression tests already specified at A7, B6, C5.
```

The three pipelines share `hybrid_rank()` as the single retrieval primitive. Future tuning (α default, structured weight, alternative q_gat constructions) becomes one knob.
