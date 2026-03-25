# Upgrade Plan: Enrich nutri_graph with FoodKG Recipe Knowledge

## Goal

Extend nutri_graph's bipartite food↔nutrient graph with **recipe** and **tag** nodes
from FoodKG (via PFoodReq's pre-processed `recipe_kg.json`), so the GAT learns
food embeddings that encode both **nutritional similarity** and **recipe co-occurrence**.

## Data Source

**File:** `mimir/PFoodReq/data/recipe_kg/recipe_kg.json` (218MB, NDJSON)

| Stat               | Count  |
|---------------------|--------|
| Tags                | 238    |
| Unique recipes      | 76,138 |
| Unique ingredients  | 11,581 |
| Recipe nutrition    | 6 keys (calories, protein, carbs, sat/mono/poly fat) |

**Format:** Each line is a JSON object keyed by a tag URI:
```json
{
  "http://idea.rpi.edu/heals/kb/tag/baja": {
    "name": ["baja"],
    "type": ["tag"],
    "neighbors": {
      "tagged_dishes": [{
        "<dish_uri>": {
          "name": ["Guacamole Ole"],
          "type": ["dish_recipe"],
          "neighbors": {
            "calories": [380],
            "protein": [12.5],
            "carbohydrates": [24.0],
            "contains_ingredients": [{
              "<ing_uri>": {
                "name": ["fresh cilantro"],
                "type": ["ingredient"]
              }
            }]
          }
        }
      }]
    }
  }
}
```

## Current Graph (Before)

```
Node types: 2  (food=0, nutrient=1)
Nodes:      74,175 foods + 477 nutrients = 74,652 total
Edges:      155,216 food→nutrient (bidirectional = 310,432 for message passing)
Edge attr:  log1p(amount)
```

Built by:
- `kb/builder.py` → DuckDB tables: `nodes_food`, `nodes_nutrient`, `edges_food_contains_nutrient`
- `graph/dataset.py` → PyG `Data` object with `edge_index`, `edge_attr`, `node_type`
- `models/gat_model.py` → `GATFrontEnd(num_nodes=74652, num_types=2)`
- `scripts/train_GAT.py` → trains, saves `food_embeddings.npy` (74175 × 64)

## Target Graph (After)

```
Node types: 4  (food=0, nutrient=1, recipe=2, tag=3)
Nodes:      74,175 foods + 477 nutrients + ~76K recipes + 238 tags ≈ 151K total
Edges:      existing food→nutrient
          + recipe→food (uses ingredient)
          + recipe→tag (has tag)
          all bidirectional for message passing
Edge attr:  log1p(amount) for food→nutrient, 1.0 for recipe→food and recipe→tag
```

**What this enables:** After 2 GAT layers, food nodes hear from recipes they appear
in, and those recipes have heard from other foods. Foods that co-occur in recipes
(garlic + olive oil + tomato) get similar embeddings even if they're nutritionally
different.

---

## Phase 1: Ingest recipe_kg.json into DuckDB

**Files to create:** `nutri_graph/kb/recipe_builder.py`, `scripts/build_recipe_kb.py`
**Difficulty:** Medium (main challenge is ingredient name matching)
**Estimated lines:** ~200

### 1.1 New DuckDB Tables

Add to `nutri_kb.duckdb` (same database, new tables):

```sql
CREATE TABLE nodes_recipe (
    recipe_id       BIGINT PRIMARY KEY,  -- auto-assigned sequential ID
    recipe_name     VARCHAR,
    source_uri      VARCHAR,             -- original FoodKG URI
    calories        DOUBLE,
    protein         DOUBLE,
    carbohydrates   DOUBLE,
    saturated_fat   DOUBLE,
    monounsaturated_fat DOUBLE,
    polyunsaturated_fat DOUBLE
);

CREATE TABLE nodes_tag (
    tag_id          BIGINT PRIMARY KEY,  -- auto-assigned sequential ID
    tag_name        VARCHAR,
    source_uri      VARCHAR              -- original FoodKG URI
);

CREATE TABLE edges_recipe_uses_food (
    recipe_id       BIGINT,              -- FK → nodes_recipe
    fdc_id          BIGINT,              -- FK → nodes_food (FoodData Central ID)
    ingredient_name VARCHAR,             -- original name from recipe_kg.json
    similarity_score DOUBLE              -- text embedding match confidence
);

CREATE TABLE edges_recipe_has_tag (
    recipe_id       BIGINT,              -- FK → nodes_recipe
    tag_id          BIGINT               -- FK → nodes_tag
);
```

### 1.2 Ingredient Name → fdc_id Mapping

The core challenge: recipe_kg.json has ingredient names like `"fresh cilantro"`,
`"frozen broccoli carrots cauliflower mix"`. These need to be mapped to
nutri_graph's 74,175 USDA FoodData Central `fdc_id`s.

**Approach:** Use nutri_rag's existing `TextEmbedder` + `FoodVectorIndex`:

```python
from nutri_rag.embedding import TextEmbedder, FoodVectorIndex, FOOD_SEARCH_INSTRUCTION

embedder = TextEmbedder()
index = FoodVectorIndex()

def match_ingredient(name: str, threshold: float = 0.45) -> tuple[int, float] | None:
    """Map an ingredient name to the best-matching fdc_id."""
    query_vec = embedder.encode([name], task_instruction=FOOD_SEARCH_INSTRUCTION)
    results = index.search(query_vec, k=1)
    if results and results[0]:
        fdc_id, score, _ = results[0][0]
        if score >= threshold:
            return fdc_id, score
    return None
```

**Batch optimization:** Pre-encode all 11,581 unique ingredient names at once,
then do a single matrix multiply against the 74K food embeddings.

```python
# Encode all unique ingredient names
unique_ingredients = list(all_ingredient_names)  # 11,581 names
query_vecs = embedder.encode(unique_ingredients, task_instruction=FOOD_SEARCH_INSTRUCTION)

# Cosine similarity against all 74K USDA foods (single matrix op)
scores = query_vecs @ index.embeddings.T  # (11581, 74175)

# Best match per ingredient
best_indices = scores.argmax(axis=1)
best_scores = scores[np.arange(len(scores)), best_indices]

# Apply threshold
ingredient_to_fdc = {}
for i, name in enumerate(unique_ingredients):
    if best_scores[i] >= THRESHOLD:
        ingredient_to_fdc[name] = (int(index.fdc_ids[best_indices[i]]), float(best_scores[i]))
```

### 1.3 Parsing Pipeline

```python
def build_recipe_kb(recipe_kg_path: str, db_path: str, threshold: float = 0.45):
    """
    Parse recipe_kg.json and insert into existing nutri_kb.duckdb.

    Steps:
    1. First pass: collect all unique ingredient names, recipe names, tag names
    2. Batch-encode ingredient names with TextEmbedder
    3. Match against FoodVectorIndex
    4. Second pass: insert recipes, tags, and edges into DuckDB
    """
```

**Expected output stats (approximate):**
- `nodes_recipe`: ~76K rows
- `nodes_tag`: 238 rows
- `edges_recipe_uses_food`: ~76K recipes × ~8 avg ingredients = ~600K edges
  (after filtering unmatched ingredients, likely ~400-500K)
- `edges_recipe_has_tag`: ~76K+ (recipes can have multiple tags)

### 1.4 Quality Validation

After building, run sanity checks:

```sql
-- How many ingredients matched?
SELECT COUNT(DISTINCT ingredient_name) FROM edges_recipe_uses_food;
-- vs 11,581 total unique ingredients

-- Match quality distribution
SELECT
    CASE
        WHEN similarity_score >= 0.7 THEN 'high (>=0.7)'
        WHEN similarity_score >= 0.55 THEN 'medium (0.55-0.7)'
        ELSE 'low (<0.55)'
    END AS quality,
    COUNT(*) AS cnt
FROM edges_recipe_uses_food
GROUP BY 1;

-- Recipes with zero matched ingredients (should be few)
SELECT COUNT(*) FROM nodes_recipe r
WHERE NOT EXISTS (
    SELECT 1 FROM edges_recipe_uses_food e WHERE e.recipe_id = r.recipe_id
);
```

---

## Phase 2: Extend PyG Graph Construction

**Files to modify:** `nutri_graph/graph/dataset.py`
**Difficulty:** Medium
**Estimated lines:** ~100 lines added

### 2.1 Modified `build_graph_from_db`

The function currently builds a bipartite graph with 2 node types. It needs to
support 4 node types and 3 edge types.

**New node index layout:**

```
Index range                              | Type | Count
[0 .. NUM_FOODS-1]                       | food=0     | 74,175
[NUM_FOODS .. NUM_FOODS+NUM_NUTRIENTS-1] | nutrient=1 | 477
[..F+N .. F+N+NUM_RECIPES-1]             | recipe=2   | ~76,138
[..F+N+R .. F+N+R+NUM_TAGS-1]            | tag=3      | 238
```

**New queries to add in `build_graph_from_db`:**

```python
# Load recipe data
recipes = con.execute("SELECT recipe_id FROM nodes_recipe").df()
tags = con.execute("SELECT tag_id FROM nodes_tag").df()
recipe_food_edges = con.execute("""
    SELECT recipe_id, fdc_id
    FROM edges_recipe_uses_food
""").df()
recipe_tag_edges = con.execute("""
    SELECT recipe_id, tag_id
    FROM edges_recipe_has_tag
""").df()

NUM_RECIPES = len(recipes)
NUM_TAGS = len(tags)

# Build index mappings
recipe_id_to_idx = {int(rid): i for i, rid in enumerate(recipes["recipe_id"])}
tag_id_to_idx = {int(tid): i for i, tid in enumerate(tags["tag_id"])}

# Global offsets
RECIPE_OFFSET = NUM_FOODS + NUM_NUTRIENTS
TAG_OFFSET = RECIPE_OFFSET + NUM_RECIPES
```

**New edge construction:**

```python
# recipe → food edges (bidirectional)
src_recipe = recipe_food_edges["recipe_id"].map(recipe_id_to_idx).to_numpy() + RECIPE_OFFSET
dst_food = recipe_food_edges["fdc_id"].map(food_id_to_idx).to_numpy()
# edge_attr = 1.0 (no amount info for recipe→food edges)

recipe_food_ei = torch.tensor(np.vstack([src_recipe, dst_food]), dtype=torch.long)
food_recipe_ei = torch.tensor(np.vstack([dst_food, src_recipe]), dtype=torch.long)

# recipe → tag edges (bidirectional)
src_recipe_t = recipe_tag_edges["recipe_id"].map(recipe_id_to_idx).to_numpy() + RECIPE_OFFSET
dst_tag = recipe_tag_edges["tag_id"].map(tag_id_to_idx).to_numpy() + TAG_OFFSET

recipe_tag_ei = torch.tensor(np.vstack([src_recipe_t, dst_tag]), dtype=torch.long)
tag_recipe_ei = torch.tensor(np.vstack([dst_tag, src_recipe_t]), dtype=torch.long)

# Combine all edges for message passing
edge_index_all = torch.cat([
    pos_edge_index, rev_edge_index,           # food ↔ nutrient (existing)
    recipe_food_ei, food_recipe_ei,           # recipe ↔ food (new)
    recipe_tag_ei, tag_recipe_ei,             # recipe ↔ tag (new)
], dim=1)

# Edge attributes: log1p(amount) for food↔nutrient, 1.0 for recipe edges
recipe_food_attr = torch.ones(recipe_food_ei.size(1), 1)
recipe_tag_attr = torch.ones(recipe_tag_ei.size(1), 1)

edge_attr_all = torch.cat([
    edge_attr_pos, edge_attr_pos,             # food ↔ nutrient
    recipe_food_attr, recipe_food_attr,       # recipe ↔ food
    recipe_tag_attr, recipe_tag_attr,         # recipe ↔ tag
], dim=0)

# Node types: 4 types
TOTAL_NODES = NUM_FOODS + NUM_NUTRIENTS + NUM_RECIPES + NUM_TAGS
node_type = torch.zeros(TOTAL_NODES, dtype=torch.long)
node_type[NUM_FOODS:NUM_FOODS+NUM_NUTRIENTS] = 1
node_type[RECIPE_OFFSET:RECIPE_OFFSET+NUM_RECIPES] = 2
node_type[TAG_OFFSET:TAG_OFFSET+NUM_TAGS] = 3
```

### 2.2 Backward Compatibility

Add a flag to `build_graph_from_db` so the old bipartite-only mode still works:

```python
def build_graph_from_db(db_path: str, include_recipes: bool = False):
    ...
    # If include_recipes=False, skip recipe/tag loading (original behavior)
    # If include_recipes=True, check if nodes_recipe table exists, load if so
```

### 2.3 Updated meta dict

```python
meta = {
    "NUM_FOODS": NUM_FOODS,
    "NUM_NUTRIENTS": NUM_NUTRIENTS,
    "NUM_RECIPES": NUM_RECIPES,       # new
    "NUM_TAGS": NUM_TAGS,             # new
    "RECIPE_OFFSET": RECIPE_OFFSET,   # new
    "TAG_OFFSET": TAG_OFFSET,         # new
    "foods": foods,
    "nutrs": nutrs,
    "food_id_to_idx": food_id_to_idx,
    "nutr_id_to_idx": nutr_id_to_idx,
    "food_to_nutrs": food_to_nutrs,
}
```

---

## Phase 3: Adjust Model and Training

**Files to modify:** `scripts/train_GAT.py` (minor), `models/gat_model.py` (minimal)
**Difficulty:** Low (for message-passing-only approach)
**Estimated lines:** ~20 lines changed

### 3.1 Model Changes

The `GATFrontEnd` architecture does NOT need structural changes. The GATv2Conv
layers are node-type-agnostic — they operate on any graph topology. The only
change is the constructor arguments:

```python
# train_GAT.py — change from:
model = GATFrontEnd(
    num_nodes=data.num_nodes,   # was 74,652 → now ~151,028
    num_types=2,                # was 2 → now 4
    ...
)
```

The `node_emb` table grows from 74,652 to ~151K rows (more parameters), and
`type_emb` grows from 2 to 4 rows. The GATv2Conv layers, decoders, and all
other parameters stay the same.

### 3.2 Training Changes

**Approach: Message-passing only (no new decoder/loss).**

The existing training loop in `trainer.py` supervises on `food → nutrient` edges
only (existence + amount prediction). Recipe and tag edges participate in
message passing (they're in `edge_index_all`) but are not directly supervised.

This means:
- Recipe nodes act as **bridges** between foods that co-occur in recipes
- Tag nodes act as **bridges** between recipes with the same cuisine/category
- The food embeddings absorb co-occurrence signal through 2-hop message passing
- **No changes to `trainer.py` needed** — it only touches `pos_edge_index`
  (food→nutrient) for the loss, and uses `edge_index_all` for encoding

**The only change in `train_GAT.py`:**

```python
# Before:
data, meta = build_graph_from_db("data/nutri_kb.duckdb")

# After:
data, meta = build_graph_from_db("data/nutri_kb.duckdb", include_recipes=True)
```

### 3.3 Training Considerations

- **Memory:** ~151K nodes × 64-dim embeddings = ~38MB for `node_emb` (was ~19MB).
  Additional edges (~1M recipe↔food + ~76K recipe↔tag) increase `edge_index`
  size. Total GPU memory increase is modest (~200MB extra).
- **Speed:** More edges = more message passing per layer. Expect ~2-3× slower
  per epoch. Consider reducing MAX_EPOCHS from 90 to 60 if convergence is faster
  with the richer graph.
- **Validation:** The supervised task (food→nutrient prediction) remains unchanged.
  Validation RMSE should still be the primary metric. Monitor whether recipe
  edges help or hurt nutrient prediction — if RMSE increases, the recipe signal
  might be adding noise.

### 3.4 Optional: Recipe-Ingredient Decoder (Future Enhancement)

If message-passing-only doesn't produce sufficient co-occurrence signal, add a
third decoder head:

```python
# In gat_model.py
self.recipe_exist_mlp = nn.Sequential(
    nn.Linear(2 * hidden, hidden),
    nn.ReLU(),
    nn.Linear(hidden, 1),
)

def decode_recipe_exist(self, h, edge_index):
    z = self.pair(h, edge_index)
    return self.recipe_exist_mlp(z).squeeze(-1)
```

With a corresponding loss in `trainer.py`:
```python
# Supervised recipe→food existence
recipe_food_logits = model.decode_recipe_exist(h, recipe_pos_edges)
neg_recipe_food_logits = model.decode_recipe_exist(h, recipe_neg_edges)
loss_recipe = bce(recipe_food_logits, 1) + bce(neg_recipe_food_logits, 0)

loss = loss_amt + 0.4 * loss_exist + 0.2 * loss_recipe
```

This is more complex and requires recipe-level negative sampling. Defer until
after evaluating Phase 3.2 results.

---

## Phase 4: Extract and Evaluate Enriched Embeddings

**Files to modify:** None (existing `train_GAT.py` extraction code works as-is)
**Difficulty:** Trivial

### 4.1 Embedding Extraction

The existing code in `train_GAT.py:82-84` already handles this:

```python
food_emb = h[: int(meta["NUM_FOODS"])]
np.save("outputs/embeddings/food_embeddings.npy", food_emb)
```

Food nodes are still indices `[0 .. NUM_FOODS-1]`, so the slice is unchanged.
The output shape is still `(74175, 64)`. nutri_rag loads this file without
knowing how it was produced — **no downstream changes needed**.

### 4.2 Evaluation: Comparing Before vs After

**Quantitative (NutriBench):**
Run the existing benchmark with old vs new embeddings:

```bash
# Save old embeddings
cp outputs/embeddings/food_embeddings.npy outputs/embeddings/food_embeddings_v1.npy

# Train with recipes, produces new food_embeddings.npy
python scripts/train_GAT.py  # (with include_recipes=True)

# Run NutriBench V2 (GAT re-ranking) with new embeddings
cd ../nutri_rag && python scripts/run_bench.py --mode v2
```

Compare accuracy and MAE between old and new embeddings.

**Qualitative (Neighbor inspection):**
For known food items, compare top-5 GAT neighbors before and after:

```python
import numpy as np

old_emb = np.load("food_embeddings_v1.npy")
new_emb = np.load("food_embeddings.npy")

# Pick "chicken breast" (find its index)
# Compare: old neighbors = nutritionally similar foods
#          new neighbors = nutritionally similar + recipe co-occurring foods
```

Expected changes:
- Before: "chicken breast" neighbors = turkey, lean pork, tilapia (pure nutrition)
- After: "chicken breast" neighbors = turkey, olive oil, garlic, lemon
  (nutrition + recipe co-occurrence)

Whether this is desirable depends on the downstream task. For NutriBench
(nutrient estimation), pure nutritional similarity may be better. For the
assistant (meal recommendations), recipe co-occurrence adds useful variety.

---

## Execution Order and Dependencies

```
Phase 1: Ingest recipe_kg.json → DuckDB
    │
    ├── 1.1 Create recipe_builder.py (new file)
    ├── 1.2 Ingredient matching (uses nutri_rag TextEmbedder)
    ├── 1.3 Build script: scripts/build_recipe_kb.py (new file)
    └── 1.4 Validation queries
    │
    ▼
Phase 2: Extend PyG graph
    │
    ├── 2.1 Modify graph/dataset.py (add recipe/tag node types + edges)
    ├── 2.2 Add include_recipes flag for backward compatibility
    └── 2.3 Update meta dict
    │
    ▼
Phase 3: Adjust training
    │
    ├── 3.1 Update train_GAT.py (num_types=4, include_recipes=True)
    ├── 3.2 Train with message-passing only (no trainer.py changes)
    └── 3.3 Monitor convergence and memory
    │
    ▼
Phase 4: Extract and evaluate
    │
    ├── 4.1 Extract food_embeddings.npy (no code changes)
    └── 4.2 Compare NutriBench scores + qualitative neighbor analysis
```

## Files Changed/Created Summary

| File | Action | Phase |
|------|--------|-------|
| `nutri_graph/kb/recipe_builder.py` | **Create** | 1 |
| `scripts/build_recipe_kb.py` | **Create** | 1 |
| `nutri_graph/graph/dataset.py` | **Modify** — add recipe/tag node loading | 2 |
| `scripts/train_GAT.py` | **Modify** — `num_types=4`, `include_recipes=True` | 3 |
| `nutri_graph/config.py` | **Modify** — add `INCLUDE_RECIPES = True` flag | 3 |
| `nutri_graph/models/gat_model.py` | **No changes** (for Phase 3.2) | — |
| `nutri_graph/training/trainer.py` | **No changes** (for Phase 3.2) | — |
| `nutri_rag/**` | **No changes** (loads food_embeddings.npy as before) | — |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low ingredient match rate (<50%) | Recipe edges too sparse to help | Lower threshold to 0.40; inspect failures; add fuzzy fallback |
| Recipe edges hurt nutrient prediction (val_RMSE increases) | Worse food embeddings | Compare with/without recipes; tune recipe edge weight; try Phase 3.4 decoder |
| Memory/speed issues with ~151K nodes | Can't train on available GPU | Subsample recipes (use top-50K by ingredient coverage); reduce batch edges |
| USDA version mismatch (SR-27 names vs FDC descriptions) | Systematic match failures for certain ingredients | Qwen3-Embedding handles synonyms well; validate on sample before full run |
