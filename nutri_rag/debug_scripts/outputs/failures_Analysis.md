5 Root Causes (ranked by frequency)
1. PARTIAL_COVERAGE — 125 samples (36%)
Only some food items get USDA matches. The model computes matched items precisely with the formula but guesses poorly on unmatched ones. The structured format makes it more likely to under-estimate unmatched items.


doc=7: "okra leaves, onion, sorghum flour, tomato, caterpillar, goat"
  → Only sorghum flour matched. Model underestimates caterpillar + goat protein.
2. WRONG_MATCH — 113 samples (33%)

Food above the 0.55 threshold but matched to the wrong USDA entry:

beans → Chickpeas, dry (21.3g/100g vs ~5g for baked beans)
apple → Pears, raw
mashed potatoes → Flour, potato (76g carb vs ~15g for mashed)
egg → Egg, whole, dried (48g protein/100g vs ~13g for cooked egg)
potato → Flour, potato
pinto beans → Pupusas con frijoles
These wrong matches have catastrophic impact — e.g., "dried egg" has 4x the protein of a normal egg.

3. NO_TERMS_EXTRACTED — 68 samples (20%)

The regex parser fails entirely on some sentence structures, producing zero food terms. The RAG block is then empty and the model generates slightly differently than pure baseline.

4. RAG_RETURNED_MINUS1 — 46 samples (13%)

Model gets confused by the per-item format and outputs {"total_protein": -1}. Often happens when most items are "no reliable match" and the model gives up rather than estimating.

5. BROKEN_TERM — 22 samples (6%)

Regex grabs too much text including other items' weights:

"powdered milk with 13 grams of sugar" → single term
"white bread with 100g of chicken" → single term
"white rice up with drizzle of 34" → single term
This merges multiple food items into one, then the search returns a completely wrong USDA match.