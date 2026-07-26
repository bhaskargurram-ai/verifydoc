# VerifyDoc human-annotation guide (campaign 2026-07, issue #68)

You are judging whether a value that an AI extracted from a document is **correct**,
by looking at the document image yourself. Your labels become the human gold standard
for two research papers — please follow the rules exactly and work independently
(**do not discuss items with the other annotators until everyone is done**).

## Setup
You received a folder with `images/`, and your own CSV file (`annotator_A.csv`,
`_B`, or `_C` — use only yours). Open the CSV in any spreadsheet app. For each row:

1. Open the image named in the `image` column.
2. Read `field` (what was being extracted — e.g. `menu[0].nm` = the name of the
   first menu item on a receipt; `total.total_price` = the grand total) and
   `predicted_value` (what the AI said).
3. Fill `label` with exactly one of: **`1`** (correct), **`0`** (incorrect), **`U`** (cannot judge).

## Decision rules
- **`1` (correct)** — the predicted value matches what the document actually says
  for that field, allowing:
  - case, spacing, and punctuation differences (`Ice Tea` vs `ICE TEA.`);
  - currency/format differences for numbers (`12,000` vs `12000`);
  - obvious OCR-style transliteration of the same token.
- **`0` (incorrect)** — any of:
  - the value contradicts the document (wrong item, wrong number, wrong person);
  - the value is not present in the document at all (invented);
  - the value belongs to a *different* field (e.g. unit price given as total);
  - only part of a multi-part answer, if the missing part changes the meaning.
- **`U` (cannot judge)** — the image is unreadable at that spot, the field is
  genuinely ambiguous (two equally valid answers), or you cannot find the
  region after a careful look (~60 seconds max per item).

## Important
- Judge against the **document**, not against what you think a database answer
  should look like.
- Do not penalize the AI for reasonable formatting; do penalize wrong content.
- `notes` is optional — use it when you pick `U` or when a case is interesting.
- Expect roughly 20–30 seconds per item; ~600 items ≈ 3–4 hours. Take breaks;
  accuracy matters more than speed.
- Label **every** row (use `U` rather than leaving blanks).

When done, send back your CSV. Agreement statistics (Fleiss'/Cohen's kappa) and
all downstream analyses are computed by `scripts/campaign2026/score_human_gold.py`.
