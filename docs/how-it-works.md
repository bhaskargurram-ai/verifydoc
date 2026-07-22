# How VerifyDoc works

VerifyDoc is a thin, model-agnostic *layer*. It never replaces your extractor —
it wraps one and adds a per-field reliability contract. The pipeline is a
sequence of independent, typed stages; each has its own tests and can be swapped.

```
document + schema
     │
     ▼
  ingest          pdf/image/text → pages with a text layer + word boxes
     │
     ▼
  adapter         any extractor behind ExtractorAdapter (one file per model):
     │            rapidocr · paddleocr-vl · docling · api-vlm · your own
     ▼
  confidence      raw per-field score from one or more signals:
     │            token-prob · verbalized · k-sample consensus · grounding · learned
     ▼
  calibration     map raw score → calibrated probability, fit ONLY on a
     │            held-out calibration split (temperature/Platt/isotonic/
     │            histogram/conformal). Never tuned on test.
     ▼
  grounding       locate each value on the page → {page, bbox, char_span, support}
     │
     ▼
  policy          accept iff confidence ≥ threshold; threshold chosen for a
     │            target selective risk α (empirical or conformal guarantee)
     ▼
  verified JSON   every leaf: value + confidence + grounding + accept/review
```

## The core idea

At an operating point you pick a target error budget α (say 2%). VerifyDoc
auto-accepts the largest set of fields whose *estimated* risk stays under α,
and routes the rest to `review` — each with its source region attached, so a
human verifies in seconds. You trade a little coverage for a controlled error
rate, instead of trusting or eyeballing everything.

## Why grounding is a trust signal

A correct value can be traced to a region on the page; a hallucinated value
often cannot. VerifyDoc turns that into a signal: the `support` score of a
value's grounding (how well the located region matches) predicts correctness.
On real receipts and forms, grounded fields are ~85% correct versus ~1% for
ungrounded — so grounding both *explains* a decision and *drives* it.

## Extending it

- **New extractor** → one file in `verifydoc/adapters/` implementing
  `extract(doc, schema) -> list[FieldPrediction]`. Nothing else changes.
- **New dataset** → a loader in `benchmark/datasets/` returning
  `BenchDocument`s. The harness scores it with the full metric suite.
- **New signal / calibrator** → drop it behind the existing interface.

See [CONTRIBUTING.md](../CONTRIBUTING.md) and [PROJECT.md](../PROJECT.md) §5–§7.
