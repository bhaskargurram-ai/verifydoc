# Examples

A runnable 30-second example. From the repo root (or anywhere after
`pip install verifydoc`):

```bash
verifydoc extract examples/invoice.txt --schema examples/invoice_schema.json
```

Or from Python:

```python
from verifydoc import verify

result = verify("examples/invoice.txt", "examples/invoice_schema.json")
for f in result.fields:
    mark = "✅" if f.decision == "accept" else "🔎"
    print(f"{mark} {f.path:12} = {f.value!r:20} conf={f.confidence:.2f}")
print(f"\n{result.n_accepted} auto-accepted, {result.n_review} to review")
```

Every field comes back with a calibrated confidence, a grounding (the page
region / character span the value was read from), and an accept/review
decision. Values that can't be traced to the page get low confidence and are
routed to review — that's how silent hallucinations get caught.
