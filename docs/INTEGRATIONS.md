# Drop-in integrations

VerifyDoc adds a trust layer to the extraction tools you already use. These
wrappers import no framework dependency — any Pydantic object or
`str -> dict` callable works — so they cover Instructor, Pydantic-AI, Outlines,
Marvin, and LangChain.

## Instructor / Pydantic / Outlines / Marvin

You already extract a typed object; VerifyDoc scores each field against the
source document (confidence + grounding + accept/review) with no extra model call:

```python
import instructor
from verifydoc.integrations.instructor import verify_instructor_result

client = instructor.from_openai(OpenAI())
invoice = client.chat.completions.create(response_model=Invoice, messages=[...])

report = verify_instructor_result(document_text, invoice, threshold=0.8)
for f in report.fields:
    if f.decision == "review":
        print("verify by hand:", f.path, "=", f.value, "grounding:", f.grounding)
print(f"{report.n_accepted} auto-accepted, {report.n_review} to review")
```

## LangChain

Wrap any extraction chain / runnable (`document -> dict` or `-> BaseModel`):

```python
from verifydoc.integrations.langchain import VerifiedExtractor

chain = prompt | llm.with_structured_output(Invoice)
extractor = VerifiedExtractor(chain.invoke, Invoice, threshold=0.8)

result = extractor(document_text)
if result.n_review:
    route_to_human(result)          # only the uncertain fields
payload = result.to_dict()          # value + confidence + grounding + decision
```

## Why this beats validation-only

Instructor/Outlines/Pydantic guarantee the invoice total is a `float`. VerifyDoc
tells you whether it's the *right* float and whether it appears on the page —
and routes the rest to review with an audit trail. They're complementary:
**validation checks format; VerifyDoc checks trustworthiness.**
