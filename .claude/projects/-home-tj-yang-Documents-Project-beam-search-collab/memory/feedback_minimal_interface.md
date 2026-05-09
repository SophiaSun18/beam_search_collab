---
name: Keep interfaces minimal and don't over-design
description: User wants minimal interfaces (3 methods max), self-contained structs, no external buffer management, no speculative abstractions.
type: feedback
---

Keep interfaces minimal. Don't add methods the caller doesn't need. Don't pass internal state externally. Don't think too long before communicating — ask if confused.

**Why:** User has corrected bloated interfaces multiple times. Extra methods and parameters create confusion and maintenance burden.

**How to apply:** When designing a class/concept, start with the absolute minimum operations the caller needs. Add more only when the user asks. Prefer self-contained structs over parameter-heavy free functions.
