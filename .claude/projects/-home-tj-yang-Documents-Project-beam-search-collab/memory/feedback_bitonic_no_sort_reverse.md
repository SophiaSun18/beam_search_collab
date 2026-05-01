---
name: No sort-then-reverse for bitonic halves
description: When implementing upper/lower half split in bitonic sorting, keep the kept half in the right order upfront instead of sorting the whole buffer then reversing.
type: feedback
---

When implementing bitonic sort merge with upper/lower half splits, do NOT sort the whole buffer ascending then reverse the kept half to make it descending. Instead, maintain the kept half in descending order from the start (e.g. reverse the input once upfront). This avoids unnecessary sort+reverse in the loop body.

**Why:** User has corrected this pattern multiple times. The sort+reverse approach is wasteful and a "bad old habit."

**How to apply:** Whenever a bitonic merge loop needs one half kept in a particular order across iterations, set that order once before the loop, not repeatedly inside it.
