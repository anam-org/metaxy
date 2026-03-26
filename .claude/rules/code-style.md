**Docstrings**: Avoid documenting implementation details. Focus on high-level concepts, properties and invariants. Avoid documenting default values as they are already documented in the code.

**Comments**: Write comments that describe the current state of the code, not how it got there. Do not reference refactoring history, deleted code, or previous implementations (e.g., avoid "using X because Y was removed" or "previously this used Z").

**Variable usage**: Avoid introducing variables that are only used once. Prefer method chaining and inline expressions. Exception: use a named variable when it clarifies non-obvious meaning or intent.
