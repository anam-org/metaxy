# Migration Example

```mermaid
---
title: Feature Graph (snapshot: 4bd49031)
---
flowchart TB
        examples_parent["<b>examples/parent</b><br/><small>(v: befcdcdd)</small><br/>---<br/>•
embeddings <small>(v: eb57f1d2)</small>"]
        examples_child["<b>examples/child</b><br/><small>(v: 6148c18f)</small><br/>---<br/>•
predictions <small>(v: bebfc10d)</small>"]
        examples_parent --> examples_child
```
