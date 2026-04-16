---
title: "Questions And Answers"
description: "Answers to frequently asked questions."
---

# Questions And Answers

## How can I redesign feature fields without invalidating everything downstream?

Do your redesign and use [metadata rebases](/guide/concepts/metadata-stores.md#rebasing-metadata-versions).

## How can I select a subset of a feature for recomputation?

Invoke [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update] with an appropriate condition passed to `staleness_predicates`.

## Why are feature definitions created via classes?

In short: it's handy for some integrations, and allows to have more type annotation goodness, but isn't strictly necessary.

!!! note
    Other interfaces will be explored in the future — see [`anam-org/metaxy#800`](https://github.com/anam-org/metaxy/issues/800).

## Do you support PostgreSQL?

Kinda.

!!! warning
    PostgreSQL is a [really bad choice](/integrations/metadata-stores/databases/postgresql.md) for a metadata store backend.

## What does the logo mean?

It's an accident. The intent was for it to resemble a galaxy, but it is what it is.

!!! info
    Yes, we've seen [this page](https://velvetshark.com/ai-company-logos-that-look-like-buttholes).
