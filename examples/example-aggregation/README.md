# N:1 Aggregation Example

This example demonstrates Metaxy's N:1 aggregation lineage, where multiple upstream records are aggregated into a single downstream record.

## Scenario

- **Audio**: Individual audio recordings, each with a `speaker_id`
- **SpeakerEmbedding**: Aggregated embedding computed from all audio recordings of a speaker

Multiple audio recordings from the same speaker are aggregated to compute a single speaker embedding.

## Running

```bash
uv run python pipeline.py
```

## Key Concepts

1. **Aggregation Lineage**: `LineageRelationship.aggregation(on=["speaker_id"])` tells Metaxy that multiple Audio records with the same `speaker_id` are aggregated into one SpeakerEmbedding.

2. **Window Functions**: Metaxy uses window functions to compute aggregated provenance without reducing rows. All audio rows for the same speaker get identical provenance values.

3. **User Aggregation**: The pipeline code performs the actual aggregation (grouping by speaker_id). Metaxy only tracks the provenance.

4. **Change Detection**: When any audio for a speaker changes, the aggregated provenance changes, triggering recomputation of that speaker's embedding.
