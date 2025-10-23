## Feature Version

Every feature has a **feature version** - hash of its complete specification:
- Feature key
- Field definitions and code versions
- Dependencies (feature-level and field-level)

Being more pedantic, the feature version is calculated by hashing together its field versions and the feature key. The field version is calculated by hashing together the field code version, key, and versions of its dependencies.

Feature and field versions are static and deterministic. They are available as class methods:

```python
class VideoProcessing(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "processing"]),
    fields=[
        FieldSpec(key=FieldKey(["frames"]), code_version=1),
        FieldSpec(key=FieldKey(["audio"]), code_version=1),
    ],
)):
    pass

# Get feature version
print(VideoProcessing.feature_version())  # "a3f8b2c1"

# Change code_version
class VideoProcessing(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "processing"]),
    fields=[
        FieldSpec(key=FieldKey(["frames"]), code_version=2),  # Changed!
        FieldSpec(key=FieldKey(["audio"]), code_version=1),
    ],
)):
    pass

print(VideoProcessing.feature_version())  # "d7e9f4a2" (different!)
```

Every metadata row includes a `feature_version` column:
```python
metadata = pl.DataFrame({
    "sample_id": [1, 2, 3],
    "data_version": [...],
    "feature_version": "a3f8b2c1",  # Auto-added by store
})
```

This enables:
- **Current vs historical**: Filter by `feature_version` to get specific versions
- **Migration tracking**: Know which rows need updating
- **Audit trail**: See what feature definition produced each row


## User-Defined Metadata Alignment

When feature definitions change, you may need custom logic to align metadata with upstream changes. Override `Feature.align_metadata_with_upstream`:

### Default Behavior (Inner Join)

```python
class MyFeature(Feature, spec=...):
    # Default behavior: inner join on sample_id
    # Only keeps samples present in ALL upstream features
    pass

# During migration:
# - Reads current metadata
# - Inner joins with upstream on sample_id
# - Only samples present in both current AND upstream survive
```

### Custom Alignment Examples

#### One-to-Many Mapping

```python
class VideoFrames(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "frames"]),
    deps=[FeatureDep(key=FeatureKey(["videos"]))],
    ...
)):
    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Each video produces 30 frames.

        When videos upstream change, regenerate all frame sample IDs.
        """
        video_samples = upstream_metadata["videos"]["sample_id"]

        # Generate frame sample IDs
        frames = []
        for video_id in video_samples:
            for frame_idx in range(30):
                frames.append({
                    "sample_id": f"{video_id}_frame_{frame_idx}",
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                })

        return pl.DataFrame(frames)
```

#### Conditional Filtering

```python
class ProcessedVideos(Feature, spec=...):
    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Only process videos longer than 10 seconds."""

        videos = upstream_metadata["videos"]

        # Filter by duration
        valid_videos = videos.filter(pl.col("duration") > 10)

        # Keep custom columns from current_metadata if available
        if len(current_metadata) > 0:
            return current_metadata.join(
                valid_videos.select(pl.col("sample_id")),
                on="sample_id",
                how="inner",  # Only keep samples that pass filter
            )
        else:
            return valid_videos.select(["sample_id", "duration", "path"])
```

#### Outer Join (Keep All Samples)

```python
class MergedFeature(Feature, spec=...):
    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Keep union of all upstream samples (outer join)."""

        all_samples = set()
        for upstream_df in upstream_metadata.values():
            all_samples.update(upstream_df["sample_id"].to_list())

        return pl.DataFrame({"sample_id": sorted(all_samples)})
```
