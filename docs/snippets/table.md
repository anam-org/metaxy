| id        | metaxy_feature_version | metaxy_data_version | metaxy_data_version_by_field                  | metaxy_provenance | metaxy_provenance_by_field                    | metaxy_created_at    | metaxy_updated_at    | metaxy_deleted_at |
| --------- | ---------------------- | ------------------- | --------------------------------------------- | ----------------- | --------------------------------------------- | -------------------- | -------------------- | ----------------- |
| video_001 | a1b2c3d4               | e7f8a9b0            | `{"audio": "a7f3c2d8", "frames": "b9e1f4a2"}` | e7f8a9b0          | `{"audio": "a7f3c2d8", "frames": "b9e1f4a2"}` | 2024-01-15T10:30:00Z | 2024-01-15T10:30:00Z | null              |
| video_002 | a1b2c3d4               | c1e4b9d8            | `{"audio": "d4b8e9c1", "frames": "f2a6d7b3"}` | c1e4b9d8          | `{"audio": "d4b8e9c1", "frames": "f2a6d7b3"}` | 2024-01-15T10:31:00Z | 2024-01-15T10:31:00Z | null              |
| video_003 | a1b2c3d4               | k1j2ah7v            | `{"audio": "custom01", "frames": "custom02"}` | a8e2f4c9          | `{"audio": "c9f2a8e4", "frames": "e7d3b1c5"}` | 2024-01-15T10:32:00Z | 2024-01-16T14:20:00Z | null              |
| video_001 | f5d6e7c8               | b2c3d4e5            | `{"audio": "b1e4f9a7", "frames": "a8c2e6d9"}` | b2c3d4e5          | `{"audio": "b1e4f9a7", "frames": "a8c2e6d9"}` | 2024-01-18T09:00:00Z | 2024-01-18T09:00:00Z | null              |

It can also contain custom user-defined columns (1).
{ .annotate }

1. and in fact, `id` is such a column, because ID columns are customizable
