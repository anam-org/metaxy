import logging

from metaxy.entrypoints import load_features

logging.basicConfig(level=logging.DEBUG)


print("Calling load_features()...")
graph = load_features()

print(f"Graph has {len(graph.features_by_key)} features")
print(f"Features: {list(graph.features_by_key.keys())}")

assert len(graph.features_by_key) == 2, (
    f"Expected 2 features, got {len(graph.features_by_key)}"
)

print("SUCCESS: All features loaded via entry points")
