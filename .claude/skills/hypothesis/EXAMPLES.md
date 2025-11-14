# Hypothesis Examples

Practical examples for property-based testing with Hypothesis.

## Basic Property Test

```python
from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.integers()))
def test_sorted_properties(lst):
    """Properties that sorted() must satisfy"""
    sorted_lst = sorted(lst)

    # Length preserved
    assert len(sorted_lst) == len(lst)

    # All elements present
    assert set(sorted_lst) == set(lst)

    # Monotonically increasing
    for i in range(len(sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]
```

## Composite Strategy

```python
from hypothesis import given, strategies as st
from hypothesis.strategies import composite


@composite
def user_with_preferences(draw):
    """Generate user dict with constrained fields"""
    age = draw(st.integers(min_value=18, max_value=100))
    name = draw(st.text(min_size=1, max_size=50))
    email = draw(st.emails())

    return {
        "name": name,
        "age": age,
        "email": email,
        "is_adult": age >= 18,
        "preferences": {
            "newsletter": draw(st.booleans()),
            "theme": draw(st.sampled_from(["light", "dark", "auto"])),
        },
    }


@given(user_with_preferences())
def test_user_invariants(user):
    """Test user data properties"""
    # Adult flag is consistent
    assert user["is_adult"] == (user["age"] >= 18)

    # Email format
    assert "@" in user["email"]

    # Theme is valid
    assert user["preferences"]["theme"] in ["light", "dark", "auto"]
```

## Testing with Polars DataFrames

```python
from hypothesis import given
import polars as pl
from polars.testing.parametric import dataframes, column
import narwhals as nw


@given(
    dataframes(
        cols=[
            column("sample_uid", dtype=pl.String, unique=True),
            column("feature_value", dtype=pl.Int64),
            column("timestamp", dtype=pl.Datetime),
        ],
        min_size=1,
        max_size=100,
    )
)
def test_feature_metadata_properties(df: pl.DataFrame):
    """Test properties of feature metadata operations"""
    # All sample_uids are unique
    assert df["sample_uid"].n_unique() == df.shape[0]

    # Test with Narwhals
    nw_df = nw.from_native(df)
    result = nw_df.select(nw.col("sample_uid"), nw.col("feature_value") * 2)

    # Properties preserved
    assert result.shape[0] == df.shape[0]
    assert "sample_uid" in result.columns
```

## Roundtrip Testing

```python
from hypothesis import given
from hypothesis import strategies as st
import json


@given(st.dictionaries(st.text(), st.one_of(st.integers(), st.text(), st.booleans())))
def test_json_roundtrip(data):
    """JSON encode/decode should be identity function"""
    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert decoded == data
```

## Stateful Testing

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition
from hypothesis import strategies as st


class CacheStateMachine(RuleBasedStateMachine):
    """Test cache behavior with stateful operations"""

    def __init__(self):
        super().__init__()
        self.cache = {}
        self.max_size = 10

    @rule(key=st.text(min_size=1), value=st.integers())
    def set(self, key, value):
        """Set a cache entry"""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value

    @rule(key=st.text(min_size=1))
    def get(self, key):
        """Get from cache"""
        if key in self.cache:
            assert self.cache[key] is not None

    @precondition(lambda self: len(self.cache) > 0)
    @rule()
    def clear(self):
        """Clear cache"""
        self.cache.clear()

    @invariant()
    def size_limit_not_exceeded(self):
        """Cache never exceeds max size"""
        assert len(self.cache) <= self.max_size


# Run the state machine
TestCache = CacheStateMachine.TestCase
```
