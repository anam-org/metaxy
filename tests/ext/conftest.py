"""Shared fixtures for integration tests."""

import socket
import uuid
from collections.abc import Generator
from typing import Any

import boto3
import pytest
from metaxy import BaseFeature
from metaxy.config import MetaxyConfig
from moto.server import ThreadedMotoServer
from pytest_cases import fixture, parametrize_with_cases

from tests.metadata_stores.shared.resolve_update import (
    FeatureGraphCases,
    FeaturePlanOutput,
    OptionalDependencyCases,
    RootFeatureCases,
)
from tests.metadata_stores.shared.versioning import FeaturePlanCases, FeaturePlanSequence


def _find_free_port() -> int:
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# ============= S3 FIXTURES =============


@pytest.fixture(scope="session")
def s3_endpoint_url() -> Any:
    """Start a moto S3 server on a random free port."""
    port = _find_free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture(scope="function")
def s3_bucket_and_storage_options(
    s3_endpoint_url: str,
) -> tuple[str, dict[str, Any]]:
    """Creates a unique S3 bucket and provides storage_options."""
    bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
    access_key = "testing"
    secret_key = "testing"
    region = "us-east-1"

    s3_resource: Any = boto3.resource(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3_resource.create_bucket(Bucket=bucket_name)

    storage_options = {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_ENDPOINT_URL": s3_endpoint_url,
        "AWS_REGION": region,
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }

    return (bucket_name, storage_options)


# ============= CONFIG FIXTURES =============


@pytest.fixture
def config_with_truncation(truncation_length: int | None) -> Generator[MetaxyConfig, None, None]:
    """Fixture that sets MetaxyConfig with hash_truncation_length.

    The test must be parametrized on truncation_length for this fixture to work.
    """
    config = MetaxyConfig.get().model_copy(update={"hash_truncation_length": truncation_length})

    with config.use():
        yield config


# ============= PARAMETRIZED CASE FIXTURES =============


@fixture
@parametrize_with_cases("root_feature", cases=RootFeatureCases)
def root_feature(root_feature: type[BaseFeature]) -> type[BaseFeature]:
    return root_feature


@fixture
@parametrize_with_cases("feature_plan_config", cases=FeatureGraphCases)
def feature_plan_config(feature_plan_config: FeaturePlanOutput) -> FeaturePlanOutput:
    return feature_plan_config


@fixture
@parametrize_with_cases("feature_plan_sequence", cases=FeaturePlanCases)
def feature_plan_sequence(feature_plan_sequence: FeaturePlanSequence) -> FeaturePlanSequence:
    return feature_plan_sequence


@fixture
@parametrize_with_cases("optional_dep_config", cases=OptionalDependencyCases)
def optional_dep_config(optional_dep_config: FeaturePlanOutput) -> FeaturePlanOutput:
    return optional_dep_config
