"""Container services for tests, run by the Dagger CLI.

The Dagger CLI provisions an engine when none is running. Inside a Dagger
exec with nested access it joins the parent session instead, so the same
fixtures work on a developer machine and inside `dagger check`.
"""

import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest


def find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def service_image(name: str, config_path: Path) -> str:
    """The image of service `name`.

    `METAXY_TEST_<NAME>_IMAGE` selects it when set, otherwise the `[services]` table in `config_path` (tests.toml).
    """
    return (
        os.environ.get(f"METAXY_TEST_{name.upper()}_IMAGE") or tomllib.loads(config_path.read_text())["services"][name]
    )


@contextmanager
def dagger_service(
    image: str,
    ports: Sequence[int],
    *,
    env: Mapping[str, str] | None = None,
    args: Sequence[str] = (),
    timeout: float = 300,
) -> Generator[dict[int, int]]:
    """Run `image` as a Dagger service and yield a map from each of its `ports` to a forwarded localhost port.

    `args` replaces the image's command; the entrypoint still runs.
    Skips the test when the Dagger CLI is not installed.
    """
    if shutil.which("dagger") is None:
        pytest.skip("Dagger CLI not found in PATH")

    local_ports = {port: find_free_port() for port in ports}
    service_args = f" --use-entrypoint --args {shlex.quote(','.join(args))}" if args else ""
    pipeline = " | ".join(
        [
            f"container | from {image}",
            *(f"with-env-variable {name} {shlex.quote(value)}" for name, value in (env or {}).items()),
            *(f"with-exposed-port {port}" for port in ports),
            f"as-service{service_args}",
            "up --ports " + ",".join(f"{local}:{port}" for port, local in local_ports.items()),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="metaxy-dagger-") as workdir:
        log_path = Path(workdir) / "dagger.log"
        with log_path.open("wb") as log:
            # Run outside the repository so the CLI does not load the Dagger workspace.
            process = subprocess.Popen(
                ["dagger", "--progress=plain", "-c", pipeline], cwd=workdir, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                deadline = time.monotonic() + timeout
                for local_port in local_ports.values():
                    _wait_for_port(process, local_port, deadline, log_path)
                yield local_ports
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _wait_for_port(process: subprocess.Popen[bytes], port: int, deadline: float, log_path: Path) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Dagger service exited with code {process.returncode}:\n{log_path.read_text()[-4000:]}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Dagger service not ready on port {port}:\n{log_path.read_text()[-4000:]}")
