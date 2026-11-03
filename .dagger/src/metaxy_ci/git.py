"""The upstream repository and the changes relative to its `main` branch."""

import asyncio

from dagger_clients.core import Workspace, core

# Read over HTTPS, so no git credentials are needed, also for forks.
UPSTREAM = "https://github.com/anam-org/metaxy"


async def tach_args(ws: Workspace) -> list[str]:
    """pytest arguments that skip tests unaffected by the changes since the merge base with `main`.

    Empty on `main` itself, where every test runs.
    """
    head = ws.git().head()
    # Not locked: the merge base must follow `main` as it moves.
    main = core().git(UPSTREAM).branch("main", no_lock=True)
    base, head_sha = await asyncio.gather(head.common_ancestor(main).commit(), head.commit())
    return [] if base == head_sha else ["--tach", f"--tach-base={base}"]
