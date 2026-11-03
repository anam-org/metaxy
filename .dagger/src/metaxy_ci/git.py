"""The upstream repository and the changes relative to its `main` branch."""

import asyncio

from dagger_clients.core import Directory, Workspace, core

# Read over HTTPS, so no git credentials are needed, also for forks.
UPSTREAM = "https://github.com/anam-org/metaxy"


async def merge_base(ws: Workspace) -> tuple[str, str]:
    """The merge base of HEAD with `main`, and HEAD. They are the same commit on `main` itself."""
    head = ws.git().head()
    # Not locked: the merge base must follow `main` as it moves.
    main = core().git(UPSTREAM).branch("main", no_lock=True)
    base, head_sha = await asyncio.gather(head.common_ancestor(main).commit(), head.commit())
    return base, head_sha


def tach_args(base: str, head: str) -> list[str]:
    """pytest arguments that skip tests unaffected by the changes since `base`.

    Empty on `main` itself, where every test runs.
    """
    return [] if base == head else ["--tach", f"--tach-base={base}"]


def base_git_dir(base: str) -> Directory:
    """A `.git` holding only the commit `base`, which `git diff <base>` compares the working tree with.

    It does not depend on HEAD, so it changes only when the merge base does.
    """
    return core().git(UPSTREAM).commit(base).tree(discard_git_dir=False, depth=1).directory(".git")
