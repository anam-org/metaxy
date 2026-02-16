---
title: "Ray integration for Metaxy"
description: "Use Metaxy with Ray for distributed computing workloads."
---

Metaxy has basic integration with [Ray](https://ray.io/) to assist with setting up [Ray Data](https://docs.ray.io/en/latest/data/data.html) jobs.

!!! tip "Ray Environment Setup"

    It's critically important for Metaxy to resolve correct configuration and feature graph on the Ray worker.

    - ensure `METAXY_CONFIG` points to the correct Metaxy config file

    - configure `worker_process_setup_hook` parameter of [RuntimeEnv](https://docs.ray.io/en/latest/ray-core/api/doc/ray.runtime_env.RuntimeEnv.html#ray-runtime-env-runtimeenv)
    to run [`metaxy.init`][metaxy.init] before anything else on the Ray worker

    ??? tip "Per-task setup"
        Additionally, `RAY_USER_SETUP_FUNCTION` can be configured to execute a Python function on every Ray **task** startup


::: metaxy.ext.ray
members: true
