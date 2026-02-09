---
title: "Ray integration for Metaxy"
description: "Use Metaxy with Ray for distributed computing workloads."
---

Metaxy has basic integration with [Ray](https://ray.io/) to assist with setting up Ray Data jobs.

!!! tip

    It's a very good idea to configure `worker_process_setup_hook` parameter of [RuntimeEnv](https://docs.ray.io/en/latest/ray-core/api/doc/ray.runtime_env.RuntimeEnv.html#ray-runtime-env-runtimeenv)
    to execute [`metaxy.init`][metaxy.init] so that it runs before any other code in the Ray process.

::: metaxy.ext.ray
members: true
