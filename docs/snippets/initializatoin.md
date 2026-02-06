??? tip "Metaxy Initialization"

    Metaxy must be explicitly initialized via:

    <!-- skip next -->
    ```py
    import metaxy as mx

    mx.init()
    ```

    This triggers feature discovery and config discovery.
    `init` is expected to be called at the beginning of the program, in the "entry point" of the application.
