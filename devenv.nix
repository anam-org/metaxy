{ pkgs, lib, config, inputs, ... }:

let
  # Allow overriding Python version via DEVENV_PYTHON_VERSION env var
  # Default to 3.10, but support 3.10, 3.11, 3.12, 3.13
  pythonVersion = builtins.getEnv "DEVENV_PYTHON_VERSION";
  defaultPythonVersion = if pythonVersion == "" then "3.10" else pythonVersion;
in
{
  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = defaultPythonVersion;
  };

  dotenv.enable = true;

  # https://devenv.sh/packages/
  packages = with pkgs; [
    # Core development tools
    stdenv.cc
    git
    uv

    # Databases and query engines
    duckdb
    clickhouse
    postgresql

    # Visualization
    graphviz

    # Node.js for documentation tooling
    nodejs_22

    # Image processing libraries for mkdocs-material social cards
    cairo
    pango
    gdk-pixbuf
    gobject-introspection
    freetype
    libffi
    libjpeg
    libpng
    zlib
    harfbuzz
    pngquant
  ];

  # Environment variables
  env = {
    # UV Python preference
    UV_PYTHON_PREFERENCE = "only-system";

    # PostgreSQL binary path for pytest-postgresql
    # This ensures fixtures use Nix-provided PostgreSQL consistently
    PG_BIN = "${pkgs.postgresql}/bin";

    # Platform-specific library paths for DuckDB/ClickHouse and mkdocs image generation
    # NOTE: We include stdenv.cc.cc.lib for libstdc++ (needed by DuckDB Python bindings)
    # but avoid glibc to prevent "stack smashing detected" errors when spawning subprocesses
    LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib  # libstdc++.so.6
      pkgs.duckdb.lib
      pkgs.clickhouse
      pkgs.cairo
      pkgs.pango
      pkgs.gdk-pixbuf
      pkgs.harfbuzz
    ]);

    DYLD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isDarwin (lib.makeLibraryPath [
      pkgs.cairo
      pkgs.pango
      pkgs.gdk-pixbuf
      pkgs.gobject-introspection
      pkgs.freetype
      pkgs.libffi
      pkgs.libjpeg
      pkgs.libpng
      pkgs.zlib
      pkgs.harfbuzz
    ]);
  };

  # https://devenv.sh/basics/
  enterShell = ''
    echo "🚀 Metaxy development environment"
    echo "Python: $(python --version)"
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    # Verify critical tools are available and work correctly
    echo "Testing Python..."
    python --version

    echo "Testing DuckDB..."
    duckdb --version

    echo "Testing PostgreSQL binaries (critical for pytest-postgresql)..."
    # Test that pg_ctl and initdb work without "stack smashing" errors
    ${pkgs.postgresql}/bin/pg_ctl --version || { echo "ERROR: pg_ctl failed"; exit 1; }
    ${pkgs.postgresql}/bin/initdb --version || { echo "ERROR: initdb failed"; exit 1; }

    echo "All tools verified successfully!"
  '';
}
