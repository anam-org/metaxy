{ pkgs, lib, config, inputs, ... }:

let
  # Map DEVENV_PYTHON_VERSION to nixpkgs packages (all pre-built)
  pythonVersionEnv = builtins.getEnv "DEVENV_PYTHON_VERSION";
  pythonVersion = if pythonVersionEnv == "" then "3.10" else pythonVersionEnv;
  pythonPackages = {
    "3.10" = pkgs.python310;
    "3.11" = pkgs.python311;
    "3.12" = pkgs.python312;
    "3.13" = pkgs.python313;
  };
  selectedPython = pythonPackages.${pythonVersion} or (throw "Unsupported Python version: ${pythonVersion}");
in
{
  # https://devenv.sh/languages/
  # Use nixpkgs' pre-built Python (faster CI, no compilation from source)
  languages.python = {
    enable = true;
    package = selectedPython;
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
