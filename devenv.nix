{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  # Map DEVENV_PYTHON_VERSION to nixpkgs packages (all pre-built)
  pythonVersionEnv = builtins.getEnv "DEVENV_PYTHON_VERSION";
  pythonVersion =
    if pythonVersionEnv == ""
    then "3.10"
    else pythonVersionEnv;
  pythonPackages = {
    "3.10" = pkgs.python310;
    "3.11" = pkgs.python311;
    "3.12" = pkgs.python312;
    "3.13" = pkgs.python313;
    "3.14" = pkgs.python314;
  };
  selectedPython = pythonPackages.${pythonVersion} or (throw "Unsupported Python version: ${pythonVersion}");

  # Minimal packages needed by all shells
  corePackages = with pkgs; [
    stdenv.cc
    git
    uv
  ];

  # System packages per integration (only those needing native deps)
  integrationPackages = with pkgs; {
    duckdb = [duckdb];
    clickhouse = [clickhouse];
    postgres = [postgresql];
    docs = [
      nodejs_22
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
  };

  # LD_LIBRARY_PATH entries per integration
  integrationLibPaths = with pkgs; {
    duckdb = [stdenv.cc.cc.lib duckdb.lib];
    clickhouse = [clickhouse];
    docs = [cairo pango gdk-pixbuf harfbuzz];
  };

  # Build LD_LIBRARY_PATH from a list of integration names
  makeLinuxLibPath = integrations:
    lib.optionalString pkgs.stdenv.isLinux (lib.makeLibraryPath (
      lib.concatMap (name: integrationLibPaths.${name} or []) integrations
    ));

  # All integration names
  allIntegrations = builtins.attrNames integrationPackages;
in {
  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = selectedPython;
  };

  dotenv.enable = true;

  # Main shell: all system packages for local development
  packages =
    corePackages
    ++ [pkgs.graphviz pkgs.git-cliff]
    ++ lib.concatMap (name: integrationPackages.${name}) allIntegrations;

  # Environment variables
  env = {
    UV_PYTHON_PREFERENCE = "only-system";
    PG_BIN = "${pkgs.postgresql}/bin";

    LD_LIBRARY_PATH = makeLinuxLibPath allIntegrations;

    DYLD_FALLBACK_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isDarwin (lib.makeLibraryPath [
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

  enterShell = ''
    echo "🚀 Metaxy development environment"
    echo "Python: $(python --version)"
  '';

  enterTest = ''
    echo "Testing Python..."
    python --version

    echo "Testing DuckDB..."
    duckdb --version

    echo "Testing PostgreSQL binaries..."
    ${pkgs.postgresql}/bin/pg_ctl --version || { echo "ERROR: pg_ctl failed"; exit 1; }
    ${pkgs.postgresql}/bin/initdb --version || { echo "ERROR: initdb failed"; exit 1; }

    echo "All tools verified successfully!"
  '';
}
