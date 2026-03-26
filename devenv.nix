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

  # Declarative config per integration: packages, library paths, and env vars.
  # Omitted fields default to empty.
  integrations = {
    duckdb = {
      packages = [pkgs.duckdb];
      libPaths = [pkgs.stdenv.cc.cc.lib pkgs.duckdb.lib];
    };
    clickhouse = {
      packages = [pkgs.clickhouse];
      libPaths = [pkgs.clickhouse];
    };
    postgres = {
      packages = [pkgs.postgresql];
      env.PG_BIN = "${pkgs.postgresql}/bin";
    };
    delta = {};
    iceberg = {};
    lancedb = {};
    bigquery = {};
    dagster = {};
    ray = {};
    sqlalchemy = {};
    sqlmodel = {};
    mcp = {};
    docs = {
      packages = with pkgs; [
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
      libPaths = with pkgs; [cairo pango gdk-pixbuf harfbuzz];
    };
  };

  # Normalize: fill in missing fields with defaults
  normalize = cfg: {
    packages = cfg.packages or [];
    libPaths = cfg.libPaths or [];
    env = cfg.env or {};
  };
  configs = builtins.mapAttrs (_: normalize) integrations;

  # Collect all values across integrations
  allPackages = lib.concatLists (lib.mapAttrsToList (_: cfg: cfg.packages) configs);
  allLibPaths = lib.concatLists (lib.mapAttrsToList (_: cfg: cfg.libPaths) configs);
  allEnvs = lib.foldlAttrs (acc: _: cfg: acc // cfg.env) {} configs;

  makeLinuxLibPath = paths:
    lib.optionalString pkgs.stdenv.isLinux (lib.makeLibraryPath paths);

  # CI profiles: core + duckdb (all tests depend on it) + integration-specific deps
  duckdbCfg = configs.duckdb;

  makeCiProfile = name: let
    cfg = configs.${name};
    profilePackages = duckdbCfg.packages ++ cfg.packages;
    profileLibPaths = duckdbCfg.libPaths ++ cfg.libPaths;
    profileEnv = duckdbCfg.env // cfg.env;
    # Force-clear env vars from other integrations that aren't in this profile
    allEnvKeys = builtins.attrNames allEnvs;
    clearedEnvs = lib.genAttrs
      (builtins.filter (k: !(profileEnv ? ${k})) allEnvKeys)
      (_: lib.mkForce "");
  in {
    packages = lib.mkForce (corePackages ++ profilePackages);
    env = {
      LD_LIBRARY_PATH = lib.mkForce (makeLinuxLibPath profileLibPaths);
      DYLD_FALLBACK_LIBRARY_PATH = lib.mkForce "";
    } // (builtins.mapAttrs (_: lib.mkForce) profileEnv) // clearedEnvs;
    enterShell = lib.mkForce "true";
    enterTest = lib.mkForce "true";
  };
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
    ++ allPackages;

  # Environment variables
  env = {
    UV_PYTHON_PREFERENCE = "only-system";
    LD_LIBRARY_PATH = makeLinuxLibPath allLibPaths;

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
  } // allEnvs;

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

  # CI profiles: one per integration + docs
  profiles = lib.listToAttrs (map (name: {
    name = "ci-${name}";
    value.module = makeCiProfile name;
  }) (builtins.attrNames integrations));
}
