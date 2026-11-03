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
    then "3.11"
    else pythonVersionEnv;
  pythonPackages = {
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
    just
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

  # dagger/nix only packages stable releases; track the 1.0 betas until 1.0.0 lands there.
  dagger = inputs.dagger.packages.${pkgs.stdenv.hostPlatform.system}.dagger.overrideAttrs (_: let
    version = "1.0.0-beta.21";
    platform =
      {
        x86_64-linux = "linux_amd64";
        aarch64-linux = "linux_arm64";
        x86_64-darwin = "darwin_amd64";
        aarch64-darwin = "darwin_arm64";
      }.${
        pkgs.stdenv.hostPlatform.system
      };
  in {
    inherit version;
    src = pkgs.fetchurl {
      url = "https://dl.dagger.io/dagger/releases/${version}/dagger_v${version}_${platform}.tar.gz";
      sha256 =
        {
          linux_amd64 = "c5ac777a5c99089869660943381136e22177b59a46412e310c2fef594760a041";
          linux_arm64 = "ba9d55638636918a3404313511bceba5cc0af22f487c1c1b6a5ba60ce70a61bb";
          darwin_amd64 = "fdbe191c965d240ee9cc90719fa9dca1586805cbb89b0a0087314a84a0937354";
          darwin_arm64 = "781f4862c022a19b9a4c498b59e1d9cfda1ee3cdc8ce210e4545bd8842a7416d";
        }.${
          platform
        };
    };
  });
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
    ++ [pkgs.graphviz pkgs.git-cliff dagger]
    ++ lib.concatMap (name: integrationPackages.${name}) allIntegrations;

  # Environment variables
  env = {
    UV_PYTHON_PREFERENCE = "only-system";
    UV_PREVIEW = "1";
    PG_BIN = "${pkgs.postgresql}/bin";

    # Silence the Material for MkDocs "MkDocs 2.0" upgrade notice.
    NO_MKDOCS_2_WARNING = "1";

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
