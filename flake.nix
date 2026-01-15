{
  description = "Cross-platform dev environment (aarch64-darwin, aarch64-linux, x86_64-linux)";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    mermaid-ascii.url = "github:AlexanderGrooff/mermaid-ascii";
  };
  outputs = { self, nixpkgs, mermaid-ascii, ... }@inputs:
  let
    systems = [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ];
    forAllSystems = f: nixpkgs.lib.genAttrs systems f;
  in
  {
    formatter = forAllSystems (system:
      nixpkgs.legacyPackages.${system}.alejandra
    );
    devShells = forAllSystems (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;
        isLinux = lib.hasInfix "linux" system;

        # Pinning PostgreSQL version ensures consistency
        pgPackage = pkgs.postgresql_18;

        imageLibs = with pkgs; [
          cairo pango gdk-pixbuf gobject-introspection freetype
          libffi libjpeg libpng zlib harfbuzz pngquant
        ];

        # Libraries Python needs at runtime
        runtimeLibs = [
          pkgs.stdenv.cc.cc.lib
          pkgs.glib
          pkgs.duckdb.lib
          pkgs.clickhouse
          pkgs.graphviz
          pgPackage.lib
        ] ++ imageLibs ++ lib.optionals isLinux [
          pkgs.gcc-unwrapped.lib
          pkgs.glibc
          pkgs.openssl
          pkgs.readline
          pkgs.icu
          pkgs.zlib
        ];

        mkPythonShell = python: pkgs.mkShell {
          packages = [
            pkgs.stdenv.cc
            pkgs.uv
            pkgs.duckdb
            pkgs.git
            pkgs.clickhouse
            pkgs.graphviz
            pkgs.nodejs_22
            python
            pgPackage
          ] ++ imageLibs ++ lib.optional (mermaid-ascii.packages.${system} ? default) mermaid-ascii.packages.${system}.default;

          buildInputs = [ pgPackage ];

          # 1. Export LD_LIBRARY_PATH so Python/uv can start and import psycopg
          LD_LIBRARY_PATH = lib.makeLibraryPath (runtimeLibs ++ [python]);

          shellHook = ''
            # Force library path export for the shell
            export LD_LIBRARY_PATH="${lib.makeLibraryPath (runtimeLibs ++ [python])}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Export Postgres variables for pytest and local tools
            export PG_BIN="${pgPackage}/bin"
            export PATH="${pgPackage}/bin:$PATH"

            # Fix CI Locales for initdb
            export LANG="C.UTF-8"
            export LC_ALL="C.UTF-8"
          '' + lib.optionalString (pkgs.stdenv.isDarwin) ''
            export DYLD_LIBRARY_PATH="${lib.makeLibraryPath (runtimeLibs ++ [python])}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            export DYLD_FALLBACK_LIBRARY_PATH="${lib.makeLibraryPath (runtimeLibs ++ [python])}''${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
          '' + lib.optionalString isLinux ''
            export LOCALE_ARCHIVE="${pkgs.glibcLocales}/lib/locale/locale-archive"
          '';
        };
      in {
        default = mkPythonShell pkgs.python310;
        "python3.10" = mkPythonShell pkgs.python310;
        "python3.11" = mkPythonShell pkgs.python311;
        "python3.12" = mkPythonShell pkgs.python312;
        "python3.13" = mkPythonShell pkgs.python313;
      });
  };
}
