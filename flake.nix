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
      let
        pkgs = import nixpkgs { inherit system; };
      in
        pkgs.alejandra
    );
    devShells = forAllSystems (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
        isLinux = lib.hasInfix "linux" system;
        postgresql = pkgs.postgresql_18;
        mermaidAscii = mermaid-ascii.packages.${system}.default or null;

        # Function to create a dev shell for a specific Python version
        mkPythonShell = python: pkgs.mkShell {
          # buildInputs is for compiling code (e.g. pip C extensions) inside the shell.
          # We only need the C compiler; Nix handles the rest.
          buildInputs = [
            pkgs.stdenv.cc
          ];

          # packages are the tools and libraries available in the shell's PATH.
          packages = (lib.filter (pkg: pkg != null) [
            pkgs.uv
            pkgs.duckdb
            pkgs.git
            pkgs.clickhouse
            pkgs.graphviz
            pkgs.nodejs_24
            postgresql # This provides pg_ctl, initdb, etc. on the PATH
            mermaidAscii
            python
          ]) ++ lib.optionals isLinux [
            # glibcLocales is needed for the LOCALE_ARCHIVE env var on Linux
            pkgs.glibcLocales
          ];

          # LD_LIBRARY_PATH is ONLY for non-Nix-aware programs (like pip-installed packages)
          # to find their required shared libraries (.so files).
          LD_LIBRARY_PATH = lib.makeLibraryPath [
            pkgs.duckdb.lib
            pkgs.clickhouse
            pkgs.graphviz
            postgresql.lib
          ];

          # Set locale on Linux to prevent errors from tools like PostgreSQL.
          shellHook = lib.optionalString isLinux ''
            export LOCALE_ARCHIVE=${pkgs.glibcLocales}/lib/locale/locale-archive
          '';
        };
      in {
        # Default shell with Python 3.10
        default = mkPythonShell pkgs.python310;
        # Individual shells for each Python version
        "python3.10" = mkPythonShell pkgs.python310;
        "python3.11" = mkPythonShell pkgs.python311;
        "python3.12" = mkPythonShell pkgs.python312;
        "python3.13" = mkPythonShell pkgs.python313;
        "python3.14" = mkPythonShell pkgs.python314;
      });
  };
}
