{
  description = "Cross-platform dev environment (aarch64-darwin, aarch64-linux, x86_64-linux)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    mermaid-ascii.url = "github:AlexanderGrooff/mermaid-ascii";
  };

  outputs = { nixpkgs, mermaid-ascii, ... }:
  let
    systems = [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ];
    forAllSystems = f: nixpkgs.lib.genAttrs systems f;
  in
  {
    formatter = forAllSystems (system:
      (import nixpkgs { inherit system; }).alejandra
    );

    devShells = forAllSystems (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
        isLinux = lib.hasInfix "linux" system;
        postgresql = pkgs.postgresql;
        mermaidAscii = mermaid-ascii.packages.${system}.default or null;

        # Function to create a dev shell for a specific Python version
        mkPythonShell = python: pkgs.mkShell {
          buildInputs = [
            pkgs.stdenv.cc
          ];

          packages = lib.filter (pkg: pkg != null) [
            pkgs.uv
            pkgs.duckdb
            pkgs.git
            pkgs.clickhouse
            pkgs.graphviz
            pkgs.nodejs_24
            postgresql
            mermaidAscii
            python
          ] ++ lib.optionals isLinux [
            pkgs.glibcLocales
          ];

          LD_LIBRARY_PATH = lib.makeLibraryPath [
             # For C++ extensions in pip packages (e.g., duckdb)
            pkgs.stdenv.cc.cc.lib
            pkgs.duckdb.lib
            pkgs.clickhouse
            pkgs.graphviz
            postgresql.lib
          ];

          shellHook = lib.optionalString isLinux ''
            export LOCALE_ARCHIVE=${pkgs.glibcLocales}/lib/locale/locale-archive
          '';
        };
      in {
        default = mkPythonShell pkgs.python310;
        "python3.10" = mkPythonShell pkgs.python310;
        "python3.11" = mkPythonShell pkgs.python311;
        "python3.12" = mkPythonShell pkgs.python312;
        "python3.13" = mkPythonShell pkgs.python313;
        "python3.14" = mkPythonShell pkgs.python314;
      });
  };
}
