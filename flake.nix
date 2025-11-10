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
        postgresqlPgConfig =
          if postgresql ? pkgs && postgresql.pkgs ? pg_config then
            postgresql.pkgs.pg_config
          else
            pkgs.writeShellScriptBin "pg_config" ''
              exec ${postgresql}/bin/pg_config "$@"
            '';
        mermaidAscii = mermaid-ascii.packages.${system}.default or null;

        # Common packages for all shells
        commonPackages =
          lib.filter (pkg: pkg != null) (
            [
              pkgs.stdenv.cc
              pkgs.uv
              pkgs.duckdb
              pkgs.git
              pkgs.clickhouse
              pkgs.graphviz
              pkgs.nodejs_22  # so basedpyright runs against it
              postgresql
              postgresqlPgConfig
              mermaidAscii
            ]
          );

        # Function to create a dev shell for a specific Python version
        mkPythonShell = python: pkgs.mkShell {
          buildInputs = with pkgs; [
            stdenv.cc.cc.lib
          ] ++ lib.optionals isLinux [
            gcc-unwrapped.lib
            glibc
          ];
          packages =
            commonPackages
            ++ lib.optionals isLinux [ pkgs.glibc pkgs.glibcLocales ]
            ++ [python];


          LD_LIBRARY_PATH = lib.makeLibraryPath (
            [
              pkgs.stdenv.cc.cc.lib
              pkgs.glib
              pkgs.duckdb.lib
              pkgs.clickhouse
              pkgs.graphviz
              postgresql.lib
              python
            ] ++ lib.optionals isLinux [
              pkgs.gcc-unwrapped.lib
              pkgs.glibc
            ]
          );
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
      });
  };
}
