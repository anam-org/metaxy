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

        # Common packages for all shells
        commonPackages = with pkgs; [
          stdenv.cc
          uv
          duckdb
          git
          clickhouse
          graphviz
          nodejs_22  # so basedpyright runs against it
          (mermaid-ascii.packages.${system}.default or null)
        ];

        # Function to create a dev shell for a specific Python version
        mkPythonShell = python: pkgs.mkShell {
          buildInputs = with pkgs; [
            stdenv.cc.cc.lib
          ] ++ lib.optionals isLinux [
            gcc-unwrapped.lib
            glibc
          ];
          packages = commonPackages ++ [python];
          LD_LIBRARY_PATH = lib.makeLibraryPath (
            [
              pkgs.stdenv.cc.cc.lib
              pkgs.glib
              pkgs.duckdb.lib
              pkgs.clickhouse
              pkgs.graphviz
              python
            ] ++ lib.optionals isLinux [
              pkgs.gcc-unwrapped.lib
              pkgs.glibc
            ]
          );
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