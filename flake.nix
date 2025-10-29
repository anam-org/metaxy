{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    mermaid-ascii.url = "github:AlexanderGrooff/mermaid-ascii";
  };

  outputs = {nixpkgs, ...} @ inputs: {
    formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.alejandra;
    devShells.x86_64-linux = let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
      };
      lib = pkgs.lib;

      # Common packages for all shells
      commonPackages = with pkgs; [
        stdenv.cc
        uv
        duckdb
        git
        clickhouse
        graphviz
        nodejs_22  # so basedpyright runs against it
        inputs.mermaid-ascii.outputs.packages.x86_64-linux.default
      ];

      # Function to create a dev shell for a specific Python version
      mkPythonShell = python: pkgs.mkShell {
        buildInputs = with pkgs; [
          stdenv.cc.cc.lib
          gcc-unwrapped.lib
          glibc
        ];
        packages = commonPackages ++ [python];
        LD_LIBRARY_PATH = lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib
          gcc-unwrapped.lib
          glibc
          glib
          duckdb.lib
          clickhouse
          graphviz
          python

          # this allows external tools (normal ones like git) to still find their expected libraries
          # Caution: This is a hack and may not work on all systems
          "/usr"
          "/usr/local"
        ]);
      };
    in {
      # Default shell with Python 3.10
      default = mkPythonShell pkgs.python310;

      # Individual shells for each Python version
      "python3.10" = mkPythonShell pkgs.python310;
      "python3.11" = mkPythonShell pkgs.python311;
      "python3.12" = mkPythonShell pkgs.python312;
      "python3.13" = mkPythonShell pkgs.python313;
    };
  };
}
