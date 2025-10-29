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
        git
        clickhouse
        graphviz
        inputs.mermaid-ascii.outputs.packages.x86_64-linux.default
      ];

      # Function to create a dev shell for a specific Python version
      mkPythonShell = python: pkgs.mkShell {
        buildInputs = [
          pkgs.stdenv.cc.cc.lib
          pkgs.gcc-unwrapped.lib
          pkgs.glibc
        ];
        packages = commonPackages ++ [python];
        LD_LIBRARY_PATH = lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.gcc-unwrapped.lib
          pkgs.glibc
          pkgs.glib
          pkgs.clickhouse
          pkgs.graphviz
          python
        ];
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
