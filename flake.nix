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
      pythons = [
        pkgs.python310
        pkgs.python311
        pkgs.python312
        pkgs.python313
        pkgs.python314
      ];
    in {
      default = pkgs.mkShell {
        buildInputs = [
          pkgs.stdenv.cc.cc.lib
          pkgs.gcc-unwrapped.lib
          pkgs.glibc
        ];
        packages = with pkgs;
          [
            stdenv.cc
            uv
            clickhouse
            graphviz
            inputs.mermaid-ascii.outputs.packages.x86_64-linux.default
          ]
          ++ pythons;
        LD_LIBRARY_PATH = lib.makeLibraryPath ([
            pkgs.stdenv.cc.cc.lib
            pkgs.gcc-unwrapped.lib
            pkgs.glibc
            pkgs.glib
            pkgs.clickhouse
            pkgs.graphviz
          ]
          ++ pythons);
      };
    };
  };
}
