{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = {nixpkgs, ...}: {
    formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.alejandra;
    devShells.x86_64-linux = let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
      };
      lib = pkgs.lib;
      python = pkgs.python310;
    in {
      default = pkgs.mkShell {
        buildInputs = [
          pkgs.stdenv.cc.cc.lib
          pkgs.gcc-unwrapped.lib
          pkgs.glibc
          pkgs.graphviz
        ];
        packages = with pkgs; [
          stdenv.cc
          uv
          python
          clickhouse
          graphviz
        ];
        LD_LIBRARY_PATH = lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.gcc-unwrapped.lib
          pkgs.glibc
          pkgs.glib
          pkgs.python310
          pkgs.clickhouse
          pkgs.graphviz
        ];
        # UV_PYTHON = "${python}/bin/python";
        shellHook = ''
          uv python pin 3.10
        '';
      };
    };
  };
}
