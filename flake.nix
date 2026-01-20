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

        # Image processing libraries for mkdocs-material social cards
        imageLibs = with pkgs; [
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

        # Runtime libraries for LD_LIBRARY_PATH
        runtimeLibs = [
          pkgs.stdenv.cc.cc.lib
          pkgs.glib
          pkgs.duckdb.lib
          pkgs.clickhouse
          pkgs.graphviz
        ] ++ imageLibs ++ lib.optionals isLinux [
          pkgs.gcc-unwrapped.lib
          pkgs.glibc
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
          ] ++ imageLibs ++ lib.optional (mermaid-ascii.packages.${system} ? default) mermaid-ascii.packages.${system}.default;

          LD_LIBRARY_PATH = lib.makeLibraryPath (runtimeLibs ++ [python]);

          shellHook = lib.optionalString (pkgs.stdenv.isDarwin) ''
            export DYLD_LIBRARY_PATH="${lib.makeLibraryPath imageLibs}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            export DYLD_FALLBACK_LIBRARY_PATH="${lib.makeLibraryPath imageLibs}''${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
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
