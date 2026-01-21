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

        basePackages = python: [
          pkgs.stdenv.cc
          pkgs.uv
          pkgs.duckdb
          pkgs.git
          pkgs.clickhouse
          pkgs.graphviz
          pkgs.nodejs_22
          python
        ] ++ imageLibs ++ lib.optional (mermaid-ascii.packages.${system} ? default) mermaid-ascii.packages.${system}.default;

        libPath = python: lib.makeLibraryPath (runtimeLibs ++ [python]);

        # Shell for NixOS - set LD_LIBRARY_PATH globally (linker and libs both from Nix)
        mkNixOSShell = python: pkgs.mkShell {
          packages = basePackages python;
          LD_LIBRARY_PATH = libPath python;
        };

        # Shell for non-NixOS Linux (e.g., Ubuntu CI) - don't set LD_LIBRARY_PATH globally
        # to avoid linker/library mismatch with non-Nix binaries (ty, GitHub runner tools).
        # Commands that need Nix libs must be wrapped or run with LD_LIBRARY_PATH set explicitly.
        mkNonNixOSShell = python: pkgs.mkShell {
          packages = basePackages python;
          shellHook = ''
            export NIX_LD_LIBRARY_PATH="${libPath python}"

            # Wrapper for commands that need Nix libraries
            nix-run() {
              LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH" "$@"
            }
            export -f nix-run
          '';
        };

        # Shell for macOS - uses DYLD_LIBRARY_PATH instead
        mkDarwinShell = python: pkgs.mkShell {
          packages = basePackages python;
          shellHook = ''
            export DYLD_LIBRARY_PATH="${lib.makeLibraryPath imageLibs}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            export DYLD_FALLBACK_LIBRARY_PATH="${lib.makeLibraryPath imageLibs}''${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
          '';
        };

        mkShellForPython = python:
          if pkgs.stdenv.isDarwin then mkDarwinShell python
          else mkNixOSShell python;

        mkCIShellForPython = python: mkNonNixOSShell python;

      in {
        # Default shells - for local development (NixOS or macOS)
        default = mkShellForPython pkgs.python310;
        "python3.10" = mkShellForPython pkgs.python310;
        "python3.11" = mkShellForPython pkgs.python311;
        "python3.12" = mkShellForPython pkgs.python312;
        "python3.13" = mkShellForPython pkgs.python313;

        # CI shells - for non-NixOS Linux (GitHub Actions)
        ci = mkCIShellForPython pkgs.python310;
        "ci-python3.10" = mkCIShellForPython pkgs.python310;
        "ci-python3.11" = mkCIShellForPython pkgs.python311;
        "ci-python3.12" = mkCIShellForPython pkgs.python312;
        "ci-python3.13" = mkCIShellForPython pkgs.python313;
      });
  };
}
