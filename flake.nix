{
  description = "pyecap";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows   = "uv2nix";
      inputs.nixpkgs.follows  = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, uv2nix, pyproject-nix
            , pyproject-build-systems, ... }:
      let
        inherit (nixpkgs) lib;
        pkgs   = nixpkgs.legacyPackages.x86_64-linux;
        python = pkgs.python311;

        # ─── Load the uv workspace (every uv project is a workspace) ─────────
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # ─── Overlay generated from uv.lock ──────────────────────────────────
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";   # prefer binary wheels when available
          # environ = { platform_release = "6.6.0"; };  # customise PEP‑508 env if needed
        };

        # ─── Extra manual fix‑ups go here (rarely needed) ────────────────────
        pyprojectOverrides = final: prev: {
  # ---- Numba: provide libtbb.so.12 ----------------------------------------
          numba = prev.numba.overrideAttrs (old: {
            # Make libtbb.so.12 available at build‑ and run‑time
            buildInputs = (old.buildInputs or []) ++ [ pkgs.tbb_2021_11 ];
          });

          # ---- Asciitree: add setuptools + wheel so the sdist can build -----------
          asciitree = prev.asciitree.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [])
              ++ final.resolveBuildSystem {
                # Both are wheels from nixpkgs/python-packages
                setuptools = [ ];
                wheel      = [ ];
              };
          });
        };

        # ─── Compose final Python package set ────────────────────────────────
        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages { inherit python; })
          .overrideScope (lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
            pyprojectOverrides
          ]);

      in
      {
      # Package a virtual environment as our main application.
      #
      # Enable no optional dependencies for production build.
      packages.x86_64-linux.default = pythonSet.mkVirtualEnv "pyecap-env" workspace.deps.default;

        # #######################################################################
        # ## nix run  →  python -m pyecap  (falls back to plain python REPL)
        # #######################################################################
        # apps.default = {
        #   type = "app";
        #   program =
        #     let drv = self.packages.${system}.default;
        #     in  "${drv}/bin/python";
        # };

        #######################################################################
        ## Development shells
        #######################################################################
        devShells.x86_64-linux = {
          # ── “Impure” mode: keep using uv directly inside an env you manage ──
          impure = pkgs.mkShell {
            packages = [ python pkgs.uv ];
            env = {
              UV_PYTHON_DOWNLOADS  = "never";
              UV_PYTHON            = python.interpreter;  # force uv to nix‑python
            } // lib.optionalAttrs pkgs.stdenv.isLinux {
              LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
            };
          shellHook = ''
            unset PYTHONPATH
          '';
        };

        # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
        # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
        #
        # This means that any changes done to your local files do not require a rebuild.
        #
        # Note: Editable package support is still unstable and subject to change.
        uv2nix =
          let
            # Create an overlay enabling editable mode for all local dependencies.
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              # Use environment variable
              root = "$REPO_ROOT";
              # Optional: Only enable editable for these packages
              # members = [ "pyecap" ];
            };

            # Override previous set with our overrideable overlay.
            editablePythonSet = pythonSet.overrideScope (
              lib.composeManyExtensions [
                editableOverlay

                # Apply fixups for building an editable package of your workspace packages
                (final: prev: {
                  pyecap = prev.pyecap.overrideAttrs (old: {
                    # It's a good idea to filter the sources going into an editable build
                    # so the editable package doesn't have to be rebuilt on every change.
                    src = lib.fileset.toSource {
                      root = old.src;
                      fileset = lib.fileset.unions [
                        (old.src + "/pyproject.toml")
                        (old.src + "/README.md")
                        (old.src + "/src/pyecap/__init__.py")
                      ];
                    };

                    # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                    #
                    # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                    # This behaviour is documented in PEP-660.
                    #
                    # With Nix the dependency needs to be explicitly declared.
                    nativeBuildInputs =
                      old.nativeBuildInputs
                      ++ final.resolveBuildSystem {
                        editables = [ ];
                      };
                  });

                })
              ]
            );

            # Build virtual environment, with local packages being editable.
            #
            # Enable all optional dependencies for development.
            virtualenv = editablePythonSet.mkVirtualEnv "pyecap-dev-env" workspace.deps.all;

          in
          pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];

            env = {
              # Don't create venv using uv
              UV_NO_SYNC = "1";

              # Force uv to use nixpkgs Python interpreter
              UV_PYTHON = python.interpreter;

              # Prevent uv from downloading managed Python's
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              # Undo dependency propagation by nixpkgs.
              unset PYTHONPATH

              # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              export PATH=$PATH:$(git rev-parse --show-toplevel)
              python -m ipykernel install --name=pyecap-dev-env
            '';
          };
      };
    };
}
