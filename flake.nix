{
  description = "pyecap";

  ##############################################################################
  ## Inputs
  ##############################################################################
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

  ##############################################################################
  ## Outputs
  ##############################################################################
  outputs = { self, nixpkgs, flake-utils, uv2nix, pyproject-nix
            , pyproject-build-systems, ... }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (nixpkgs) lib;
        pkgs   = nixpkgs.legacyPackages.${system};
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

        # ─── Convenience: a virtualenv for dev shells that should be rebuilt
        #      only once per lock‑file change
        devEnv = pythonSet.mkVirtualEnv "pyecap-dev-env" workspace.deps.all;
      in
      {
        #######################################################################
        ## Reproducible package / nix build
        #######################################################################
        packages.default =
          pythonSet.mkVirtualEnv "pyecap-env" workspace.deps.default;

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
        devShells = {
          # ── “Impure” mode: keep using uv directly inside an env you manage ──
          impure = pkgs.mkShell {
            packages = [ python pkgs.uv ];
            env = {
              UV_PYTHON            = python.interpreter;  # force uv to nix‑python
              UV_PYTHON_DOWNLOADS  = "never";
            } // lib.optionalAttrs pkgs.stdenv.isLinux {
              LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
            };
            shellHook = "unset PYTHONPATH";
          };

          # ── Pure, editable mode via uv2nix (PEP‑660) ───────────────────────
          uv2nix =
            let
              editableOverlay = workspace.mkEditablePyprojectOverlay {
                root = "$REPO_ROOT";
              };

              editablePythonSet = pythonSet.overrideScope (
                lib.composeManyExtensions [
                  editableOverlay
                  # example: add `editables` wheel for hatch‑vcs editable builds
                  (final: prev: {
                    pyecap = prev.pyecap.overrideAttrs (old: {
                      nativeBuildInputs =
                        old.nativeBuildInputs
                        ++ final.resolveBuildSystem { editables = [ ]; };
                    });
                  })
                ]);

              editableEnv = editablePythonSet.mkVirtualEnv
                              "pyecap-editable-env" workspace.deps.all;
            in
            pkgs.mkShell {
              packages = [ editableEnv pkgs.uv pkgs.git];
              env = {
                UV_NO_SYNC          = "1";                 # keep uv from venv‑sync
                UV_PYTHON           = "${editableEnv}/bin/python";
                UV_PYTHON_DOWNLOADS = "never";
              };
              shellHook = ''
                unset PYTHONPATH
                export REPO_ROOT=${toString ./.}
              '';
            };
        };
      });
}
