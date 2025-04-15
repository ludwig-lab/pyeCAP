{
  description = "Python project built with uv and packaged reproducibly through uv2nix";

  inputs = {
    # Pick a nixpkgs revision / channel you trust
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";

    # Helper utilities (matrix over x86_64‑linux, aarch64‑darwin, …)
    flake-utils.url = "github:numtide/flake-utils";

    # uv‑to‑Nix bridge
    uv2nix.url      = "github:pyproject-nix/uv2nix";
  };

  outputs = { self, nixpkgs, flake-utils, uv2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs   = import nixpkgs { inherit system; };
        python = pkgs.python312;               # pick the interpreter you need
        uvLib  = uv2nix.lib { inherit pkgs python; };
      in
      {
        #######################################################################
        # 1.  Reproducible build of the project (`nix build`, `nix run`)
        #######################################################################
        packages.default = uvLib.mkUvApplication {
          projectDir = self;       # contains pyproject.toml + uv.lock
          # Optional extra build inputs for wheels that need native deps
          # nativeBuildInputs = [ pkgs.openssl pkgs.pkg-config ];
        };

        #######################################################################
        # 2.  Dev shell (`nix develop`) with the exact same uv‑locked deps
        #######################################################################
        devShells.default = pkgs.mkShell {
          # uvLib.mkUvEnvironment gives you a fully populated $VIRTUAL_ENV
          packages = [
            # (uvLib.mkUvEnvironment { projectDir = self; })
            pkgs.uv      # uv CLI itself (handy for `uv pip …` inside shell)
            pkgs.git
          ];
          # convenience
          PYTHONBREAKPOINT = "ipdb.set_trace";
        };

        #######################################################################
        # 3.  nix run  (optional – runs your project’s console‑script entrypoint)
        #######################################################################
        # apps.default = flake-utils.lib.mkApp {
        #   drv     = self.packages.${system}.default;
        #   exePath = "/bin/${self.packages.${system}.default.pname}";
        # };
      });
}

