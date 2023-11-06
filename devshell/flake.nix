{
  description = "Python Shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      python_package = "python3";
      pypkgs = ps: with ps; [
        filterpy
        ipython
        jupyter
        numpy
        openpyxl
        pandas
        pillow
        plotly
        scikit-learn
        scipy
        seaborn
        # matplotlib.override { enableGtk3 = true; }
        matplotlib.override { enableQt = true; }
      ];
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        py = pkgs.${python_package}.withPackages pypkgs;
      in {
        devShells.default = pkgs.mkShell {

          packages = [ 
            py
            pkgs.black
            pkgs.texlive.combined.scheme-full
            # pkgs.tectonic
            # pkgs.texlab
            pkgs.inkscape
            pkgs.pandoc
          ];

          buildInputs = [
            # pkgs.qt5.qtwayland
          ];

          shellHook = ''
            alias jnb='jupyter notebook'
          '';

          # QT_PLUGIN_PATH = with pkgs.qt5; "${qtbase}/${qtbase.qtPluginPrefix}";
        };
      }
    );
}
