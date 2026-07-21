{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          # Hadoop dependencies
          openjdk17
          hadoop

          # Python environment
          python312
          python312Packages.pip

          # Build tools
          gcc
          pkg-config

          # Scientific libraries (keep if your project needs them)
          zlib
          hdf5
          openblas
          lapack

          # Utilities
          git
          wget
          curl
        ];

        shellHook = ''
          # Java setup
          export JAVA_HOME=${pkgs.openjdk17}

          # Hadoop setup
          export HADOOP_HOME=${pkgs.hadoop}
          export HADOOP_CONF_DIR=$PWD/config/hadoop
          export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

          # Native libraries
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.hdf5
            pkgs.openblas
            pkgs.lapack
          ]}:$LD_LIBRARY_PATH

          export HDF5_DIR=${pkgs.hdf5}

          echo "--------------------------------"
          echo "Hadoop development environment"
          echo "Java:"
          java -version
          echo "Hadoop:"
          hadoop version | head -n 3
          echo "--------------------------------"
        '';
      };
    };
}
