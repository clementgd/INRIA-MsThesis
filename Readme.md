# Master Thesis - Improving Memory Management in the Linux Kernel for NUMA Architectures



## Using the tools

### Correlation analysis

This script will print out the correlation coefficients between the events / metrics defined in the script and the runtime of the specified application. \
It is located at `INRIA-scripts/applications/correlate_runtime.py` \
Example command : `./correlate_runtime.py --run ~/npb/NPB3.4-OMP/bin/cg.C.x`

### Huge page false sharing profiler

This script will print out stats about huge page false sharing and L3 misses fetched from remote dram. \
It is located at `INRIA-scripts/applications/profile_page_sharing.py` \
Example command : `./profile_page_sharing.py --run ~/npb/NPB3.4-OMP/bin/cg.C.x`



## Reproducing the experiments

### Download NAS benchmarks
- Get the source from https://www.nas.nasa.gov/assets/npb/NPB3.4.3.tar.gz
- Unzip the tarball and set the desired benchmarks to be compiled in config/suite.def
- Run `make suite` in `NPB3.4-OMP` directory

### Run the experiments
- Edit the lanch-experiments.py script to run the appropriate test function
- Run it on the desired hardware, installing any missing packages along the way
- Run the analysis jupyter notebook 



## Contact

Email : clem.gd@gmail.com \
Linkedin : https://www.linkedin.com/in/cl%C3%A9ment-gachod-b058891aa/
