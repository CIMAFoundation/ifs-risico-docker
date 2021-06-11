# Docker container for the RISICO model

This repo contains a Docker container that allow to run the RISICO model using IFS output.

## Content of the repo

* src: source directory for the containers
  * RISICO2015: RISICO binary
  * adapter: input and ouput python adapters for RISICO
  * risico: static data and configuration for the RISICO model
* build.sh: build script for the container
* run.sh: run script for the RISICO model inside the container

## Build

To build the image, after you clone the repo locally, run the ```build.sh``` shell script

## Run RISICO

* Create a ```data``` folder in the host machine
* Copy the IFS output to the ```data/ifs``` directory
* Optionally copy the static files to the ```data/static``` directory
* Run the model in your host, replacing ```/path/to/data/```with the absolute path of the data directory
* pass the following environmental variable:
    
    *RISICO_RUN_DATE* in YYYYMMDDHHMM format (HH and MM should be 0000).
    
    *CELLS_FILE* optional cells definition file (lon, lat, aspect, slope, veg_type)
    
    *PVEG_FILE* optional vegetation file (see _src/risico/pveg_world.csv_ for reference)
  

Example script:
```bash
  docker run  -it \
    -v /path/to/data/:/home/risico/data \
    --env RISICO_RUN_DATE=201908170000 \
    --env CELLS_FILE=data/static/region.txt \
    --env PVEG_FILE=data/static/pveg.txt \
    risico-ifs
```

The output of the RISICO model will be placed in the data/output directory
