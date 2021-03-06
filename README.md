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
* Copy the IFS output to the ```data/ifs```
* Run the model in your host, replacing ```/path/to/data/```with the absolute path of the data directory and 
```RUNDATE``` in YYYYMMDDHHMM format (HH and MM should be 0000).
```bash
  docker run  -it \
    -v /path/to/data/:/home/risico/data \
    risico-ifs \
    $RUNDATE
```
* the output of the RISICO model will be placed in the data/output directory
