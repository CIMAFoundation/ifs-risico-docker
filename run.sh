# replace /path/to/data/ with the absolute path for the data directory
docker run  -it \
    -v `pwd`/data/:/home/risico/data \
    risico-ifs \
    201807020000
