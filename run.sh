# replace /path/to/data/ with the absolute path for the data directory
docker run -it \
    -v `pwd`/data/:/home/risico/data \
    --env RISICO_RUN_DATE=202206230000 \
    --env CELLS_FILE=data/static/ireland.txt \
    --env PVEG_FILE=data/static/pveg.txt \
    --entrypoint /bin/bash \
    risico-hyds 
    
