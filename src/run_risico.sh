#!/bin/bash

echo "Container args: $@"

if [ "$#" -eq 1 ]; then
  RUN_DATE=$1
else
  RUN_DATE="${RISICO_RUN_DATE}"
fi



cd /home/risico/
export PYTHONPATH=$PYTHONPATH:/home/risico/adapter/
echo "Convert IFS files"
python3 adapter/importer.py data/ifs/ input/ input_files.txt $CELLS_FILE

echo "Run RISICO"
./RISICO2015 $RUN_DATE risico/configuration.txt input_files.txt

mkdir -p data/output
RISICO_OUTPUT="data/output/risico_$RUN_DATE.nc"
RISICO_AGGR_OUTPUT="data/output/risico_aggr_$RUN_DATE.nc"
echo "Export files: $RISICO_OUTPUT $RISICO_AGGR_OUTPUT"

python3 adapter/exporter.py risico/OUTPUT/ $RISICO_OUTPUT $RISICO_AGGR_OUTPUT