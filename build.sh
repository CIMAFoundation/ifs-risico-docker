#!/bin/bash
mkdir -p dist
tar cvzf dist/adapter.tgz -C src --exclude="adapter/.venv" adapter
tar cvzf dist/risico.tgz -C src risico
cp src/RISICO2015 dist/RISICO2015
cp src/requirements.txt dist/requirements.txt
cp src/run_risico.sh dist/run_risico.sh
cp src/Dockerfile dist/Dockerfile
pushd .
cd dist
docker build -t risico-hyds .
popd