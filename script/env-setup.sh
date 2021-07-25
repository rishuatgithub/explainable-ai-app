#!/bin/bash

PROJECTENV="explainable_app"
HOME_DIR=`pwd`
ENV=$HOME_DIR/env

echo "Installing Python Environment in Local Mode"
echo "Project Env. Name: $PROJECTENV"

echo "1. Creating project env."
python3 -m venv $ENV/$PROJECTENV/env

echo "2. Pip upgrade"
source $ENV/$PROJECTENV/env/bin/activate
pip install pip --upgrade

echo "3. Installing Dependencies"
pip3 install -r requirements.txt

echo "4. Activating the env."
source env/$PROJECTENV/env/bin/activate

exit 0
