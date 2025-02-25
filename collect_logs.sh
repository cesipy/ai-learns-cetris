#!/bin/sh 

LOGS_DEST_DIR="res/collected_logs"
CONTAINER_LOGS_DIR="/app/logs"
CONTAINER_PLOTS_DIR="/app/res"

mkdir -p $LOGS_DEST_DIR
# if there are any logs, delete them. 
rm -rf $LOGS_DEST_DIR/*

# Get all containers (including stopped ones)
containers=$(docker ps -aq)      # -a includes stopped containers, -q for quiet mode (IDs only)

for cont in $containers; do
    echo "Processing container: $cont"
    mkdir -p "$LOGS_DEST_DIR/$cont"


    docker cp "$cont:$CONTAINER_LOGS_DIR/." "$LOGS_DEST_DIR/$cont/"
    docker cp "$cont:$CONTAINER_PLOTS_DIR/." "$LOGS_DEST_DIR/$cont/"
    docker cp "$cont:$CONTAINER_PLOTS_DIR/../models" "$LOGS_DEST_DIR/$cont/"

done