#!/bin/sh
trap 'rm -rf fifo_*' EXIT
rm -rf fifo* && \
cd cpp && \
make clean && \
make && \
cd .. && \
python main.py