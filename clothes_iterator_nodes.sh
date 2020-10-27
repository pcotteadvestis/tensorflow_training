#!/bin/bash

# Iterate on node number
n_nodes_0=32
n_nodes_f=2048
n_nodes=$n_nodes_0

load_venv tensorflow

while [ $n_nodes -le $n_nodes_f ] ; do
  echo "Using $n_nodes nodes..."
  if [ $n_nodes -eq $n_nodes_0 ] ; then
    python clothes.py --units $n_nodes,10 --kind nodes --rewrite
  else
    python clothes.py --units $n_nodes,10 --kind nodes
  fi
  (( n_nodes = n_nodes + n_nodes ))
done

