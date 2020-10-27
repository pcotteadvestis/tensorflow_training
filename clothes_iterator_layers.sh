#!/bin/bash

# Iterate on numberof layers
n_layers_0=1
n_layers_f=5
n_layers=$n_layers_0
nodes=512

load_venv tensorflow

while [ $n_layers -le $n_layers_f ] ; do
  echo "Using $n_layers layers..."
  i=1
  units=$nodes
  while [ ${i} -lt $n_layers ] ; do
    units=$units,$nodes
    (( i = i + 1 ))
  done
  units=$units,10
  echo $units

  if [ $n_layers -eq $n_layers_0 ] ; then
    python clothes.py --units $units --kind layers --rewrite
  else
    python clothes.py --units $units --kind layers
  fi
  (( n_layers = n_layers + 1 ))
done

