#!/bin/bash

case $1 in
  Sine)
    echo Run experiments on Sine dataset.
    # sine
    python script_wspn.py --wspn_type=1 --train_type=1 --data_type=sine --threshold=0.5 --n_min_slice=1000
    python script_wspn.py --wspn_type=2 --train_type=1 --data_type=sine --threshold=0.3 --n_min_slice=1000
    python script_wspn.py --wspn_type=3 --train_type=1 --data_type=sine --threshold=0.3 --n_min_slice=1000
    python script_wspn.py --wspn_type=1 --train_type=2 --data_type=sine --threshold=0.5 --n_min_slice=1000
    python script_wspn.py --wspn_type=2 --train_type=2 --data_type=sine --threshold=0.3 --n_min_slice=1000
    python script_wspn.py --wspn_type=3 --train_type=2 --data_type=sine --threshold=0.3 --n_min_slice=1000
    ;;
  MNIST)
    echo Run experiments on MNIST dataset.

    # mnist
    python script_wspn.py --wspn_type=1 --train_type=1 --data_type=2 --threshold=0.7 --n_min_slice=100
    python script_wspn.py --wspn_type=1 --train_type=2 --data_type=2 --threshold=0.7 --n_min_slice=100
    python script_wspn.py --wspn_type=2 --train_type=1 --data_type=2 --threshold=0.7 --n_min_slice=100
    python script_wspn.py --wspn_type=2 --train_type=2 --data_type=2 --threshold=0.7 --n_min_slice=100
    python script_wspn.py --wspn_type=3 --train_type=1 --data_type=2 --threshold=0.7 --n_min_slice=100
    python script_wspn.py --wspn_type=3 --train_type=2 --data_type=2 --threshold=0.7 --n_min_slice=100
    ;;
  SP)
    echo Run experiments on SP dataset.
    # SP
    python script_wspn.py --wspn_type=1 --train_type=1 --data_type=3 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=1 --train_type=2 --data_type=3 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=2 --train_type=1 --data_type=3 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=2 --train_type=2 --data_type=3 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=3 --train_type=1 --data_type=3 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=3 --train_type=2 --data_type=3 --threshold=0.8 --n_min_slice=5
    ;;
  Stock)
    echo Run experiments on Stock dataset.
    # Stock
    python script_wspn.py --wspn_type=1 --train_type=1 --data_type=4 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=1 --train_type=2 --data_type=4 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=2 --train_type=1 --data_type=4 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=2 --train_type=2 --data_type=4 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=3 --train_type=1 --data_type=4 --threshold=0.8 --n_min_slice=5
    python script_wspn.py --wspn_type=3 --train_type=2 --data_type=4 --threshold=0.8 --n_min_slice=5
    ;;
  Billiards)
    echo Run experiments on Billiards dataset.
    # Billiards
    python script_wspn.py --wspn_type=1 --train_type=1 --data_type=5 --threshold=0.7 --n_min_slice=200
    python script_wspn.py --wspn_type=1 --train_type=2 --data_type=5 --threshold=0.7 --n_min_slice=200
    python script_wspn.py --wspn_type=2 --train_type=1 --data_type=5 --threshold=0.7 --n_min_slice=200
    python script_wspn.py --wspn_type=2 --train_type=2 --data_type=5 --threshold=0.7 --n_min_slice=200
    python script_wspn.py --wspn_type=3 --train_type=1 --data_type=5 --threshold=0.7 --n_min_slice=200
    python script_wspn.py --wspn_type=3 --train_type=2 --data_type=5 --threshold=0.7 --n_min_slice=200
    ;;
  *)
    echo Wrong dataset name, must be Sine, MNIST, SP, Stock or Billiards.
    ;;
esac

# VAR
#?python script_wspn.py --wspn_type=2 --train_type=1 --data_type=6 --threshold=0.3 --n_min_slice=1000
#?python script_wspn.py --wspn_type=2 --train_type=2 --data_type=6 --threshold=0.3 --n_min_slice=1000
