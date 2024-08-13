#!/bin/bash

grep "Target MLM Accuracy reached at" slurm-$1.out

echo ""
echo "got `grep "Target MLM Accuracy reached at" slurm-$1.out | wc -l` accuracy"
echo ""

grep "0: {'e2e_time':" slurm-$1.out
echo ""
echo "got `grep "0: {'e2e_time':" slurm-$1.out | wc -l` stops"
echo ""
