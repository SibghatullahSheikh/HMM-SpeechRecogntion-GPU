#!/bin/bash

awk '/\(float\)fullComputationTime:/ {print $2}'  $1 
