#!/bin/bash

bash slope_analysis.sh svd 0.1 2 'linear_dec'
bash slope_analysis.sh tt 0.1 2 'linear_dec'
bash slope_analysis.sh tk 0.1 2 'linear_dec'
bash slope_analysis.sh 'cp' 0.1 2 'linear_dec'

bash slope_analysis.sh svd 0.1 2 'flat'
bash slope_analysis.sh tt 0.1 2 'flat'
bash slope_analysis.sh tk 0.1 2 'flat'
bash slope_analysis.sh 'cp' 0.1 2 'flat'
