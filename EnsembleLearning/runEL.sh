#!/bin/bash

# pip install -U numpy pandas scipy
# cd "$(dirname "$0")"

echo "implement_adaboost.py (problem 2(a))" 
# python implement_adaboost.py
python3 implement_adaboost.py

echo "implement_bagging.py (problem 2(b))" 
# python implement_bagging.py
python3 implement_bagging.py

echo "implement_bagging_bias_variance.py (problem 2(c))" 
# python implement_bagging_bias_variance.py
python3 implement_bagging_bias_variance.py

echo "implement_random_forest.py (problem 2(d))" 
# python implement_random_forest.py
python3 implement_random_forest.py

echo "implement_random_forest_bias_variance.py (problem 2(e))" 
# python implement_random_forest_bias_variance.py
python3 implement_random_forest_bias_variance.py