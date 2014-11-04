#!/bin/bash
# Example of a smart way to find the classification score files we'd like. This one finds all file which is newer than one hour.

find ../../data/wavelets -name "classification*.csv" -ctime -1 | xargs python submissions.py
