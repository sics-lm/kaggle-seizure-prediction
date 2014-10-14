#!/usr/bin/env python
"""Small script for finding the channel names"""

import csv
import os
import os.path
import fnmatch
import re
import sys
import random

from collections import defaultdict

def collect_channels(root):
    """Returns a dictionary of channel names to the files these names occurs in"""
    return collect_file_channels(find_segment_files(root))

def collect_file_channels(segment_files):
    """Returns the channel names in the segment files. The channel names are always assumed to be the first line of each csv file."""
    results = defaultdict(set)
    for i, f in enumerate(segment_files):
        amtDone = (i+1.0)/len(segment_files)
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))
        with open(f) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter="\t")
            for line in csv_reader:
                try:
                    channel_i=line['channel_i']

                    channel_j=line['channel_j']
                    results[channel_i].add(f)
                    results[channel_j].add(f)
                except KeyError as e:
                    print("No channel columns in file {}".format(f))

    return results


def find_segment_files(root, 
                       score_pattern="*cross_correlation_5.0s.csv", 
                       do_sample=True, 
                       sample_size=1):
    """
    Finds all files from root matching the given unix shell stype pattern. If *do_sample* is True, only *sample_size* files from each folder will be included.
    """
    matched_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        matches = fnmatch.filter(filenames, score_pattern)
        if do_sample and len(matches) > sample_size:
            matches = random.sample(matches, sample_size)
            
        matched_files.extend([os.path.join(dirpath, f) for f in matches])

    return matched_files



def write_scores(root, output):
    score_files = find_classification_scores(root)
    collected_scores = collect_scores(score_files)
    fixed_scores = fix_segment_names(collected_scores)
    csv_writer = csv.writer(output)
    csv_writer.writerow(['clip', 'preictal'])
    for segment_name, prob in sorted(fixed_scores.items()):
        csv_writer.writerow([segment_name, prob])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script for collecting channel names")

    parser.add_argument("root", help="The root folder to search for classification files")
    parser.add_argument("-o", "--output", 
                        help="The file to write the results to")
    args = parser.parse_args()

    if args.output is None:
        output = sys.stdout
    else:
        output = open(args.output, 'w')
    
    try:
        write_scores(args.root, output)

    finally:
        if args.output:
            output.close()
