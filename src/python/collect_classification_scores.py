#!/usr/bin/env python
"""Module for collection classification scores"""
import csv
import os
import os.path
import fnmatch
import re
import sys

from collections import defaultdict

def collect_scores(score_files, only_newest=True):
    """Returns the score from the list of score files. If *only_newest* is False, the returned score might contain duplicates if there are many scores for the same segment."""
    results = defaultdict(dict)
    for f in score_files:
        ctime=os.path.getctime(f)
        with open(f) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for line in csv_reader:
                filename = line["file"]
                value= float(line["preictal"])
                results[filename][ctime]=value
    
    if only_newest:
        newest_results = dict()
        for filename, time_values in results.items():
            time, value = max(time_values.items())
            newest_results[filename] = value
        return newest_results
    else:
        return { filename: list(time_values.values()) 
                 for filename, time_values in sorted(results.items()) }


def find_classification_scores(root, score_pattern="*classification*.csv"):
    """
    Returns all classification score files in subfolders of *root*.
    *score_pattern* is a string which is used to match the score filenames using unix shell syntax.
    """
    matched_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        matches = fnmatch.filter(filenames, score_pattern)
        matched_files.extend([os.path.join(dirpath, f) for f in matches])
    return matched_files
            

def fix_segment_names(scores, 
                      segment_match=r"(.*)_cross_correlation.*\.csv",
                      suffix=".mat"):
    """
    Replaces the names in the scores dictionary with the segment names the scores are based on. *segment_match* should be a regular expression used to match the part of the key which corresponds to the segment name. *suffix* will be appended to the final segment name.
    """
    prog = re.compile(segment_match)
    new_names = dict()
    for key, value in scores.items():
        match = re.match(prog, key)
        if match is not None:
            key = match.group(1) + suffix
        new_names[key] = value
    return new_names


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
    parser = argparse.ArgumentParser(description="Script for collecting the classification scores. Will only collect the scores from the newest files.")

    parser.add_argument("root", help="The root folder to search for classification files")
    parser.add_argument("-o", "--output", 
                        help="The file to write the results to")
    # parser.add_argument("--match", help="A unix shell style expression used to match the files in the subdirectories of root")
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
