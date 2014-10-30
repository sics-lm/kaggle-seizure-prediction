#!/usr/bin/env python
"""Module for collection classification scores"""
import csv
import os
import os.path
import fnmatch
import re
import sys
import datetime

import numpy as np
from collections import defaultdict

def collect_scores(score_files, only_newest=True):
    """Returns the score from the list of score files. If *only_newest* is False, the returned score might contain duplicates if there are many scores for the same segment."""
    results = defaultdict(dict)
    for f in score_files:
        ctime=os.path.getctime(f)
        with open(f) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            try:
                for line in csv_reader:
                    filename = line["file"]
                    value= float(line["preictal"])
                    results[filename][ctime]=value
            except KeyError as e:
                print('The file {} is missing a column: {}'.format(f, e))

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


def normalize_scores(all_scores, pattern=r"([A-Za-z]*_[0-9])*", old_normalization=True):
    """
    'Normalizes' the scores for the different subjects, so that their score are in the interval [0,1]. This is done on a per-subject basis.
    The subjects are divided by their prefix, given by the first match group of the regular expression *pattern*.
    """
    subject_scores = defaultdict(list)
    subject_segments = defaultdict(list)

    matcher = re.compile(pattern)

    for segment, score in all_scores.items():
        match = re.match(matcher, segment)
        if match:
            subject = match.group(1)
            subject_scores[subject].append(score)
            subject_segments[subject].append(segment)

    normalized_scores = dict()
    for subject, scores in subject_scores.items():
        if old_normalization:
            max_score = max(scores)
            min_score = min(scores)
            if max_score > 0:
                for segment in subject_segments[subject]:
                    segment_score = all_scores[segment]
                    segment_score -= min_score
                    segment_score /= float(max_score)
                    normalized_scores[segment] = segment_score
        else:
            print("Using new normalization")
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            if score_mean > 0 and score_std > 0:
                for segment in subject_segments[subject]:
                    segment_score = all_scores[segment]
                    segment_score -= score_mean
                    segment_score /= score_std
                    segment_score += 0.5
                    if segment_score < 0:
                        segment_score = 0
                    elif segment_score > 1:
                        segment_score = 1
                    print("Normalized score is: ", segment_score)
                    normalized_scores[segment] = segment_score

    return normalized_scores


def get_scores(root, normalize=False, old_normalization=True):
    """
    Gathers all the scores in subdirectories of the folder *root*
    """
    score_files = find_classification_scores(root)
    collected_scores = collect_scores(score_files)
    fixed_scores = fix_segment_names(collected_scores)

    if normalize:
        fixed_scores = normalize_scores(fixed_scores, old_normalization=old_normalization)


    return fixed_scores


def write_scores(root, output, normalize=False, new_normalization=False):
    scores = get_scores(root, normalize=normalize, old_normalization=not new_normalization)
    csv_writer = csv.writer(output)
    csv_writer.writerow(['clip', 'preictal'])
    for segment_name, score in sorted(scores.items()):
        csv_writer.writerow([segment_name, score])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script for collecting the classification scores. Will only collect the scores from the newest files.")

    parser.add_argument("root", help="The root folder to search for classification files")
    parser.add_argument("-o", "--output",
                        help="The file to write the results to. If the supplied path is a directory, a file with a name on the current time will be generated in the folder.")
    parser.add_argument("-n", "--normalize",
                        help="Normalize the scores on a per-subject basis to the range [0,1]",
                        action='store_true')
    parser.add_argument("--new-normalization",
                        help="Use the new normalization where we standarize the per subject scores by subtracting the mean, dividing by the std, shift +0.5 and clip at 0 and 1.",
                        action='store_true')

    # parser.add_argument("--match", help="A unix shell style expression used to match the files in the subdirectories of root")
    args = parser.parse_args()

    if args.output is None:
        output = sys.stdout
    else:
        if os.path.isdir(args.output):
            timestamp = datetime.datetime.now().replace(microsecond=0)
            submission_file = "submission_{}.csv".format(timestamp)
            output = open(os.path.join(args.output, submission_file), 'w')
        else:
            output = open(args.output, 'w')

    try:
        write_scores(args.root, output, args.normalize, args.new_normalization)

    finally:
        if args.output:
            output.close()
