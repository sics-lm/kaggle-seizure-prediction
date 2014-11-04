"""Module for dealing with submissions"""

import sys
import os.path
import re
import json
import csv
from collections import defaultdict

import numpy as np

CANONICAL_NAMES_FILE = '../../data/test_segment_names.json'


def generate_canonical_names(name_file=CANONICAL_NAMES_FILE):
    import subprocess
    filenames = subprocess.check_output(['find', '../../data/', '-name', '*test*.mat']).split()
    decoded = [os.path.basename(name.decode()) for name in filenames]
    matched_names = [re.match(r"([DP][a-z]*_[1-5]_[a-z]*_segment_[0-9]{4}).*", name) for name in decoded]
    only_matches = [match.group(1) for match in matched_names if match]
    only_matches.sort()
    formatted_names = ["{}.mat".format(name) for name in only_matches]
    with open(name_file, 'w') as fp:
        json.dump(formatted_names, fp, indent=4, separators=(',', ': '))
    return set(formatted_names)


def load_canonical_names(name_file=CANONICAL_NAMES_FILE):
    """Loads the canonical names as a set of names"""
    try:
        with open(name_file, 'r') as fp:
            names = json.load(fp)
            return set(names)
    except FileNotFoundError:
        return generate_canonical_names(name_file)


def get_segment_name(name):
    """Returns the canonical segment name for a string *name*. The canonical
    segment name is the one identifying the original matlab data file and will
    be inferred by the prefix of the basename using a regular expression. If the
    name can't be matched, the argument is returned."""

    pattern = r"([DP][a-z]*_[1-5]_[a-z]*_segment_[0-9]{4}).*"
    basename = os.path.basename(name)  # Remove any directories from the name
    match = re.match(pattern, basename)
    if match is None:
        return name
    else:
        return match.group(1) + '.mat'


def scores_to_submission(score_dicts, canonical_names_file=CANONICAL_NAMES_FILE, **kwargs):
    """
    Returns a list of dictionaries with 'clip' and 'preictal' keys, suitable
    for writing to a submission file.
    """
    all_scores = merge_scores(score_dicts)
    canonical_names = load_canonical_names(canonical_names_file)
    submission = create_submission_rows(all_scores, canonical_names=canonical_names, **kwargs)
    return submission


def create_submission_rows(score_dict,
                           do_normalize=True,
                           old_normalization=True,
                           canonical_names=None,
                           default_score=0.0):
    """
    Produce a list of scores in a format suitable for writing to a submissions file.
    Args:
        score_dict: A dictionary with scores. The keys should be filenames with
                    a prefix which identifies the segment file it's based on.
                    The values should be floating point scores for each of the
                    files
        do_normalize: If True, the scores of each subject will be normalized to
                      be more evenly distributed over the [0,1] range.
        old_normalization: If True, the normalization will just subtract the
                           minimum value and scale by the (max - min), instead
                           of doing a standardization to mean 0.5 and std of 0.5.
        canonical_names: If this is a set of names, they will be used to decide
                         if any segments are missing from the scores dict. These
                         segments will then be given the default score.
        default_score: Which score to give segments which aren't present in
                       score_dict.
    Returns:
        A list of dictionaries, where each inner dictionary represents one of
        the scores. The inner dictionaries has the keys 'clip' and 'preictal',
        where the 'clip' item gives the clip filename. The list is sorted by
        clip.
    """
    clip_scores = dict()
    present_segments = set()
    for name, score in score_dict.items():
        segment_name = get_segment_name(name)
        present_segments.add(segment_name)
        clip_scores[segment_name] = score

    # We normalize before adding missing segment names, so it doesn't affect the
    # distribution of the scores we're actually interested in
    if do_normalize:
        if old_normalization:
            clip_scores = old_normalize_scores(clip_scores)
        else:
            clip_scores = normalize_scores(clip_scores)

    if canonical_names is not None:
        missing_segments = canonical_names - present_segments
        for segment_name in missing_segments:
            clip_scores[segment_name] = default_score

    return [dict(clip=segment_name, preictal=segment_score)
            for segment_name, segment_score
            in sorted(clip_scores.items())]


def get_subject(segment_name):
    """Returns the subject prefix from the segment name"""
    subject_pattern = r"([PD][a-z]*_[1-5])=.*"
    subject = re.match(subject_pattern, segment_name).group(1)
    return subject


def old_normalize_scores(clip_scores):
    """
    Normalizes the scores per subject.
    """
    # subject_scores is a dictionary with subjects as keys and a list of all
    # scores for that subject as values
    subject_scores = collect_subject_scores(clip_scores)
    subject_max = {subject: np.max(scores) for subject, scores in subject_scores.items()}
    subject_min = {subject: np.min(scores) for subject, scores in subject_scores.items()}

    new_clip_scores = dict()
    for segment, score in clip_scores.items():
        subject = get_subject(segment)
        new_score = old_normalize_score(score, subject_max[subject], subject_min[subject])
        new_clip_scores[segment] = new_score
    return new_clip_scores


def normalize_scores(clip_scores):
    """
    Normalizes the scores per subject.
    """
    # subject_scores is a dictionary with subjects as keys and a list of all
    # scores for that subject as values
    subject_scores = collect_subject_scores(clip_scores)

    subject_means = {subject: np.mean(scores) for subject, scores in subject_scores.items()}
    subject_stds = {subject: np.std(scores) for subject, scores in subject_scores.items()}

    new_clip_scores = dict()
    for segment, score in clip_scores.items():
        subject = get_subject(segment)
        new_score = normalize_score(score, subject_means[subject], subject_stds[subject])
        new_clip_scores[segment] = new_score
    return new_clip_scores


def collect_subject_scores(clip_scores):
    """Returns a dictionary with the subjects as keys and a list with the score for the subject as values"""
    subject_scores = defaultdict(list)
    for segment, score in clip_scores.items():
        subject = get_subject(segment)
        subject_scores[subject].append(score)
    return subject_scores


def normalize_score(score, mean, std):
    """Normalizes the score by considering a normal distribution centered on 0.5
       with a std of 0.5"""
    score -= mean
    score /= std*2  # We don't want to clip to much of the distribution
    score += 0.5
    if score < 0:
        score = 0
    elif score > 1:
        score = 1
    return score


def old_normalize_score(score, scores_max, scores_min):
    """Simple normalization which subtracts the minimum score and scales in
       proportion to the maximum score."""
    score -= scores_min
    score /= scores_max - scores_min
    return score


def read_score_file(filename):
    """Returns a dictionary of the scores in the csv file *filename*"""
    with open(filename, 'r') as fp:
        csv_file = csv.reader(fp)
        _ = next(csv_file)
        return {segment_name: float(score) for segment_name, score in csv_file}


def collect_scores(filenames):
    """
    Collects the scores from multiple files into a score file. The files will be
    read in sorted order, so scores with the same segment name will use the
    scores from the last filename in the list. If the filenames are timestamped
    this means that the latest score will be used.
    """
    return [read_score_file(filename) for filename in sorted(filenames)]


def merge_scores(score_dicts):
    """Merges the collection of score dictionaries to a single one. The dictionaries will be merged in the order they're supplied, so the last dictionary in the collection will be the ones who's scores are kept if multiple dictionaries have the same keys"""
    scores = dict()
    for score_dict in score_dicts:
        scores.update(score_dict)
    return scores


def write_scores(classification_files,
                 output=sys.stdout,
                 do_normalize=False,
                 default_score=0):
    """Writes the given classification_files to output in submission format."""
    scores = collect_scores(classification_files)
    submissions = scores_to_submission(scores, do_normalize=do_normalize, default_score=default_score)
    csv_writer = csv.DictWriter(output, fieldnames=['clip', 'preictal'])
    csv_writer.writeheader()
    csv_writer.writerows(submissions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Script for producing submission files""")

    parser.add_argument("classification_files",
                        help="""The files containing the classification scores. The score files should be csv:s with two columns, the first with the segment names and the second with the segment scores""",
                        nargs='+')
    parser.add_argument("--output", "-o",
                        help="The file the submission scores should be written two, the default is stdout",
                        dest='output')
    parser.add_argument("-n", "--normalize",
                        help="Enable normalization per subject of the scores.",
                        action='store_true',
                        dest='do_normalize',
                        default=False)
    parser.add_argument("--default-score",
                        help="The score to use for segments which aren't in the classification files.",
                        type=float,
                        default=0.0)
    args = parser.parse_args()

    if args.output is not None:
        with open(args.output, 'w') as fp:
            write_scores(args.classification_files,
                         output=fp,
                         do_normalize=False,
                         default_score=args.default_score)
    else:
        write_scores(args.classification_files,
                     do_normalize=False,
                     default_score=args.default_score)
