"""Module for dealing with submissions"""
from __future__ import absolute_import
import sys
import csv
from collections import defaultdict

import numpy as np

from ..datasets import fileutils


def scores_to_submission(score_dicts, **kwargs):
    """
    Returns a list of dictionaries with 'clip' and 'preictal' keys, suitable
    for writing to a submission file.

    :param score_dicts: List of dictionaries containing segment scores.
    :param kwargs: keword arguments to the create_submission_rows function.
    :return: A list of dictionaries, where each inner dictionary represents one of the scores. The inner dictionaries
    has the keys 'clip' and 'preictal', where the 'clip' item gives the clip filename. The list is sorted by clip.
    """
    all_scores = merge_scores(score_dicts)
    canonical_names = fileutils.load_testsegment_names()
    submission = create_submission_rows(all_scores, canonical_names=canonical_names, **kwargs)
    return submission


def create_submission_rows(score_dict,
                           do_normalize=True,
                           old_normalization=False,
                           canonical_names=None,
                           default_score=0.0):
    """
    Produce a list of scores in a format suitable for writing to a submissions file.

    :param score_dict: A dictionary with scores. The keys should be filenames with a prefix which identifies the
    segment file it's based on. The values should be floating point scores for each of the files
    :param do_normalize: If True, the scores of each subject will be normalized to be more evenly distributed over the
    [0,1] range.
    :param old_normalization: If True, the normalization will just subtract the minimum value and scale by the
    (max - min), instead of doing a standardization to mean 0.5 and std of 0.5.
    :param canonical_names: If this is a set of names, they will be used to decide if any segments are missing from the
    scores dict. These segments will then be given the default score.
    :param default_score: Which score to give segments which aren't present in score_dict.
    :return: A list of dictionaries, where each inner dictionary represents one of the scores. The inner dictionaries
    has the keys 'clip' and 'preictal', where the 'clip' item gives the clip filename. The list is sorted by clip.
    """

    clip_scores = dict()
    present_segments = set()
    for name, score in score_dict.items():
        segment_name = fileutils.get_segment_name(name)
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


def old_normalize_scores(clip_scores):
    """
    Normalizes the scores per subject. This version just stretches the scores to the maximum an minimum values.

    :param clip_scores: A dictionary with segment name to score mappings.
    :return: A dictionary with segment name to score mappings, where the scores for each subject has been normalized.
    """
    # subject_scores is a dictionary with subjects as keys and a list of all
    # scores for that subject as values
    subject_scores = collect_subject_scores(clip_scores)
    subject_max = {subject: np.max(scores) for subject, scores in subject_scores.items()}
    subject_min = {subject: np.min(scores) for subject, scores in subject_scores.items()}

    new_clip_scores = dict()
    for segment, score in clip_scores.items():
        subject = fileutils.get_subject(segment)
        new_score = old_normalize_score(score, subject_max[subject], subject_min[subject])
        new_clip_scores[segment] = new_score
    return new_clip_scores


def normalize_scores(clip_scores):
    """
    Normalizes the scores per subject. This version centers the score to 0.5, scales it by 2 standard deviations and
    clips any values less than 0 or greater than 1.

    :param clip_scores: A dictionary with segment name to score mappings.
    :return: A dictionary with segment name to score mappings, where the scores for each subject has been normalized.
    """
    # subject_scores is a dictionary with subjects as keys and a list of all
    # scores for that subject as values
    subject_scores = collect_subject_scores(clip_scores)

    subject_means = {subject: np.mean(scores) for subject, scores in subject_scores.items()}
    subject_stds = {subject: np.std(scores) for subject, scores in subject_scores.items()}

    new_clip_scores = dict()
    for segment, score in clip_scores.items():
        subject = fileutils.get_subject(segment)
        new_score = normalize_score(score, subject_means[subject], subject_stds[subject])
        new_clip_scores[segment] = new_score
    return new_clip_scores


def collect_subject_scores(clip_scores):
    """
    Groups the scores per subject.
    :param clip_scores: A dictionary with segment name to score mappings.
    :return: A dictionary with subject to score mappings. The keys are the subject names and the values are lists of
             scores for the subject in question.
    """
    subject_scores = defaultdict(list)
    for segment, score in clip_scores.items():
        subject = fileutils.get_subject(segment)
        subject_scores[subject].append(score)
    return subject_scores


def normalize_score(score, mean, std):
    """
    Normalizes the score by centering it at 0.5 and scale it by 2 standard deviations. Values below 0 or above 1
    are clipped.
    :param score: The score to standardize.
    :param mean: The mean score for this subject.
    :param std: The standard deviation of scores for the subject.
    :return: The score centered at 0.5 and scaled by 2 standard deviations.
    """
    score -= mean
    if std != 0:
        score /= std*2  # We don't want to clip to much of the distribution
    score += 0.5
    if score < 0:
        score = 0
    elif score > 1:
        score = 1
    return score


def old_normalize_score(score, scores_max, scores_min):
    """
    Normalizes the score by subtracting the minimum score for the subject and scale it by the maximum score.
    :param score: The score to normalize.
    :param scores_max: The maximum score for the subject.
    :param scores_min: The minimum score for the subject.
    :return: The normalized score.
    """
    score -= scores_min
    score /= scores_max - scores_min
    return score


def read_score_file(filename):
    """
    Reads the given score file.
    :param filename: A CSV file with segment scores.
    :return: A dictionary with segment name to score mappings.
    """
    """Returns a dictionary of the scores in the csv file *filename*"""
    with open(filename, 'r') as fp:
        csv_file = csv.reader(fp)
        _ = next(csv_file)
        return {segment_name: float(score) for segment_name, score in csv_file}


def collect_file_scores(filenames):
    """
    Reads multiple score files.

    :param filenames: The files to read scores from.
    :return: A list of dictionaries with segment name to score mappings.
    """
    return [read_score_file(filename) for filename in sorted(filenames)]


def merge_scores(score_dicts):
    """
    Merges the collection of score dictionaries to a single one. The dictionaries will be merged in the order they're
    supplied, so the last dictionary in the collection will be the ones who's scores are kept if multiple dictionaries
    have the same keys

    :param score_dicts: A list of score dictionaries, containing segment name to score mappings.
    :return: A dictionary containing segment name to score mappings.
    """
    scores = dict()
    for score_dict in score_dicts:
        scores.update(score_dict)
    return scores


def write_scores(scores,
                 output=sys.stdout,
                 do_normalize=True,
                 default_score=0):
    """
    Writes the given segment scores to a submission file. If a expected segment is missing, it will be filled with
    a default value.

    :param scores: A list of score dictionaries. Each if the inner dictionaries should have two keys: 'clip' and
                   'preictal'. 'clip' is the name of the segment and 'preictal' is the class probability of that
                    segment being preictal.
    :param output: The file-like object which the scores should be written to.
    :param do_normalize: If True, scores will be normalized per subject.
    :param default_score: The value to use for missing segments in the *scores* list.
    :return: None. The scores are written to the output file-like object.
    """
    submissions = scores_to_submission(scores, do_normalize=do_normalize, default_score=default_score)
    csv_writer = csv.DictWriter(output, fieldnames=['clip', 'preictal'])
    csv_writer.writeheader()
    csv_writer.writerows(submissions)


def submission_from_files(classification_files, **kwargs):
    """
    Creates a submission file based on the given subject score files.
    :param classification_files: Files containing segment scores.
    :param kwargs: keyword arguments which will be sent to *write_scores*
    :return: None.
    """
    scores = collect_file_scores(classification_files)
    write_scores(scores, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="""Script for producing submission files""")

    parser.add_argument("classification_files",
                        help=("The files containing the classification scores. The score files should be"
                              "csv:s with two columns, the first with the segment names and the second with"
                              " the segment scores"),
                        nargs='+')
    parser.add_argument("-o", "--output",
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
            submission_from_files(args.classification_files,
                                  output=fp,
                                  do_normalize=args.do_normalize,
                                  default_score=args.default_score)
    else:
        submission_from_files(args.classification_files,
                              do_normalize=args.do_normalize,
                              default_score=args.default_score)


if __name__ == '__main__':
    main()