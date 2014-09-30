"""Module holding some utilities for doing file related stuff"""

import os.path


def expand_paths(filenames, recursive=True):
    """Goes through the list of *filenames* and expands any directory to the files included in that directory.
    If *recursive* is True, any directories in the base directories will be expanded as well. If *recursive* is
    False, only normal files in the directories will be included.
    The returned list only includes non-directory files."""
    new_files = []
    for file in filenames:
        if os.path.isdir(file):
            if recursive:
                #We recurse over all files contained in the directory and add them to the list of files
                for dirpath, dirnames, subfilenames in os.walk(file):
                    new_files.extend([os.path.join(dirpath, fn) for fn in subfilenames])
            else:
                #No recursion, we just do a listfile on the files of any directoy in filenames
                for subfile in os.listdir(file):
                    if os.path.isfile(subfile):
                        new_files.append(os.path.join(file, subfile))
        elif os.path.isfile(file):
            new_files.append(file)
    return new_files


def load_modules(module_names):
    """Loads the give list of python files as modules and returns the list of module objects."""
    import imp

    modules = []
    for filename in module_names:
        modname = os.path.basename(filename)
        mod = imp.load_source(modname, filename)
        modules.append(mod)
    return modules

def get_preictal_files(dirname):
    """
    Returns all .mat files in the directory which correspond to preictal segments.
    """
    all_files = expand_paths([dirname])
    return list(filter(lambda x: '.mat' in x and 'preictal' in x, all_files))


def get_interictal_files(dirname):
    """
    Returns all .mat files in the directory which correspond to interictal segments.
    """
    all_files = expand_paths([dirname])
    return list(filter(lambda x: '.mat' in x and 'interictal' in x, all_files))


def get_test_files(dirname):
    """
    Returns all .mat files in the directory which correspond to test segments.
    """
    all_files = expand_paths([dirname])
    return list(filter(lambda x: '.mat' in x and 'test' in x, all_files))


def process_segments(files, fun, output_format="{basename}_{fun_name}.txt"):
    """Loads each of the files in the list *files* as segments and applies the function *fun* to each of the segment.
    The function *fun* should return an iteration of rows which will be written to files using the given output_format.
     *basename* will be the name of the files without extension, *fun_name* is the __name__ attribute of the function
     object.
    """
    import segment

    for file in files:
        seg = segment.Segment(file)
        fun_name = fun.__name__
        base_name, ext = os.path.splitext(file)
        output_name = output_format.format(**{'basename' : base_name, 'fun_name' : fun_name })
        with open(output_name, 'w') as fp:
            fp.writelines(fun(seg))
