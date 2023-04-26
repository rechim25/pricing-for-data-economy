import csv, os, time, json, typer, traceback, typer, logging
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from dataprofiler import Data, Profiler, ProfilerOptions
from dataprofiler.data_readers.csv_data import CSVData
from dataprofiler.data_readers.json_data import JSONData
from dataprofiler.data_readers.parquet_data import ParquetData
from dataprofiler.dp_logging import set_verbosity
from typing import Optional
from functools import partial
from tqdm import tqdm


# GLOBAL VARIABLES
# # File types to be excluded from feature extraction
EXCLUDED_FILE_TYPES = ['.pptx', '.lua', '.patch', '.iml', '.yml', '.ipynb', '.R', '.pdf', '.c', '.js', '.py', '.go']
# # File types that can be read by profiler
PROFILE_COMPATIBLE_FILE_TYPES = ['.csv', '.CSV', '.xls', '.xlsx', '.parquet', '.avro', ]
# # Known file types (not necesserily supported by profiler)
KNOWN_FILE_TYPES = PROFILE_COMPATIBLE_FILE_TYPES + ['.db', '.sql', '.sqlite', '.yaml', '.npy', '.xml'] + EXCLUDED_FILE_TYPES
# # Format of lastUpdate date 
LAST_UPDATED_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
# # Max num rows to load in memory
MAX_NROWS = 300000



# Set dataprofiler logging level
set_verbosity(logging.CRITICAL)
# Profilers options (disable unused features to reduce compute time)
profiler_options = ProfilerOptions()
# # Structured options
profiler_options.structured_options.multiprocess.is_enabled = False
profiler_options.structured_options.chi2_homogeneity.is_enabled = False
# # # Numerical (int, float)
# # # # Integer
profiler_options.structured_options.int.min.is_enabled = False
profiler_options.structured_options.int.max.is_enabled = False
profiler_options.structured_options.int.num_zeros.is_enabled = False
profiler_options.structured_options.int.num_negatives.is_enabled = False
profiler_options.structured_options.int.bias_correction.is_enabled = False
# # # # Float
profiler_options.structured_options.float.min.is_enabled = False
profiler_options.structured_options.float.max.is_enabled = False
profiler_options.structured_options.float.num_zeros.is_enabled = False
profiler_options.structured_options.float.num_negatives.is_enabled = False
profiler_options.structured_options.float.bias_correction.is_enabled = False
# # # Text
profiler_options.structured_options.text.vocab.is_enabled = False
profiler_options.structured_options.text.min.is_enabled = False
profiler_options.structured_options.text.max.is_enabled = False
profiler_options.structured_options.text.mode.is_enabled = False
profiler_options.structured_options.text.median.is_enabled = False
profiler_options.structured_options.text.sum.is_enabled = False
profiler_options.structured_options.text.variance.is_enabled = False
profiler_options.structured_options.text.skewness.is_enabled = False
profiler_options.structured_options.text.kurtosis.is_enabled = False
profiler_options.structured_options.text.histogram_and_quantiles.is_enabled = False
profiler_options.structured_options.text.median_abs_deviation.is_enabled = False
profiler_options.structured_options.text.bias_correction.is_enabled = False


app = typer.Typer()


def get_missing_datasets_ref_list(folders: list[str]) -> list[str]:
    missing_folders = []
    for folder in folders:
        if not os.path.isdir(folder):
            missing_folders.append(folder)
    return missing_folders


def get_folders_with_child_subfolders(folders: list[str]) -> bool:
    folder_with_nonfiles = []
    for folder in folders:
        folder_items = os.listdir(folder)
        for item in folder_items:
            if not os.path.isfile(os.path.join(folder, item)):
                folder_with_nonfiles.append(folder)
    return folder_with_nonfiles


def flatten_2d_list(list_2d: list) -> list:
    return [val for sublist in list(list_2d) for val in sublist]


def get_folder_to_child_files_dict(folders: list[str]) -> dict[str, list[str]]:
    dict = {} # key: folder str name, value: list of child file paths
    for folder in folders:
        file_paths = []
        for path, subdirs, files in os.walk(folder):
            for file_name in files:
                if Path(file_name).suffix.lower() in KNOWN_FILE_TYPES:
                    file_paths.append(os.path.join(path, file_name))
        if len(file_paths) > 0:
            dict[folder] = file_paths
        else:
                print('\nFolder empty after file type filtering: folder_name=%s' % folder)
    return dict


def get_unique_file_types(folder_to_file_names: dict[str, list[str]]) -> list[str]:
    remaining_file_types = set()
    for _, files in folder_to_file_names.items():
        for file_path_str in files:
            type = Path(file_path_str).suffix
            remaining_file_types.add(type)
    return list(remaining_file_types)


def remove_excluded_file_types(folder_to_file_paths: dict[str, list[str]], allowed_file_types: list[str]) -> None:
    # Remove file paths that have excluded type
    for folder, files in folder_to_file_paths.items():
        files_to_remove_from_folder = []
        for file_path_str in files:
            type = Path(file_path_str).suffix
            if type not in allowed_file_types:
                files_to_remove_from_folder.append(file_path_str)
        if len(files_to_remove_from_folder) > 0:
            print("Removing from folder '%s' the following files: %s" % (os.path.basename(folder), files_to_remove_from_folder))
            for file_to_remove in files_to_remove_from_folder:
                folder_to_file_paths[folder].remove(file_to_remove)
    # Remove folders that have only excluded file types
    for folder, files in folder_to_file_paths.items():
        if len(files) == 0:
            print('Removing empty folder %s' % folder)
            folder_to_file_paths.pop(folder)


def get_attributes_dict():
    return {
        # Consistency
        'n_int_features': 0,
        'n_float_features': 0,
        'n_string_features': 0,
        'n_datetime_features': 0,
        'n_unknown_features': 0,
        'n_categorical_features': 0,
        'n_noncateg_features': 0,
        'is_datetime_consistent': 1,
        'n_ordered_features': 0,
        'file_type': '',
        'is_file_structured': 0,
        'is_file_semi_structured': 0,
        'is_file_unstructured': 0,
        'encoding': '',
        'has_duplicated_rows': 0,
        'has_null_rows': 0,
        'has_null_feature_values': 0,
        # Completeness
        'n_features': 0,
        'n_rows': 0,
        'n_rows_times_features': 0,
        'avg_ratio_duplication_noncateg_features': 0,
        'avg_ratio_duplication_categ_features': 0,
        'ratio_features_at_least_one_null': 0,
        'avg_ratio_null_values_features': 0,
        'avg_variance_numerical_features': 0,
        'avg_skewness_numerical_features': 0,
        'avg_kurtosis_numerical_features': 0,
        'avg_median_dev_numerical_features': 0,
        'avg_n_outliers_numerical_features': 0,
        'avg_unalikeability_categ_features': 0,
        # Accuracy
        'avg_margin_error_numerical_features': 0,
        'avg_gini_impurity_categ_features': 0,
        # Timeliness
        'time_diff_current_and_last_update': 0,
        # Privacy
        'num_pii_entities': 0,
    }


def add_consistency_attributes(dict, feature_stats: dict, global_stats: dict):
    STRUCTURE_TO_EXTENSION = {
        'structured': set(['csv', 'parquet', 'xls', 'xlsx', 'hdf5', 'sqlite', 'sql', 'pickle', 'pgn']),
        'semi-structured': set(['xml', 'json', 'geojson', 'html', 'css']),
        # Everything else is unstructured
    }
    # Num of integer features (int)
    if feature_stats['data_type'] == 'int':
        dict['n_int_features'] += 1
    # Num of float features (int)
    elif feature_stats['data_type'] == 'float':
        dict['n_float_features'] += 1
    # Num of string features (int)
    elif feature_stats['data_type'] == 'string' or feature_stats['data_type'] == 'str':
        dict['n_string_features'] += 1
    # Num of datetime features (int)
    elif feature_stats['data_type'] == 'datetime':
        dict['n_datetime_features'] += 1
    else:
        dict['n_unknown_features'] += 1

    # Num of categorical features (int)
    if feature_stats['categorical'] is True:
        dict['n_categorical_features'] += 1
    else:
        dict['n_noncateg_features'] += 1

    # Are datetime types consistent throughout the dataset? (one-hot) TODO: do for numerical types as well
    if 'is_datetime_consistent' == 1 and feature_stats['data_type'] == 'datetime':
        if 'format' in feature_stats['statistics']:
            format_list_str = feature_stats['statistics']['format']
            n_datetime_formats = len(format_list_str.split(','))
            if format_list_str == '[]' or n_datetime_formats > 1:
                dict['is_datetime_consistent'] = 0

    # Num of ordered features? (int)
    if feature_stats['order'] != 'random':
        dict['n_ordered_features'] += 1

    # File format extension (type)? (e.g., csv, xlsx, txt, etc.)
    dict['file_type'] = global_stats['file_type']

    # Structure type? (i.e., structured, semi-structured, unstructured)
    if global_stats['file_type'] in STRUCTURE_TO_EXTENSION['structured']:
        dict['is_file_structured'] = 1
    elif global_stats['file_type'] in STRUCTURE_TO_EXTENSION['semi-structured']:
        dict['is_file_semi_structured'] = 1
    else:
        dict['is_file_unstructured'] = 1

    # Contains duplicated rows?
    dict['has_duplicated_rows'] = 0 if global_stats['duplicate_row_count'] == 0 else 1

    # Contains null rows?
    dict['has_null_rows'] = 0 if global_stats['row_is_null_ratio'] == 0 else 1

    # Contains null feature values?
    dict['has_null_feature_values'] = 0 if global_stats['row_has_null_ratio'] == 0 else 1


def get_num_outliers_for_numerical_feature(profiler_dataframe: pd.DataFrame, feature_stats: dict) -> int:
    # Average number of outliers across numerical features
    q1 = feature_stats['statistics']['quantiles'][0]
    q2 = feature_stats['statistics']['quantiles'][1]
    q3 = feature_stats['statistics']['quantiles'][2]
    iqr = q3 - q1
    lav = q1 - 1.5 * iqr
    uav = q3 + 1.5 * iqr

    data_type = 'int64'
    if feature_stats['data_type'] == 'float':
        data_type = 'float64'
    
    try:
        df_column = profiler_dataframe[feature_stats['column_name']].to_numpy(dtype=data_type, na_value=q2)
    except Exception as e:
        # TODO: exception message to include ref and column name
        tqdm.write(str(e))
        return 0
    return len(np.where((df_column > uav) & (df_column < lav)))


def add_completeness_attributes(dict, feature_stats: dict, global_stats: dict, feature_data: pd.Series) -> None:
    NUMERICAL_TYPES = ['integer', 'float']

    # Num of features
    dict['n_features'] = global_stats['column_count']

    # Num rows
    dict['n_rows'] = global_stats['column_count']

    # Num rows x num features
    dict['n_rows_times_features'] = dict['n_features'] * dict['n_rows']

    if feature_stats['categorical'] is False:
        # Average ratio of duplication across non-categorical features
        dict['avg_ratio_duplication_noncateg_features'] += feature_stats['statistics']['unique_ratio']

    if feature_stats['categorical'] is True:
        # Average ratio of duplication across categorical features
        dict['avg_ratio_duplication_categ_features'] += feature_stats['statistics']['unique_ratio']

    # Ratio of features that contain at least one null value
    dict['ratio_features_at_least_one_null'] = global_stats['row_has_null_ratio']

    # Average ratio of null values across features
    dict['avg_ratio_null_values_features'] += feature_stats['statistics']['null_count'] / dict['n_rows'] 

    if feature_stats['data_type'] in NUMERICAL_TYPES:
        # Average variance across features for int, float (numerical) data types
        dict['avg_variance_numerical_features'] += feature_stats['statistics']['variance']

        # Average skewness across features for int, float (numerical) data types
        dict['avg_skewness_numerical_features'] += feature_stats['statistics']['skewness']

        # Average kurtosis across features for int, float (numerical) data types
        dict['avg_kurtosis_numerical_features'] += feature_stats['statistics']['kurtosis']
        
        # Average median absolute deviation across features for int, float (numerical) data types
        dict['avg_median_dev_numerical_features'] += feature_stats['statistics']['median_abs_deviation']

        # Average number of outliers across numerical features
        dict['avg_n_outliers_numerical_features'] += get_num_outliers_for_numerical_feature(feature_data, feature_stats)

    # Average unalikeability impurity across categorical features
    if feature_stats['categorical'] is True and 'unalikeability' in feature_stats['statistics'] and feature_stats['statistics']['unalikeability'] is not None:
        dict['avg_unalikeability_categ_features'] += feature_stats['statistics']['unalikeability']


def add_accuracy_attributes(dict, feature_stats) -> None:
    NUMERICAL_TYPES = ['integer', 'float']
    # Precision - average margin of error for numerical (int, float) features
    if feature_stats['data_type'] in NUMERICAL_TYPES:
        dict['avg_margin_error_numerical_features'] += feature_stats['statistics']['precision']['margin_of_error']

    # Average Gini impurity across categorical features
    if feature_stats['categorical'] is True and 'gini_impurity' in feature_stats['statistics'] and feature_stats['statistics']['gini_impurity'] is not None:
        dict['avg_gini_impurity_categ_features'] += feature_stats['statistics']['gini_impurity']


def add_timeliness_attributes(dict, last_update_time) -> None:
    # Difference between current time and last update time
    current_time = time.time()
    if last_update_time > current_time:
        tqdm.write('Last update time is a future date: %s' % last_update_time)
        tqdm.write("Setting attribute 'time_diff_current_and_last_update' to 0")
    else:
        dict['time_diff_current_and_last_update'] = current_time - last_update_time


def add_privacy_attributes(dict, feature_stats) -> None:
    PII_LABELS = ['ADDRESS', 'CREDIT_CARD', 'DRIVERS_LICENSE', 'EMAIL_ADDRESS', 'UUID', 'IPV4', 'IPV6', 'MAC_ADDRESS', 'PERSON', 'PHONE_NUMBER', 'SSN']
    predicted_label_ratios = feature_stats['statistics']['data_label_representation']
    if predicted_label_ratios is not None:
        for pii_label in PII_LABELS:
            dict['num_pii_entities'] += predicted_label_ratios[pii_label] * dict['n_rows'] 


def create_profiler_report(data, profiler_options: ProfilerOptions):
    # Set max sample size to 500k
    sample_size = min(len(data.data), MAX_NROWS)
    start = time.time()
    # Calculate Statistics, Entity Recognition, etc
    profile = Profiler(data, samples_per_update=sample_size, options=profiler_options) 
    end = time.time()
    tqdm.write('Profiled data in %.2fs with sample_size=%d' % (end - start, sample_size))

    start = time.time()
    report = profile.report(report_options={"output_format":"serializable"})
    end = time.time()
    tqdm.write('Create output format report in %.2fs' % (end - start))
    return report, sample_size


def create_attributes_from_file_report(data, report: dict):
    attributes_dict = get_attributes_dict()

    global_stats = report['global_stats']
    for feature_stats in report['data_stats']:
        add_consistency_attributes(attributes_dict, feature_stats, global_stats)
        add_completeness_attributes(attributes_dict, feature_stats, global_stats, data.data)
        add_accuracy_attributes(attributes_dict, feature_stats)
        add_privacy_attributes(attributes_dict, feature_stats)
    
    # Average out attributes across features (if needed)
    if attributes_dict['n_noncateg_features'] != 0:
        attributes_dict['avg_ratio_duplication_noncateg_features'] /= attributes_dict['n_noncateg_features']
    if attributes_dict['n_categorical_features'] != 0:
        attributes_dict['avg_ratio_duplication_categ_features'] /= attributes_dict['n_categorical_features']
        attributes_dict['avg_unalikeability_categ_features'] /= attributes_dict['n_categorical_features']
        attributes_dict['avg_gini_impurity_categ_features'] /= attributes_dict['n_categorical_features']
    if attributes_dict['n_features'] != 0:
        attributes_dict['avg_ratio_null_values_features'] /= attributes_dict['n_features']
    n_numerical_features = attributes_dict['n_int_features'] + attributes_dict['n_float_features']
    if n_numerical_features != 0:
        attributes_dict['avg_variance_numerical_features'] /= n_numerical_features
        attributes_dict['avg_skewness_numerical_features'] /= n_numerical_features
        attributes_dict['avg_kurtosis_numerical_features'] /= n_numerical_features
        attributes_dict['avg_median_dev_numerical_features'] /= n_numerical_features
        attributes_dict['avg_n_outliers_numerical_features'] /= n_numerical_features
        attributes_dict['avg_margin_error_numerical_features'] /= n_numerical_features

    return attributes_dict

def add_dict_values(dict1: dict, dict2: dict):
    added_dict = {}
    for key in dict1.keys():
        added_dict[key] = dict1[key] + dict2[key]
    return added_dict

AVERAGED_ATTRIBUTE_KEYS = [
    'is_datetime_consistent',
    'is_file_structured' ,
    'is_file_semi_structured',
    'is_file_unstructured',
    'has_duplicated_rows',
    'has_null_rows',
    'has_null_feature_values',
    'avg_ratio_duplication_noncateg_features',
    'avg_ratio_null_values_features',
    'avg_variance_numerical_features',
    'avg_skewness_numerical_features',
    'avg_kurtosis_numerical_features',
    'avg_median_dev_numerical_features',
    'avg_n_outliers_numerical_features',
    'avg_unalikeability_categ_features',
    'avg_margin_error_numerical_features',
    'avg_gini_impurity_categ_features',
]

def filter_missing_datasets(selected_folders: list[str]):
    # Collect missing datasets
    missing_folders = get_missing_datasets_ref_list(selected_folders)
    print('\nWARNING - %d datasets missing: %s' % (len(missing_folders), missing_folders))

    if len(missing_folders) == len(selected_folders):
        print('\nERROR - All %d folders missing' % len(selected_folders))
        return

    # Remove refs of missing datasets (that got exception during download)
    size_before = len(selected_folders)
    for missing_folder in missing_folders:
        selected_folders.remove(missing_folder)
    print('\nIgnoring %d missing folders' % (size_before - len(selected_folders)))

    return selected_folders

def map_folders_to_files(folders: list[str]) -> dict[str, list[str]]: 
    # For each dataset, collect the name of all files into a dict.
    folder_to_file_paths = get_folder_to_child_files_dict(folders)
    print('\nFound %d folders, %d files total' % (len(folder_to_file_paths), len(flatten_2d_list(folder_to_file_paths.values()))))
    
    # Find existing file types in all folders
    file_types = get_unique_file_types(folder_to_file_paths)

    # Count remaining number of unique file types
    print('\nFound %d known file types: %s' % (len(file_types), str(file_types)))

    return folder_to_file_paths


# TODO: refactor this function
def load_data_from_file(file_path: str) -> Data:
    ENCODINGS = ['utf-8', 'latin1', 'ascii', 'iso-8859-1', 'cp1252']
    file_type = Path(file_path).suffix.lower()
    # Default data loading class, uses universal file type and encoding detectors (not always accurate)
    dataprofiler_loader_class = Data
    # Change data loading class based on file type
    # do not let profiler choose as it makes mistakes
    data = None
    if file_type == '.json':
        dataprofiler_loader_class = JSONData
    elif file_type == '.parquet':
        dataprofiler_loader_class = ParquetData
    elif file_type == '.csv':
        # For csv files try loading data into dataframe first and then pass it to dataprofiler
        for encoding in ENCODINGS:
            try:
                data = pd.read_csv(file_path, encoding=encoding, nrows=MAX_NROWS)
                tqdm.write("READ SUCCESS for csv file with encoding '%s' for file '%s'" % (encoding, file_path))
                return Data(data=data, data_type='csv'), encoding
            except Exception as e:
                tqdm.write(f"READ FAILED for csv file '{file_path}' with encoding '{encoding}'. Got exception: {str(e)}")
        dataprofiler_loader_class = CSVData
    elif file_type in ['.xls', '.xlsx']:
        # For excel files must load into dataframe first before passing to dataprofiler
        for encoding in ENCODINGS:
            try:
                data = pd.read_excel(file_path, nrows=MAX_NROWS)
                tqdm.write("READ SUCCESS for excel file with encoding '%s' for file '%s'" % (encoding, file_path))
                return Data(data=data, data_type='csv'), encoding
            except Exception as e:
                tqdm.write(f"READ FAILED for excel file '{file_path}' with encoding '{encoding}'. Got exception: {str(e)}")
        dataprofiler_loader_class = Data
    # Try multiple encodings with dataprofiler data readers
    for encoding in ENCODINGS:
        try:
            data = dataprofiler_loader_class(file_path, options={'encoding': encoding, 'nrows': MAX_NROWS})
            tqdm.write("READ SUCCESS using dataprofiler loaders for file '%s'" % (file_path))
            return data, encoding
        except Exception as e:
            tqdm.write(f"READ FAILED for file '{file_path}' with encoding '{encoding}'. Got exception: {str(e)}")
    tqdm.write("WARNING: Could not load data for file '%s'" % file_path)
    return None, None


def extract_file_attributes_to_df(folder_to_file_paths: dict[str, list[str]]) -> pl.DataFrame:
    folder_count = 0
    max_files_threshold = 300
    exceptions = []
    n_folders = len(folder_to_file_paths)
    list_of_file_attribute_dicts = []

    start_time_total = time.time()

    # Extract attributes for each datasets (can have multiple files)
    for folder, files in folder_to_file_paths.items():
        folder_count += 1
        start_time_folder = time.time()

        n_files = len(files)
        if n_files == 0:
            print("WARNING folder '%s' has 0 files. Will skip folder...")
            continue
        files_sample = files
        # If num files is large (above threshold) perform random sampling
        if n_files > max_files_threshold:
            print('Num files (%d) greater than threshold. Will perform random sampling across files...' % len(files))
            files_sample = list(np.random.choice(files, size=max_files_threshold, replace=False))
        n_sample_files = len(files_sample)
        print("\nSTART extraction for folder '%s', %d files" % (folder, n_sample_files))
        for file in tqdm(files_sample, desc=f'Processing files (folder %d/%d)' % (folder_count, n_folders)):
            if Path(file).suffix in PROFILE_COMPATIBLE_FILE_TYPES:
                str_header = "\n\nFILE '%s'" % file
                str_underline = '-' * len(str_header)
                tqdm.write(str_header + '\n' + str_underline)
                try:
                    # Auto-Detect & Load: CSV, AVRO, Parquet, JSON, Text
                    start = time.time()
                    data, encoding = load_data_from_file(file)
                    if data is None:
                        tqdm.write(f"WARNING failed to read, skipping file '{file}'")
                        continue
                    end = time.time()
                    tqdm.write('READ file in %.2fs' % (end - start))

                    tqdm.write("PROFILING data '%s'" % file)
                    report, sample_size = create_profiler_report(data, profiler_options)

                    start = time.time()
                    attribute_dict = create_attributes_from_file_report(data, report)

                    # Add sample size to dict
                    attribute_dict['profiler_sample_size'] = sample_size

                    attribute_dict['encoding'] = encoding

                    attribute_dict['num_total_files'] = n_files

                    attribute_dict['num_sample_files'] = n_sample_files
                    
                    # Add dataset folder name to dict
                    file_basename = os.path.basename(folder)
                    attribute_dict['ref_no_slash'] = file_basename

                    # Add file attributes dict to list
                    list_of_file_attribute_dicts.append(attribute_dict)

                    end = time.time()
                    tqdm.write('EXTRACTED ATTRIBUTES in %.2fs for file %s' % (end - start, file_basename))
                except Exception as e:
                    tr = str(traceback.format_exc())
                    exceptions.append((folder_count, os.path.basename(folder), file, str(tr)))
                    tqdm.write("EXCEPTION occured while extracting attributes for file '%s'..." % file)
                    tqdm.write("TRACEBACK: %s" % str(tr))
                    tqdm.write("ERROR MESSAGE: %s" % str(e))
        end_time_folder = time.time()
        print("DONE %d/%d attributes for folder '%s' in %.2fs" % (folder_count, n_folders, folder, end_time_folder - start_time_folder))
    end_time_total = time.time()
    print('FINISHED all files in %.2fs' % (end_time_total - start_time_total))
    
    # Return dataframe from all aggregated attribute dicts
    return pl.from_dicts(list_of_file_attribute_dicts), exceptions
    
@app.command()
def folders_stats(datasets_root_folder: str):
    # Load refs of downloaded datasets from file
    print('\nLoading datasets from file...')
    df_selected = pl.read_csv('datasets_group_100_sampled.csv')
    folders = df_selected['ref_no_slash'].to_numpy()

    # Remove illegal char '/' from refs to get folder names and append root directory of folders
    selected_folders = [os.path.join(datasets_root_folder, folder) for folder in folders]

    # Filter out downloaded datasets (prepare for feature extraction)
    folders_filtered = filter_missing_datasets(selected_folders) #folders_to_exclude=['husnantaj_review-scrapper'])

    # Create map of folder names to file paths
    folder_to_files = map_folders_to_files(folders_filtered)

    return folder_to_files


@app.command(name="extract")
def extract_attributes(
    input_refs_file: str,
    datasets_root_folder: str,
    output_file: str,
    range: str = typer.Option(
        None, 
        "--range",
        "-r",
        help="""Index range of first and last (inclusive) folder to extract attributes from.
                Format '<start>-<end>' where <start> is start index and <end> is end index.
                Default is None, in which case attributes from all 
                folders will be extracted, unless idx argument is set."""
    ),
    idx: int = typer.Option(
        None,
        "--index",
        "-i",
        help="""Index of single folder to extract attributes from.  
                Default is None."""
    )
):  
    if range is not None and idx is not None:
        raise Exception("Cannot specifiy both index and range")
    use_range_argument = False
    use_idx_argument = False
    start_idx, end_idx = None, None
    if range is not None:
        idxs = range.split('-')
        if len(idxs) == 0 or len(idxs) > 2:
            raise Exception("Invalid range argument: format is '<start>-<end>' where <start> is start index and <end> is end index.")
        start_idx = int(idxs[0])
        end_idx = int(idxs[1])
        if start_idx < 0 or end_idx < 0 or start_idx >= end_idx:
            raise Exception("Invalid range argument: start index must be strictly smaller than end index and both must be greater than or equal to 0")
        else:
            use_range_argument = True
    if idx is not None and idx >= 0:
        use_idx_argument = True
    
    # print(tf.__version__)
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Load refs of downloaded datasets from input file
    print("\LOADING datasets from input file '%s' ..." % input_refs_file)
    df_selected = pl.read_csv(input_refs_file)
    folders = df_selected['ref_no_slash'].to_numpy()

    # Remove illegal char '/' from refs to get folder names and append root directory of folders
    selected_folders = [os.path.join(datasets_root_folder, folder) for folder in folders]

    # Filter out downloaded datasets (prepare for feature extraction)
    folders_filtered = filter_missing_datasets(selected_folders) #folders_to_exclude=['husnantaj_review-scrapper'])

    # Create map of folder names to file paths
    folder_to_files = map_folders_to_files(folders_filtered)

    # Select folder based on given index/range arguments
    index_selected_folders = list(folder_to_files.keys())
    if use_range_argument is True:
        index_selected_folders = index_selected_folders[start_idx:end_idx + 1]
    elif use_idx_argument is True:
        index_selected_folders = [index_selected_folders[idx]]

    selected_folder_to_files = {}
    for folder in index_selected_folders:
        selected_folder_to_files[folder] = folder_to_files[folder]
    
    print("\nSTART attribute extraction for all folders to csv output file '%s'" % output_file)
    df_attributes, exceptions = extract_file_attributes_to_df(selected_folder_to_files)

    # Get num of bytes and add as attribute to dict
    # df_dataset_row = df_datasets.filter((pl.col('ref_no_slash') == os.path.basename(folder)))
    # agg_attribute_dict['n_bytes_total'] = df_dataset_row['totalBytesNullable'][0] or 0

    # # Add timeliness attribute to dict
    # last_update_time = time.strptime(df_dataset_row['lastUpdated'][0] , '%Y-%m-%d %H:%M:%S')
    # last_update_time = time.mktime(last_update_time)
    # add_timeliness_attributes(agg_attribute_dict, last_update_time)

    # Write attributes to CSV file
    df_attributes.write_csv(output_file)

    # Write exceptions to file
    n_exceptions = len(exceptions)
    print('\nExceptions occured in %d files' % n_exceptions)
    if n_exceptions > 0:
        exceptions_filename = str(os.path.splitext(output_file)[0]) + '_exceptions.csv'
        print("\nWriting exceptions to file %s" % exceptions_filename)
        df_exceptions = pd.DataFrame(exceptions, columns=['folder_num', 'ref_no_slash', 'file_path', 'exception_str'])
        df_exceptions.to_csv(exceptions_filename)
    print('\nDONE')

if __name__ == '__main__':
    app()