"""Train the model.

This script builds a gender classification model on the dataset of the Author Profiling task at the PAN 2018
shared task. A linear Support Vector classifier is trained on text features.

The *main_development()* function is run for the development phase and the *main_tira_evaluation()* function for the
**TIRA** evaluation phase.
"""

import argparse
import base64
from datetime import datetime
import hashlib
import logging
import os
import pickle
import re
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from usermodeling import process_data_files
from usermodeling import utils

# Change the level of the loggers of some of the imported modules
logging.getLogger("matplotlib").setLevel(logging.INFO)


def load_datasets_development(preset_key):
    """Load the PAN dataset for the development phase.

    This function loads the PAN training dataset and truth by calling the *ProcessDataFiles* module,
    then splits the dataset into training and test sets.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'dataset_name': 'PAN 2018 English',
                                            'xmls_directory': 'data/PAN 2018, Author Profiling/en/text',
                                            'truth_path': 'data/PAN 2018, Author Profiling/en/en.txt',
                                            'txts_destination_directory': 'data/PAN 2018, Author Profiling/TXT Files/en',
                                            },
                          'PAN18_Spanish': {'dataset_name': 'PAN 2018 Spanish',
                                            'xmls_directory': 'data/PAN 2018, Author Profiling/es/text',
                                            'truth_path': 'data/PAN 2018, Author Profiling/es/es.txt',
                                            'txts_destination_directory': 'data/PAN 2018, Author Profiling/TXT Files/es',
                                            },
                          'PAN18_Arabic': {'dataset_name': 'PAN 2018 Arabic',
                                            'xmls_directory': 'data/PAN 2018, Author Profiling/ar/text',
                                            'truth_path': 'data/PAN 2018, Author Profiling/ar/ar.txt',
                                            'txts_destination_directory': 'data/PAN 2018, Author Profiling/TXT Files/ar',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]

    # Load the PAN 2018 training dataset and the truth from the files into lists
    logger.info("Loading the %s training dataset and the truth...", PRESET['dataset_name'])
    merged_tweets_of_authors, truths, author_ids, original_tweet_lengths =\
        process_data_files.load_pan_data(PRESET['xmls_directory'], PRESET['truth_path'],
                                         False, PRESET['txts_destination_directory'])

    # Split the dataset into balanced (stratified) training and test sets:
    docs_train, docs_test, y_train, y_test, author_ids_train, author_ids_test,\
    original_tweet_lengths_train, original_tweet_lengths_test =\
        train_test_split(merged_tweets_of_authors, truths, author_ids, original_tweet_lengths,
                         test_size=0.4, random_state=42, stratify=truths)
    # ↳ *stratify=truths* selects a balanced sample from the data, with the same class proportion as the *truths* list.

    # • Sort all lists in the ascending order of *author_ids* (separately, for the training and test set)
    # This is only done for the sakes of consistency between the *load_datasets_development()* and
    # *load_datasets_tira_evaluation()* functions, because the output of the latter is sorted by *author_ids*, while the
    # former is shuffled by the *train_test_split()* function.
    # Sort the training set
    author_ids_train, docs_train, y_train, original_tweet_lengths_train = [list(tuple) for tuple in zip(*sorted(zip(
        author_ids_train, docs_train, y_train, original_tweet_lengths_train)))]
    # Sort the test set
    author_ids_test, docs_test, y_test, original_tweet_lengths_test = [list(tuple) for tuple in zip(*sorted(zip(
        author_ids_test, docs_test, y_test, original_tweet_lengths_test)))]

    # # TEMP: Used for producing a mimic of the **TIRA** environment
    # ProcessDataFiles.split_train_and_test_files(author_ids_train, author_ids_test, y_train, y_test, preset_key)

    return docs_train, docs_test, y_train, y_test


def load_datasets_tira_evaluation(test_dataset_main_directory, preset_key):
    """Load the PAN dataset for **Tira** evaluation.

    This function loads the PAN training and test dataset and truth by calling the *ProcessDataFiles* module twice,
    then passes them along with Author IDs of the test dataset.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'dataset_name': 'PAN 2018 English',
                                            'xmls_subdirectory': 'en/text',
                                            'truth_subpath': 'en/truth.txt',
                                            },
                          'PAN18_Spanish': {'dataset_name': 'PAN 2018 Spanish',
                                            'xmls_subdirectory': 'es/text',
                                            'truth_subpath': 'es/truth.txt',
                                            },
                          'PAN18_Arabic': {'dataset_name': 'PAN 2018 Arabic',
                                            'xmls_subdirectory': 'ar/text',
                                            'truth_subpath': 'ar/truth.txt',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]

    # Define the constant and the paths
    TRAINING_DATASET_MAIN_DIRECTORY =\
        "//VBOXSVR/training-datasets/author-profiling/pan18-author-profiling-training-dataset-2018-02-27"

    # # TEMP (TIRA): For local testing on TIRA
    # TRAINING_DATASET_MAIN_DIRECTORY = "E:/author-profiling/pan18-author-profiling-training-dataset-2018-02-27"

    xmls_directory_train = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['xmls_subdirectory'])
    truth_path_train = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['truth_subpath'])
    xmls_directory_test = os.path.join(test_dataset_main_directory, PRESET['xmls_subdirectory'])
    # ↳ Note: truth_path_test will not be provided to the participants.

    # Load the PAN 2018 training dataset and truth from the files into lists
    logger.info("Loading the %s training dataset and truth...", PRESET['dataset_name'])
    docs_train, y_train, author_ids_train, original_tweet_lengths_train = \
        process_data_files.load_pan_data(xmls_directory_train, truth_path_train, False, None)

    # Load the PAN 2018 test dataset from the files into lists
    logger.info("Loading the %s test dataset...", PRESET['dataset_name'])
    docs_test, y_test, author_ids_test, original_tweet_lengths_test = \
        process_data_files.load_pan_data(xmls_directory_test, None, False, None)
    # ↳ Note: truth_path_test will not be provided to the participants. As a result, *truths_test* will be empty.

    return docs_train, docs_test, y_train, author_ids_test


def preprocess_tweet(tweet):
    """Pre-process a tweet.

    The following pre-processing operations are done on the tweet:
    - Replace repeated character sequences of length 3 or greater with sequences of length 3
    - Lowercase
    - Replace all URLs and username mentions with the following tags:
        URL		    <URLURL>
        @Username   <UsernameMention>

    Args:
        tweet: String
    Returns:
        The pre-processed tweet as String

    IMPROVEMENTS TO MAKE:
    - Instead of tokenizing and detokenizing, which is messy, the strings should be directly replaced using regex.
    """

    replaced_urls = []  # Create an empty list
    replaced_mentions = []  # Create an empty list

    # Tokenize using NLTK
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)

    # Iterate over tokens
    for index, token in enumerate(tokens):
        # Replace URLs
        if token[0:8] == "https://":
            replaced_urls.append(token)
            tokens[index] = "<URLURL>"
            # ↳ *tokens[index]* will directly modify *tokens*, whereas any changes to *token* will be lost.

        # Replace mentions (Twitter handles; usernames)
        elif token[0] == "@" and len(token) > 1:
            # ↳ Skip the single '@' tokens
            replaced_mentions.append(token)
            tokens[index] = "<UsernameMention>"

    # Detokenize using NLTK's Treebank Word Detokenizer
    detokenizer = TreebankWordDetokenizer()
    processed_tweet = detokenizer.detokenize(tokens)

    # *replaced_urls* and *replaced_mentions* will contain all of the replaced URLs and Mentions of the input string.
    return processed_tweet


def extract_features(docs_train, docs_test, preset_key):
    """Extract features

    This function builds a transformer (vectorizer) pipeline,
    fits the transformer to the training set (learns vocabulary and idf),
    transforms the training set and the test set to their TF-IDF matrix representation,
    and builds a classifier.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'dataset_name': 'PAN 2018 English',
                                            'word_ngram_range': (1, 3),
                                            'perform_dimentionality_reduction': True,
                                            },
                          'PAN18_Spanish': {'dataset_name': 'PAN 2018 Spanish',
                                            'word_ngram_range': (1, 2),
                                            'perform_dimentionality_reduction': False,
                                            },
                          'PAN18_Arabic': {'dataset_name': 'PAN 2018 Arabic',
                                            'word_ngram_range': (1, 2),
                                            'perform_dimentionality_reduction': False,
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]

    # Build a vectorizer that splits strings into sequences of i to j words
    word_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                      analyzer='word', ngram_range=PRESET['word_ngram_range'],
                                      min_df=2, use_idf=True, sublinear_tf=True)
    # Build a vectorizer that splits strings into sequences of 3 to 5 characters
    char_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                     analyzer='char', ngram_range=(3, 5),
                                     min_df=2, use_idf=True, sublinear_tf=True)
    # %% Trying out count vectorizer
    # vectorizer = CountVectorizer(ngram_range=(1,1), analyzer='word', min_df=1)

    # Build a transformer (vectorizer) pipeline using the previous analyzers
    # *FeatureUnion* concatenates results of multiple transformer objects
    ngrams_vectorizer = Pipeline([('feats', FeatureUnion([('word_ngram', word_vectorizer),
                                                         ('char_ngram', char_vectorizer),
                                                         ])),
                                 # ('clff', LinearSVC(random_state=42))
                                 ])

    # Fit (learn vocabulary and IDF) and transform (transform documents to the TF-IDF matrix) the training set
    X_train_ngrams_tfidf = ngrams_vectorizer.fit_transform(docs_train)
    '''
    ↳ Check the following attributes of each of the transformers (analyzers)—*word_vectorizer* and *char_vectorizer*:
    vocabulary_ : dict. A mapping of terms to feature indices.
    stop_words_ : set. Terms that were ignored
    '''
    logger.info("@ %.2f seconds: Finished fit_transforming the training dataset", time.process_time())
    logger.info("Training set word & character ngrams .shape = %s", X_train_ngrams_tfidf.shape)

    feature_names_ngrams = [word_vectorizer.vocabulary_, char_vectorizer.vocabulary_]

    # # TEMP: For debugging purposes
    # ProcessDataFiles.write_iterable_to_csv(list(feature_names_ngrams[0].items()), "word_vectorizer.vocabulary_",
    #                                     logger.handlers[1].baseFilename)
    # ProcessDataFiles.write_iterable_to_csv(list(feature_names_ngrams[1].items()), "char_vectorizer.vocabulary_",
    #                                     logger.handlers[1].baseFilename)

    '''
    Extract the features of the test set (transform test documents to the TF-IDF matrix)
    Only transform is called on the transformer (vectorizer), because it has already been fit to the training set.
    '''
    X_test_ngrams_tfidf = ngrams_vectorizer.transform(docs_test)
    logger.info("@ %.2f seconds: Finished transforming the test dataset", time.process_time())
    logger.info("Test set word & character ngrams .shape = %s", X_test_ngrams_tfidf.shape)

    # • Dimensionality reduction using truncated SVD (aka LSA)
    if PRESET['perform_dimentionality_reduction']:
        # Build a truncated SVD (LSA) transformer object
        svd = TruncatedSVD(n_components=300, random_state=42)
        # Fit the LSI model and perform dimensionality reduction
        X_train_ngrams_tfidf_reduced = svd.fit_transform(X_train_ngrams_tfidf)
        logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) on the training dataset", time.process_time())
        X_test_ngrams_tfidf_reduced = svd.transform(X_test_ngrams_tfidf)
        logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) on the test dataset", time.process_time())

        X_train = X_train_ngrams_tfidf_reduced
        X_test = X_test_ngrams_tfidf_reduced
    else:
        X_train = X_train_ngrams_tfidf
        X_test = X_test_ngrams_tfidf

    # # Extract features: offensive words
    # X_train_offensive_words_tfidf_reduced, X_test_offensive_words_tfidf_reduced, feature_names_offensive_words =\
    #     extract_features_offensive_words(docs_train, docs_test)
    #
    # # Combine the n-grams with additional features:
    # X_train_combined_features = np.concatenate((X_train_offensive_words_tfidf_reduced,
    #                                            X_train_ngrams_tfidf_reduced
    #                                            ), axis=1)
    # X_test_combined_features = np.concatenate((X_test_offensive_words_tfidf_reduced,
    #                                           X_test_ngrams_tfidf_reduced
    #                                           ), axis=1)
    # feature_names_combined_features = np.concatenate((feature_names_offensive_words,
    #                                                 feature_names_ngrams
    #                                                 ), axis=0)

    '''
    Build a classifier: Linear Support Vector classification
    - The underlying C implementation of LinearSVC uses a random number generator to select features when fitting the
    model.
    References:
        http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    '''
    clf = LinearSVC(random_state=42)
    # ↳ *dual=False* selects the algorithm to solve the primal optimization problem, as opposed to dual.
    # Prefer *dual=False* when n_samples > n_features. Source:
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    return X_train, X_test, clf, feature_names_ngrams


def extract_features_offensive_words(docs_train, docs_test):
    """Extract offensive words features

    This function performs the following tasks for the training and test datasets:
        1. Gets the counts of offensive words from the *count_offensive_words()* function.
        2. Concatenates the count arrays for the desired Flame levels into X_train and X_test.
        3. Transforms the count matrix (NumPy array) to a normalized TF or TF-IDF representation.
        4. Performs dimensionality reduction on the normalized matrix using truncated SVD (aka LSA).
    Moreover, the function collects and returns the feature names (offensive expressions) for the desired Flame
    levels in the following format: “expression (flame level)”

    Important constants:
    DESIRED_FLAME_LEVELS: (Tuple: Ints) Desired flame levels. You can select any of the levels: 1, 2, 3, 4, and 5.
    """

    # Count the number of occurrences of all the offensive expressions in the training set and test set
    counts_of_offensive_words_dict_train = count_offensive_words(
        docs_train, "pickles/counts_of_offensive_words_dict_train, <HASH>.pickle")
    counts_of_offensive_words_dict_test = count_offensive_words(
        docs_test, "pickles/counts_of_offensive_words_dict_test, <HASH>.pickle")

    # Load the Flame Dictionary (to produce the list of feature names)
    flame_dictionary, flame_expressions_dict = process_data_files.load_flame_dictionary()
    '''
    ↳
    *flame_dictionary*
        Keys:   (string) Expression
        Values: (int)    Flame level
    *flame_expressions_dict*
        Keys:   (int)           Flame level
        Values: (list: strings) Expressions
    '''

    # Log the min, max, and shape of the offensive words count arrays (just to make sure the pickles were loaded
    # correctly.
    for flame_index in range(1, 6):
        array = counts_of_offensive_words_dict_train[flame_index]
        logger.debug("Flame level %d: min = %d | max = %-3d | shape = %s",
                     flame_index, array.min(), array.max(), array.shape)
    for flame_index in range(1, 6):
        array = counts_of_offensive_words_dict_test[flame_index]
        logger.debug("Flame level %d: min = %d | max = %-3d | shape = %s",
                     flame_index, array.min(), array.max(), array.shape)

    # Create empty lists
    arrays_list_train = []
    arrays_list_test = []
    feature_names_offensive_words = []

    # Concatenate the counts NumPy arrays and the feature names for the desired Flame levels
    DESIRED_FLAME_LEVELS = (1, 2, 3, 4, 5)
    for flame_index in DESIRED_FLAME_LEVELS:
        arrays_list_train.append(counts_of_offensive_words_dict_train[flame_index])
        arrays_list_test.append(counts_of_offensive_words_dict_test[flame_index])
        # Add the expressions to the list of feature names in the form: “expression (flame level)”
        for expression in flame_expressions_dict[flame_index]:
            feature_names_offensive_words.append("{} ({})".format(expression, flame_index))
    X_train_offensive_words_counts = np.concatenate(arrays_list_train, axis=1)
    X_test_offensive_words_counts = np.concatenate(arrays_list_test, axis=1)

    # • Transform the count matrix (NumPy array) to a normalized TF or TF-IDF representation
    # Build a TF-IDF transformer object
    tfidf_transformer = TfidfTransformer(norm='l2', use_idf=False, sublinear_tf=False)
    # ↳ With these parameters, the transformer does not make any changes: norm=None, use_idf=False, sublinear_tf=False
    '''
    ↳ With normalization, each row (= author) is normalized to have a sum of absolute values / squares equal to 1.
    L^1-norm: Sum of the absolute value of the numbers (here, TF or TF-IDF of the offensive expressions)
    L^2-norm: Sum of the square         of the numbers ”...
    More info: http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
    '''
    # Fit and transform
    X_train_offensive_words_tfidf = tfidf_transformer.fit_transform(X_train_offensive_words_counts)
    X_test_offensive_words_tfidf = tfidf_transformer.transform(X_test_offensive_words_counts)

    # • Dimensionality reduction using truncated SVD (aka LSA)
    # Build a truncated SVD (LSA) transformer object
    svd_offensive_words = TruncatedSVD(n_components=10, random_state=42)
    # Fit the LSI model and perform dimensionality reduction
    x_train_offensive_words_tfidf_reduced = svd_offensive_words.fit_transform(X_train_offensive_words_tfidf)
    logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) in *extract_features_offensive_words()* on "
                "the training dataset", time.process_time())
    x_test_offensive_words_tfidf_reduced = svd_offensive_words.transform(X_test_offensive_words_tfidf)
    logger.info("@ %.2f seconds: Finished dimensionality reduction (LSA) in *extract_features_offensive_words()* on "
                "the test dataset", time.process_time())

    return x_train_offensive_words_tfidf_reduced, x_test_offensive_words_tfidf_reduced, feature_names_offensive_words


def count_offensive_words(docs, pickle_path_pattern=None):
    """Count the number of offensive words in the documents.

    This function counts the number of occurrences of all the expressions inside the Flame Dictionary.
    If the pickled results of the function (corresponding to the same input) already exists, the function is
    bypassed. If not, after the function is done, the results are stored as pickles.

    Args:
        - docs:         (list: strings) List of documents. Each row represents an author and contains one string.
        - pickle_path_pattern: (string) The path pattern for the pickle. This needs to include “<HASH>”, which will be
    replaced with the hash of the input of the function. Refer to the docstring of the *generate_pickle_path* function.

    Returns:
        - counts_of_expressions_dict: A Python dictionary
            • Keys:   (int)         Flame level
            • Values: (NumPy array) Counts of occurrences of expressions in that Flame level. Each row
                represents an author, and each column represents an expression in the Flame level of the key.

    Note: List of expressions can be accessed by calling *ProcessDataFiles.load_flame_dictionary*.
    """

    pickle_path = generate_pickle_path(docs, pickle_path_pattern)

    # Bypass: If the pickled results already exist, load (unpickle) and return them and skip the rest of the function
    if (pickle_path is not None) and os.path.isfile(pickle_path):
        with open(pickle_path, 'rb') as pickle_input_file:
            unpickled_object = pickle.load(pickle_input_file)
        logger.info('Function bypassed: The counts of offensive words was loaded from pickle "%s" instead.', pickle_path)
        return unpickled_object

    # Load the Flame Dictionary
    # %% TODO: Prevent loading the dictionary every time...
    flame_dictionary, flame_expressions_dict = process_data_files.load_flame_dictionary()
    '''
    ↳
    *flame_dictionary*
        Keys:   (string) Expression
        Values: (int)    Flame level
    *flame_expressions_dict*
        Keys:   (int)           Flame level
        Values: (list: strings) Expressions
    '''

    # keys_dict_view = flame_dictionary.keys()
    # expressions = list(keys_dict_view)

    # Preprocess the merged tweets of authors
    preprocessed_docs = []  # Create an empty list
    for author_index, doc in enumerate(docs):
        preprocessed_docs.append(preprocess_tweet(doc))

        # Log after preprocessing the merged tweets of every 200 authors
        if author_index % 200 == 0:
            logger.debug("@ %.2f seconds, progress: Preprocessed the tweets of author_index = %d",
                         time.process_time(), author_index)
    logger.info("@ %.2f seconds: Finished preprocessing the tweets in *count_offensive_words()*",
                time.process_time())

    # Create a dictionary of five NumPy arrays full of zeros
    counts_of_expressions_dict = {}  # Create an empty dictionary
    for flame_index in range(1, 6):
        counts_of_expressions_dict[flame_index] = np.zeros((len(preprocessed_docs),
                                                        len(flame_expressions_dict[flame_index])), dtype=int)

    # Compile regex patterns into regex objects for all expressions, and store them in five separate lists, based on
    # Flame level (similar to *flame_expressions_dict*).
    '''
    - Most regex operations are available as module-level functions as well as methods on compiled
    regular expressions. The functions are shortcuts that don’t require you to compile a regex object first,
    but miss some fine-tuning parameters.
    - Compiling a regex pattern and storing the resulting regex object for reuse is more efficient when the
    expression will be used several times in a single program. Even though the most recent patterns passed to
    re.compile() and the module-level matching functions are cached, the size of this cache is limited.
    More info: https://docs.python.org/3/library/re.html#re.compile
    Here, we are dealing with 2,600+ expressions, so the built-in cache cannot help. Storing the regex objects,
    decreased the processing time of each Author from 1.6 seconds to 0.7 seconds (on my machine).
    '''
    '''
    - In Python code, Regular Expressions will often be written using the raw string notation (r"text").
    Without it, every backslash in a regular expression would have to be prefixed with another one to escape it.
    - The shorthand \b matches a word boundary, without consuming any characters. Word boundary characters
    include space, . ! " ' - * and much more.
    - Some examples of matches of the /\bWORD\b/ pattern: WORD's, prefix-WORD, WORD-suffix, "WORD".
    %% TODO: To increase the performance of regex:
        1. You can combine the patterns using | for all expressions of the same level of Flame.
        https://stackoverflow.com/questions/1782586/speed-of-many-regular-expressions-in-python#comment1669596_1782712
        2. You can first use str.find to find potential matches, and then check those matches with regex. 
    '''
    regex_objects_dict = {1: [], 2: [], 3: [], 4: [], 5: []}  # Create a dictionary of 5 empty lists
    for flame_index in range(1, 6):
        for expression in flame_expressions_dict[flame_index]:
            regex_pattern = r'\b' + expression + r'\b'
            regex_object = re.compile(regex_pattern, re.IGNORECASE)
            regex_objects_dict[flame_index].append(regex_object)
    logger.info("@ %.2f seconds: Finished compiling the regex patterns into regex objects.",
                time.process_time())

    # Count the matches of each expression for each author
    for author_index, merged_tweets_of_author in enumerate(preprocessed_docs):
        for flame_index in range(1, 6):
            for expression_index in range(len(flame_expressions_dict[flame_index])):
                # ↳ Note: We are assuming that the lists inside *flame_expressions_dict* have not been manipulated since
                # the lists inside *regex_objects_dict* were created.
                list_of_matches = regex_objects_dict[flame_index][expression_index].findall(merged_tweets_of_author)
                count = len(list_of_matches)
                # count = merged_tweets_of_author.count(expression)
                counts_of_expressions_dict[flame_index][author_index, expression_index] = count

        # Log after counting the offensive words for every 100 authors
        if author_index % 100 == 0:
            logger.debug("@ %.2f seconds, progress: Counted (regex) the offensive words for author_index = %d",
                         time.process_time(), author_index)

    logger.info("@ %.2f seconds: Finished counting the occurrences of offensive words", time.process_time())

    # Pickle the output variable
    if pickle_path is not None:
        # Create the directory if it does not exist.
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        # Pickle
        with open(pickle_path, 'wb') as pickle_output_file:
            pickle.dump(counts_of_expressions_dict, pickle_output_file)
        logger.info('The counts of offensive words was pickled to: "%s"', pickle_path)

    return counts_of_expressions_dict


def generate_pickle_path(input_object, pickle_path_pattern):
    """Generate a "pickle path" to store the pickled output of a function

    Args:
        input_object: An object, which is the input of the function that we want to pickle its output.
        pickle_path_pattern: A string, which holds the pattern of the pickle path.

    Returns:
        The pickle path (String)

    Raises:
        ValueError: If *pickle_path_pattern* does not contain the *HASH_PLACEHOLDER* sub-string.

    Here's how the "pickle path" is generated:
    1. *input_object* is an object with any type. We calculate its SHA1 digest (hash value) encoded with the
    Base 32 encoding. The result is a 32-character, upper-case string (160 ÷ 5 = 32).
    2. *pickle_path_pattern* is the pattern of the pickle path, containing the placeholder sub-string, “<HASH>”.
    We replace this sub-string with the hash calculated in step 1, and return the result.

    Remarks:
    - If *pickle_path_pattern* is None, the function returns None.
    """

    # Define the constant
    HASH_PLACEHOLDER = "<HASH>"

    # Bypass 1: If *pickle_path_pattern* is None, return None and skip the rest of the function.
    if pickle_path_pattern is None:
        return None

    # Bypass 2: If *pickle_path_pattern* does not contain the *HASH_PLACEHOLDER* substring, raise an exception.
    if HASH_PLACEHOLDER not in pickle_path_pattern:
        raise ValueError('The pickle path pattern should contain the hash placeholder, "%s".' % HASH_PLACEHOLDER)
        # ↳ This is printf-style String Formatting.

    # Convert the input object to a *bytes* object (the pickled representation of the input object as a *bytes* object)
    input_object_as_bytes = pickle.dumps(input_object)
    # ↳ An inferior alternative could be *str(input_object).encode("utf-8")*

    # Create a hash object that uses the SHA1 algorithm
    hash_object = hashlib.sha1()

    # Update the hash object with the *bytes* object. This will calculate the hash value.
    hash_object.update(input_object_as_bytes)

    '''
    • Get a digest (hash value) suitable for filenames—alpha-numeric, case insensitive, and
    relatively short: Base 32
    - The SHA1 algorithm produces a 160-bit (20-Byte) digest (hash value).
    - *hash_object.hexdigest()* returns the digest (hash value) as a string object, containing only hexadecimal
    digits. Each hexadecimal (also “base 16”, or “hex”) digit represents four binary digits (bits). As a result,
    a SHA1 hash represented as hex will have a length of 160 ÷ 4 = 40 characters.
    - *hash_object.digest()* returns the digest (hash value) as a *bytes* object.
    - *base64.b32encode()* encodes a *bytes* object using the Base32 encoding scheme—specified in RFC 3548—and
    returns the encoded *bytes* object.
    - The Base 32 encoding is case insensitive, and uses an alphabet of A–Z followed by 2–7 (32 characters).
    More info: https://tools.ietf.org/html/rfc3548#section-5
    Each Base 32 character represents 5 bits (2^5 = 32). As a result, a SHA1 hash represented as Base 32 will have
    a lenght of 160 ÷ 5 = 32 characters.
    '''
    hash_value_as_bytes = hash_object.digest()
    hash_value_as_base32_encoded_bytes = base64.b32encode(hash_value_as_bytes)
    hash_value_as_base32_encoded_string = hash_value_as_base32_encoded_bytes.decode()
    # ↳ *.decode()* returns a string decoded from the given bytes. The default encoding is "utf-8".

    # Replace the *HASH_PLACEHOLDER* sub-string in path pattern with the Base 32 encoded digest (hash value)
    pickle_path = pickle_path_pattern.replace(HASH_PLACEHOLDER, hash_value_as_base32_encoded_string)

    return pickle_path


def hex_hash_object(input_object):
    """Generates the SHA1 digest (hash value) of an object.

    Args:
        input_object: An object with any type
    Returns:
        The SHA1 digest (hash value) of the *input_object* as a string, containing 40 hexadecimal digits.
    """

    # Convert the input object to a *bytes* object (the pickled representation of the input object as a *bytes* object)
    input_object_as_bytes = pickle.dumps(input_object)
    # ↳ An inferior alternative could be *str(input_object).encode("utf-8")*

    # Create a hash object that uses the SHA1 algorithm
    hash_object = hashlib.sha1()

    # Update the hash object with the *bytes* object. This will calculate the hash value.
    hash_object.update(input_object_as_bytes)

    # Get the hexadecimal digest (hash value)
    hex_hash_value = hash_object.hexdigest()

    return hex_hash_value


def cross_validate_model(clf, X_train, y_train):
    """Evaluates the classification model by k-fold cross-validation.

    The model is trained and tested k times, and all the scores are reported.
    """

    # Build a stratified k-fold cross-validator object
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    '''
    Evaluate the score by cross-validation
    This fits the classification model on the training data, according to the cross-validator
    and reports the scores.
    Alternative: sklearn.model_selection.cross_validate
    '''
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=skf)

    logger.info("@ %.2f seconds: Cross-validation finished", time.process_time())

    # Log the cross-validation scores, the mean score and the 95% confidence interval, according to:
    # http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
    # https://en.wikipedia.org/wiki/Standard_error#Assumptions_and_usage
    logger.info("Scores = %s", scores)
    logger.info("%%Accuracy: %0.2f (±%0.2f)" % (scores.mean()*100, scores.std()*2*100))
    # ↳ https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html


def train_and_test_model(clf, X_train, y_train, X_test, y_test):
    """Train the classifier and test it.

    This function trains the classifier on the training set,
    predicts the classes on the test set using the trained model,
    and evaluates the accuracy of the model by comparing it to the truth of the test set.
    """

    # Fit the classification model on the whole training set (as opposed to cross-validation)
    clf.fit(X_train, y_train)

    ''' Predict the outcome on the test set
        Note that the clf classifier has already been fit on the training data.
    '''
    y_predicted = clf.predict(X_test)

    logger.info("@ %.2f seconds: Finished training the model and predicting class labels for the test set",
                time.process_time())

    # Simple evaluation using numpy.mean
    logger.info("np.mean %%Accuracy: %f", np.mean(y_predicted == y_test) * 100)

    # Log the classification report
    logger.info("Classification report:\n%s", metrics.classification_report(y_test, y_predicted))

    # Log the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
    logger.info("Confusion matrix:\n%s", confusion_matrix)

    # Plot the confusion matrix
    plt.matshow(confusion_matrix)
    plt.set_cmap('jet')
    plt.show()


def train_model_and_predict(clf, X_train, y_train, X_test, author_ids_test, preset_key,
                            write_to_xml_files=True, xmls_destination_main_directory=None, ):
    """Train the classifier on the training set and predict the classes for the test set.

    This function is used only in **TIRA** evaluation.
    The difference between *train_model_and_predict* and the *train_and_test_model* function is that this function
    does not get *y_test* as an input, and hence, does not evaluate the accuracy of the model. Instead, it gets the
    Author IDs of the test dataset and writes the predictions as XML files for out-sourced evaluation.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'dataset_name': 'PAN 2018 English',
                                            'language_code': 'en',
                                            },
                          'PAN18_Spanish': {'dataset_name': 'PAN 2018 Spanish',
                                            'language_code': 'es',
                                            },
                          'PAN18_Arabic': {'dataset_name': 'PAN 2018 Arabic',
                                            'language_code': 'ar',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]

    # Fit the classification model on the whole training set (as opposed to cross-validation)
    clf.fit(X_train, y_train)

    ''' Predict the outcome on the test set
        Note that the clf classifier has already been fit on the training data.
    '''
    y_predicted = clf.predict(X_test)

    logger.info("@ %.2f seconds: Finished training the model and predicting class labels for the test set",
                time.process_time())

    if write_to_xml_files:
        logger.info("Writing the predictions to XML files.")
        process_data_files.write_predictions_to_xmls(author_ids_test, y_predicted,
                                                     xmls_destination_main_directory, PRESET['language_code'])


def rank_importance_of_features(clf, feature_names, write_to_file):
    """Rank the importance of the features

    This function ranks the features based on their importance in the classification model—absolute feature
    weight. It then writes the rankings to a CSV file (optional), and plots a number of top-ranking features and
    their weights.

    Args:
        clf: (LinearSVC) A classifier object. This classifier should be trained.
        feature_names: (list: strings) Feature names. You can get these from the vectorizer, when it is fit on the
    training set. For instance, the *vocabulary_* attribute of the vectorizers in scikit-learn.

    Returns:
        sorted_feature_weights: (list: floats) Feature weights, sorted by their absolute value in descending order.
        sorted_feature_names: (list: strings) Feature names, corresponding to the *sorted_feature_weights* list.
    """

    # The NumPy array of feature weights (coefficients in the primal problem)
    feature_weights = list(clf.coef_.flatten())
    # ↳ *clf.coef_* is a NumPy array with the shape (1, n), where n is the number of features. *.flatten()*—a NumPy
    # function—collapses this array into one dimension, hence, giving an array with the shape (n,). We then convert
    # this array into a list.

    list_of_tuples = list(zip(feature_weights, feature_names))
    # ↳ *zip()* makes an iterator that can be called only once. In order to reuse it, we convert it into a list.
    # Here, the result would be a list of n tuples, n being the number of features: (featureWeight, featureName)

    # Sort the list of tuples based on the absolute value of the feature weights in descending order.
    sorted_list_of_tuples = sorted(list_of_tuples, key=lambda tuple: abs(tuple[0]), reverse=True)
    # ↳ *sorted* sorts the items in an iterable based on a *key* function (optional), and returns a list.

    # Split the sorted list of tuples into two lists
    sorted_feature_weights, sorted_feature_names = [list(a) for a in zip(*sorted_list_of_tuples)]
    # ↳ - zip(*sorted_list_of_tuples) returns an iterator of two tuples: (featureWeight1, ...) and (featureName1, ...)
    # - List comprehension: *list(a) for a in* is used to convert those two tuples into lists.
    # - An asterisk (*) denotes “iterable unpacking”.

    # Write the rankings to a CSV file
    if write_to_file:
        process_data_files.write_feature_importance_rankings_to_csv(sorted_feature_weights, sorted_feature_names,
                                                                    log_file_path=logger.handlers[1].baseFilename)

    # Define constant: Number of top ranking features to plot
    PLOT_TOP = 30

    plt.barh(range(PLOT_TOP), sorted_feature_weights[:PLOT_TOP], align='center')
    plt.yticks(range(PLOT_TOP), sorted_feature_names[:PLOT_TOP])
    plt.xlabel("Feature weight")
    plt.ylabel("Feature name")
    plt.title("Top %d features based on absolute feature weight" % PLOT_TOP)

    # Flip the y axis
    plt.ylim(plt.ylim()[::-1])
    # ↳ plt.ylim() gets the current y-limits as a tuple. [::-1] reverses the tuple. plt.ylim(...) sets the new y-limits.
    # [start:end:step] is the “extended slicing” syntax. Some more info available at:
    # https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy

    plt.show()

    return sorted_feature_weights, sorted_feature_names


def main_development():
    """The "main" function for the development phase.

    Every time the script runs, it will call this function.
    """

    # %% Only in English
    for presetKey in ("PAN18_English",
                      # "PAN18_Spanish",
                      # "PAN18_Arabic",
                      ):

        logger.info("Running main_development() for preset: %s", presetKey)

        docs_train, docs_test, y_train, y_test = load_datasets_development(presetKey)
        X_train, X_test, clf, feature_names = extract_features(docs_train, docs_test, presetKey)
        cross_validate_model(clf, X_train, y_train)
        train_and_test_model(clf, X_train, y_train, X_test, y_test)

        # %% TODO: This function has bugs.
        # rank_importance_of_features(clf, feature_names, True)

        # Log run time
        logger.info("@ %.2f seconds: Run finished\n", time.process_time())


def main_tira_evaluation():
    """The "main" function for the Tira evaluation phase.

    Every time the script runs, it will call this function.
    """

    logger.info("sys.argv = %s", sys.argv)

    '''
    Parse the command line arguments
    According to PAN, the submitted script will be executed via command line calls with the following format:
        interpreter.exe script.py -c $inputDataset -o $outputDir
    
    For local testing on your machine, you can use the following command:
        ~/python.exe ~/train_model.py -c $inputDataset -o $outputDir
    Notes:
        - Replace $inputDataset and $outputDir.
        - Replace the ~ characters with the path of your Python interpreter and the path of this script.
    '''
    # Build a parser
    command_line_argument_parser = argparse.ArgumentParser()
    command_line_argument_parser.add_argument("-c")
    command_line_argument_parser.add_argument("-o")

    # Parse arguments
    command_line_arguments = command_line_argument_parser.parse_args()
    test_dataset_main_directory = command_line_arguments.c
    # ↳ This will be ignored for now.
    prediction_xmls_destination_main_directory = command_line_arguments.o

    # # TEMP (TIRA): For local testing on TIRA
    # test_dataset_main_directory = "E:/author-profiling/pan18-author-profiling-training-dataset-2018-02-27"
    # output_folder_name = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    # prediction_xmls_destination_main_directory = os.path.join("output", output_folder_name)
    # os.makedirs(prediction_xmls_destination_main_directory, exist_ok=True)  # Create the directory if it does not exist

    logger.info("test_dataset_main_directory = %s", test_dataset_main_directory)
    logger.info("prediction_xmls_destination_main_directory = %s\n", prediction_xmls_destination_main_directory)

    for preset_key in ("PAN18_English", "PAN18_Spanish", "PAN18_Arabic"):
        logger.info("Running main_tira_evaluation() for preset: %s", preset_key)

        docs_train, docs_test, y_train, author_ids_test =\
            load_datasets_tira_evaluation(test_dataset_main_directory, preset_key)
        # ↳ There is no *y_test* because the truth of the test dataset will not be provided to the participants.

        # # TEMP (TIRA): For fast debugging and testing
        # docs_train = docs_train[:100]
        # docs_test = docs_test[:100]
        # y_train = y_train[:100]
        # author_ids_test = author_ids_test[:100]

        X_train, X_test, clf, feature_names = extract_features(docs_train, docs_test, preset_key)
        cross_validate_model(clf, X_train, y_train)
        train_model_and_predict(clf, X_train, y_train, X_test, author_ids_test, preset_key,
                                True, prediction_xmls_destination_main_directory)

        # Log run time
        logger.info("@ %.2f seconds: Run finished\n", time.process_time())


'''
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to "__main__" when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == "__main__":
    logger = utils.configure_root_logger()
    utils.set_working_directory()
    main_development()
    # main_tira_evaluation()
