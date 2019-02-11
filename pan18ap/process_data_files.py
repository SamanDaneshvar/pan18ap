"""Deal with files: Load datasets from files and write output files.

This module loads the dataset of the Author Profiling task at PAN 2018 and pre-processes it.
For more information refer to the docstring of the *load_pan_data* function.

Remarks:
- The *xml.etree.ElementTree* module is not secure against maliciously constructed data. Make sure the XML files
are from a trusted source.
"""

import csv
import fnmatch
import logging
import os
import shutil
import sys
import time

from xml.etree import ElementTree


def load_pan_data(xmls_directory, truth_path, write_to_txt_files=False, txts_destination_directory=None):
    """Load PAN data

    This function loads the PAN dataset and the truth, parses the XML and returns:
    Merged tweets of the authors, the truth, Author IDs, and the original length of the tweets.
    It also writes the tweets to TXT files (optional).

    Args:
        xmls_directory: The directory where the XML files of the dataset reside.
        truth_path: The path of the truth file.
        write_to_txt_files: (boolean) If True, the XML files will also be written as TXT files after being parsed.
        txts_destination_directory: The TXT files will be written to this directory.

    Returns:
        merged_tweets_of_authors: List. Each item is all of the tweets of an author, merged into one string.
            Refer to the list of replacements in the remarks.
        truths: List of truths for authors.
        author_ids: List of Author IDs.
        original_tweet_lengths: List of original tweet lengths.

    Raises:
        RuntimeError: If a non-XML file exists inside the *xmls_directory*

    Remarks:
        - Since *xml_filenames* is sorted in ascending order, all the returned lists will also be in the same order
        (sorted in ascending order of the Author IDs).
        - List of replacements:
            Line feed		<LineFeed>
            End of Tweet	<EndOfTweet>
    """

    ''' 
    *os.listdir* returns a list containing the name of all files and folders in the given directory.
    Normally, the list is created in ascending order. However, the Python documentation states,
    “the list is in arbitrary order”.
    To ensure consistency and avoid errors in syncing the order of the items among
    different lists (e.g., *author_ids*, *truths*), we sort the list by calling *sorted*.
    *sorted()* returns a new sorted list (in ascending lexicographical order) of all the items in an iterable.
    '''
    xml_filenames = sorted(os.listdir(xmls_directory))

    # Store the Author IDs in a list
    # The Author IDs list will have the same order as the XML filenames list.
    author_ids = []  # Create an empty list
    for xml_filename in xml_filenames:
        author_ids.append(xml_filename[:-4])

    # Skip loading truth if path input is None. Else, load the truth from the file.
    if truth_path is None:
        logger.info("*truth_path* is None => Skipped loading the truth")
        truths = None
        # This scenario will happen when loading the test dataset for **TIRA** evaluation, where the truth of the test
        # set is not provided.
    else:
        truths = load_truth(truth_path, author_ids)

    if write_to_txt_files:
        logger.info("The parsed XMLs will also be written to TXT files.")
        # Create the directory if it does not exist.
        os.makedirs(txts_destination_directory, exist_ok=True)

    # Initialize the lists.
    # The lists will have the same order as the XML filenames list (refer to: “Iterate over XML Files”)
    original_tweet_lengths = []  # Create an empty list
    # ↳ Every row will represent an author, every column will represent a tweet.
    merged_tweets_of_authors = []  # Create an empty list
    # ↳ Each cell will contain all 100 tweets of an author, merged.

    # Iterate over XML files
    for author_index, xml_filename in enumerate(xml_filenames):
        # Make sure only XML files go through
        if not fnmatch.fnmatch(xml_filename, '*.xml'):
            logger.error("Encountered a non-XML file inside the directory: %s >>> The program will now exit.",
                         xml_filename)
            raise RuntimeError('Encountered a non-XML file inside the directory: %s' % xml_filename)
            # ↳ This is printf-style String Formatting.

        # Read the XML file and parse it into a tree
        # Parser is explicitly defined to ensure UTF-8 encoding.
        tree = ElementTree.parse(os.path.join(xmls_directory, xml_filename),
                                 parser=ElementTree.XMLParser(encoding="utf-8"))
        root = tree.getroot()
        '''
        root is the root element of the parsed tree
        root[0], ..., root[m-1] are the children of root—elements one level below the root.
        root[0][0], ..., root[0][n-1] are the children of root[0].
        and so on.
        
        Each element has a tag, a dictionary of attributes, and sometimes some text:
            root[i][j].tag, ”.attrib, ”.text 
        '''

        # Add an empty new row to the list. Each row represents an author.
        original_tweet_lengths.append([])

        # Initialize the list. Note that this list resets in every author (XML file) loop.
        tweets_of_this_author = []  # Create an empty list

        # Iterate over the tweets within this parsed XML file:
        # Record the tweet length, replace line feeds, and append the tweet to a list
        for child in root[0]:
            # Element.text accesses the element's text content,
            # which is saved with the following format in the XML files: <![CDATA[some text]]>
            tweet = child.text
            original_tweet_lengths[author_index].append(len(tweet))

            # Replace line feed (LF = \n) with “ <LineFeed> ”
            # Note: There were no carriage return (CR = \r) characters in any of the 3,000 XML files.
            tweet = tweet.replace('\n', " <LineFeed> ")

            # Create a list of the tweets of this author, to write to a text file and merge, after the loop terminates.
            '''
            Google Python Style Guide: Avoid using the + and += operators to accumulate a string within a loop.
            Since strings are immutable, this creates unnecessary temporary objects and results in quadratic rather
            than linear running time.
            Avoid: merged_tweets_of_authors[author_index] += tweet + " <EndOfTweet> "
            Instead, append each substring to a list and ''.join the list after the loop terminates.
            '''
            tweets_of_this_author.append(tweet)

        # Write the tweets of this author to a TXT file
        # Note that in these tweets, the line feed characters are replaced with a tag.
        if write_to_txt_files:
            # Create a TXT file with the Author ID as the filename (same as the XML files) in the write mode
            with open(os.path.join(txts_destination_directory, author_ids[author_index] + ".txt"),
                      'w', encoding="utf-8") as txt_output_file:
                txt_output_file.write('\n'.join(tweets_of_this_author))
                # ↳ '\n'.join adds a newline character between every two strings,
                # so there won't be any extra line feeds on the last line of the file.

        # Concatenate the tweets of this author, and append it to the main list
        merged_tweets_of_this_author = " <EndOfTweet> ".join(tweets_of_this_author) + " <EndOfTweet>"
        # ↳ " <EndOfTweet> ".join adds the tag between every two strings, so we need to add another tag to the end.
        merged_tweets_of_authors.append(merged_tweets_of_this_author)

    logger.info("@ %.2f seconds: Finished loading the dataset", time.process_time())

    return merged_tweets_of_authors, truths, author_ids, original_tweet_lengths


def load_truth(truth_path, author_ids):
    """Load the truth

    This function loads the truth from the TXT file, and makes sure the order of the Truth list is the same as
    the Author IDs list.

    Args:
        truth_path: The path of the truth file.
        author_ids: The list of Author IDs.

    Returns:
        The list of the truths.

    Raises:
        RuntimeError: If for any reason, the function was unable to sync the order of the Truth list and the
        Author ID list. This error should not happen, but the exception is put in place as a measure of caution.
    """

    # Load the Truth file, sort its lines (in ascending order), and store them in a list
    temp_sorted_author_ids_and_truths = []  # Create an empty list
    # ↳ Each row represents an author. Column 0: Author ID,  column 1: Truth.
    with open(truth_path, 'r') as truth_file:
        for line in sorted(truth_file):
            # ↳ “for line” automatically skips the last line if it only contains a newline character.
            # ↳ *sorted()* returns a new sorted list (in ascending lexicographical order) of all the items in an
            # iterable—here, the lines in truth_file.
            # Remove the ending newline character from each line (line is a string)
            line = line.rstrip('\n')
            # str.split returns a list of the parts of the string which are separated by the specified separator string.
            temp_sorted_author_ids_and_truths.append(line.split(":::"))

    truths = []  # Create an empty list
    # Make sure the rows in *temp_sorted_author_ids_and_truths* and *author_ids* have the same order,
    # and store the truth in the *truths* list.
    for i, row in enumerate(temp_sorted_author_ids_and_truths):
        # Compare the Author ID in the two lists
        if row[0] == author_ids[i]:
            # ↳ row[0] is the Author ID of this row, and row[1] is the truth of this row.
            # Add the truth to the truths list
            truths.append(row[1])
        else:
            logger.error("Failed to sync the order of the Truth list and the Author ID list."
                         "Row number: %d >>> The program will now exit.", i)
            raise RuntimeError('Failed to sync the order of the Truth list and the Author ID list. Row number: %d' % i)
            # ↳ This is printf-style String Formatting.

    return truths


def split_train_and_test_files(author_ids_train, author_ids_test, truths_train, truths_test, preset_key):
    """Split the dataset files into training and test sets

    This function splits the XML files of the dataset into training and test sets according to the results of
    sklearn's *train_test_split* function. It also writes two separate TXT files for the truth of the training
    and test sets. This function is used for mimicking the **TIRA** environment for local testing.
    """

    # Define the dictionary of presets. Each “preset” is a dictionary of some values.
    PRESETS_DICTIONARY = {'PAN18_English': {'dataset_name': 'PAN 2018 English',
                                            'xmls_source_directory': 'data/PAN 2018, Author Profiling/en/text/',
                                            'xmls_destination_subdirectory': 'en/text/',
                                            'truth_destination_subpath': 'en/truth.txt',
                                            },
                          'PAN18_Spanish': {'dataset_name': 'PAN 2018 Spanish',
                                            'xmls_source_directory': 'data/PAN 2018, Author Profiling/es/text/',
                                            'xmls_destination_subdirectory': 'es/text/',
                                            'truth_destination_subpath': 'es/truth.txt',
                                            },
                          'PAN18_Arabic': {'dataset_name': 'PAN 2018 Arabic',
                                            'xmls_source_directory': 'data/PAN 2018, Author Profiling/ar/text/',
                                            'xmls_destination_subdirectory': 'ar/text/',
                                            'truth_destination_subpath': 'ar/truth.txt',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]

    # Define the constants: Destination main directory of the training and test datasets
    TRAIN_DESTINATION_MAIN_DIR = "data/PAN 2018, Author Profiling/TiraDummy/train directory/"
    TEST_DESTINATION_MAIN_DIR = "data/PAN 2018, Author Profiling/TiraDummy/test directory/"

    # Assemble the destination directories and paths
    xmls_destination_directory_train = os.path.join(TRAIN_DESTINATION_MAIN_DIR, PRESET['xmls_destination_subdirectory'])
    xmls_destination_directory_test = os.path.join(TEST_DESTINATION_MAIN_DIR, PRESET['xmls_destination_subdirectory'])
    truth_destination_path_train = os.path.join(TRAIN_DESTINATION_MAIN_DIR, PRESET['truth_destination_subpath'])
    truth_destination_path_test = os.path.join(TEST_DESTINATION_MAIN_DIR, PRESET['truth_destination_subpath'])

    # Create the destination directories if they do not exist.
    for directory in [xmls_destination_directory_train, xmls_destination_directory_test,
                      os.path.dirname(truth_destination_path_train), os.path.dirname(truth_destination_path_test)]:
        os.makedirs(directory, exist_ok=True)

    # Copy the XML files of the split training and test dataset to two different destinations.
    for author_id in author_ids_train:
        shutil.copy(os.path.join(PRESET['xmls_source_directory'], author_id + ".xml"), xmls_destination_directory_train)
    for author_id in author_ids_test:
        shutil.copy(os.path.join(PRESET['xmls_source_directory'], author_id + ".xml"), xmls_destination_directory_test)

    # • Write the truth of the split training and test datasets to two different text files.
    # Create empty lists
    lines_to_write_train = []
    lines_to_write_test = []
    # Iterate over the authors in the training and test set, and keep the lines to be written
    for author_id, gender in zip(author_ids_train, truths_train):
        lines_to_write_train.append(author_id + ":::" + gender)
    for author_id, gender in zip(author_ids_test, truths_test):
        lines_to_write_test.append(author_id + ":::" + gender)
    # Write the lines to the files
    with open(truth_destination_path_train, 'w') as truthFile_train:
        truthFile_train.write('\n'.join(lines_to_write_train))
    with open(truth_destination_path_test, 'w') as truthFile_test:
        truthFile_test.write('\n'.join(lines_to_write_test))
        # ↳ '\n'.join adds a newline character between every two strings,
        # so there won't be any extra line feeds on the last line of the file.

    logger.info("@ %.2f seconds: Finished splitting the files of the training and test datasets (to mimic TIRA)",
                time.process_time())


def load_flame_dictionary(path="data/Flame_Dictionary.txt"):
    """Load the Flame dictionary

    This function loads the Flame dictionary from a text file.
    If there are any duplicate expressions in the text file, the value of the first instance is kept, and the
    other instances are reported as duplicates.

    Args:
        path: The path of the Flame dictionary text file

    Returns:
        *flame_dictionary*: A Python dictionary with all the entries
            Keys:   (string) Expression
            Values: (int)    Flame level
        *flame_expressions_dict*: A Python dictionary with the entries, separated by Flame level, into five lists
            Keys:   (int)           Flame level
            Values: (list: strings) Expressions
    """

    logger.info("Loading the Flame Dictionary from path: %s", os.path.realpath(path))

    flame_dictionary = {}  # Create an empty dictionary
    duplicates = []  # Create an empty list
    flame_expressions_dict = {1: [], 2: [], 3: [], 4: [], 5: []}  # Create a dictionary of 5 empty lists

    with open(path, 'r') as flame_dictionary_file:
        for line in flame_dictionary_file:
            # ↳ “for line” automatically skips the last line if it only contains a newline character.
            # Remove the ending newline character from each line (line is a string)
            line = line.rstrip('\n')

            # Split the line into the flame level and the expression
            flame_level = int(line[0])
            # ↳ int() converts the string into an integer.
            expression = line[2:]

            # Add the entry to the dictionary, and to the corresponding list within *flame_expressions_dict*.
            # If it already exists in the dictionary, ignore it and keep a record of it in the *duplicates* list.
            if expression in flame_dictionary:
                duplicates.append(expression)
            else:
                flame_dictionary[expression] = flame_level
                flame_expressions_dict[flame_level].append(expression)

    # Report the duplicate items to the user
    if len(duplicates) > 0:
        logger.warning("%d duplicate expressions found in the Flame Dictionary: %s",
                       len(duplicates), duplicates)

    return flame_dictionary, flame_expressions_dict


def write_predictions_to_xmls(author_ids_test, y_predicted, xmls_destination_main_directory, language_code):
    """Write predictions to XML files

    This function is only used in **TIRA** evaluation.
    It writes the predicted results to XML files with the following format:
        <author id="author-id" lang="en|es" gender_txt="female|male" gender_img="N/A" gender_comb="N/A" />
    """

    # Add the alpha-2 language code (“en” or “es”) subdirectory to the end of the output directory
    xmls_destination_directory = os.path.join(xmls_destination_main_directory, language_code)

    # Create the directory if it does not exist.
    os.makedirs(xmls_destination_directory, exist_ok=True)

    # Iterate over authors in the test set
    for author_id, predicted_gender in zip(author_ids_test, y_predicted):
        # Create an *Element* object with the desired attributes
        root = ElementTree.Element('author', attrib={'id': author_id,
                                                     'lang': language_code,
                                                     'gender_txt': predicted_gender,
                                                     'gender_img': "N/A",
                                                     'gender_comb': "N/A",
                                                     })
        # Create an ElementTree object
        tree = ElementTree.ElementTree(root)
        # Write the tree to an XML file
        tree.write(os.path.join(xmls_destination_directory, author_id + ".xml"))
        # ↳ ElementTree sorts the dictionary of attributes by name before writing the tree to file.
        # ↳ The final file would look like this:
        # <author gender_comb="N/A" gender_img="N/A" gender_txt="female|male" id="author-id" lang="en|es" />
    logger.info("@ %.2f seconds: Finished writing the predictions to XML files", time.process_time())


def write_feature_importance_rankings_to_csv(sorted_feature_weights, sorted_feature_names, log_file_path):
    """Write the feature importance rankings to a CSV file.

    This function writes the feature importance rankings to a CSV file, next to the log file.
    Refer to the docstring of the *write_iterable_to_csv()* function.
    """

    # Determine the path of the output CSV file based on the path of the log file, such that the leading date and
    # time of the two filenames are the same.
    log_file_directory = os.path.dirname(log_file_path)
    log_file_name_without_extension = os.path.splitext(os.path.basename(log_file_path))[0]
    CSV_PATH = os.path.join(log_file_directory, log_file_name_without_extension + "; Feature importance rankings.csv")

    # Create the directory if it does not exist.
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Write to the CSV file
    with open(CSV_PATH, 'w', newline='', encoding="utf-8") as csv_output_file:
        csv_writer = csv.writer(csv_output_file)
        csv_writer.writerow(["Feature weights:", "Feature names:"])
        csv_writer.writerows(zip(sorted_feature_weights, sorted_feature_names))

    logger.info('List of features based on their importance in the classification model (absolute feature weight) '
                'was written to CSV file: "%s"', CSV_PATH)


def write_iterable_to_csv(iterable, iterable_name, log_file_path):
    """Write an iterable to a CSV file.

    This function writes any iterable object to a CSV file next to the log file.
    - You can get *log_file_path* by calling *logger.handlers[1].baseFilename* in the root module, assuming that
    the file handler is the second handler of the logger.

    • CSV Writer objects remarks:
    - *csvwriter.writerow(row)*:   A row must be an iterable of strings or numbers.
    - *csvwriter.writerows(rows)*: *rows* must be a list of row objects, described above.
    """

    # Determine the path of the output CSV file based on the path of the log file, such that the leading date and
    # time of the two filenames are the same.
    log_file_directory = os.path.dirname(log_file_path)
    log_file_name_without_extension = os.path.splitext(os.path.basename(log_file_path))[0]
    CSV_PATH = os.path.join(log_file_directory, log_file_name_without_extension + "; " + iterable_name + ".csv")

    # • Find out if the iterable is an “iterable of iterables”. For example, [[1, 2], [3, 4]] is an iterable
    # of iterables—each item in it is also an iterable; however, [1, 2, 3] isn't.

    # Select the first item in the iterable. We will only test this item.
    item = iterable[0]

    # The following is “the try statement”.
    try:
        iterator = iter(item)
        # ↳ This will raise a TypeError exception if *item* is not iterable.
    except TypeError:
        # This means *item* is not iterable.
        item_is_iterable = False
    else:
        # This means *item* is an iterable.
        item_is_iterable = True

    # If *item* is a string, it means it escaped from us! Strings are considered iterables, but here, we are
    # looking for iterables such as lists and tuples, not strings.

    # If *item* is not iterable or it is a string, convert *iterable* to a list of lists of one item each.
    # For example: (1, 2, 3) → [[1], [2], [3]]
    if not item_is_iterable or isinstance(item, str):
        iterable = [[item] for item in iterable]
    # Now *iterable* is an “iterable of iterables”!

    # Create the directory if it does not exist.
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Write to the CSV file
    with open(CSV_PATH, 'w', newline='', encoding="utf-8") as csv_output_file:
        csv_writer = csv.writer(csv_output_file)
        csv_writer.writerow([iterable_name])
        csv_writer.writerows(iterable)

    logger.info('%s was written to CSV file: "%s"', iterable_name, CSV_PATH)


'''
The following lines will be executed any time this .py file is run as a script or imported as a module.
'''
# Create a logger object. The root logger would be the parent of this logger
# Note that if you run this .py file as a script, this logger will not function, because it is not configured.
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # The following lines will be executed only if this .py file is run as a script,
    # and not if it is imported as a module.
    print("Module was executed directly.")
