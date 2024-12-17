import argparse
import glob
import logging
import re
import sys

import numpy as np
import pandas as pd

from collections import Counter

from pypinyin import pinyin, lazy_pinyin, Style
from scipy import stats

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)


def read_file(file_name, dataframe):
    with open(file_name, "r") as f:
        file_text = f.read()
    logger.info(f"file: {file_name}, len (character count): {len(file_text)}")
    dataframe['file'] = [file_name]
    return file_text, dataframe


def keep_chinese_only(text: str) -> str:
    chinese_re = re.compile(r'[\u4e00-\u9FFF]+')
    text_list = chinese_re.findall(text)
    text = ''.join(text_list)
    return text


def strip_transcription(transcription: list, lazy: bool, add_tones=None) -> str:
    output = ""
    if add_tones:
        if lazy:
            for index, character in enumerate(transcription):
                output += f"{character}{add_tones[index]} "
        else:
            for index, character in enumerate(transcription):
                output += f"{character[0]}{add_tones[index]} "
    else:
        if lazy:
            for character in transcription:
                output += f"{character} "
        else:
            for character in transcription:
                output += f"{character[0]} "

    return output


def get_tones(pinyin3: str) -> list:
    tones = []
    for syllable in pinyin3.split():
        tones.append(syllable[-1])
    return tones


def transcribe_zhcn(text: str, api=pinyin) -> dict[str]:
    if api == lazy_pinyin:
        lazy = True
    else:
        lazy = False
    # logger.info(f"api: {api}, lazy: {lazy}")
    # for tone annotations, lazy_pinyin doesn't include the diacritics
    transcriptions = {"py": strip_transcription(api(text), lazy=False),
                      "py2": strip_transcription(api(text, style=Style.TONE2), lazy=lazy),
                      "py3": strip_transcription(api(text, style=Style.TONE3), lazy=lazy)}
    tones = get_tones(transcriptions["py3"])
    transcriptions["wg"] = strip_transcription(api(text, style=Style.WADEGILES), lazy=lazy, add_tones=tones)
    transcriptions["bpmf"] = strip_transcription(api(text, style=Style.BOPOMOFO), lazy=lazy)

    conf_length = 0
    for key, value in transcriptions.items():
        # dataframe[f'{key}_transcription'] = [value]
        if conf_length == 0:
            conf_length = len(value.split())
        else:
            if len(value.split()) != conf_length:
                logger.error(f"{key} length does not match {conf_length} at {len(value.split())}")
    logger.info(f"transcription dict char count: {len(transcriptions['bpmf'].split())}")
    return transcriptions


def string_counter(text: str):
    counter = Counter(text).items()
    return counter


def convert_dict_to_counter(transcription_dict: dict, dataframe):
    counters = {}
    for key, value in transcription_dict.items():
        counter = string_counter(value)
        counters[key] = counter
        dataframe[f'{key}_counter'] = [counter]

    return counters, dataframe


def iterate_through(transcription, number):
    # works like pairwise and triplewise in itertools but can take any number
    list = []
    for i in range(len(transcription) - 2):
        list.append(transcription[i:i+number])
    return list


def create_mapping(transcriptions: dict, dataframe, context):
    mappings = {}
    for key, transcription in transcriptions.items():
        character_mapping = {
            letter: index for index, letter in enumerate(set(transcription))
        }
        if context > 1:
            mapping_list = iterate_through(transcription, context)
            mapping_set = set(mapping_list)
            context_mapping = {
                "".join(map(str, item)): index for index, item in enumerate(mapping_set)
            }
        else:
            context_mapping = character_mapping

        mappings[key] = (context_mapping, character_mapping)
        dataframe[f"{key}_char_mapping"] = [character_mapping]
        dataframe[f"{key}_context_mapping"] = [context_mapping]
        dataframe[f"{key}_vocab_size"] = [len(character_mapping)]
    return mappings, dataframe


def create_array(dimensions, smoothing) -> np.array:
    return np.full(dimensions, smoothing, dtype=float)


def calculate_ngram_surprisals(phonetic_transcriptions: dict, mappings: dict, context: int,
                               smoothing_factor: int) -> np.array:
    ngram_surprisals = {}

    for key, phonetic_transcription in phonetic_transcriptions.items():
        # mapping context mapping x character mapping
        context_mapping = mappings[key][0]
        character_mapping = mappings[key][1]

        print(context_mapping)

        if context > 0:
            dimensions = [len(context_mapping), len(character_mapping)]
        else:
            dimensions = [len(context_mapping), ]
        char_count_matrix = create_array(tuple(dimensions), smoothing_factor)

        for i in range(len(phonetic_transcription) - 1):
            if context == 0:
                char_count_matrix[context_mapping[phonetic_transcription[i]]] += 1
            elif context == 1:
                char_count_matrix[
                    context_mapping[phonetic_transcription[i]], character_mapping[phonetic_transcription[i + 1]]] += 1
            else:
                context_chars = ""
                loops = 0
                if (i + context) < len(phonetic_transcription):
                    while len(context_chars) < context:
                        context_chars += phonetic_transcription[i + loops]
                        loops += 1
                    if len(context_chars) == context:
                            char_count_matrix[
                                context_mapping[context_chars], character_mapping[phonetic_transcription[i + context]]] += 1
        if context == 0:
            char_count_matrix[context_mapping[phonetic_transcription[-1]]] += 1

        # logger.info(f"{key}: {char_count_matrix}")
        ngram_surprisals[key] = char_count_matrix

    return ngram_surprisals


def info(probs):
    return -np.log2(np.array(probs))


def calculate_info(counts: dict) -> dict:
    info_content = {}
    for key, surprisal in counts.items():
        normalized_vector = surprisal / sum(surprisal)
        information_content = info(normalized_vector)
        info_content[key] = (normalized_vector, information_content)
    return info_content


def calculate_entropies(information_content: dict, dataframe):
    entropies = {}
    for key, surprisal_tuple in information_content.items():
        entropy = stats.entropy(surprisal_tuple[0].flatten(), base=2).round(3)
        entropies[key] = entropy
        dataframe[f"{key}_entropy"] = entropy
    return entropies, dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="chinese surprisal calculations")
    parser.add_argument(
        "-l",
        "--lazy",
        required=False,
        help="-l for lazy_pinyin api (otherwise defaults to regular)",
        action="store_false",
        default="False"
    )
    parser.add_argument(
        "-s",
        "--smoothing",
        required=False,
        help="default to 0",
        action="store"
    )
    parser.add_argument(
        "-c",
        "--context",
        required=True,
        help="context window to consider for probabilities",
        action="store"
    )
    args = parser.parse_args()

    if args.lazy:
        api_type = lazy_pinyin
    else:
        api_type = pinyin

    smoothing_quantity = args.smoothing
    if not smoothing_quantity:
        smoothing_quantity = 0.1

    prior_context = int(args.context)

    output_df = pd.DataFrame()

    genre_list = ["bc", "bn", "mz", "nw", "tc", "wb"]
    zh_path = "Desktop/ontonotes/chinese/"

    # genre_list = ["tt"]

    for genre in genre_list:
        files = glob.glob(f"{zh_path}{genre}/*/*/*.name")
        # files = glob.glob(f"tt/*.txt")
        logger.info(f"Number of files in {genre}: {len(files)}")
        for file in files:
            logger.info(f"data read in: {file}")
            data, output_df = read_file(file, output_df)
            data_stripped = keep_chinese_only(data)
            output_df['file_length'] = [len(data_stripped)]
            output_df['genre'] = [genre]
            output_df['context'] = [prior_context]
            data_transcribed = transcribe_zhcn(data_stripped, api=api_type)

            # counts and frequencies
            data_counters, output_df = convert_dict_to_counter(data_transcribed, output_df)

            # get mapping for calculations
            mapping, output_df = create_mapping(data_transcribed, output_df, prior_context)

            # surprisal and entropy
            unigram_surprisals = calculate_ngram_surprisals(data_transcribed, mapping, prior_context,
                                                            smoothing_factor=smoothing_quantity)
            info_counts = calculate_info(unigram_surprisals)
            entropies, output_df = calculate_entropies(info_counts, output_df)
            logger.info(f"context considered {prior_context}: {entropies}")

            output_df.to_csv(f"outputs.csv", mode="a", header=False, index=True)
