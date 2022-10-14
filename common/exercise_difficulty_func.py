# import relevant libraries
import re
from statistics import quantiles

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import scipy.stats as stat
import stanza
import textstat
import wordfreq
from autocorrect import Speller
from scipy import stats
from spellchecker import SpellChecker
from textblob import TextBlob
from textstat.textstat import textstatistics
from wordfreq import (
    get_frequency_dict,
    tokenize,
    top_n_list,
    word_frequency,
    zipf_frequency
)


def get_token_list(text):
    """
    Tokenize the text and add new tokens to token list.
    Parameters
    ----------
    text : str
        A text string.
    """

    tokens = tokenize(text, language)

    for token in tokens:
        if token not in token_list:
            token_list.extend(tokens)


def find_word_difficulty(word, token_list, language):
    """
    Calculate the average sentence length.
    Parameters
    ----------
    text : str
        A text string.
    Returns
    -------
    pandas series
        The length of the word. 
        The zipf frequency of the word.
        The difficulty score of the word.

    """
    lmax = len(max(token_list, key=len))
    length_word = len(word)
    zipf_word = zipf_frequency(word, language)
    relative_length = length_word/lmax
    relative_freq = (8 - zipf_word)
    score = relative_length * relative_freq

    return pd.Series([length_word, zipf_word, score])


def transform_difficulty(score_column):
    """
    Boxcox Transformation of the difficulty scores of the words.
    """
    return stat.boxcox(score_column)


def find_difficulty_quantiles(score_column, quantiles=32):
    return pd.qcut(
        x=score_column,
        q=quantiles,
        labels=False,
        retbins=True,
        duplicates="drop"
    )


def find_difficulty_level(quantile_ranks):
    """
    Calculate the difficulty level of the words based on their quantile rank.
    """

    for q in quantile_ranks:
        if q < 2:
            return 'A1'
        if q < 4:
            return 'A2'
        if q < 8:
            return 'B1'
        return 'B2' if q < 16 else 'C1'


def set_word_difficulty(text, word_diff_df):
    score = word_diff_df[word_diff_df["word"] == text]["boxcox_score"]
    difficulty = word_diff_df[word_diff_df["word"] == text]["difficulty"]

    return pd.Series([score, difficulty])


def get_sentence_length(text, nlp):
    """
    Calculate the length of the text.
    Parameters
    ----------
    text : str
        A text string.
    Returns
    -------
    int
        The length of the text.
    """
    doc = nlp(text)
    return sum(len(sent.tokens) for sent in doc.sentences)


def find_wLengthMax(text):
    """
    Find the length of the longest word in the sentence
    Parameters
    ----------
    text : str
        A text string.
    Returns
    -------
    int
        The length of the longest word.
    """
    word_list = list(set(re.findall(r"[\w\='‘’]+", text.lower())))
    return len(max(word_list, key=len))


def find_wSRarest(text, language):
    """
    Find the difficulty of the rarest word in the sentence
    Parameters
    ----------
    text : str
        A text string.
    Returns
    -------
    float
        The difficulty of the rarest word.
    """

    word_list = list(set(re.findall(r"[\w\='‘’]+", text.lower())))
    word_freq = [zipf_frequency(word, language) for word in word_list]
    return 8 - min(word_freq) if word_freq else 8 - 0


def find_wSavg(text, word_diff_df):
    """
    Find the average difficulty of the words in a sentence
    Parameters
    ----------
    text : str
        A text string.
    Returns
    -------
    float
        The average difficulty of the words.
    """
    word_list = list(set(re.findall(r"[\w\='‘’]+", text.lower())))
    avg_word_diff = [
        word_diff_df[word_diff_df["word"] == word]["score"].values[0]
        for word in word_list if word in word_diff_df["word"].unique()
    ]
    return np.mean(avg_word_diff)


def find_SScore(text):
    """
    Calculate the sentence difficulty score.
    Parameters
    ----------
    text : str
        A text string.
    Returns
    -------
    float
        The sentence difficulty score.
    """
    return get_sentence_length(text) * find_wSavg(text, word_diff_df) * find_wSRarest(text, language)


def find_all_scores(exo_df, token_list):
    # split the exercise dataset by exercise types
    word_exo_df = exo_df[exo_df["Exo_objective"].isin(word_exo_objs)]
    sent_exo_df = exo_df[exo_df["Exo_objective"].isin(sent_exo_objs)]

    # find all the tokens in the exercise dataset
    word_exo_df["Full_sentence"].apply(get_token_list)
    sent_exo_df["Full_sentence"].apply(get_token_list)

    # make the token list into a pandas dataframe
    word_diff_df = pd.DataFrame({"word": token_list})

    # get the difficulty of the words in the token list and sort by their difficulty
    word_diff_df[['length', 'zipf_freq', 'score']] = word_diff_df["word"].apply(
        lambda text: find_word_difficulty(text, token_list, language))
    word_diff_df.sort_values(
        by='score',
        ascending=True,
        inplace=True,
        ignore_index=True
    )

    # do a boxcox transformation on the word difficulty scores
    word_diff_df["boxcox_score"] = word_diff_df["score"].apply(
        lambda score: transform_difficulty(score))

    # get the level of the transformed difficulty score
    quantile_ranks = word_diff_df["boxcox_score"].apply(
        lambda score: find_difficulty_quantiles(score))
    word_diff_df["Difficulty"] = find_difficulty_level(quantile_ranks)

    # set the difficulty for the word exercises
    word_exo_df[["Score_target_word", "Difficulty"]] = word_exo_df["Full_sentence"].apply(
        lambda text: set_word_difficulty(text, word_diff_df))
    word_exo_df = word_exo_df.merge(word_diff_df[["boxcox_score", "Difficulty"]],
                                    how="left", left_on="Full_sentence", right_on="word").drop(columns=["word"])

    # get the average sentence length for each full sentence in the exercise dataset
    sent_exo_df["Length_sentence"] = sent_exo_df["Full_sentence"].apply(
        lambda text: get_sentence_length(text, nlp))

    # get length of right answers i.e. target words
    sent_exo_df["Length_traget_word"] = sent_exo_df["Full_sentence"].apply(
        lambda x: len(x))

    # get length of the longest word
    sent_exo_df["Length_longest_word"] = sent_exo_df["Full_sentence"].apply(
        lambda text: find_wLengthMax(text))

    # get the difficulty score of the right answers (target words), rarest word in the sentence and the sentence
    sent_exo_df["Score_target_word"] = sent_exo_df["Right_answer"].apply(
        lambda text: set_word_difficulty(text, word_diff_df)[0])
    sent_exo_df["Score_rarest_word"] = sent_exo_df["Right_answer"].apply(
        lambda text: find_wSRarest(text, language))
    sent_exo_df["Score_sentence"] = sent_exo_df["Right_answer"].apply(
        lambda text: find_SScore(text))

    # get the average difficulty score of the words in the sentences
    sent_exo_df["Score_sentence_average"] = sent_exo_df["Right_answer"].apply(
        lambda text: find_wSavg(text, word_diff_df))

    # get the difficulty level of the sentences
    quantile_ranks = sent_exo_df["Score_sentence"].apply(
        lambda score: find_difficulty_quantiles(score))
    sent_exo_df["Difficulty"] = find_difficulty_level(quantile_ranks)

    return word_exo_df, sent_exo_df


exo_df = pd.read_excel("../en/English_Exercises.xlsx")
language = "en"
word_exo_objs = ["Learning vocabulary"]
sent_exo_objs = ["Useful Sentences", "Grammar", "Verb_Conjugation"]

quantiles = 32
token_list = []
word_diff_df = pd.DataFrame()

stanza.download(language)
nlp = stanza.Pipeline(language)

word_exo_df, sent_exo_df = find_all_scores(exo_df, token_list)

print(word_exo_df.head())
print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
print(sent_exo_df.head())
