# import relevant libraries
import re

import numpy as np
import pandas as pd
import scipy.stats as stat
from wordfreq import (
    tokenize, 
    zipf_frequency
)

class Difficulty:

    def __init__(self, exo_df, language, word_exo_objs, sent_exo_objs):
        self.exo_df = exo_df
        self.language = language
        self.word_exo_objs = word_exo_objs
        self.sent_exo_objs = sent_exo_objs

        self.quantiles = 32
        self.token_list = []
        self.word_diff_df = pd.DataFrame()

    def get_token_list(self, text):
        """
        Tokenize the text and add new tokens to token list.
        Parameters
        ----------
        text : str
            A text string.
        """

        tokens = tokenize(text, self.language)

        for token in tokens:
            self.token_list.extend(tokens)
    
    def find_word_difficulty(self, word):
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
        lmax = len(max(self.token_list, key = len))
        length_word = len(word)
        zipf_word = zipf_frequency(word, self.language)
        relative_length = length_word/lmax
        relative_freq = (8 - zipf_word)
        score = relative_length * relative_freq
        return pd.Series([length_word, zipf_word, score])

    def find_difficulty_quantiles(self, score_column):
        return pd.qcut(score_column, 32, labels=False, duplicates = "drop")

    def find_difficulty_level(self, q):
        """
        Calculate the difficulty level of the words based on their quantile rank.
        """
        
        if q < 2:
            return 'A1'
        if q < 4:
            return 'A2'
        if q < 8:
            return 'B1'
        
        return 'B2' if q < 16 else 'C1'

    def set_word_difficulty(self, text):
        return str(self.word_diff_df[self.word_diff_df["word"] == text.lower()]["Difficulty"].values[0])

    def sentence_length(self, text):
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
        return len(tokenize(text, self.language))

    def find_wLengthMax(self, text):
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
        word_list = list(set(tokenize(text, self.language)))
        return len(max(word_list, key = len))

    def find_wSRarest(self, text):
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

        word_list = list(set(tokenize(text, self.language)))
        word_freq = [zipf_frequency(word, self.language) for word in word_list]
        return 8 - min(word_freq) if word_freq else 8 - 0

    def find_wSavg(self, text):
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
        word_list = list(set(tokenize(text, self.language)))
        avg_word_diff = [
            self.word_diff_df[self.word_diff_df["word"] == word]["score"].values[0]
            for word in word_list if word in self.word_diff_df["word"].unique()
        ]
        return np.mean(avg_word_diff)
    
    def find_SScore(self, text):
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
        return self.sentence_length(text) * self.find_wSavg(text) * self.find_wSRarest(text)
    

    def find_all_scores(self):
        # split the exercise dataset by exercise types
        word_exo_df = self.exo_df[self.exo_df["Exo_objective"].isin(self.word_exo_objs)]
        sent_exo_df = self.exo_df[self.exo_df["Exo_objective"].isin(self.sent_exo_objs)]

        # find all the tokens in the exercise dataset
        word_exo_df["Full_sentence"].apply(lambda text: self.get_token_list(str(text)))
        sent_exo_df["Full_sentence"].apply(lambda text: self.get_token_list(str(text)))

        # make the token list into a pandas dataframe
        self.token_list = list(set(self.token_list))
        self.word_diff_df = pd.DataFrame({"word": self.token_list})

        # get the difficulty of the words in the token list and sort by their difficulty
        self.word_diff_df[['length','zipf_freq','score']] = self.word_diff_df["word"].apply(lambda text: self.find_word_difficulty(str(text)))
        self.word_diff_df.sort_values(by = 'score', ascending = True, inplace = True, ignore_index = True)

        # do a boxcox transformation on the word difficulty scores
        self.word_diff_df["boxcox_score"] = stat.boxcox(self.word_diff_df["score"])[0]

        # get the level of the transformed difficulty score
        quantile_ranks = self.find_difficulty_quantiles(self.word_diff_df["boxcox_score"])
        self.word_diff_df["Difficulty"] = list(map(self.find_difficulty_level, quantile_ranks))

        if not word_exo_df.empty:
            # set the difficulty for the word exercises
            average_word_score = word_exo_df["Full_sentence"].apply(lambda text: self.find_wSavg(str(text)))

            # get the difficulty level of the sentences
            quantile_ranks = self.find_difficulty_quantiles(average_word_score)
            word_exo_df["Difficulty"] = list(map(self.find_difficulty_level, quantile_ranks))

            # get the average sentence length for each full sentence in the exercise dataset
            sent_exo_df["Length_sentence"] = sent_exo_df["Full_sentence"].apply(lambda text: self.sentence_length(str(text)))

        if not sent_exo_df.empty:
            # get length of right answers i.e. target words
            sent_exo_df["Length_traget_word"] = sent_exo_df["Right_answer"].apply(lambda text: len(str(text)))

            # get length of the longest word
            sent_exo_df["Length_longest_word"] = sent_exo_df["Full_sentence"].apply(lambda text: self.find_wLengthMax(str(text)))

            # get the difficulty score of the right answers (target words), rarest word in the sentence and the sentence
            # sent_exo_df["Score_target_word"] = sent_exo_df["Right_answer"].apply(lambda text: self.set_word_difficulty(str(text.lower())))
            sent_exo_df["Frequency_rarest_word"] = sent_exo_df["Right_answer"].apply(lambda text: self.find_wSRarest(str(text)))
            sent_exo_df["Score_sentence"] = sent_exo_df["Full_sentence"].apply(lambda text: self.find_SScore(str(text)))

            # get the average difficulty score of the words in the sentences
            # sent_exo_df["Score_sentence_average"] = sent_exo_df["Right_answer"].apply(lambda text: self.find_wSavg(str(text)))

            # get the difficulty level of the sentences
            quantile_ranks = self.find_difficulty_quantiles(sent_exo_df["Score_sentence"])
            sent_exo_df["Difficulty"] = list(map(self.find_difficulty_level, quantile_ranks))

        return word_exo_df.append(sent_exo_df,ignore_index = True)