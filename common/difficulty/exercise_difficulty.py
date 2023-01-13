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
        text : string
            Sentence as text string.
        """

        tokens = tokenize(text, self.language)

        for token in tokens:
            self.token_list.extend(tokens)
    
    def find_word_difficulty(self, word):
        """
        Calculate the average sentence length.
        Parameters
        ----------
        text : string
            word from tiken list as string.
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
        """
        Calculates the 32 quantile ranks for a score column
        Parameters
        ----------
        score_column: array
            The array of scores.
        Returns
        -------
        the 32 quantile ranks for the score column
        """
        return pd.qcut(score_column, 32, labels=False, duplicates = "drop")

    def find_difficulty_level(self, q_rank):
        """
        Calculate the difficulty level of the words based on their quantile rank.
        Parameters
        ----------
        q_rank: array
            The array of quantile ranks.
        Returns
        -------
        the difficulty level according to the quantile ranks.
        """
        
        if q_rank < 2:
            return 'A1'
        if q_rank < 4:
            return 'A2'
        if q_rank < 8:
            return 'B1'
        
        return 'B2' if q_rank < 16 else 'C1'
    
    def set_word_difficulty(self, word):
        """
        Find the difficulty level of the words in vocab exos.
        Parameters
        ----------
        word: string
            The word from vocab exos as string.
        Returns
        -------
        the difficulty level of the word.
        """
        return str(self.word_diff_df[self.word_diff_df["word"] == word.lower()]["Difficulty"].values[0])

    def sentence_length(self, text):
        """
        Calculate the length of the text.
        Parameters
        ----------
        text : str
            Sentence as text string.
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
        text : string
            Sentence as text string.
        Returns
        -------
        int
            The length of the longest word.
        """
        word_list = list(set(tokenize(text, self.language)))
        return len(max(word_list, key = len))

    def find_wSRarest(self, text):
        """
        Find the frequency of the rarest word in the sentence
        Parameters
        ----------
        text : string
            Sentence as text string.
        Returns
        -------
        float
            The frequency of the rarest word.
        """

        word_list = list(set(tokenize(text, self.language)))
        word_freq = [zipf_frequency(word, self.language) for word in word_list]
        return 8 - min(word_freq) if word_freq else 8 - 0

    def find_wSavg(self, text):
        """
        Find the average difficulty score of the words in a sentence
        Parameters
        ----------
        text : string
            Sentence as text string.
        Returns
        -------
        float
            The average difficulty score of the words.
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
        text : string
            Sentence as text string.
        Returns
        -------
        float
            The sentence difficulty score.
        """
        return (self.sentence_length(text) / self.max_sent_length) * self.find_wSavg(text) * self.find_wSRarest(text)
    
    def get_right_answer(self, propositions, index):
        """
        Find the right answer of the sentence.
        Parameters
        ----------
        propositions
            The propositions of the sentence.
        index
            The index of the right answer.
        Returns
        -------
        string
            The right answer of the sentence.
        """
        return propositions.split('-')[int(index)]
    
    def find_all_scores(self):
        """
        Calculates the score for all exercises in the dataset.
        Returns
        -------
        DataFrame
            The exercise dataset with their respective difficulty scores.
        """

        # split exercise dataset by exercise types
        word_exo_df = self.exo_df[self.exo_df["Exo_objective"].isin(self.word_exo_objs)]
        sent_exo_df = self.exo_df[self.exo_df["Exo_objective"].isin(self.sent_exo_objs)]

        # find all unique tokens in the exercise dataset
        word_exo_df["Full_sentence"].apply(lambda text: self.get_token_list(str(text)))
        sent_exo_df["Full_sentence"].apply(lambda text: self.get_token_list(str(text)))

        # convert token list into a pandas dataframe
        self.token_list = list(set(self.token_list))
        self.word_diff_df = pd.DataFrame({"word": self.token_list})

        # get the difficulty of the words in the token list and sort by their difficulty
        self.word_diff_df[['length','zipf_freq','score']] = self.word_diff_df["word"].apply(lambda text: self.find_word_difficulty(str(text)))
        self.word_diff_df.sort_values(by = 'score', ascending = True, inplace = True, ignore_index = True)

        # perform a boxcox transformation on the calculated word difficulty scores
        self.word_diff_df["boxcox_score"] = stat.boxcox(self.word_diff_df["score"])[0]

        # get the level of the transformed difficulty score
        quantile_ranks = self.find_difficulty_quantiles(self.word_diff_df["boxcox_score"])
        self.word_diff_df["Difficulty"] = list(map(self.find_difficulty_level, quantile_ranks))

        if not word_exo_df.empty:
            # get average difficulty score for each word/phrase in the word exercises
            word_exo_df["Score_sentence"] = word_exo_df["Full_sentence"].apply(lambda text: self.find_wSavg(str(text)))

            self.max_sent_length = max(list(word_exo_df["Score_sentence"]))

            # get the difficulty level of the word exercises
            quantile_ranks = self.find_difficulty_quantiles(word_exo_df["Score_sentence"])
            word_exo_df["Difficulty"] = list(map(self.find_difficulty_level, quantile_ranks))

        if not sent_exo_df.empty:
            # get the average sentence length for each full sentence
            sent_exo_df["Length_sentence"] = sent_exo_df["Full_sentence"].apply(lambda text: self.sentence_length(str(text)))

            # get the right answer for each sentence
            right_answers = sent_exo_df.apply(lambda x: self.get_right_answer(x["Propositions"], x["Right_answer_id"]), axis=1)

            # get the average difficulty score of the words in each sentences
            # sent_exo_df["Score_sentence_average"] = sent_exo_df["Right_answer"].apply(lambda text: self.find_wSavg(str(text)))
            sent_exo_df["Score_sentence_average"] = right_answers.apply(lambda text: self.find_wSavg(str(text)))
            
            # get the frequency of the rarest word in each sentence
            sent_exo_df["Frequency_rarest_word"] = sent_exo_df["Right_answer"].apply(lambda text: self.find_wSRarest(str(text)))
            
            # get the difficulty score of each sentence
            # sent_exo_df["Score_sentence"] = sent_exo_df["Full_sentence"].apply(lambda text: self.find_SScore(str(text)))
            sent_exo_df["Frequency_rarest_word"] = right_answers.apply(lambda text: self.find_wSRarest(str(text)))

            # get the difficulty level of each sentence
            quantile_ranks = self.find_difficulty_quantiles(sent_exo_df["Score_sentence"])
            sent_exo_df["Difficulty"] = list(map(self.find_difficulty_level, quantile_ranks))
            
        df_list = [word_exo_df, sent_exo_df]
        
        return pd.concat(df_list)
