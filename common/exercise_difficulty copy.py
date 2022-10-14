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

class Difficulty:

    def __init__(self, exo_df, language, word_exo_objs, sent_exo_objs):
        self.exo_df = exo_df
        self.language = language
        self.word_exo_objs = word_exo_objs
        self.sent_exo_objs = sent_exo_objs

        self.quantiles = 32
        self.token_list = []
        self.word_diff_df = pd.DataFrame()

        stanza.download(language)
        self.nlp = stanza.Pipeline(language)

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
            if token not in self.token_list:
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
        return length_word, zipf_word, score

    def transform_difficulty(self, score_column):
        """
        Boxcox Transformation of the difficulty scores of the words.
        """
        return stat.boxcox(score_column)

    def find_difficulty_quantiles(self, score_column):
        return pd.qcut(
            x = score_column, 
            q = self.quantiles, 
            labels = False, 
            retbins = True,
            duplicates = "drop"
        )

    def find_difficulty_level(self, quantile_ranks):
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

    def set_word_difficulty(self, text):
        score = self.word_diff_df[self.word_diff_df["word"] == text]["boxcox_score"]
        difficulty = self.word_diff_df[self.word_diff_df["word"] == text]["difficulty"]

        return score, difficulty

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
        doc = self.nlp(text)
        return sum(len(sent.tokens) for sent in doc.sentences)

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
        word_list = list(set(re.findall(r"[\w\='‘’]+", text.lower())))
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

        word_list = list(set(re.findall(r"[\w\='‘’]+", text.lower())))
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
        word_list = list(set(re.findall(r"[\w\='‘’]+", text.lower())))
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
        return self.avg_sentence_length(text) * self.find_wSavg(text) * self.find_wSRarest(text)
    


    def find_all_scores(self):
        # split the exercise dataset by exercise types
        word_exo_df = self.exo_df[self.exo_df["Exo_objective"].isin(self.word_exo_objs)]
        sent_exo_df = self.exo_df[self.exo_df["Exo_objective"].isin(self.sent_exo_objs)]

        # find all the tokens in the exercise dataset
        for i in range(len(word_exo_df)):
            self.get_token_list(word_exo_df["Full_sentence"][i])

        for i in range(len(sent_exo_df)):
            self.get_token_list(sent_exo_df["Full_sentence"][i])

        # make the token list into a pandas dataframe
        self.word_diff_df = pd.DataFrame({"word": self.token_list})

        # get the difficulty of the words in the token list and sort by their difficulty
        for i in range(len(self.word_diff_df)):
            results = self.find_word_difficulty(self.word_diff_df["word"][i])
            self.word_diff_df['length'][i] = results[0]
            self.word_diff_df['zipf_freq',][i] = results[1]
            self.word_diff_df['score'][i] = results[2]

        # do a boxcox transformation on the word difficulty scores
        for i in range(len(self.word_diff_df)):
            self.word_diff_df["boxcox_score"][i] = self.transform_difficulty(self.word_diff_df["score"][i])

        # get the level of the transformed difficulty score
        quantile_ranks = [
            self.find_difficulty_quantiles(self.word_diff_df["boxcox_score"][i]) 
            for i in range(len(self.word_diff_df))
        ]

        self.word_diff_df["Difficulty"] = self.find_difficulty_level(quantile_ranks)

        # set the difficulty for the word exercises
        for i in range(len(word_exo_df)):
            results = self.set_word_difficulty(word_exo_df["Full_sentence"][i])
            word_exo_df["Score_target_word"] = results[0]
            word_exo_df["Difficulty"] = results[1]
        
        for i, text in enumerate(word_exo_df["Full_sentence"]):
            word_exo_df["boxcox_score"][i] = self.word_diff_df[self.word_diff_df["word"] == text]["boxcox_score"][i]
            word_exo_df["Difficulty"][i] = self.word_diff_df[self.word_diff_df["word"] == text]["Difficulty"][i]
        
        for i in range(len(sent_exo_df)):
            # get the average sentence length for each full sentence in the exercise dataset
            sent_exo_df["Length_sentence"][i] = self.avg_sent_length(sent_exo_df["Full_sentence"][i])

            # get length of right answers i.e. target words
            sent_exo_df["Length_traget_word"][i] = len(sent_exo_df["Right_answer"][i])

            # get length of the longest word
            sent_exo_df["Length_longest_word"][i] = self.find_wLengthMax(sent_exo_df["Full_sentence"][i])

            # get the difficulty score of the right answers (target words), rarest word in the sentence and the sentence
            sent_exo_df["Score_target_word"][i] = self.set_word_difficulty(sent_exo_df["Right_answer"][i])[0]
            sent_exo_df["Score_rarest_word"][i] = self.find_wSRarest(sent_exo_df["Right_answer"][i])
            sent_exo_df["Score_sentence"][i] = self.find_SScore(sent_exo_df["Right_answer"][i])

            # get the average difficulty score of the words in the sentences
            sent_exo_df["Score_sentence_average"] = sent_exo_df["Right_answer"].apply(lambda text: self.find_wSavg(text))

        # get the level of the transformed difficulty score
        quantile_ranks = [
            self.find_difficulty_quantiles(self.sent_exo_df["Score_sentence"][i]) 
            for i in range(len(self.word_diff_df))
        ]
        sent_exo_df["Difficulty"] = self.find_difficulty_level(quantile_ranks)

        return word_exo_df, sent_exo_df

data = pd.read_excel("../en/English_Exercises.xlsx")

# # Spellchecker using pyspellchecker
# # After little test using pyspellchecker, textblob and autocorrect, pyspellchecker performs better

# spell = SpellChecker()
# def correct_spelling(text):
#     corrected_text = []
#     misspelled_words = spell.unknown(tokenize(text,'en'))
#     for word in text.split():
#         next_word = word

#         if word in misspelled_words:
#             next_word = spell.correction(word)

#         if next_word is not None :
#             corrected_text.append(next_word)
#         else:
#             corrected_text.append(word)


#     return " ".join(corrected_text)


# # second package which performs well is autocorrect
# # we pass the code through 2 spellcheckers to create an idea of an "ensemble"
# speller = Speller(lang = 'en')

# def autocorrect_speller(text):
#     return speller(text)

# # spellcheck
# data["Full_sentence"] = data["Full_sentence"].apply(lambda x:correct_spelling(x))
# data["Full_sentence"] = data["Full_sentence"].apply(lambda x:autocorrect_speller(x))

word_exo_objs = ["Learning vocabulary"]
sent_exo_objs = ["Useful Sentences", "Grammar", "Verb_Conjugation"]

difficulty_class = Difficulty(exo_df = data, language = "en", word_exo_objs = word_exo_objs, sent_exo_objs = sent_exo_objs)

word_exo_df, sent_exo_df = difficulty_class.find_all_scores()

print(word_exo_df.head())
print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
print(sent_exo_df.head())