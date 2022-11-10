The **difficulty class** calculates the difficulty of the exercises based on word vocabulary and sentence flashcards/exercises.

## Dependencies

- numpy
- pandas
- scipy
- wordfreq

<!-- If you have screenshots you'd like to share, include them here. -->

## Install

```python
pip install -r requirements.txt # (not done yet i.e TBD)
```

## Usage

- Open `exercise_difficulty.ipynb`, run the cells: it import the difficulty class, calls relevant functions, calculate our exercises' difficulty.
- Each section will take the exercise data from a respective language as such Arabic, English and Hindi
- Export of the new exercise data with difficulty will be send into an excel file.

```python
# import relevant libraries
import pandas as pd

# import class to calculate exercise difficulty
from exercise_difficulty import Difficulty

"""
<< Language >> Exercise Difficulty
"""

# import exercise dataset
exo_df = pd.read_excel("path to exercise excel file")

# define exercise objectives to split by
word_exo_objs = ["Vocabulary"] # objs of word exos
sent_exo_objs = ["Useful Sentences", "Grammar"] # objs of sent exos

# define difficulty class
difficulty_class = Difficulty(
    exo_df = exo_df, 
    language = "language code", 
    word_exo_objs = word_exo_objs, 
    sent_exo_objs = sent_exo_objs
)

# call function to calculate exercise difficulty from class object
complete_exodf = difficulty_class.find_all_scores()

# export the exercise dataset with their respective difficulty scores as an excel file
complete_exo_hi_df.to_excel(
    "path to save exercise difficulty excel file", 
    index = False
)
```

## List of functions

### Define the class

```python
difficulty_class = Difficulty(exo_df, lang_code, word_exo_objs, sent_exo_objs)
```

### Token list

* `difficulty_class.get_token_list(text)`: adds new tokens to the tokens list by tokenizing text using wordfreq library using wordfreq.tokenize(text, lang_code)

### Word difficulty

* `difficulty_class.find_word_difficulty(word)`: returns the length (int), zipf_frequency (float) and difficulty score (float) for word in the token list. <br>The difficulty of a word is calculated based on an adapted version of the algorithm 2 from [Jagoda & Boiński (2019)](https://www.researchgate.net/publication/322996917_Assessing_Word_Difficulty_for_Quiz-Like_Game), with the formula :

    $$
    word \; difficulty = \frac{word \; length}{length \; of \; longest \; word} * (8 - zipf \; frequency \; of \; word)
    $$
    
    > We subtracted with 8 due to the upper bound of the zipf frequency is 8 and the concept of the formula is to have higher difficulty scores for the combination of longest words and rarest words (less frequent) and lower difficulty scores for short and common (more frequent) words , we used relative length and relative frequency as scaling factors.

* `difficulty_class.find_difficulty_quantiles(score_column)`: returns the respective quantile rank (between 1 and 32) of each score in the difficulty score column.

* `difficulty_class.find_difficulty_level(q_rank)`: Returns the difficulty level of a word based on their quantile rank.
  * range: `A1`,`A2`,`B1`,`B2`,`C1`
    If the quantile value is:
    - <= 2nd quantiles: A1 level.
    - <= 4th quantiles: A2 level.
    - <= 8th quantiles: B1 level.
    - <= 16th quantiles: B2 level.
    - <= 32nd quantiles: C1 level.

### Word Exercise Difficulty

* `set_word_difficulty(word)`: returns the word’s respective difficulty level.

### Sentence Exercise Difficulty

* `difficulty_class.sentence_length(text)`: returns the sentence length.
* `difficulty_class.find_wLengthMax(text)`: returns length of the longest word in the sentence.
* `difficulty_class.find_wSRarest(text)`: returns frequency of the rarest word in the sentence.
* `difficulty_class.find_wSavg(text)`: returns average difficulty score of the words in a sentence.
* `difficulty_class.find_SScore(text)`: returns the sentence difficulty score. Its formula is:

    $$
    sentence \ length * average \ difficulty \ score \ of \ sentence * frequency \ of \ rarest \ word
    $$

* `difficulty_class.find_all_scores()`: returns a modified version of the exercise dataset with their respective difficulty scores.

### Theoretical background
`.find_all_scores()` works as follow :
1. splits the exercise dataset by exercise types into word and sentence exercises
2. find all unique tokens in the exercise dataset
3. convert token list into a pandas dataframe
4. get the difficulty of the words in the token list and sort by their difficulty
5. perform a boxcox transformation on the calculated word difficulty scores
6. get the level of the transformed difficulty score using 32 quantile ranks
7. if the word exercise dataframe is not empty for each word/phrase in the exercises get the following:   
   1. average difficulty score
   2. difficulty level using 32 quantile ranks
8. if the sentence exercise dataframe is not empty for each sentence in the exercises get the following:
   1. average sentence length
   2. average difficulty score of the words
   3. frequency of the rarest word
   4. difficulty score
   5. difficulty level
9. concatenate word and sentence exercise dataframes and retruns it

## Project Status
Project is: _**in progress**_, the class code has only been tested for Arabic, English and Hindi.
