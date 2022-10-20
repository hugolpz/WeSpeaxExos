# WeSpeaxExos

## General Information

In this part of the project we have created a class that calculates the difficulty of the exercises (word vocabulary and sentence flashcards/exercises).

## Libraries Used

- numpy
- pandas
- scipy
- wordfreq

<!-- If you have screenshots you'd like to share, include them here. -->

## Setup

```python
pip install -r requirements.txt # (not done yet i.e TBD)
```

## Usage

- Run the cells in `exercise_difficulty.ipynb` which will import the `Difficulty` class and call function to calculate exercise difficulty from class.

- Each section will take the exercise data from a respective language as such Arabic, English and Hindi and extract the new exercise data with difficulty into an excel file.

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

## Class Code Description

### Token List

```python
def get_token_list(self, text)
```

* Tokenizes a sentence (text) using wordfreq library using `wordfreq.tokenize(text, language)` 

* Adds new tokens to the tokens list.

* Parameters: sentence (text) as string

### Word Difficulty

```python
def find_word_difficulty(self, word)
```

* Calculates the word difficulty score of all words from the corpus that is obtained after the tokenization process.

* Parameters: word from the token as string
- Returns: a pandas series of the length (integer), the zipf frequency (float) and the difficulty score (float) of the word.
* The difficulty of a word is calculated based on an adapted version of the algorithm 2 from Jagoda & Boiński (2019) [(PDF) Assessing Word Difficulty for Quiz-Like Game](https://www.researchgate.net/publication/322996917_Assessing_Word_Difficulty_for_Quiz-Like_Game)

* The final formula to calculate the word difficulty is : 

$$
difficulty = \frac{lw}{lmax} * (8 - zipf\_word)
$$

    where :

1. lw : length of the word

2. lmax : the length of the longest word in our corpus

3. zipfword : zipf frequency of the word using `wordfreq` library

>  We subtracted with 8 due to the upper bound of the zipf frequency is 8 and the concept of the formula is to have higher difficulty scores for the combination of longest words and rarest words (less frequent) and lower difficulty scores for short and common (more frequent) words , we used relative length and relative frequency as scaling factors

```python
def find_difficulty_quantiles(self, score_column)
```

* Calculates the difficulty quantile ranks for a difficulty score column

* Parameters: score value column as array

* Returns: the respective quantile rank (between 1 and 32) of each score in the difficulty score column.

```python
def find_difficulty_level(self, q_rank)
```

- Calculates the difficulty level of the words based on their quantile rank.

- Parameters: quantile rank as integer

- Returns: the difficulty level according to the quantile rank as string.

- The inputed quantile rank value (between 1 to 32) is from the quantiles column
  outputed from the function find_difficulty_quantiles, and if the quantile value lies:
  
  - Less than equal to 2nd quantiles: the word gets assigned to A1 level.
  
  - Less than equal to 4th quantiles: the word gets assigned to A2 level.
  
  - Less than equal to 8th quantiles: the word gets assigned to B1 level.
  
  - Less than equal to16th quantiles: the word gets assigned to B2 level.
  
  - Less than equal to32nd quantiles: the word gets assigned to C1 level.

### Word Exercise Difficulty

```python
def set_word_difficulty(self, word)
```

- Find the difficulty level of the word in vocab exercises.

- Parameters: word from vocab exos as string.

- Returns: the word’s respective difficulty level as string.

### Sentence Exercise Difficulty

```python
def sentence_length(self, text)
```

* Calculates the length of the text.

* Parameters: sentence as string

* Returns: the sentence length. as integer

```python
def find_wLengthMax(self, text)
```

* Find the length of the longest word in the sentence

* Parameters: sentence as string

* Returns: length of the longest word as integer

```python
def find_wSRarest(self, text)
```

* Find the frequency of the rarest word in the sentence.

* Parameters: sentence as string

* Returns: frequency of the rarest word in the sentence as float

```python
def find_wSavg(self, text)
```

* Find the average difficulty score of the words in a sentence

* Parameters: sentence as string

* Returns: average difficulty score of the words in a sentence as float

```python
def find_SScore(self, text)
```

* Calculate the sentence difficulty score

* Parameters: sentence as string

* Returns: sentence difficulty score as float 

* The final formula to calculate the sentence difficulty is :

$$ sentence \ length * average \ difficulty \ score \ of \ sentence * frequency \ of \ rarest \ word$$

```python
def find_all_scores(self)
```

* Calculates the score for all exercises in the dataset

* Returns: the exercise dataset with their respective difficulty scores

* How the function works:
  
  * split exercise dataset by exercise types into word and sentence exercises
  
  * find all unique tokens in the exercise dataset
  
  * convert token list into a pandas dataframe
  
  * get the difficulty of the words in the token list and sort by their difficulty
  
  * perform a boxcox transformation on the calculated word difficulty scores
  
  * get the level of the transformed difficulty score using 32 quantile ranks
  
  * if the word exercise dataframe is not empty for each word/phrase in the exercises get the following:
    
    * average difficulty score 
    
    * difficulty level using 32 quantile ranks
  
  * if the sentence exercise dataframe is not empty for each sentence in the exercises get the following:
    
    * average sentence length
    
    * average difficulty score of the words
    
    * frequency of the rarest word
    - difficulty score
    * difficulty level
  
  * concatenate word and sentence exercise dataframes and retrun it

## Project Status

Project is: _**in progress**_, the class code has only been tested for Arabic, English and Hindi
