Corporeal
========================

An interactive command-line Python package for transforming, ordering and visualizing text corpora, for Python 3 (tested only on mac). Current version is pre-alpha, still under construction. Please let me know about any bugs you may find.


<h2> Requirements </h2>

Corporeal uses a few well-known Python packages. Check requirements.txt and use your favorite package manager to install these. Pip-users (https://pypi.python.org/pypi/pip) may run getRequirements (.sh) from terminal to get pip to install the packages.


<h2> Usage </h2>
 
Navigate to the folder Corporeal is located in, and enter `python corporeal.py`. For those who don't like command-line manoeuvring: just click the executable (.sh). 

Corporeal expects to be placed in a folder that also contains a data folder. It will ask for that folder on startup. User should enter a directory name starting from current.

Corporeal uses simple command-line commands. Answering its questions, it expects either a number (indicated by `INT` in the response line), a string (indicated by `STR`) or a simple yes/no (indicated by `Y/N`). Characters are case insensitive.


*E.g. "data/combi"*

Corporeal expects to find .txt files in this data folder, preferrably titled by subcorpus, and optionally with an en-dash to identify separate parts of the subcorpus (if the corpus has not been segmented yet, Corporeal can do that for you).

*E.g. "Austen.txt", "James.txt" or "Austen-1.txt" to "Austen-112.txt"*

Note that Corporeal expects to be used for parsing English texts (in terms of  stopword removal and tokenizing); inputting other languages may yield less desirable results.


<h3> Features </h3>

<h4>Chunking</h4>
Segments input text files in smaller size text files, and places them into new folder. User can determine the size of the chunks (i.e., the number of words). As in all modules, the program normalizes the input texts by case-folding the word tokens and removing punctuation, in order to achieve base equivalence classing. Output files are named in such a way that Corporeal can analyze them through other functions.

<h4>Stemming</h4>
Stems all words per file. Fast, but less reliable than lemmatization. User selects output: 
* One .csv file in the root folder, based on the aggregate stemmed texts, containing the top 100 words; 
* Multiple .txt files with the stemmed texts. These textt files (with their reduced inflectional word forms) may then be used as input for other functions. 

<h4>POS tagging / filtering</h4>
POS tags all words per file. User selects output:
* One .csv file in the root folder, based on the aggregate tagged texts, containing the top 100 words; 
* Multiple .txt files with the tagged texts. These text files (with their reduced inflectional word forms) may then be used as input for other functions. 

User may also choose to filter POS output for nouns, pronouns or verbs. Note however that this can also be done at a later point with the POS filter function.

<h4>POS filter</h4>
POS filter that takes in a folder with general POS-tagged files, created by the POS tagging function described above. It allows the user to filter these files for nouns, pronouns or verbs, and outputs a folder with these filtered files.  

<h4>Lemmatization</h4>
Lemmatizes all words per file. Slower than stemming, but more precise. User selects output:
* One .csv file in the root folder, based on the aggregate lemmatized texts, containing the top 100 words; 
* Multiple .txt files with the lemmatized texts. These text files (with their reduced inflectional word forms) may then be used as input for other functions. 

<h4>Word count</h4>
Runs through all files and outputs word counts per subcorpus, as well as the total.

<h4>Top words</h4>
Runs through all files and outputs most-frequent words for the whole corpus. User can choose to make use of regular word tokens or POS-tagged word tokens, the latter allowing for a search for nouns, pronouns and verbs only. User is asked for number of words to be outputted.

<h4>Word finder</h4>
User is asked for word, output is two graphs:
* The relative frequency of the search word, normalized to the total amount of words per subcorpus;
* The relative frequency of the search word, normalized to the total amount of that word in the entire corpus.
User can opt for a .csv file with these relative frequencies of the word. 

<h4>Concordances</h4>
User is asked for a word. Output is the lexical context of this chosen word per file in the subcorpus. The program iterates randomly through the corpus (instead of alphabetically, as with the other functions): this can be useful if you want to manually check the lexical context of a certain word in the corpus.

<h4>Top clusters (bi- or trigrams)</h4>
User is asked for a word and whether to look for bi- or trigrams. The program searches through the entire corpus and finally outputs the top-N most frequent bi- or trigrams involving the chosen word. User selects how many top bi- or trigrams are found (between 1 and 100).

<h4>Lexical variety (means and TTR)</h4>
Calculates and visualizes mean word use and TTF scores. User can choose to make use of regular word tokens or POS-tagged word tokens, if the user hasn't created a POS-tagged the corpus herself yet (note that this takes significantly longer, and that the user may wish to create POS-tagged files herself first). The output consists of:
* Mean word frequency, i.e. the amount of word tokens divided by the amount of word types. The result is the average amount of times in which a word type is used in that file/subcorpus.
* The Type-Token Ratio (TTR) score, i.e. the amount of word types dividied by the amount of word tokens, and its result multiplied by 100.
* Bar chart output of compared means and TTR scores per subcorpus. 
Note that lexical variety will always be lower if the texts are longer: only if the input corpus consists of files of roughly the same size, some degree of comparison might be possible. If input files are of a significantly different size, chunking up front is recommended.
Computation depends on input: if user inputs list of separately named subcorpora (*e.g. "austen.txt" and "james.txt"*), the script goes through those files; if the user inputs list of chunked subcorpora (*e.g. "austen-1" to "austen-100" and "james-1" to "james-100"*), the script sorts the files per subcorpora.

<h4>Distinctive words</h4>
User is asked for a subcorpus name (e.g. "Austen"), output is a list of distinctive words for that subcorpus by comparing average rates of that word in that subcorpus vs. the entire corpus. The difference between these rates is calculated as distinctiveness. User can opt for a .csv file with these distinctive words. 

<h4>Euclidian distances</h4>
Calculates Euclidian distances between all subcorpora in the data folder based on their word counts. If user inputs split corpora, these will be concatenated into subcorpora before analysis. Output consists of two graphs:
* 2D plot
* Dendogram

<h4>TF-IDF cosine distances</h4>
Calculates cosine distances between all subcorpora in the data folder based on their TF-IDF scores. If user inputs split corpora, they will be concatenated into subcorpora before analysis. Output consists of two graphs:
* 2D plot
* Dendogram