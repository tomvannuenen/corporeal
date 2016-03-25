CORPOREAL
========================

An interactive command-line Python package for transforming, ordering and visualizing text corpora, for Python 3 on mac.


<h2> Requirements </h2>

Makes use of the following python modules:
numpy, nltk, sklearn, matplotlib, scipy
Run getRequirements (.sh) from terminal to get pip to install these for you.


<h2> Usage </h2>
 
Navigate to the folder Corporeal is installed in in terminal, and enter "python corporeal.py"
For those who don't like command-line manoeuvring: just click the executable (.sh). 

CORPOREAL expects to be placed in a folder that also contains a data folder. It will ask for that folder on startup. User must enter folder starting from current.
E.g. "data/combi"
It also expects data in the data folder to be titled by subcorpus--optionally with an en-dash to separate parts of the subcorpus.
E.g. "Austen.txt, James.txt" or "Austen-1.txt, Austen-112.txt, James-1.txt, James-34.txt"


<h3> Features </h3>

<h4>Chunking</h4>
Segments input text files in smaller size into new folder. User can determine the size of the chunks (number of words). 

<h4>Word count</h4>
Runs through all files and outputs word count per file.

<h4>Top words</h4>
Runs through all files and outputs most-frequent words for the whole corpus. User is asked for number of words to be outputted.

<h4>Word finder</h4>
User is asked for word, output is two graphs:
* The relative frequency of the search word, normalized to the total amount of words per subcorpus;
* The relative frequency of the search word, normalized to the total amount of that word in the entire corpus.
User can opt for a .csv file with these relative frequencies of the word. 

<h4>Lexical variety (means and TTR)</h4>
Calculates and visualizes mean word use and TTF scores. User can choose to make use of regular word tokens or POS-tagged word tokens (using NLTK's Penn Treebank Part-of-Speech Tagset). The output consists of:
* Mean word frequency, i.e. the amount of word tokens divided by the amount of word types. The result is the average amount of times in which a word type is used in that file/subcorpus.
* The Type-Token Ratio (TTR) score, i.e. the amount of word types dividied by the amount of word tokens, and its result multiplied by 100.
* Bar chart output of compared means and TTR scores per subcorpus. 
Note that lexical variety will always be lower if the texts are longer: only if the input corpus consists of files of roughly the same size, some degree of comparison might be possible. If input files are of a significantly different size, chunking up front is recommended.
Computation depends on input: if user inputs list of sperarately names subcorpora (e.g. "austen.txt" and "james.txt"), the script goes through those files; if the user inputs list of chunked subcorpora (e.g. "austen-1" to "austen-100" and "james-1" to "james-100"), the script sorts the files per subcorpora.

<h4>Stemming</h4>
Stems all words per file, outputs stemmed files in a new folder named 
"[userfolder]-stem".

<h4>POS tagging</h4>
POS tags all words per file, outputs tagged files in a new folder named 
"[userfolder]-POS".

<h4>Distinctive words</h4>
User is asked for a subcorpus name (e.g. "Austen"), output is a list of distinctive words for that subcorpus by comparing average rates of that word in that subcorpus vs. the entire corpus. The difference between these rates is calculated as distinciveness.
User can opt for a .csv file with these distinctive words. 

<h4>Euclidian distances</h4>
Calculates Euclidian distances between all files in the data folder based on their word counts. Outputs two graphs based on these distances:
* 2D plot
* Dendogram

<h4>TF-IDF cosine distances</h4>
Calculates cosine distances between all files in the data folder based on theuir TF-IDF scores. Outputs two graphs based on these distances:
* 2D plot
* Dendogram

