#!/usr/bin/env python
import os
from os import listdir
import csv
import re
import numpy as np
from numpy import linspace
import string
from collections import Counter
from collections import OrderedDict
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from unicodedata import category
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('ggplot')

def main():
    print("-------------------------------")
    print("CORPOREAL 0.2, by Tom van Nuenen")
    print("-------------------------------")
#    myDir = 'problogs/data-test'
#    main_menu(myDir)      
    myDir = input("Please enter directory within current, containing .txt files. \nExample: data/data-folder \nIts files may be named with a hypen between subcorpus and id. \nExamples: austen-102.txt, houellebecq-1.txt, fullcorpus.txt\n>>> ")
    if os.path.isdir(myDir):
        fileList, fileNo = list_textfiles(myDir)
        if not fileList:
            print("There are no .txt files in this folder. Exiting...")
            exit()        
        if fileNo == 1:
            print("This folder contains " + str(fileNo) + " file: " + fileList[0].split("/")[-1])
        if fileNo > 1:
            print("This folder contains " + str(fileNo) + " files, from " + fileList[0].split("/")[-1] + " to " + fileList[-1].split("/")[-1] + "\n")
        main_menu(myDir)         
    else:
        print("That folder does not exist. Exiting...")
        exit()

def main_menu(myDir):
    """Main menu the user starts off in"""
    fileList, noFiles = list_textfiles(myDir)
    print("--------------MAIN MENU--------------")
    userInput = input("""Please select:
    [0] for chunking
    [1] for word count
    [2] for top words 
    [3] for word finder
    [4] for lexical variety (means and TTR)
    [5] for stemming
    [6] for POS tagging
    [7] for distinctive words
    [8] for Euclidian distances
    [9] for TF-IDF cosine distances
    [x] to exit \n>>> """)

    if userInput == "0":
        chunking(myDir)
    if userInput == "1":
        word_count(myDir)
    elif userInput == "2":    
        top_words(myDir)
    elif userInput == "3":    
        word_find(myDir)
    elif userInput == "4":
        # If the user is using split files, we go to a different function
        for f in fileList:
            author = f.split("/")[-1]
            if "-" in author:        # We could ask the user for a different escape char at te start too
                lexical_variety_split(myDir)
            else:
                lexical_variety(myDir)
    elif userInput == "5":
        stemmer(myDir)
    elif userInput == "6":
        tagger(myDir)
    elif userInput == "7":
        distinctive(myDir)
    elif userInput == "8":
        euclidian(myDir)
    elif userInput == "9":
        cosine(myDir)
    elif userInput == "x" or "X":
        exit()
    else:
        print("Please try again or x to exit")
        main()

# --- SUPPORTING FUNCTIONS ---

def list_textfiles(directory):
    "Return a list of filenames ending in '.txt' in DIRECTORY. Removes files that are empty."
    textFiles = []
    # We are sorting because different operating systems may list files in different orders
    for fileName in sorted(listdir(directory)):
        if fileName.endswith(".txt"):
            if os.stat(directory + "/" + fileName).st_size > 70:
                textFiles.append(directory + "/" + fileName)
    fileNo = 1 + len(textFiles)
    return textFiles, fileNo

def read_file(filename):
    "Read the contents of FILENAME and return as a string."
    infile = open(filename) 
    contents = infile.read()
    infile.close()
    return contents

def chunking(myDir):
    """Copy text file into new folder with chunks of that text.
    Size of the chunks is determined by user."""
    fileList, noFiles = list_textfiles(myDir)
    chunks = []
    try:
        chunkLength = int(input("how many words should each chunks be? Enter a number\n(suggested: 500 or 1000).\n>>> "))
    except ValueError:
        print("Please enter a number")
        chunking(myDir)
    for filePath in fileList:
        chunk_counter = 0
        texts = split_text(filePath, chunkLength)
        for text in texts:
            chunk = {'text': text, 'number': chunk_counter, 'filename': filePath}
            chunks.append(chunk)
            chunk_counter += 1         
    output_dir = myDir + '-data_chunks'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)     
    for chunk in chunks:
        basename = os.path.basename(chunk['filename']).replace('.txt', '')
        fn = os.path.join(output_dir,
                          "{}{:04d}".format(basename, chunk['number']) + '.txt')
        with open(fn, 'w') as f:
            f.write(str(chunk['text']))

def split_text(filePath, n_words):    
    """Split text into chunks. Used in chunking function"""
    tokens = get_tokens(filePath)  
    chunks = []
    current_chunk_words = []
    current_chunk_word_count = 0
    for word in tokens:
        current_chunk_words.append(word)
        current_chunk_word_count += 1
        if current_chunk_word_count == n_words:
            chunks.append(' '.join(current_chunk_words))
            current_chunk_words = []
            current_chunk_word_count = 0
    chunks.append(' '.join(current_chunk_words) )        
    return chunks

def get_tokens(fn):
    """Get tokens, presented as a list, for analysis"""
    with open(fn, 'r') as f:
        text = f.read()
        lowers = text.lower()
        no_punctuation = ''.join(ch for ch in lowers if category(ch)[0] != 'P')
        # Another way of removing punct, but only works for ASCII
        # no_punctuation = lowers.translate(string.punctuation)
        tokens = no_punctuation.split()
        # Another way of tokenizing, seems less optimal
        # tokens = nltk.word_tokenize(lowers)
        return tokens

def get_POS_tokens(fn):
    """Get POS-tagged tokens, presented as a list, for analysis.
    The tokens are modified based on their POS."""
    # To check which tagger we're using
    # print(nltk.tag._POS_TAGGER)   
    # All the tags in the Penn Treebank Part-of-Speech Tagset
    # See also https://web.stanford.edu/~jurafsky/slp3/9.pdf
    with open(fn, 'r') as f:
        text = f.read()
        lowers = text.lower()
        no_punctuation = ''.join(ch for ch in lowers if category(ch)[0] != 'P')
        tokens = no_punctuation.split()
        pos = nltk.pos_tag(tokens)
        posTokens = []
        for tup in pos:
            posTokens.append(''.join(tup))              
    return posTokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
            
# --- MAIN FUNCTIONS ---

def word_count(myDir):
    fileList, noFiles = list_textfiles(myDir)
    print("WORD COUNTS PER FILE")
    print('%-*s %s' % (20, "Word", "Frequency"))
    print("------------------------------")
    totalWordCount = 0
    for filePath in fileList:
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]
        myWordCounter = 0
        tokens = get_tokens(filePath)  
        print("%-*s %i" % (20, fName, len(tokens)))
        totalWordCount += len(tokens)
    print("\nTotal word count")
    print(str(totalWordCount) + "\n")

def top_words(myDir):
    """finds top N words, based on user input"""
    inp = input("How many top words should I find? \n>>> ")
    try:
       val = int(inp)
    except ValueError:
       print("Please enter a number!")
       top_words()
    condCSV = 0
    while condCSV == 0:
        userAns = input("Do you want a .csv file with the top words?\n>>> ").lower()
        if userAns == "no" or userAns == "n":
            condCSV = 1
        elif userAns == "yes" or userAns == "y":
            outFile = "top_words.csv"
            f = open(outFile, "a", newline='')
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow( ("word", "count") )    
            condCSV = 2
        else:
            continue
    allWords = []
    fileList, noFiles = list_textfiles(myDir)
    for filePath in fileList:
        with open(filePath, 'r') as f:
            text = f.read()
        wordList = text.lower().split()
        allWords.extend(wordList)
    # Create a frequency distribution
    allWords = [word for word in allWords if len(word) > 1]
    condStop = 0
    while condStop == 0:
        inp2 = input("Remove stopwords? (EN only)\n>>> ").lower()
        valid = ["yes", "y", "no", "n"]
        if inp2 in valid:
            if inp2 == "no" or inp2 == "n":
                words = allWords
                condStop = 1
            elif inp2 == "yes" or inp2 == "y":
                stopWords = set(stopwords.words('english'))
                cleanWords = [w for w in allWords if not w in stopWords]
                words = cleanWords
                condStop = 2    
        else:
            print("please enter 'yes' or 'no'!")
            continue
    fdist = nltk.FreqDist(words)
    # Show the top N words in the list, with counts
    print("TOP N WORDS IN CORPUS")
    print('%-*s %s' % (20, "Word", "Frequency"))
    print("------------------------------")
    for word, frequency in fdist.most_common(int(inp)):
        print('%-*s %d' % (20, word, frequency))        
        if condCSV == 2:
            with open(outFile, "a", newline='') as f:
                writer.writerow( (word, frequency) )
    exit()
            
def word_find(myDir):
    """finds word of choice; generates a .csv file with (relative) frequencies,
    as well as two plots"""
    myWord = input("What word should I look for?\n>>> ").lower()    
    cond = 0
    while cond == 0:
        userFile = input("Do you want a .csv file with (relative) frequencies of the word?\n>>> ").lower()
        valid = ["yes", "y", "no", "n"]
        if userFile in valid:
            if userFile == "yes" or userFile == "y":
                outFile = "relative_word_freq-" + myWord + ".csv"
                f = open(outFile, "a", newline='')
                writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow( ("filename", "wordcount", "word: %s" % myWord, "relFreq: %s" % myWord, "relFreqTotal: %s" % myWord) )    
                cond = 2
            elif userFile == "no" or userFile == "n":
                cond = 1
    totalMyWord = 0
    totalWords = 0
    relFreqList = []
    relFreqTotalList = []
    fList = []
    fileList, noFiles = list_textfiles(myDir)
    for filePath in fileList:
        myWordCounter = 0
        words = get_tokens(filePath)  
        wordCount = len(words)
        totalWords += wordCount
        for w in words:
            if myWord in w:
                    myWordCounter += 1
        totalMyWord += myWordCounter
    for filePath in fileList:
        myWordCounter = 0
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]
        fList.append(fName)
        words = get_tokens(filePath)
        wordCount = len(words)
        for w in words:
            if myWord in w:
                    myWordCounter += 1
        relFreq = ((myWordCounter / wordCount) * 100)
        relFreqList.append(relFreq)
        relFreqTotal = ((myWordCounter / totalMyWord) * 100)
        relFreqTotalList.append(relFreqTotal)
        if cond == 2:
            with open(outFile, "a", newline='') as f:
                writer.writerow( (fName, wordCount, myWordCounter, relFreq, relFreqTotal) )
    
    print("Generating output figure 1...\n")
    # Output figure 1: the search word normalized to the total no. of words in the subcorpus
    fig, ax = plt.subplots()
    N = len(relFreqList)
    x = np.arange(1, N+1)
    y = [num for num in relFreqList]
    labels = [s for s in fList]
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("relative frequency")
    plt.xticks(x - 1 + width, labels, rotation='30')
    # Set 'minor ticks' so that they are located halfway between the major ticks
    # First, hide major tick labels
    #ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # Then, customize minor tick labels
    #ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    #ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    # Auto-limiting graph to the size of the plot
    plt.xlim(1, len(fList) + 1) 
    plt.title("Word: %s" % myWord + ", normalized to total no. words in subcorpus", fontsize=15)    
    plt.show()
    
    print("Generating output figure 2...\n")
    # Output figure 2: the search word normalized to the total no. of that word in the whole corpus    
    labels = [s for s in fList]

    sizes = [num for num in relFreqTotalList]
    colors = ["lightcoral", "yellowgreen", "gold", "lightskyblue"]
    # if we want, we could explode the largest number
    #explodeList = []
    #for i in relFreqTotalList:
    #    if i != max(relFreqTotalList):
    #        explodeList.append(0)
    #    elif i == max(relFreqTotalList):
    #        explodeList.append(0.1)
    #explode = tuple(explodeList)
    # add explode=explode in next param
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)
    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.3,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title("Word: '%s'" % myWord + ", normalized to frequency of '%s'" % myWord + " in total corpus", fontsize=15, y=1.08)    
    plt.show()    
    exit()

def lexical_variety(myDir):
    """Calculates and visualizes mean word use and TTF scores"""
    print("Found unsplit subcorpora in folder. If the files are big, consider splitting them using chunking")
    fileList, noFiles = list_textfiles(myDir)
    cond = 0
    while cond == 0:
        userFile = input("Do you want a .csv file with means and TTF scores per file?\n>>> ").lower()
        valid = ["yes", "y", "no", "n"]
        if userFile in valid:
            if userFile == "yes" or userFile == "y":
                outFile = "lexical_variety.csv"
                f = open(outFile, "a", newline='')
                writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow( ("filename", "mean", "TTF") )    
                cond = 2
            elif userFile == "no" or userFile == "n":
                cond = 1
        else:
            print("Please try again.")
    totalWordCounter = 0
    totalTypeCounter = 0
    allMeans = {}
    allTTR = {}
    tokensCond = 0
    while tokensCond == 0:
        userTokens = input("Do you want to use [1] regular tokens or [2] POS-tagged tokens?\n>>> ")
        valid = ["1", "2"]
        if userTokens in valid:
            tokensCond = int(userTokens)
        else:
            print("Please try again.")

    print("LEXICAL VARIETY PER FILE")
    print('%-*s %-*s %s' % (30, "File Name", 20, "Mean Word Freq", "Type-Token Ratio"))       
    for filePath in fileList:
        tokenCounter = 0
        typeCounter = 0
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]        
        if tokensCond == 1:
            tokens = get_tokens(filePath)
        else:
            tokens = get_POS_tokens(filePath)            
        tokenCounter += len(tokens)    
        typeCounter = len(set(tokens))
        totalWordCounter += len(tokens)
        totalTypeCounter += len(set(tokens))
        mean = tokenCounter / typeCounter # average times in which word types are used
        allMeans[fName] = mean
        TTR = typeCounter / tokenCounter
        allTTR[fName] = TTR
        print('%-*s %-*f %f' % (30, fName, 20, mean, TTR*100))        
        if cond == 2:
            with open(outFile, "a", newline='') as f:
                writer.writerow( (fName, mean, TTR*100) )
    # Calculate total mean value based on the counting we've been doing
    totalMeans = totalWordCounter / totalTypeCounter
    # Dict comprehension to normalize values, subtracting total mean from the mean of every text
    allMeans = {key:value-totalMeans for key, value in allMeans.items()} 
    
    # Plotting means, sorted by values
    fig, ax = plt.subplots()
    sortedMeans = OrderedDict(sorted(allMeans.items(), key=lambda t: t[1]))
    N = len(sortedMeans)
    x = np.arange(1, N+1)
    y = [num for num in sortedMeans.values()]
    labels = sorted(allMeans, key=allMeans.get, reverse=False)
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("mean word use")
    # Hide major format
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # Insert minor format
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    plt.xlim(1, len(sortedMeans) +1)
    plt.title("Lexical repetitiveness by normalized mean word use per file, ordered by value", fontsize=14, y=1.03)    
    plt.show()

    # Plotting TTR, sorted by values
    fig, ax = plt.subplots()
    sortedTTR = OrderedDict(sorted(allTTR.items(), key=lambda t: t[1]))
    N = len(sortedTTR)
    x = np.arange(1, N+1)
    y = [num for num in sortedTTR.values()]
    labels = sorted(allTTR, key=allTTR.get, reverse=False)
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("TTR")
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    plt.xlim(1, len(sortedTTR) +1)
    plt.title("Lexical variety by TTR value per file, ordered by value", fontsize=14, y=1.03)    
    plt.show()
    exit()
    
def lexical_variety_split(myDir):
    """Calculates and visualizes mean word use and TTF scores. The parts of the subcorps
    are organized per subcorpus"""
    print("Found split subcorpora in folder. Will concatenate for evaluation.")
    fileList, noFiles = list_textfiles(myDir)
    cond = 0
    while cond == 0:
        userFile = input("Do you want a .csv file with means and TTF scores per file?\n>>> ").lower()
        valid = ["yes", "y", "no", "n"]
        if userFile in valid:
            if userFile == "yes" or userFile == "y":
                outFile = "lexical_variety_split.csv"
                f = open(outFile, "a", newline='')
                writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow( ("filename", "mean", "TTF") )    
                cond = 2
            elif userFile == "no" or userFile == "n":
                cond = 1
        else:
            print("Please try again.")
    totalWordCounter = 0
    totalTypeCounter = 0
    allMeans = {}
    allTTR = {}
    tokensCond = 0
    userTokens = input("Do you want to use [1] regular tokens or [2] POS-tagged tokens?\n>>> ")
    valid = ["1", "2"]
    if userTokens in valid:
        tokensCond = int(userTokens)
    else:
        print("Please try again.")
    totalWordCounter = 0
    totalTypeCounter = 0
    allMeans = {}
    allTTR = {}
    listIndex = 0
    myDict = {}
    myList = []     
    # Starts the loop that will put all the authors/subcorpora in their separate lists inside myDict
    for filePath in fileList:
        fName = filePath.split("/")[-1].split("-")[0]        
        if fName not in myList:
            author = fileList[listIndex].split("/")[-1].split("-")[0]
            myDict[fName] = listFromAuthor(author, fileList)        
            myList.append(fName)
        listIndex += 1
    meansDict = {}
    TTRDict = {}
    for key, values in myDict.items():
        authorTTR = []
        authorMean = []
        for value in values:
            tokenCounter = 0
            typeCounter = 0
            fSmall = os.path.split(value)[1] 
            fName = os.path.splitext(fSmall)[0]        
            if tokensCond == 1:
                tokens = get_tokens(value)
            else:
                tokens = get_POS_tokens(value) 
            tokenCounter += len(tokens)    
            typeCounter = len(set(tokens))
            totalWordCounter += len(tokens)
            totalTypeCounter += len(set(tokens))
            mean = tokenCounter / typeCounter
            TTR = typeCounter / tokenCounter
            allTTR[fName] = TTR 
            authorMean.append(mean)
            authorTTR.append(TTR)
        meansDict[key] = sum(authorMean) / len(authorMean)
        TTRDict[key] = sum(authorTTR) / len(authorTTR)
    print("LEXICAL VARIETY PER SUBCORPUS")
    print('%-*s %-*s %s' % (30, "Corpus Name", 20, "Mean Word Freq", "Type-Token Ratio"))
    for (k,v), (k2,v2) in zip(meansDict.items(), TTRDict.items()):
        print('%-*s %-*f %f' % (30, k, 20, v, v2))     
        if cond == 2:
            with open(outFile, "a", newline='') as f:
                writer.writerow( (k, v, v2) )
    # Calculate total means
    totalMeans = totalWordCounter / totalTypeCounter
    # Dict comprehension to normalize values, recalculating meansDict
    meansDict = {key:value-totalMeans for key, value in meansDict.items()}

    # Plotting means, sorted by values
    fig, ax = plt.subplots()
    sortedMeans = OrderedDict(sorted(meansDict.items(), key=lambda t: t[1]))
    N = len(sortedMeans)
    x = np.arange(1, N+1)
    y = [num for num in sortedMeans.values()]
    labels = sorted(meansDict, key=meansDict.get, reverse=False)
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("mean word use")
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    plt.xlim(1, len(sortedMeans) +1)
    plt.title("Lexical repetitiveness by normalized mean word use per subcorpus, ordered by value", fontsize=14, y=1.03)    
    plt.show()
        
    # Plotting TTF, sorted by values
    fig, ax = plt.subplots()
    sortedTTR = OrderedDict(sorted(TTRDict.items(), key=lambda t: t[1]))
    N = len(sortedTTR)
    x = np.arange(1, N+1)
    y = [num for num in sortedTTR.values()]
    labels = sorted(TTRDict, key=TTRDict.get, reverse=False)
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("TTR")
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    plt.xlim(1, len(sortedTTR) +1)
    plt.title("Lexical repetitiveness by TTR score per subcorpus, ordered by value", fontsize=14, y=1.03)    
    plt.show()
    exit()

def listFromAuthor(author, fileList):
    myDict = {}
    myDict[author] = []  
    for filePath in fileList:
        j = filePath.split("/")[-1].split("-")[0]
        if j == author:
            myDict[author].append(filePath)
    return myDict[author]

def stemmer(myDir):
    """Stems words. Creates directory in current directory with stemmed files."""
    stemDir = myDir + "-stem"                    
    if not os.path.exists(stemDir):
        os.makedirs(stemDir)    
    fileList, noFiles = list_textfiles(myDir)
    for filePath in fileList:
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]
        tokens = get_tokens(filePath)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        stemmer = SnowballStemmer("english")
        stemmed = stem_tokens(filtered, stemmer)
        count = Counter(stemmed)
        top = count.most_common(100)    
        with open(os.curdir + "/" + stemDir + "/" + fName + "-stemmed" + ".csv", "a", newline='') as f:
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            for j in top:
                writer.writerow(j)
    exit()

def tagger(myDir):
    """POS tags words. Creates directory in current directory with POS tagged files."""
    posDir = myDir + "-POS"               
    if not os.path.exists(posDir):
        os.makedirs(posDir)  
    fileList, noFiles = list_textfiles(myDir)
    for filePath in fileList:
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]
        tokens = get_tokens(filePath)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        pos = nltk.pos_tag(filtered)
        count = Counter(pos)
        top = count.most_common(100)         
        with open(os.curdir + "/" + posDir + "/" + fName + "-POS" + ".csv", "a", newline='') as f:
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            for i in top:
                writer.writerow(i)
    exit()
    
def distinctive(myDir):
    """Compare the average rate at which words are used in (A) the user-defined subcorpus and (B) the
    rest of the corpus. We calculate the difference between the rates to calculate their
    distinciveness in the subcorpus that the user enters"""
    fileList, noFiles = list_textfiles(myDir)
    userName = input("Please enter name of subcorpus you want to see distinctive features of\n (e.g. 'Austen')\n>>> ").lower()  
    testDir = []
    for i in fileList:
        fSmall = os.path.split(i)[1] 
        fName = os.path.splitext(fSmall)[0]
        try: 
            i = fName.split("-")[0]
            testDir.append(i)
        except:
            testDir.append(i)
    if userName not in testDir:
        print("That name does not seem to exist in this folder.. Try again.")
        distinctive(myDir)
    userNo = input("How many distinctive features should I find?\n>>> ").lower()
    try:
       val = int(userNo)
    except ValueError:
       print("Please enter a number!")
       distinctive(myDir)    
    condCSV = 0
    while condCSV == 0:
        userAns = input("Do you want a .csv file with the top words?\n>>> ").lower()
        if userAns == "no" or userAns == "n":
            condCSV = 1
        elif userAns == "yes" or userAns == "y":
            outFile = "distinctive_words-" + userName + ".csv"
            # We could add a condition here that if the file exists, ask user to overwrite            
            f = open(outFile, "a", newline='')
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow( ("subcorpus = " + userName,) )
            writer.writerow( ("word", "keyness") )    
            condCSV = 2
        else:
            continue  
    vectorizer = CountVectorizer(input='filename') # Input expects a list of filenames
    dtm = vectorizer.fit_transform(fileList)
    vocab = np.array(vectorizer.get_feature_names())
    dtm = dtm.toarray()
    rates = 1000 * dtm / np.sum(dtm, axis=1, keepdims=True)
    userNameIndices = []
    otherNamesIndices = []
    fileList, noFiles = list_textfiles(myDir)
    for index, fn in enumerate(fileList):
        if userName in fn:
            userNameIndices.append(index)
        else:
            otherNamesIndices.append(index)
    userNameRates = rates[userNameIndices, :]
    otherNamesRates = rates[otherNamesIndices, :]        
    userNameRatesAvg = np.mean(userNameRates, axis=0)
    otherNamesRatesAvg = np.mean(otherNamesRates, axis=0)
    distinctive_indices = (userNameRatesAvg * otherNamesRatesAvg) == 0
    ranking = np.argsort(userNameRatesAvg[distinctive_indices] + otherNamesRatesAvg
                     [distinctive_indices])[::-1] # from highest to lowest; [::-1] reverses order
    dtm = dtm[:, np.invert(distinctive_indices)]
    rates = rates[:, np.invert(distinctive_indices)]
    vocab = vocab[np.invert(distinctive_indices)]
    # recalculate variables that depend on rates
    userNameRates = rates[userNameIndices, :]
    otherNamesRates = rates[otherNamesIndices, :]
    userNameRatesAvg = np.mean(userNameRates, axis=0)
    otherNamesRatesAvg = np.mean(otherNamesRates, axis=0)
    keyness = np.abs(userNameRatesAvg - otherNamesRatesAvg)
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order
    rates_avg = np.mean(rates, axis=0)
    keyness = np.abs(userNameRatesAvg - otherNamesRatesAvg) / rates_avg
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order.
    topKeyness = [i for i in keyness[ranking][0:int(userNo)]]
    topVocab = [i for i in vocab[ranking][0:int(userNo)]]
    d = dict(zip(topVocab, topKeyness))
    print("TOP " + userNo + " DISTINCTIVE WORDS PER SUBCORPUS '" + str(userName).upper() + "' BY COMPARING AVG RATES")     
    print("%-*s %s" % (20, "Word", "Keyness"))
    for k, v in d.items():
        print("%-*s %f" % (20, k, v))
        if condCSV == 2:
            with open(outFile, "a", newline='') as f:
                for k, v in d.items():
                    writer.writerow( (k,v) )  
    exit()    
    
    # Note that as of now this only works for 1 subcorpus; it might be nice to be able to
    # compare the distinctiveness for all subcorpora. 
    # What we then need to do is to calculate the average rate of word use across all texts for
    # each author, and then look for cases where the average rate is zero for one author.
    
    
def euclidian(myDir):
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.manifold import MDS
    from scipy.cluster.hierarchy import ward, dendrogram
    import matplotlib.pyplot as plt

    fileList, noFiles = list_textfiles(myDir)    
    vectorizer = CountVectorizer(input='filename', stop_words='english', min_df=5)
    dtm = vectorizer.fit_transform(fileList)  # a sparse matrix
    vocab = vectorizer.get_feature_names()  # a list    
    dtm = dtm.toarray()  # convert to a regular array
    vocab = np.array(vocab)
    dist = euclidean_distances(dtm)
    
    #2D plot
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    names = [os.path.basename(fn).replace('.txt', '') for fn in fileList] 
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "spotted" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)
    plt.title("2D Euclidian distances between subcorpora")
    plt.show()
    
    #dendogram
    linkage_matrix = ward(dist)
    dendrogram(linkage_matrix, orientation="right", labels=names);
    plt.tight_layout()  # fixes margins
    plt.title("Dendogram")
    plt.show()
    exit()
    
def cosine(myDir):
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import MDS
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.cluster.hierarchy import ward, dendrogram
    from sklearn.feature_extraction.text import TfidfVectorizer

    fileList, noFiles = list_textfiles(myDir)
    vectorizer = TfidfVectorizer(input='filename', stop_words='english', min_df=5)
    tfidf = vectorizer.fit_transform(fileList)
    dtm = tfidf.toarray()

    # Cosine similarity is a measure of similarity so we need to 'flip' the measure so that a
    # larger angle receives a larger value. The distance measure derived from cosine similarity
    # is thus one minus the cosine similarity between two vectors.
    dist = 1 - cosine_similarity(dtm)

    # 2d plot
    mds = MDS(n_components=2, dissimilarity="precomputed")
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    names = [os.path.basename(fn).replace('.txt', '') for fn in fileList]
    for x, y, name in zip(xs, ys, names):
        color = 'blue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)  
    plt.title("2D cosine distances between subcorpora")
    plt.show()
    
    #dendogram
    linkage_matrix = ward(dist)
    dendrogram(linkage_matrix, orientation="right", labels=names);
    plt.tight_layout()  # fixes margins
    plt.title("Dendogram")
    plt.show()
    exit()

if __name__ == '__main__':
    main()