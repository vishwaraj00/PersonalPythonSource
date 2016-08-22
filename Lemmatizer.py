#!/usr/bin/python 
from __future__ import division, unicode_literals
from textblob import TextBlob as tb
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import numpy as np
import os, pdb, re, subprocess, math, time

raw_folder = 'raw_data/'
norm_folder = 'norm_data/'

def run(command):
    '''
    Function for running shel commands, returns command output (python 2.7 or higher version)
    '''
    output = subprocess.check_output(command, shell=True)
    return output

def tf(word, blob):
    '''
    Counts term frequency
    '''
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    '''
    Count document frequency
    '''
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    '''
    Compute idf (inverse document frequency)
    '''
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    '''
    Compute tfidf
    '''
    return tf(word, blob) * idf(word, bloblist)

def get_wordnet_pos(treebank_tag):
    '''
    Return part-of-speech (wordnet set of POS tags)
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # treat all the rest pos tags as noun

def normalize_data(raw_folder,filename,norm_folder):
    '''
    Data cleaning:
    Convert to lowercase, join thouthands, remove other symbols and etc.
    Remove stopwords and lemmatize.
    '''
    lmtzr = WordNetLemmatizer()
    stopword_list = stopwords.words('english')
    os.system("mv "+raw_folder+"tmp "+raw_folder+filename)
    f_reader = open(raw_folder+filename,'r')
    f_writer = open(norm_folder+filename,'w')
    for line in f_reader.readlines():
        #convert to lowercase and remove unicode characters
        line = line.lower().decode('unicode_escape').encode('ascii','ignore')
        #protect abbreviations/acronyms
        line = re.sub('_',' ',line)
        line = re.sub(' [a-z]\. ',' ',line)
        line = re.sub('([a-z])\.([a-z])','\g<1>_\g<2>',line)
        line = re.sub('_([a-z])\.','_\g<1>_',line)
        line = re.sub('_([a-z])\.','_\g<1>_',line)
        #replace all \s=[ \t\n\r\f\v] symbols with single whitespace
        line = re.sub('[\s]',' ',line)
        #normalize numbers: join thousands '1,000,000 => 1000000', replace decimal marks '1.5 => 1 point 5' and '.5 => 0 point 5'
        line = re.sub('(\d),(\d)','\g<1>\g<2>',line)
        line = re.sub('(\d)\.(\d)','\g<1> point \g<2>',line)
        line = re.sub('( +)\.(\d)',' 0 point \g<2>',line)
        #remove all non-alphanumeric symbols (except underscore)
        line = re.sub('(!+|\?+|;+|:+|\.+|\++|-+|,+|\*+|:+|/+|>+|<+|=+|\^+|%+|\$+|#+|@+|\|+|~+|`+|\"+|`+|\(+|\)+|\[+|\]+|{+|}+|<+|>+|\\+|\'s|\'+ | \'+)',' ',line)
        #replace multiple spaces with single space
        line = re.sub(' +',' ',line)
        #remove leading and trailing whitespaces
        line = line.split()
        try:
        #remove stopwords and lemmatize by using POS tag
            filtered_line = [lmtzr.lemmatize(word.strip().strip("'"),pos=get_wordnet_pos(pos_tag((word,''))[0][1])) for word in line if word not in stopword_list]
        except:
            pdb.set_trace()
        f_writer.write(' '.join(filtered_line)+'\n')
    f_reader.close()
    f_writer.close()

def create_vocab_from_tfidf(norm_folder,categories,top_keywords=10):
    '''
    Create vocab consisting of important terms only (based on tf-idf score).
    '''
    tic = time.time()
    os.system("rm -f "+norm_folder+"vocab.txt"+" "+norm_folder+"all_articles")
    for filename in categories:
        os.system("cat "+norm_folder+filename+" >> "+norm_folder+"all_articles")
    f_r = open(norm_folder+'all_articles','r')
    articles = f_r.readlines()
    f_r.close()
    f_w = open(norm_folder+"vocab.txt",'w')
    bloblist = [tb(article.strip()) for article in articles]
    print "Total num of articles: "+str(len(articles))
    for i, blob in enumerate(bloblist): 
        print i
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in scores.iteritems():
            f_w.write(word+' '+str(score)+'\n')
    f_w.close()
    print "Time spent for creating vocabulary based on tf-idf scores: "+str(time.time()-tic)
    return 'vocab.txt'

def replace_with_unk(norm_folder, categories, vocab):
    '''
    Replace words not in vocab with <unk> token
    '''
    f_vocab = open(norm_folder+vocab,'r')
    vocab_tmp = f_vocab.read().split()
    f_vocab.close()
    for filename in categories:
        f_reader = open(norm_folder+filename,'r')
        f_writer = open(norm_folder+"filtered_"+filename,'w')
        for line in f_reader.readlines():
            line_tmp = line.split()
            ind = -1
            for word in line.split():
                ind += 1
                if word not in vocab_tmp:
                    line_tmp[ind] = '<unk>'
            f_writer.write(' '.join(line_tmp)+'\n')
        f_reader.close()
        f_writer.close()


def article_to_vect(norm_folder,categories,vocab):
    '''
    Converts every article into vector of Os and 1s, i.e. vector of size vocabulary(|V|) is initilaized with 0, 
    then corresponding index of each word (index with respect to vocab position) in the article is set to 1.
    if there are two or more similar words coressponding index is set to 1 not 2 or more.
    Generates matrix 'X' of term-by-documents and corresponding label vector 'y'.
    '''
    os.system("rm -f "+norm_folder+"filtered_all_articles")
    f_vocab = open(norm_folder+vocab,'r')
    vocab_tmp = f_vocab.read().split()
    f_vocab.close()
    column_size = len(vocab_tmp) # number of attributes(features) equal to the size of vocab
    label = 0
    for filename in categories:
        #merge all articles into one file, put topic label of each article to the first field
        os.system("cat "+norm_folder+"filtered_"+filename+" | sed -e 's/^/"+str(label)+" /' > "+norm_folder+"tmp")
        os.system("mv "+norm_folder+"tmp "+norm_folder+"filteredL_"+filename)
        os.system("cat "+norm_folder+"filteredL_"+filename+" >> "+norm_folder+"filtered_all_articles")
        label += 1
    os.system("sort -R "+norm_folder+"filtered_all_articles > "+norm_folder+"rand_filt_all_articles") #shuffle all the articles randomly
    # Start building article vectors
    row_size = run('cat '+norm_folder+"rand_filt_all_articles | wc -l").split()[0] #number of samples(artciles)
    matrix_tmp = np.zeros((int(row_size),column_size), dtype=np.int8) #create matrix of term-by-document
    # label_tmp.ndim must be equal to 1
    label_tmp = np.zeros((int(row_size)), dtype=np.int8)
    f_reader = open(norm_folder+"rand_filt_all_articles",'r')
    count = 0
    for line in f_reader:
        line_tmp = line.split()
        label_tmp[count] = int(line_tmp[0])
        for word in line_tmp[1:]:
            try:
                matrix_tmp[count][vocab_tmp.index(word)] = 1
            except:
                pdb.set_trace()
        count += 1
    f_reader.close() 
    np.savez(norm_folder+'dataset.npz',x=matrix_tmp,y=label_tmp)

if __name__  == '__main__':
    if not os.path.exists(raw_folder):
        print "*Raw data folder is not found!!!"
        exit()
    if not os.path.exists(norm_folder):
        os.system('mkdir '+norm_folder)
    categories = run('ls '+raw_folder).splitlines()
    #categories = ['20001.txt','20015.txt','20002.txt','20013.txt']
    #20001 - Asian financial crisis 1998
    #20015 - Iraq nuclear weapon inspection, 1998
    #20002 - Whitehouse scandal (B.Clinton and M.Lewinsky), 1998
    #20013 - Winter olympic games 1998, Japan
    if len(categories) < 1:
        print "*Raw data folder is empyty!!!"
        exit()

    print "Categories are: "+' | '.join(categories)

    if 0:   # normalize data, done only once, very time counsuming part
        for filename in categories:
            print "...Normalizing "+filename
            normalize_data(raw_folder,filename,norm_folder)
            print "...Finished normalizing "+filename
    if 0:   # create vocab, must be executed everytime after normalization
        print "...Creating vocabulary"
        vocab_size = 5000 #takes 'vocab_size' words with highest tf-idf score, after taking only unique words vocabulary size reduces further
        #vocab = create_vocab_from_tfidf(norm_folder,categories,top_keywords=10) # very time consuming, use once only
        vocab = 'vocab.txt'
        pdb.set_trace()
        bash_cmd = '''cat '''+norm_folder+vocab+''' | grep -v 'e-' | grep -v "^[0-9']" | sort -rn -k2,2 | sed -n 1,'''+
            str(vocab_size)+'''p | cut -d' ' -f1 | sort -u > '''+norm_folder+'tmp'
        os.system(bash_cmd)
        os.system('mv '+norm_folder+'tmp '+norm_folder+'vocab_tfidf.txt')
        os.system('rm -f '+norm_folder+'tmp')
        vocab = 'vocab_tfidf.txt'
        os.system("echo '<unk>' >> "+norm_folder+vocab) #insert <unk> token to the vocab
        print "...Finished creating vocabulary"
    if 0:   # replace word not in vocab with the <unk. token
        print "...Replacing with <unk>"
        replace_with_unk(norm_folder,categories,vocab)
        print "...Finished replacing with <unk>"
    if 0:   # create article vectors, done only once
        print "...Creating article vectors"
        article_to_vect(norm_folder,categories,vocab)
        print "...Finished creating article vectors"
