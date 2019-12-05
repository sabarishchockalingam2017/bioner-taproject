import re
import pandas as pd
from progressbar import ProgressBar
from nltk.tokenize import word_tokenize

def load_biocreative(bcfilepath='./data/train/train.in', bcevalpath='./data/train/GENE.eval', procout='./data/processed.txt'):
    """ Function to parse and format the biocreative dataset.
    Arguments:
        bcfilepath: file path to the train/test corpus.
        bcevalpath: file path to the labels.
        procout: file path to save the processed output text file."""

    bcfile = open(bcfilepath)

    # list to contain text information
    templist = []

    for line in bcfile.readlines():
        linere = re.match(r'[^\s]+', line)
        lineid = linere.group(0)
        txtstart = linere.span()[1]+1
        text = line[txtstart:].replace('\n', '')
        text = word_tokenize(text)
        templist.append([lineid, text])

    bcfile.close()
    tempdf = pd.DataFrame(templist, columns=['id', 'text'])

    # calculating indexes of each word. this will be used to join with label data later
    tempdf = tempdf.explode('text')
    tempdf['textlen'] = tempdf['text'].apply(len)
    tempg = tempdf.groupby('id')
    tempdf['cummulative'] = tempg['textlen'].cumsum()
    tempdf['textpos'] = tempg['cummulative'].rank(ascending='True')
    tempdf['textind'] = tempdf.cummulative - tempdf.textlen
    tempdf.textind = tempdf.textind.astype(int)

    bceval = open(bcevalpath, encoding='utf8')
    # list to contain label data
    evaltemp = []
    for line in bceval.readlines():
        lineid = re.match(r'[^|]+', line).group(0)
        entstart = re.findall(r'\|(\d*)\s', line)[0]
        entend = re.findall(r'\s(\d*)\|', line)[0]
        # entstart = re.match(r'\d*', line[15:]).group(0)
        # entend = re.match(r'\d*', line[15+len(entstart)+1:]).group(0)
        indtilnow = len(lineid+entstart+entend)+3
        entstart = int(entstart)
        entend = int(entend)
        ent = line[indtilnow:].replace('\n','').split(' ')
        evaltemp.append([lineid, entstart, entend, ent])

    bceval.close()

    evaldf = pd.DataFrame(evaltemp, columns=['id', 'entity_start', 'entity_end', 'entity'])
    evaldf = evaldf.explode('entity')
    evaldf['ent_len'] = evaldf['entity'].apply(len)
    # calculating indexes to help with joining later
    evaldf = evaldf.sort_values(['id', 'entity_start', 'entity_end'])
    evalg = evaldf.groupby(['id', 'entity_start', 'entity_end'])
    evaldf['cummulative'] = evalg['ent_len'].cumsum()
    evaldf['entpos'] = evalg['cummulative'].rank(ascending=True)
    evaldf['entind'] = evaldf.entity_start + evaldf.cummulative - evaldf.ent_len
    evaldf.entind = evaldf.entind.astype(int)
    evaldf['label'] = evaldf.apply(lambda x: 'B' if x.entpos == 1.0 else 'I', axis=1)

    tempdf['id'] = tempdf['id'].astype(str)
    evaldf['id'] = evaldf['id'].astype(str)

    # joining per text id and index of entity
    finaldf = pd.merge(tempdf, evaldf,
                       how='left',
                       left_on=['id', 'textind'],
                       right_on=['id', 'entind'],
                       suffixes=('_t', '_e'))
    # text not in the evaluated data are considered to O (outside)
    finaldf.label = finaldf['label'].fillna('O')
    finaldf = finaldf[['id', 'text', 'textind', 'entity', 'entind','label', 'entity_start', 'entity_end']]

    towritefile = open(procout, 'w+')

    pbar = ProgressBar()
    for pid in pbar(finaldf.id.unique()):
        for tx, lb in finaldf[finaldf.id == pid][['text', 'label']].values:
            towritefile.write('{}\t{}'.format(tx, lb))
            towritefile.write('\n')
        towritefile.write('\n')

    towritefile.close()



def load_data(training_file_path):
    sentence_tokens = []
    sentence_categories = []
    training_data=[]
    for line in open(training_file_path, encoding='utf8').readlines():
        stripped = line.strip()
        if stripped:
            token, category = stripped.split('\t')
            sentence_tokens.append(token)
            sentence_categories.append(category)
        else:
            sentence = list(zip(sentence_tokens, sentence_categories))
            training_data.append(sentence)
            sentence_tokens = []
            sentence_categories = []
    return training_data

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word_suffix': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def sent2labels(sent):
    return [label for token, label in sent]

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def text2features(intext):
    sent = [(w, 'O') for w in word_tokenize(intext)]
    return [word2features(sent, i) for i in range(len(sent))]
