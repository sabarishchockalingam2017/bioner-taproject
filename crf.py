from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
import re
import pandas as pd
from progressbar import ProgressBar


def load_biocreative(bcfilepath='./data/train/train.in', bcevalpath='./data/train/GENE.eval', procout='./data/processed.txt'):
    """ Function to parse and format the biocreative dataset.
    Arguments:
        bcfilepath: file path to the train/test corpus.
        bcevalpath: file path to the labels"""

    bcfile = open(bcfilepath)

    # list to contain text information
    templist = []

    for line in bcfile.readlines():
        linere = re.match(r'[^\s]+', line)
        lineid = linere.group(0)
        txtstart = linere.span()[1]+1
        text = line[txtstart:].replace('\n', '')
        templist.append([lineid, text.split(' ')])

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
            towritefile.write('{} {}'.format(tx, lb))
            towritefile.write('\n')
        towritefile.write('\n')

    towritefile.close()

