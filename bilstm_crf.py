# The code works with tensorflow 1.15.0 and keras 2.2.4,
# I couldn't get the keras_contrib CRF package to work with later versions of tensorflow or keras ...

import logging

import gensim
import numpy as np
from keras.layers import Bidirectional, concatenate, SpatialDropout1D
from keras.layers import LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

EMBEDDING_DIM=20
EMBEDDINGS_MODEL_FILE= None

PAD = "PAD_SPECIAL_SYMBOL"
UNK = "UNK_SPECIAL_SYMBOL"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_dict_train_file_ner(train_texts, DICTFILE=None):
    dictionary = gensim.corpora.Dictionary(train_texts)
    dictionary.add_documents([[PAD, UNK]])
    if DICTFILE:
        dictionary.save(DICTFILE)
    return dictionary


def load_data(training_file_path):
    sentence_tokens = []
    sentence_categories = []
    training_data = []
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


def train_ner(examples, model_file=None, epochs=30, validation_split=0.05, test_split=0.1, batch_size=64, patience=3, pretrained_embedding=True):
    examples = [_f for _f in examples if _f]
    cutoff = int(len(examples) * test_split)
    test_examples = examples[:cutoff]
    training_examples = examples[cutoff:]
    tags = set()
    texts = []
    lengths = []
    words = set()
    for training_example in training_examples:
        tags.update(list(map(lambda x: x[1], training_example)))
        text = list(map(lambda x: x[0], training_example))
        for word in text:
            words.add(word)
        lengths.append(len(text))
        texts.append(text)

    n_tags = len(tags)

    dictionary = create_dict_train_file_ner(texts + [[UNK, PAD]], DICTFILE="./dictionaries/biocrdict.dict")
    n_words = len(dictionary)
    print("dictionary size: %s" % n_words)
    a = np.array(lengths)
    max_len = int(np.percentile(a, 90))
    print("max len text: %s" % max_len)

    word2idx = dictionary.token2id
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: w for w, i in tag2idx.items()}

    X = [[word2idx[w[0]] for w in s] for s in training_examples]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word2idx[PAD])

    X_test = [[word2idx[w[0]] if w[0] in word2idx else word2idx[UNK] for w in s] for s in test_examples]
    X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=word2idx[PAD])

    y = [[tag2idx[w[1]] for w in s] for s in training_examples]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx['O'])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    y_test = [[tag2idx[w[1]] for w in s] for s in test_examples]
    y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx['O'])
    y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

    input = Input(shape=(max_len,))

    if pretrained_embedding:
        embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_MODEL_FILE, binary=True)
        embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
        for word, i in dictionary.token2id.items():
            embedding_vector = embeddings_index[word] if word in embeddings_index else None
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model = Embedding(input_dim=n_words + 1,
                          output_dim=EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=max_len,
                          trainable=False, mask_zero=True)(input)
    else:

        model = Embedding(input_dim=n_words + 1, output_dim=20,
                          input_length=max_len, mask_zero=True)(input)  # 20-dim embedding

    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    # applies dense layer to each of the time steps
    model = TimeDistributed(Dense(50, activation="relu"))(model)
    crf = CRF(n_tags)  # CRF layer
    # the dense layers feeds a crf layer
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    print(model.summary())

    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    history = model.fit(X, np.array(y),
                        batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1,
                        callbacks=[earlystopper])
    val_loss = history.history['val_loss'][-1]
    val_crf_viterbi_accuracy = history.history['val_crf_viterbi_accuracy'][-1]
    if model_file:
        save_load_utils.save_all_weights(model, model_file)
    test_pred = model.predict(X_test, verbose=1)
    pred_labels = pred2label(test_pred, idx2tag)
    test_labels = pred2label(y_test, idx2tag)
    print(classification_report(test_labels, pred_labels))
    return val_loss, val_crf_viterbi_accuracy, len(examples), model, idx2tag


def pred2label(pred, idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out


if __name__ == '__main__':
    training_data = load_data("./data/processed_train.txt")
    _, _, _, model, idx2tag = train_ner(training_data, model_file='./models/bilstmcrf.model', pretrained_embedding=False)
