from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
import pickle
import os
import errno
from helpers import *


if __name__ == '__main__':
    training_data = load_data('./data/processed_train.txt')
    testing_data = load_data('./data/processed_test.txt')

    X = [sent2features(s) for s in training_data]
    y = [sent2labels(s) for s in training_data]

    testX = [sent2features(s) for s in testing_data]
    testy = [sent2labels(s) for s in testing_data]

    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=False)

    pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

    report = flat_classification_report(y_pred=pred, y_true=y)
    print(report)
    crf.fit(X,y)

    modelfile = "./models/bestbioner.model"
    if not os.path.exists(os.path.dirname(modelfile)):
        try:
            os.makedirs(os.path.dirname(modelfile))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    pickle.dump(crf, open(modelfile, 'wb'))

    testpred = crf.predict(testX)
    testreport = flat_classification_report(y_pred=testpred, y_true=testy)
    print("Test Report: ")
    print(testreport)


