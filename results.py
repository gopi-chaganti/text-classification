from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import re
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import nltk
import random
import sys

def read_dir(dirname, data, target):
    listing = os.listdir(dirname)
    for infile in listing:
        pathname = os.path.join(dirname, infile)
        fill_text_from_dir(pathname, data, target)

def fill_text_from_dir(dirname, data, target):
    if os.path.isfile(dirname):
        return
    listing = os.listdir(dirname)
    for infile in listing:
        pathname = os.path.join(dirname, infile)
        with open(pathname, 'r', encoding='utf-8', errors='ignore') as myfile:
            content = myfile.read().replace('\n', ' ')
            body = re.sub(r'^(.*) Lines: (\d)+ ', "", content)
            #tokens = nltk.word_tokenize(body)
            data.append(body)
            target.append(dirname.split("/")[1])

def testmodel(classifier, clf, train_data, train_target, test_data, test_target):
    _ = clf.fit(train_data, train_target)
    predicted = clf.predict(test_data)
    print(classifier)
    #print("Accuracy :", np.mean(predicted == test_target))
    print("f1 Score : {0:.3f}".format(f1_score(test_target, predicted, average='macro')))
    print("precision : {0:.3f}".format(precision_score(test_target, predicted, average='macro')))
    print("recall : {0:.3f}".format(recall_score(test_target, predicted, average='macro')))

def shuffle(data, target, length):
    shuffled_data = []
    shuffled_target = []
    indexes = [i for i in range(len(data))]
    random.shuffle(indexes)
    for index in indexes:
        shuffled_data.append(data[index])
        shuffled_target.append(target[index])
    return shuffled_data[:length], shuffled_target[:length]

def get_f1_score(clf, train_data, train_target, test_data, test_target, data_sizes):
    scores = []
    for length in data_sizes:
        sh_data, sh_train = shuffle(train_data, train_target, length)
        _ = clf.fit(sh_data, sh_train)
        predicted = clf.predict(test_data)
        scores.append(f1_score(test_target, predicted, average='macro'))
    return scores

def stem_dataset(data):
    new_data = []
    sno = nltk.stem.SnowballStemmer('english')
    for file in data:
        words = file.split(" ")
        singles = [sno.stem(word) for word in words]
        new_data.append(' '.join(singles))
    return new_data

if __name__ == '__main__':

    train = ''
    test = ''

    if len(sys.argv) == 3:
        train = sys.argv[1]
        test = sys.argv[2]
    else:
        print('Wrong number of arguments. Usage:\nmy-best-config.py <Train - train folder> <test - test folder>')

    train_data = []
    train_target = []
    test_data = []
    test_target = []

    read_dir(train, train_data, train_target)
    read_dir(test, test_data, test_target)


    print("Unigram models:")
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                            ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
    ])
    testmodel("Naive Bayes", text_clf, train_data, train_target, test_data, test_target)

    svd_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
    ])
    testmodel("SVD ", svd_clf, train_data, train_target, test_data, test_target)

    log_clf = Pipeline(steps=[('vect', CountVectorizer(ngram_range=(1,1))),
                                ('tfidf', TfidfTransformer()),
                              ('logistic', LogisticRegression())])
    testmodel("Logistic Classifier", log_clf, train_data, train_target, test_data, test_target)


    log_clf = Pipeline(steps=[('vect', CountVectorizer(ngram_range=(1,1))),
                            ('tfidf', TfidfTransformer()),
                             ('random', RandomForestClassifier()),])
    testmodel("Random Forest", log_clf, train_data, train_target, test_data, test_target)


    print("Bigram models:")
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),
                            ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
    ])
    testmodel("Naive Bayes", text_clf, train_data, train_target, test_data, test_target)

    svd_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
    ])
    testmodel("SVD ", svd_clf, train_data, train_target, test_data, test_target)

    log_clf = Pipeline(steps=[('vect', CountVectorizer(ngram_range=(2,2))),
                                ('tfidf', TfidfTransformer()),
                              ('logistic', LogisticRegression())])
    testmodel("Logistic Classifier", log_clf, train_data, train_target, test_data, test_target)


    log_clf = Pipeline(steps=[('vect', CountVectorizer(ngram_range=(2,2))),
                            ('tfidf', TfidfTransformer()),
                             ('random', RandomForestClassifier()),])
    testmodel("Random Forest", log_clf, train_data, train_target, test_data, test_target)


    #learning curve
    data_sizes = [(i+1)*100 for i in range(20)]
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                            ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
    ])
    naive_scores = get_f1_score(text_clf, train_data, train_target, test_data, test_target, data_sizes)

    svd_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
    ])
    svd_scores = get_f1_score(svd_clf, train_data, train_target, test_data, test_target, data_sizes)

    log_clf = Pipeline(steps=[('vect', CountVectorizer(ngram_range=(1,1))),
                                ('tfidf', TfidfTransformer()),
                              ('logistic', LogisticRegression())])
    log_scores = get_f1_score(log_clf, train_data, train_target, test_data, test_target, data_sizes)

    log_clf = Pipeline(steps=[('vect', CountVectorizer(ngram_range=(1,1))),
                            ('tfidf', TfidfTransformer()),
                             ('random', RandomForestClassifier()),])
    random_scores = get_f1_score(log_clf, train_data, train_target, test_data, test_target, data_sizes)



    plt.figure()
    plt.title("Learning curve")
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel("No of Training examples")
    plt.ylabel("F1 Score")
    plt.plot(data_sizes, naive_scores, 'o-', color="r",
                 label="Naive Bayes")
    plt.plot(data_sizes, svd_scores, 'o-', color="b",
                 label="SVM")
    plt.plot(data_sizes, log_scores, 'o-', color="g",
                 label="Logistic")
    plt.plot(data_sizes, random_scores, 'o-', color="y",
                 label="Random forest")
    plt.legend(loc="best")
    plt.show()



    # my best config exploration
    # with tfidf
    svd_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
    ])
    testmodel("with tfidf", svd_clf, train_data, train_target, test_data, test_target)

    #without tfidf
    svd_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                         ('clf', LinearSVC()),
    ])
    testmodel("without tfidf", svd_clf, train_data, train_target, test_data, test_target)

    #with preprocessing
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,1), lowercase=True)),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
    ])
    testmodel("stop words + lowercase", svd_clf, train_data, train_target, test_data, test_target)

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            stemmer = nltk.stem.SnowballStemmer("english")
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

    svd_clf = Pipeline([('vect', StemmedCountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
    ])
    testmodel("Stemming + Stop words ", svd_clf, train_data, train_target, test_data, test_target)

    '''
    #stemming data set
    def stem_dataset(data):
        new_data = []
        sno = nltk.stem.PorterStemmer()
        for file in data:
            words = file.split(" ")
            singles = [sno.stem(word) for word in words]
            new_data.append(' '.join(singles))
        return new_data

    stemmed_train = stem_dataset(train_data)
    stemmed_test = stem_dataset(test_data)
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,1))),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
    ])
    testmodel("SVD ", svd_clf, stemmed_train, train_target, stemmed_test, test_target)
    '''

    #feature selection
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
                         ('clf', LinearSVC(C=1.0)),
    ])
    testmodel("feature selection l1 reg ", svd_clf, train_data, train_target, test_data, test_target)

    #feature selection
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                         ('clf', LinearSVC(C=1.0)),
    ])
    testmodel("feature selection l2 reg ", svd_clf, train_data, train_target, test_data, test_target)


    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    #feature selection univariate
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                        ( 'univariate', SelectKBest(chi2, k=2000)),
                         ('clf', LinearSVC(C=1.0)),
    ])
    testmodel("feature selection k best ", svd_clf, train_data, train_target, test_data, test_target)


    #hyper paramter tuning
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                         ('clf', LinearSVC(C=100.0, loss="squared_hinge", penalty="l2")),
    ])
    testmodel("C=100", svd_clf, train_data, train_target, test_data, test_target)

    #hyper paramter tuning
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                         ('clf', LinearSVC(C=1.0, loss="squared_hinge", penalty="l2")),
    ])
    testmodel("C=1", svd_clf, train_data, train_target, test_data, test_target)

    #hyper paramter tuning
    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                         ('clf', LinearSVC(C=1.0, loss="hinge", penalty="l2")),
    ])
    testmodel("hinge loss", svd_clf, train_data, train_target, test_data, test_target)


    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                        ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                         ('clf', LinearSVC(C=1.0, loss="squared_hinge", penalty="l2")),
    ])
    testmodel("sq hinge loss", svd_clf, train_data, train_target, test_data, test_target)

    svd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=1)),
                        ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC(C=1.0, loss="squared_hinge", penalty="l2")),
    ])
    testmodel("sq hinge loss", svd_clf, train_data, train_target, test_data, test_target)
