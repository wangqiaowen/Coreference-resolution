from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from gold_standard import load_data, nested_dict
import csv
import re


class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns  # list of column to encode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''

        output = X.copy()

        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)

        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def load_gold_data(filename='gold_standard2.csv'):
    cols_names = ['Class', 'dist', 'PoS1', 'gf1', 'PoS2', 'gf2', 'sPoS', 'sgf', 'sh', 'match']

    df = (pd.read_csv(filename, names=cols_names, header=0)
          .replace({'?': 'unknown'}))  # NaN are represented by '?'

    row_len = int(len(df)/3)
    return df.sample(frac=1).reset_index(drop=True)[:row_len]


def transform_data(file):
    df = load_gold_data(filename=file)
    print('data loaded')
    X_train = df.drop(columns='Class')
    y_train = df['Class'].copy()
    le = MultiColumnLabelEncoder(columns=['PoS1', 'gf1', 'PoS2', 'gf2'])
    X_train_le = le.fit_transform(X_train)

    return X_train_le, y_train


def create_test_set(test_data):
    pos_tags = ['NOUN', 'ART', 'PRON', 'PROPN']
    feature_dict = nested_dict()
    sentence_id = 1
    prev_doc_id = None
    prev_token_id = 0
    for line in test_data:
        if line and len(line) > 1:
            if "#begin document" not in line:
                if "#end document" not in line:
                    line = line.strip('\n')
                    linecols = line.split('\t')
                    doc_id = linecols[0]
                    token_id = linecols[1]
                    if prev_doc_id != doc_id:
                        sentence_id = 1
                        prev_doc_id = doc_id
                    else:
                        prev_doc_id = doc_id
                    if int(token_id) < prev_token_id:
                        sentence_id += 1
                        prev_token_id = int(token_id)
                    else:
                        prev_token_id = int(token_id)

                    pos_tag = linecols[5]
                    fnct = linecols[9]
                    head = linecols[8]
                    person_number_gender = linecols[7].split('|')
                    person = None
                    number = None
                    gender = None
                    for png in person_number_gender:
                        if png.startswith("Person"):
                            person = png.split('=')[1]
                        elif png.startswith("Number"):
                            number = png.split("=")[1]
                        elif png.startswith("Gender"):
                            gender = png.startswith("Gender")
                    if pos_tag in pos_tags:
                        feature_dict[doc_id][str(sentence_id)][token_id] = [pos_tag, fnct, head,
                                                                            (person, number, gender)]
    return feature_dict


def get_labels(labeled_data):
    true_pos = nested_dict()
    tokens = []
    sentence_id = 1
    prev_doc_id = None
    prev_token_id = 0
    idx = 0
    for line in labeled_data:
        if line and len(line) > 1:
            if "#begin document" not in line:
                if "#end document" not in line:
                    line = line.strip('\n')
                    linecols = line.split('\t')
                    doc_id = linecols[0]
                    token_id = linecols[2]
                    if prev_doc_id != doc_id:
                        sentence_id = 1
                        prev_doc_id = doc_id
                    else:
                        prev_doc_id = doc_id
                    if int(token_id) < prev_token_id:
                        sentence_id += 1
                        prev_token_id = int(token_id)
                    else:
                        prev_token_id = int(token_id)
                    ids = linecols[-1].split('|')
                    for identry in ids:
                        _id = re.search(r'\d+', identry)
                        if _id:
                            _id = int(_id.group(0))
                            if identry.startswith('(') and identry.endswith(')'):
                                true_pos[doc_id][str(sentence_id)][idx] = tuple([token_id])
                            else:
                                if identry.startswith('('):
                                    tokens.append(token_id)
                                elif identry.endswith(')'):
                                    tokens.append(token_id)
                                    true_pos[doc_id][str(sentence_id)][idx] = tuple(tokens)
                                    tokens = []
                    idx += 1
    return true_pos


def pair_test_samples():
    with open('test_data.csv', 'w') as f:
        filewriter = csv.writer(f, delimiter=',')
        filewriter.writerow(['Label', 'Distance', 'POS1', 'Grammar1', 'POS2', 'Grammar2',
                             'samePOS', 'sameGrammar', 'sameHead', 'Match'])
    # features: distance, pos1, grammar1, pos2, grammar2, int(same_pos), int(same_grammar), int(same_head), int(match)
    num_true_pos = 0
    num_negs = 0
    df_test = load_data('test.sync.txt')
    df_true = load_data('test.coref.txt')
    true_pos = get_labels(df_true)
    print('test data loaded')
    feat_dict = create_test_set(df_test)
    print('create test pairs')
    for doc in feat_dict:
        features = feat_dict[doc]
        for sentence in features:
            sentence_feats = features[sentence]
            for word_feats in sentence_feats:
                pos_tag1 = sentence_feats[word_feats][0]
                fnct1 = sentence_feats[word_feats][1]
                head1 = sentence_feats[word_feats][2]
                gen_num_cas1 = sentence_feats[word_feats][3]
                trues = true_pos[doc][sentence].values()

                # find potential "partner" within the same sentence
                word_idx = list(sentence_feats.keys()).index(word_feats)
                for word_feats2 in list(sentence_feats.keys())[word_idx + 1:]:
                    pos_tag2 = sentence_feats[word_feats2][0]
                    fnct2 = sentence_feats[word_feats2][1]
                    head2 = sentence_feats[word_feats2][2]
                    gen_num_cas2 = sentence_feats[word_feats2][3]

                    if pos_tag1 == 'NOUN':
                        if pos_tag2 == 'NOUN' or pos_tag2 == 'PROPN':
                            match = gen_num_cas1 == gen_num_cas2
                            same_grammar = fnct1 == fnct2
                            same_pos = pos_tag1 == pos_tag2
                            same_head = head1 == head2
                            distance = int(word_feats2) - int(word_feats)
                            if tuple([word_feats, word_feats2]) in trues:
                                label = 1
                                num_true_pos += 1
                            else:
                                label = 0
                                num_negs += 1

                            with open('test_data.csv', 'a') as f:
                                filewriter = csv.writer(f, delimiter=',')
                                filewriter.writerow([label, distance, pos_tag1, fnct1, pos_tag2, fnct2,
                                                     int(same_pos), int(same_grammar), int(same_head), int(match)])
                    elif pos_tag1 == 'PROPN':
                        if pos_tag2 == 'PROPN' or pos_tag2 == 'NOUN':
                            match = gen_num_cas1 == gen_num_cas2
                            same_grammar = fnct1 == fnct2
                            same_pos = pos_tag1 == pos_tag2
                            same_head = head1 == head2
                            distance = int(word_feats2) - int(word_feats)
                            if tuple([word_feats, word_feats2]) in trues:
                                label = 1
                                num_true_pos += 1
                            else:
                                label = 0
                                num_negs += 1

                            with open('test_data.csv', 'a') as f:
                                filewriter = csv.writer(f, delimiter=',')
                                filewriter.writerow([label, distance, pos_tag1, fnct1, pos_tag2, fnct2,
                                                     int(same_pos), int(same_grammar), int(same_head), int(match)])
                    elif pos_tag1 == 'ART':
                        if pos_tag2 == 'NOUN' or pos_tag2 == 'PROPN':
                            match = gen_num_cas1 == gen_num_cas2
                            same_grammar = fnct1 == fnct2
                            same_pos = pos_tag1 == pos_tag2
                            same_head = head1 == head2
                            distance = int(word_feats2) - int(word_feats)
                            if tuple([word_feats, word_feats2]) in trues:
                                label = 1
                                num_true_pos += 1
                            else:
                                label = 0
                                num_negs += 1

                            with open('test_data.csv', 'a') as f:
                                filewriter = csv.writer(f, delimiter=',')
                                filewriter.writerow([label, distance, pos_tag1, fnct1, pos_tag2, fnct2,
                                                     int(same_pos), int(same_grammar), int(same_head), int(match)])
                    elif pos_tag1 == 'PRON':
                        if pos_tag2 == 'NOUN' or pos_tag2 == 'PROPN':
                            match = gen_num_cas1 == gen_num_cas2
                            same_grammar = fnct1 == fnct2
                            same_pos = pos_tag1 == pos_tag2
                            same_head = head1 == head2
                            distance = int(word_feats2) - int(word_feats)
                            if tuple([word_feats, word_feats2]) in trues:
                                label = 1
                                num_true_pos += 1
                            else:
                                label = 0
                                num_negs += 1

                            with open('test_data.csv', 'a') as f:
                                filewriter = csv.writer(f, delimiter=',')
                                filewriter.writerow([label, distance, pos_tag1, fnct1, pos_tag2, fnct2,
                                                     int(same_pos), int(same_grammar), int(same_head), int(match)])

                # find potential "partner" within the next sentence
                sentence_idx = list(features.keys()).index(sentence)
                for sentence2 in list(features.keys())[sentence_idx + 1:sentence_idx + 2]:
                    sentence_feats2 = features[sentence2]
                    trues2 = true_pos[doc][sentence2].values()
                    for word_feats2 in sentence_feats2:
                        pos_tag2 = sentence_feats2[word_feats2][0]
                        fnct2 = sentence_feats2[word_feats2][1]
                        head2 = sentence_feats2[word_feats2][2]
                        gen_num_cas2 = sentence_feats2[word_feats2][3]

                        if pos_tag1 == 'NOUN':
                            if pos_tag2 == 'NOUN' or pos_tag2 == 'PROPN':
                                match = gen_num_cas1 == gen_num_cas2
                                same_grammar = fnct1 == fnct2
                                same_pos = pos_tag1 == pos_tag2
                                same_head = head1 == head2
                                distance = int(word_feats2) - int(word_feats)
                                if tuple([word_feats, word_feats2]) in trues2:
                                    label = 1
                                    num_true_pos += 1
                                else:
                                    label = 0
                                    num_negs += 1

                                with open('test_data.csv', 'a') as f:
                                    filewriter = csv.writer(f, delimiter=',')
                                    filewriter.writerow([label, distance, pos_tag1, fnct1, pos_tag2, fnct2,
                                                         int(same_pos), int(same_grammar), int(same_head), int(match)])
                        elif pos_tag1 == 'PROPN':
                            if pos_tag2 == 'PROPN' or pos_tag2 == 'NOUN' or pos_tag2:
                                match = gen_num_cas1 == gen_num_cas2
                                same_grammar = fnct1 == fnct2
                                same_pos = pos_tag1 == pos_tag2
                                same_head = head1 == head2
                                distance = int(word_feats2) - int(word_feats)
                                if tuple([word_feats, word_feats2]) in trues2:
                                    label = 1
                                    num_true_pos += 1
                                else:
                                    label = 0
                                    num_negs += 1

                                with open('test_data.csv', 'a') as f:
                                    filewriter = csv.writer(f, delimiter=',')
                                    filewriter.writerow([label, distance, pos_tag1, fnct1, pos_tag2, fnct2,
                                                         int(same_pos), int(same_grammar), int(same_head), int(match)])

                        elif pos_tag1 == 'PRON':
                            if pos_tag2 == 'NOUN' or pos_tag2 == 'PROPN':
                                match = gen_num_cas1 == gen_num_cas2
                                same_grammar = fnct1 == fnct2
                                same_pos = pos_tag1 == pos_tag2
                                same_head = head1 == head2
                                distance = int(word_feats2) - int(word_feats)
                                if tuple([word_feats, word_feats2]) in trues2:
                                    label = 1
                                    num_true_pos += 1
                                else:
                                    label = 0
                                    num_negs += 1

                                with open('test_data.csv', 'a') as f:
                                    filewriter = csv.writer(f, delimiter=',')
                                    filewriter.writerow([label, distance, pos_tag1, fnct1, pos_tag2, fnct2,
                                                         int(same_pos), int(same_grammar), int(same_head), int(match)])

    print('created test data')
    print('number positives', num_true_pos)
    print('number negatives', num_negs)


def sgd_param_selection(X, y, nfolds, model):
    loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    alpha = [0.5, 0.1, 0.01, 0.001, 0.0001]
    param_grid = {'loss': loss, 'alpha': alpha}
    grid_search = GridSearchCV(model, param_grid, cv=nfolds, n_jobs=1)
    grid_search.fit(X, y)
    return grid_search.best_params_


if __name__ == '__main__':
    pair_test_samples()
    X_train_le, y_train = transform_data(file='gold_standard2.csv')
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_train_ohe = ohe.fit_transform(X_train_le)
    del X_train_le
    print('train data transformed')
    X_test_le, y_test = transform_data(file='test_data.csv')
    X_test_ohe = ohe.transform(X_test_le)
    del X_test_le
    print('test data transformed')

    # SGD classifier
    print('start training SGD')
    sgd = SGDClassifier(random_state=42, loss='log', alpha=0.001, max_iter=10, verbose=1)
    sgd.fit(X_train_ohe, y_train)
    print('start predicting')
    y_test_predict = sgd.predict(X_test_ohe)
    print()
    print('SGD')
    print('precision')
    print(precision_score(y_test, y_test_predict))
    print('recall')
    print(recall_score(y_test, y_test_predict))
    print('f1 score')
    print(f1_score(y_test, y_test_predict))
    print()
    del y_test_predict

    # p = sgd_param_selection(X_train_ohe, y_train, 5, sgd)
    del sgd
    print('best params')
    print(p)
    clf = SGDClassifier(loss=p[0], alpha=p[1], verbose=1)
    clf.fit(X_train_ohe, y_train)
    print('start predicting')
    y_test_predict = clf.predict(X_test_ohe)
    print()
    print('SGD grid')
    print('precision')
    print(precision_score(y_test, y_test_predict))
    print('recall')
    print(recall_score(y_test, y_test_predict))
    print('f1 score')
    print(f1_score(y_test, y_test_predict))
    print()
    del clf
    del y_test_predict

    print('start Naive Bayes')
    clf = BernoulliNB()
    clf.fit(X_train_ohe, y_train)
    print('start predicting')
    y_test_predict = clf.predict(X_test_ohe)

    print('precision')
    print(precision_score(y_test, y_test_predict))
    print('recall')
    print(recall_score(y_test, y_test_predict))
    print('f1 score')
    print(f1_score(y_test, y_test_predict))
    del clf
    del y_test_predict

    print()
    print('perceptron')
    clf = Perceptron(tol=1e-5, random_state=0)
    clf.fit(X_train_ohe, y_train)
    print('start predicting')
    y_test_predict = clf.predict(X_test_ohe)
    print('precision')
    print(precision_score(y_test, y_test_predict))
    print('recall')
    print(recall_score(y_test, y_test_predict))
    print('f1 score')
    print(f1_score(y_test, y_test_predict))
    print()
    del clf
    del y_test_predict

    print("MLP")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu',
                        hidden_layer_sizes=(5, 2), random_state=1, verbose=True, early_stopping=True)
    clf.fit(X_train_ohe, y_train)
    print('start predicting')
    y_test_predict = clf.predict(X_test_ohe)
    print('precision')
    print(precision_score(y_test, y_test_predict))
    print('recall')
    print(recall_score(y_test, y_test_predict))
    print('f1 score')
    print(f1_score(y_test, y_test_predict))
    print()
    del clf
    del y_test_predict

    print("KNN")
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train_ohe, y_train)
    print('start predicting')
    y_test_predict = neigh.predict(X_test_ohe)
    print('precision')
    print(precision_score(y_test, y_test_predict))
    print('recall')
    print(recall_score(y_test, y_test_predict))
    print('f1 score')
    print(f1_score(y_test, y_test_predict))
    print()
    del neigh
    del y_test_predict

