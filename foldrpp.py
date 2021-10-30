from utils import load_data, split_xy, split_X_by_Y, decode_rules, \
    split_data, get_scores, justify_data, decode_justification
from algo import fold, predict, classify, flatten_rules, justify
import pickle


class Classifier:
    def __init__(self, attrs=None, numeric=None, label=None, pos=None):
        self.attrs = attrs
        self.numeric = numeric
        self.label = label
        self.pos = pos
        self.rules = None
        self.frs = None
        self.asp_rules = None
        self.seq = 1
        self.translation = None

    def load_data(self, file, amount=-1):
        data, self.attrs = load_data(file, self.attrs, self.label, self.numeric, self.pos, amount)
        X, Y = split_xy(data)
        return X, Y

    def fit(self, X, Y, ratio=0.5):
        X_pos, X_neg = split_X_by_Y(X, Y)
        self.rules = fold(X_pos, X_neg, ratio=ratio)

    def predict(self, X):
        return predict(self.rules, X)

    def classify(self, x):
        return classify(self.rules, x)

    def asp(self):
        if self.asp_rules is None and self.rules is not None:
            self.frs = flatten_rules(self.rules)
            self.asp_rules = decode_rules(self.frs, self.attrs)
        return self.asp_rules

    def print_asp(self):
        for r in self.asp():
            print(r)

    def justify(self, x, all_flag=False):
        all_pos = justify(self.frs, x, all_flag=all_flag)
        if len(all_pos) == 0:
            print('no answer \n')
        k = 1
        for rs in all_pos:
            print('answer ', k, ':')
            for r in decode_justification(rs, x, attrs=self.attrs):
                print(r)
            print(justify_data(rs, x, attrs=self.attrs), '\n')
            k += 1


def save_model_to_file(model, file):
        f = open(file, 'wb')
        pickle.dump(model, f)
        f.close()


def load_model_from_file(file):
    f = open(file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret
