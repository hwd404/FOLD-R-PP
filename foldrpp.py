from utils import load_data, split_xy, split_X_by_Y, flatten_rules, decode_rules, split_data, get_scores
from algo import fold, predict, classify
import pickle


class Classifier:
    def __init__(self, attrs=None, numeric=None, label=None, pos=None):
        self.attrs = attrs
        self.numeric = numeric
        self.label = label
        self.pos = pos
        self.rules = None
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
            frs = flatten_rules(self.rules)
            self.asp_rules = decode_rules(frs, self.attrs)
        return self.asp_rules

    def print_asp(self):
        for r in self.asp():
            print(r)

    def save_model_to_file(self, file):
        f = open(file, 'wb')
        pickle.dump(self, f)
        f.close()


def load_model_from_file(file):
    f = open(file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret
