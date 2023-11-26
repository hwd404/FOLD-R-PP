from utils import load_data, split_xy, split_X_by_Y, \
    split_data, get_scores, justify_data, decode_rules, proof_tree, zip_rule, simplify_rule
from algo import fold, predict, classify, flatten_rules, justify, rebut
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
        self.simple = None
        self.translation = None

    def load_data(self, file, amount=-1):
        data, self.attrs = load_data(file, self.attrs, self.label, self.numeric, self.pos, amount)
        return data

    def fit(self, X, Y, ratio=0.5):
        X_pos, X_neg = split_X_by_Y(X, Y)
        self.rules = fold(X_pos, X_neg, ratio=ratio)

    def predict(self, X):
        return predict(self.rules, X)

    def classify(self, x):
        return classify(self.rules, x)

    def asp(self, simple=False):
        if (self.asp_rules is None and self.rules is not None) or self.simple != simple:
            self.simple = simple
            self.frs = flatten_rules(self.rules)
            self.frs = [zip_rule(r) for r in self.frs]
            self.asp_rules = decode_rules(self.frs, self.attrs)
            if simple:
                self.asp_rules = [simplify_rule(r) for r in self.asp_rules]
        return self.asp_rules

    def print_asp(self, simple=False):
        for r in self.asp(simple):
            print(r)

    def explain(self, x, all_flag=False):
        ret = ''
        self.asp()
        all_pos = justify(self.frs, x, all_flag=all_flag)
        k = 1
        if len(all_pos) == 0:
            all_neg = rebut(self.frs, x)
            for rs in all_neg:
                ret += 'rebuttal ' + str(k) + ':\n'
                for r in decode_rules(rs, attrs=self.attrs, x=x):
                    ret += r + '\n'
                ret += str(justify_data(rs, x, attrs=self.attrs)) + '\n'
                k += 1
        else:
            for rs in all_pos:
                ret += 'answer ' + str(k) + ':\n'
                for r in decode_rules(rs, attrs=self.attrs, x=x):
                    ret += r + '\n'
                ret += str(justify_data(rs, x, attrs=self.attrs)) + '\n'
                k += 1
        return ret

    def proof(self, x, all_flag=False):
        ret = ''
        self.asp()
        all_pos = justify(self.frs, x, all_flag=all_flag)
        k = 1
        if len(all_pos) == 0:
            all_neg = rebut(self.frs, x)
            for rs in all_neg:
                ret += 'rebuttal ' + str(k) + ':\n'
                for r in proof_tree(rs, attrs=self.attrs, x=x):
                    ret += r
                ret += str(justify_data(rs, x, attrs=self.attrs)) + '\n'
                k += 1
        else:
            for rs in all_pos:
                ret += 'answer ' + str(k) + ':\n'
                for r in proof_tree(rs, attrs=self.attrs, x=x):
                    ret += r
                ret += str(justify_data(rs, x, attrs=self.attrs)) + '\n'
                k += 1
        return ret


def save_model_to_file(model, file):
    f = open(file, 'wb')
    pickle.dump(model, f)
    f.close()


def load_model_from_file(file):
    f = open(file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret
