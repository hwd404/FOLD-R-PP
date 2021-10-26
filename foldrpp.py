from utils import load_data, split_xy, split_X_by_Y, flatten, decode_rules
from utils import fold, predict, classify, decode_test_data, split_data, get_scores
import pickle
import subprocess
import tempfile


class Classifier:
    def __init__(self, attrs=None, numeric=None, label=None, pos=None):
        self.attrs = attrs
        self.numeric = numeric
        self.label = label
        self.pos = pos
        self.rules = None
        self.asp = None
        self.seq = 1
        self.translation = None

    def load_data(self, file, amount=-1):
        data, _, self.attrs = load_data(file, self.attrs, [self.label], self.numeric, self.pos, amount)
        X, Y = split_xy(data)
        return X, Y

    def fit(self, X, Y, ratio=0.5):
        X_pos, X_neg = split_X_by_Y(X, Y)
        self.rules = fold(X_pos, X_neg, ratio=ratio)

    def predict(self, X):
        return predict(self.rules, X)

    def classify(self, x):
        return classify(self.rules, x)

    def print_asp(self):
        if self.asp is None and self.rules is not None:
            frs = flatten(self.rules)
            self.asp = decode_rules(frs, self.attrs)
        for r in self.asp:
            print(r)

    def decode_data(self, X, show_with_asp=False):
        X_ext = []
        for x in X:
            x.append(0)
            X_ext.append(x)
        ret = decode_test_data(X_ext, self.attrs)
        if show_with_asp:
            for r in self.asp:
                print(r)
            print()
            for r in ret:
                print(r)
        return ret

    def save_model_to_file(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self, f)
        f.close()

    def save_asp_to_file(self, file_name):
        if self.asp is None and self.rules is not None:
            frs = flatten(self.rules)
            self.asp = decode_rules(frs, self.attrs)
        f = open(file_name, 'w')
        for r in self.asp:
            f.write(r + '\n')
        f.close()

    def load_translation(self, file_name):
        self.translation = []
        f = open(file_name, 'r')
        for line in f.readlines():
            self.translation.append(line.strip('\n'))

    def scasp_query(self, x):
        if self.asp is None and self.rules is not None:
            frs = flatten(self.rules)
            self.asp = decode_rules(frs, self.attrs)

        tf = tempfile.NamedTemporaryFile()
        for r in self.asp:
            tf.write((r + '\n').encode())

        if self.translation is not None:
            for t in self.translation:
                tf.write((t + '\n').encode())

        data_pred = decode_test_data([x], self.attrs, self.seq)
        for d in data_pred:
            tf.write((d + '\n').encode())

        extra = 'goal(X):-' + self.label.lower().replace(' ', '_') + '(X,\'' + self.pos.lower().replace(' ', '_') + '\').\n'
        tf.write(extra.encode())
        query = '?- goal(' + str(self.seq) + ').'
        tf.write(query.encode())
        tf.flush()
        tf.seek(0)
        print(tf.read().decode('utf-8'))
        self.seq += 1
        command = 'scasp' + ' -s1 --tree --human ' + tf.name
        res = subprocess.run([command], stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        tf.close()
        return res


def load_model_from_file(file_name):
    f = open(file_name, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret
