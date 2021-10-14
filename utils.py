import random
from foldrpp import *


def load_data(file, attrs=[], label=[], numerics=[], pos='', amount=-1):
    f = open(file, 'r')
    attr_idx, num_idx, lab_idx = [], [], -1
    ret, i, k = [], 0, 0
    head = ''
    for line in f.readlines():
        if i == 0:
            line = line.strip('\n').split(',')
            attr_idx = [j for j in range(len(line)) if line[j] in attrs]
            num_idx = [j for j in range(len(line)) if line[j] in numerics]
            for j in range(len(line)):
                if line[j] in label:
                    lab_idx = j
                    head += line[j].lower().replace(' ', '_')
                    head += '(X,'
                    if isinstance(pos, str):
                        head += '\'' + pos.lower().replace(' ', '_') + '\')'
                    else:
                        head += pos.lower().replace(' ', '_') + ')'
        else:
            line = line.strip('\n').split(',')
            r = [j for j in range(len(line))]
            for j in range(len(line)):
                if j in num_idx:
                    try:
                        r[j] = float(line[j])
                    except:
                        r[j] = line[j]
                else:
                    r[j] = line[j]
            r = [r[j] for j in attr_idx]
            if lab_idx != -1:
                y = 1 if line[lab_idx] == pos else 0
                r.append(y)
            ret.append(r)
        i += 1
        amount -= 1
        if amount == 0:
            break
    n_idx = []
    i = 0
    for j in attr_idx:
        if j in num_idx:
            n_idx.append(i)
        i += 1
    random.shuffle(ret)
    attrs.append(head)
    return ret, n_idx, attrs


def split_set(data, ratio=0.8):
    random.shuffle(data)
    num = int(len(data) * ratio)
    train, test = data[: num], data[num:]
    return train, test


def split_xy(data):
    feature, label = [], []
    for d in data:
        feature.append(d[: -1])
        label.append(int(d[-1]))
    return feature, label


def split_X_by_Y(X, Y):
    n = len(Y)
    X_pos = [X[i] for i in range(n) if Y[i]]
    X_neg = [X[i] for i in range(n) if not Y[i]]
    return X_pos, X_neg


def get_scores(Y_hat, Y):
    n = len(Y)
    if n == 0:
        return 0, 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(n):
        tp = tp + 1.0 if Y[i] and Y_hat[i] == Y[i] else tp
        tn = tn + 1.0 if not Y[i] and Y_hat[i] == Y[i] else tn
        fn = fn + 1.0 if Y[i] and Y_hat[i] != Y[i] else fn
        fp = fp + 1.0 if not Y[i] and Y_hat[i] != Y[i] else fp
    if tp < 1:
        p = 0 if fp < 1 else tp / (tp + fp)
        r = 0 if fn < 1 else tp / (tp + fn)
    else:
        p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = 0 if r * p == 0 else 2 * r * p / (r + p)
    return (tp + tn) / n, p, r, f1


def flatten(rules):
    ret = []
    rule_map = dict()
    flatten.ab = -2

    def _eval(i):
        if isinstance(i, tuple) and len(i) == 3:
            return i
        elif isinstance(i, tuple):
            return _func(i)

    def _func(rule, root=False):
        t = (tuple(rule[1]), tuple([_eval(i) for i in rule[2]]))
        if t not in rule_map:
            rule_map[t] = -1 if root else flatten.ab
            _ret = rule_map[t]
            ret.append((_ret, t[0], t[1]))
            if not root:
                flatten.ab -= 1
        return rule_map[t]

    for r in rules:
        _func(r, root=True)
    return ret


def decode_rules(rules, attrs):
    ret = []
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}

    def _f1(it):
        if isinstance(it, tuple) and len(it) == 3:
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -i - 2
                r = nr[r]
            k = attrs[i].lower().replace(' ', '_')
            if isinstance(v, str):
                v = v.lower().replace(' ', '_')
                v = 'null' if len(v) == 0 else '\'' + v + '\''
            if r == '==':
                return k + '(X,' + v + ')'
            elif r == '!=':
                return 'not ' + k + '(X,' + v + ')'
            else:
                return k + '(X,' + 'N' + str(i) + ')' + ',N' + str(i) + r + str(round(v, 3))
        elif it == -1:
            return attrs[-1]
        else:
            return'ab' + str(abs(it)) + '(X)'

    def _f2(rule):
        head = _f1(rule[0])
        body = ''
        for i in list(rule[1]):
            body = body + _f1(i) + ','
        tail = ''
        for i in list(rule[2]):
            t = _f1(i)
            if 'not' not in t:
                tail = tail + 'not ' + _f1(i) + ','
            else:
                t = t.replace('not ', '')
                tail = tail + t + ','
        _ret = head + ':-' + body + tail
        chars = list(_ret)
        chars[-1] = '.'
        _ret = ''.join(chars)
        _ret = _ret.replace('<=', '=<')
        return _ret

    for _r in rules:
        ret.append(_f2(_r))
    ret.sort()
    return ret


def function(rule):
    def _neg(it):
        if len(it) == 3:
            i, r, v = it[0], it[1], it[2]
            return -i - 2, r, v
        elif len(it) == 4:
            if len(it[1]) == 1 and len(it[2]) == 0:
                return _neg(it[1][0])
            elif len(it[1]) == 0 and len(it[2]) == 1:
                return it[2][0]
            else:
                if it[3] < 1:
                    return -1, [_neg(i) for i in it[1]] + [i for i in it[2]], [], it[3] ^ 1
                else:
                    if len(it[2]) > 0:
                        return -1, [(-1, [_neg(i) for i in it[1]], [], 0), (-1, [i for i in it[2]], [], 0)], [], 1
                    else:
                        return -1, [_neg(i) for i in it[1]], [], 0

    _def = rule[1]
    _add = []
    if len(rule[2]) > 0:
        if rule[3] > 0:
            pass
        if rule[3] < 1:
            for r in rule[2]:
                _add.append(_neg(function(r)))
            if len(_add) > 1:
                _def.extend(_add)
            else:
                _def.append(_add[0])
            ret = (-1, _def, [], 0)
        else:
            for r in rule[2]:
                _add.append(_neg(function(r)))
            if len(_def) == 0:
                ret = (-1, _add, [], 1)
            else:
                ret = (-1, [(-1, _def, [], 1), (-1, _add, [], 1)], [], 0)
    else:
        ret = (-1, _def, [], rule[3])
    return ret


def flatten_rule(rule):
    def _mul(a, b):
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a
        _ret = []
        for i in a:
            for j in b:
                if isinstance(i, tuple) and len(i) == 3:
                    i = [i]
                if isinstance(j, tuple) and len(j) == 3:
                    j = [j]
                k = []
                for _i in i:
                    if _i not in k:
                        k.append(_i)
                for _j in j:
                    if _j not in k:
                        k.append(_j)
                if k not in _ret:
                    _ret.append(k)
        return _ret

    def _func(i):
        if isinstance(i, tuple) and len(i) == 3:
            return [i]
        if isinstance(i, tuple) and len(i) == 4:
            return _dfs(i)

    def _dfs(_rule):
        _ret = []
        if _rule[3] == 1:
            for i in _rule[1]:
                if isinstance(i, tuple) and len(i) == 4:
                    for j in _func(i):
                        _ret.append(j)
                else:
                    _ret.append(_func(i))
            return _ret
        else:
            _ret = [[]]
            for i in _rule[1]:
                _ret = _mul(_ret, _func(i))
        return _ret

    rules = _dfs(rule)
    ret = []
    for r in rules:
        ret.append((-1, r, [], 0))
    return ret


def denot(rule, X_neg):
    def _func(_x, _i, _r, _v):
        if _i < -1:
            return _func(_x, -_i - 2, _r, _v) ^ 1
        if isinstance(_v, str):
            if _r == '==':
                return _x[_i] == _v
            elif _r == '!=':
                return _x[_i] != _v
            return False
        elif isinstance(_x[_i], str):
            return False
        elif _r == '<=':
            return _x[_i] <= _v
        elif _r == '>':
            return _x[_i] > _v
        return False

    def_pos, def_neg = [], []
    for i in rule[1]:
        if i[0] < -1:
            def_neg.append(i)
        elif i[0] > -1:
            def_pos.append(i)
    r = (-1, def_pos, [], 0)
    X_fp = [x for x in X_neg if cover(r, x, 1)]
    for i in def_neg:
        if any([_func(x, i[0], i[1], i[2]) ^ 1 for x in X_fp]):
            def_pos.append(i)
    return -1, def_pos, [], 0


def zip_rule(rule):
    def _func(t):
        if t[0] > -1 or isinstance(t[2], str):
            return t
        nr = {'<=': '>', '>': '<='}
        return -t[0] - 2, nr[t[1]], t[2]

    dft, tab = [], dict()
    def_pos = rule[1]
    for i in def_pos:
        i = _func(i)
        if isinstance(i[2], str):
            dft.append(i)
        else:
            if tab.get(i[0]) is None:
                tab[i[0]] = []
            if i[1] == '<=':
                tab[i[0]].append([float('-inf'), i[2]])
            elif i[1] == '>':
                tab[i[0]].append([i[2], float('inf')])
    for i in tab:
        left, right = float('inf'), float('-inf')
        for j in tab[i]:
            if j[0] == float('-inf'):
                left = min(left, j[1])
            else:
                right = max(right, j[0])
        if left == float('inf'):
            dft.append((i, '>', right))
        elif right == float('-inf'):
            dft.append((i, '<=', left))
        else:
            dft.append((i, '>', right))
            dft.append((i, '<=', left))
    return -1, dft, [], 0


def post_proc(rules, X_pos, X_neg):
    shifted_rules = [function(r) for r in rules]
    flattened_rules = []
    for r in shifted_rules:
        for f in flatten_rule(r):
            if len([i for i in range(len(X_pos)) if classify([f], X_pos[i]) > 0]) > 0:
                flattened_rules.append(f)
    denoted_rules = [denot(r, X_neg) for r in flattened_rules]
    zipped_rules = [zip_rule(r) for r in denoted_rules]
    final_rules = []
    for zr in zipped_rules:
        if zr not in final_rules:
            final_rules.append(zr)
    return final_rules


def num_predicates(rules):
    def _n_pred(rule):
        return len(rule[1] + rule[2])
    ret = 0
    for r in rules:
        ret += _n_pred(r)
    return ret


def decode_test_data(data, attrs):
    ret = []
    _, n = np.shape(data)
    i = 1
    for d in data:
        ret.append('id(' + str(i) + ').')
        for j in range(n - 1):
            if isinstance(d[j], float) or isinstance(d[j], int):
                pred = attrs[j].lower() + '(' + str(i) + ',' + str(d[j]) + ').'
            elif len(d[j]) > 0:
                pred = attrs[j].lower() + '(' + str(i) + ',\'' + str(d[j]).lower().replace(' ', '_') + '\').'
            else:
                pred = attrs[j].lower() + '(' + str(i) + ',\'' + 'null' + '\').'
            ret.append(pred)
        ret.append(' ')
        i += 1
    return ret
