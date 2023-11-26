import random
from algo import evaluate, justify_one


def load_data(file, attrs, label, numerics, pos='', amount=-1):
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
                if line[j] == label:
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
    attrs.append(head)
    return ret, attrs


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


def split_data(data, ratio=0.8, rand=True):
    if rand:
        random.shuffle(data)
    num = int(len(data) * ratio)
    train, test = data[: num], data[num:]
    return train, test


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


def justify_data(frs, x, attrs):
    ret = []
    for r in frs:
        d = r[1]
        for j in d:
            ret.append(attrs[j[0]] + ': ' + str(x[j[0]]))
    return set(ret)


def decode_rules(rules, attrs, x=None):
    ret = []
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}

    def _f1(it):
        prefix, not_prefix = '', ''
        if isinstance(it, tuple) and len(it) == 3:
            if x is not None:
                prefix = '[T]' if evaluate(it, x) else '[F]'
                not_prefix = '[T]' if prefix == '[F]' else '[F]'
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -2 - i
                r = nr[r]
            k = attrs[i].lower().replace(' ', '_')
            if isinstance(v, str):
                v = v.lower().replace(' ', '_')
                v = 'null' if len(v) == 0 else '\'' + v + '\''
            if r == '==':
                return prefix + k + '(X,' + v + ')'
            elif r == '!=':
                return 'not ' + not_prefix + k + '(X,' + v + ')'
            else:
                return prefix + k + '(X,' + 'N' + str(i) + ')' + ', N' + str(i) + r + str(round(v, 3))
        elif it == -1:
            if x is not None:
                prefix = '[T]' if justify_one(rules, x, it)[0] else '[F]'
            return prefix + attrs[-1]
        else:
            if x is not None:
                if it not in [r[0] for r in rules]:
                    prefix = '[U]'
                else:
                    prefix = '[T]' if justify_one(rules, x, it)[0] else '[F]'
            return prefix + 'ab' + str(abs(it) - 1) + '(X)'

    def _f2(rule):
        head = _f1(rule[0])
        body = ''
        for i in list(rule[1]):
            body = body + _f1(i) + ', '
        tail = ''
        for i in list(rule[2]):
            t = _f1(i)
            if 'not' not in t:
                tail = tail + 'not ' + _f1(i) + ', '
            else:
                t = t.replace('not ', '')
                tail = tail + t + ', '
        _ret = head + ' :- ' + body + tail
        chars = list(_ret)
        chars[-2] = '.'
        _ret = ''.join(chars)
        _ret = _ret.replace('<=', '=<')
        return _ret

    for _r in rules:
        ret.append(_f2(_r))
    return ret


def zip_rule(rule):
    tab, dft = {}, []
    for i in rule[1]:
        if isinstance(i[2], str):
            dft.append(i)
        else:
            if i[0] not in tab:
                tab[i[0]] = []
            if i[1] == '<=':
                tab[i[0]].append([float('-inf'), i[2]])
            else:
                tab[i[0]].append([i[2], float('inf')])
    nums = [t for t in tab]
    nums.sort()
    for i in nums:
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
    return rule[0], dft, rule[2], 0


def simplify_rule(rule):
    head, body = rule.split(' :- ')
    items = body.split(', ')
    items = list(dict.fromkeys(items))
    body = ', '.join(items)
    return head + ' :- ' + body


def proof_tree(rules, attrs, x):
    ret = []
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}

    def _f1(it):
        if isinstance(it, tuple) and len(it) == 3:
            suffix = ' (DOES HOLD) ' if evaluate(it, x) else ' (DOES NOT HOLD) '
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -2 - i
                r = nr[r]
            k = attrs[i].lower().replace(' ', '_')
            if isinstance(v, str):
                v = v.lower().replace(' ', '_')
                v = 'null' if len(v) == 0 else '\'' + v + '\''
            xi = x[i]
            if isinstance(xi, str):
                xi = xi.lower().replace(' ', '_')
                xi = '\'null\'' if len(xi) == 0 else xi
            if r == '==':
                return 'the value of ' + k + ' is \'' + str(xi) + '\' which should equal ' + v + suffix
            elif r == '!=':
                return 'the value of ' + k + ' is \'' + str(xi) + '\' which should not equal ' + v + suffix
            else:
                if r == '<=':
                    return 'the value of ' + k + ' is ' + str(xi) + ' which should be less equal to ' + str(round(v, 3)) + suffix
                else:
                    return 'the value of ' + k + ' is ' + str(xi) + ' which should be greater than ' + str(round(v, 3)) + suffix
        elif it == -1:
            suffix = ' DOES HOLD ' if justify_one(rules, x, it)[0] else ' DOES NOT HOLD '
            return attrs[-1] + suffix
        else:
            if it not in [r[0] for r in rules]:
                suffix = ''
            else:
                suffix = ' DOES HOLD ' if justify_one(rules, x, it)[0] else ' DOES NOT HOLD '
            return 'exception ab' + str(abs(it) - 1) + suffix

    def _f2(rule, indent=0):
        head = '\t' * indent + _f1(rule[0]) + 'because \n'
        body = ''
        for i in list(rule[1]):
            body = body + '\t' * (indent + 1) + _f1(i) + '\n'
        tail = ''
        for i in list(rule[2]):
            for r in rules:
                if i == r[0]:
                    tail = tail + _f2(r, indent + 1)
        _ret = head + body + tail
        chars = list(_ret)
        _ret = ''.join(chars)
        return _ret

    for _r in rules:
        if _r[0] == -1:
            ret.append(_f2(_r))
    return ret


def num_predicates(rules):
    def _n_pred(rule):
        return len(rule[1] + rule[2])
    ret = 0
    for r in rules:
        ret += _n_pred(r)
    return ret


