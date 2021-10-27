import numpy as np


def evaluate(rule, x):
    def _func(i, r, v):
        if i < -1:
            return _func(-i - 2, r, v) ^ 1
        if isinstance(v, str):
            if r == '==':
                return x[i] == v
            elif r == '!=':
                return x[i] != v
            else:
                return False
        elif isinstance(x[i], str):
            return False
        elif r == '<=':
            return x[i] <= v
        elif r == '>':
            return x[i] > v
        else:
            return False

    def _eval(i):
        if len(i) == 3:
            return _func(i[0], i[1], i[2])
        elif len(i) == 4:
            return evaluate(i, x)

    if len(rule) == 0:
        return 0
    if rule[3] == 0 and not all([_eval(i) for i in rule[1]]):
        return 0
    if rule[3] == 1 and not any([_eval(i) for i in rule[1]]):
        return 0
    if len(rule[2]) > 0 and any([_eval(i) for i in rule[2]]):
        return 0
    return 1


def cover(rules, x, y):
    return int(evaluate(rules, x) == y)


def classify(rules, x):
    return int(any([evaluate(r, x) for r in rules]))


def predict(rules, X):
    ret = []
    for x in X:
        ret.append(classify(rules, x))
    return ret


def ig(tp, fn, tn, fp):
    if tp + tn < fp + fn:
        return float('-inf')
    ret = 0
    tot_p, tot_n = float(tp + fp), float(tn + fn)
    tot = float(tot_p + tot_n)
    if tp > 0:
        ret += tp / tot * np.log(tp / tot_p)
    if fp > 0:
        ret += fp / tot * np.log(fp / tot_p)
    if tn > 0:
        ret += tn / tot * np.log(tn / tot_n)
    if fn > 0:
        ret += fn / tot * np.log(fn / tot_n)
    return ret


def best_ig(X_pos, X_neg, i, used_items=[]):
    xp, xn, cp, cn = 0, 0, 0, 0
    pos, neg = dict(), dict()
    xs, cs = set(), set()

    for d in X_pos:
        if pos.get(d[i]) is None:
            pos[d[i]], neg[d[i]] = 0, 0
        if isinstance(d[i], str):
            cs.add(d[i])
            pos[d[i]] += 1.0
            cp += 1.0
        else:
            xs.add(d[i])
            pos[d[i]] += 1.0
            xp += 1.0

    for d in X_neg:
        if neg.get(d[i]) is None:
            pos[d[i]], neg[d[i]] = 0, 0
        if isinstance(d[i], str):
            cs.add(d[i])
            neg[d[i]] += 1.0
            cn += 1.0
        else:
            xs.add(d[i])
            neg[d[i]] += 1.0
            xn += 1.0

    xs = list(xs)
    xs.sort()
    for j in range(1, len(xs)):
        pos[xs[j]] += pos[xs[j - 1]]
        neg[xs[j]] += neg[xs[j - 1]]

    best, v, r = float('-inf'), float('-inf'), ''

    for x in xs:
        if (i, '<=', x) not in used_items and (i, '>', x) not in used_items:
            ifg = ig(pos[x], xp - pos[x] + cp, xn - neg[x] + cn, neg[x])
            if best < ifg:
                best, v, r = ifg, x, '<='
            ifg = ig(xp - pos[x], pos[x] + cp, neg[x] + cn, xn - neg[x])
            if best < ifg:
                best, v, r = ifg, x, '>'

    for c in cs:
        if (i, '==', c) not in used_items and (i, '!=', c) not in used_items:
            ifg = ig(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c])
            if best < ifg:
                best, v, r = ifg, c, '=='
            ifg = ig(cp - pos[c] + xp, pos[c], neg[c], cn - neg[c] + xn)
            if best < ifg:
                best, v, r = ifg, c, '!='
    return best, r, v


def best_feat(X_pos, X_neg, used_items=[]):
    if len(X_pos) == 0 and len(X_neg) == 0:
        return -1, '', ''
    n = len(X_pos[0]) if len(X_pos) > 0 else len(X_neg[0])
    _best = float('-inf')
    i, r, v = -1, '', ''
    for _i in range(n):
        bg, _r, _v = best_ig(X_pos, X_neg, _i, used_items)
        if _best < bg:
            _best = bg
            i, r, v = _i, _r, _v
    return i, r, v


def fold(X_pos, X_neg, used_items=[], ratio=0.5):
    ret = []
    while len(X_pos) > 0:
        rule = learn_rule(X_pos, X_neg, used_items, ratio)
        tp = [i for i in range(len(X_pos)) if cover(rule, X_pos[i], 1)]
        X_pos = [X_pos[i] for i in range(len(X_pos)) if i not in set(tp)]
        if len(tp) == 0:
            break
        ret.append(rule)
    return ret


def learn_rule(X_pos, X_neg, used_items=[], ratio=0.5):
    items = []
    flag = False
    while True:
        t = tuple(best_feat(X_pos, X_neg, used_items + items))
        items.append(t)
        rule = (-1, items, [], 0)
        X_tp = [X_pos[i] for i in range(len(X_pos)) if cover(rule, X_pos[i], 1)]
        X_fp = [X_neg[i] for i in range(len(X_neg)) if cover(rule, X_neg[i], 1)]
        if t[0] == -1 or len(X_fp) <= len(X_tp) * ratio:
            if t[0] == -1:
                items.pop()
                rule = (-1, items, [], 0)
            if len(X_fp) > 0 and t[0] != -1:
                flag = True
            break
        X_pos = X_tp
        X_neg = X_fp
    if flag:
        ab = fold(X_fp, X_tp, used_items + items, ratio)
        if len(ab) > 0:
            rule = (rule[0], rule[1], ab, 0)
    return rule