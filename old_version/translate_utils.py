# This file only provide two functions "translate_rules" and "translate_proof"
# if you need to use these two functions:
# ------------------------------------------------------------
# from translate_utils import translate_rules, translate_proof
# print(translate_rules(model, file='path to template'))
# for x in X:
#     print(translate_proof(model, x, file='path to template'))
# ------------------------------------------------------------
# Please find more examples in main and titanic_test functions in this file.


from algo import evaluate, justify_one
from foldrpp import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta


def load_template(file):
    f = open(file, 'r')
    ret = dict()
    for line in f.readlines():
        line = line.strip('\n')
        if len(line) == 0:
            continue
        if line[0] == '#' and 'pred' in line:
            strs = line.split('pred')[1]
            strs = strs.split('::')
            head = strs[0].strip(' ')
            tail = strs[1].strip(' ')
            heads = head.split('(')
            k, paras = heads[0].lower().replace(' ', '_'), heads[1].strip(')').split(',')
            if len(paras) > 1 and isinstance(paras[1], str) and not ('A' <= paras[1] <= 'Z'):
                ret[(k, 'parameter', paras[1])] = paras
                if tail[0] == '\'':
                    tails = tail.split('\'')
                    tail = tails[1]
                elif tail[0] == '\"':
                    tails = tail.split('\"')
                    tail = tails[1]
                ret[(k, 'phrase', paras[1])] = tail
            else:
                ret[(k, 'parameter')] = paras
                if tail[0] == '\'':
                    tails = tail.split('\'')
                    tail = tails[1]
                elif tail[0] == '\"':
                    tails = tail.split('\"')
                    tail = tails[1]
                ret[(k, 'phrase')] = tail
    return ret


def translate(rules, attrs, tmpl={}):
    ret = []
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}

    def _f0(i, r, v):
        k = attrs[i].lower().replace(' ', '_')
        if isinstance(v, str):
            v = v.lower().replace(' ', '_')
            v = 'null' if len(v) == 0 else v
        if isinstance(v, str) and (k, 'parameter', v) in tmpl:
            para, s = tmpl[(k, 'parameter', v)], tmpl[(k, 'phrase', v)]
        elif (k, 'parameter') in tmpl:
            para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
        else:
            if r == '==':
                s = 'the value of ' + k + ' is \'' + v + '\''
            elif r == '!=':
                s = 'the value of ' + k + ' is not \'' + v + '\''
            else:
                if r == '<=':
                    s = 'the value of ' + k + ' is less equal to ' + str(round(v, 3))
                else:
                    s = 'the value of ' + k + ' is greater than ' + str(round(v, 3))
            return s
        if len(para) < 2:
            s = s.replace('@(' + para[0] + ')', 'X')
            if r == '!=':
                s = ' not ' + s
            return s
        else:
            if r == '==':
                s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', str(v))
                return s
            s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', 'N' + str(i))
            if r == '!=':
                s = s + ' where N' + str(i) + ' is not ' + v
            elif r == '<=':
                s = s + ' where N' + str(i) + ' is less equal to ' + str(round(v, 3))
            else:
                s = s + ' where N' + str(i) + ' is greater than ' + str(round(v, 3))
            return s

    def _f1(it):
        if isinstance(it, tuple) and len(it) == 3:
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -i - 2
                r = nr[r]
            return _f0(i, r, v)
        elif it == -1:
            heads = attrs[-1].split('(')
            k, paras = heads[0].lower().replace(' ', '_'), heads[1].replace('\'', '').strip(')').split(',')
            if len(paras) < 2 and (k, 'parameter') in tmpl:
                para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
                s = s.replace('@(' + para[0] + ')', 'X')
            else:
                v = paras[1]
                if (k, 'parameter', v) in tmpl:
                    para, s = tmpl[(k, 'parameter', v)], tmpl[(k, 'phrase', v)]
                    s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', str(v))
                else:
                    s = attrs[-1]
            return s
        else:
            k = 'ab' + str(abs(it) - 1)
            if (k, 'parameter') in tmpl:
                para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
                s = s.replace('@(' + para[0] + ')', 'X')
            else:
                s = 'exception ab' + str(abs(it) - 1)
            return s

    def _f3(rule, indent=0):
        head = '\t' * indent + _f1(rule[0]) + ' is True, if \n'
        body = ''
        for i in list(rule[1]):
            body = body + '\t' * (indent + 1) + _f1(i) + ' and\n'
        tail = ''
        for i in list(rule[2]):
            for r in rules:
                if i == r[0]:
                    tail = tail + '\t' * (indent + 1) + _f1(i) + ' is False and\n'
        _ret = head + body + tail
        chars = list(_ret)
        _ret = ''.join(chars)
        if _ret.endswith('and\n'):
            _ret = _ret[:-4] + '\n\n'
        return _ret

    for _r in rules:
        ret.append(_f3(_r))
    return ret


def proof_trans(rules, attrs, x, tmpl={}):
    ret = []
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}

    def _f0(i, r, v):
        k = attrs[i].lower().replace(' ', '_')
        if isinstance(v, str):
            v = v.lower().replace(' ', '_')
            v = 'null' if len(v) == 0 else v
        if isinstance(v, str) and (k, 'parameter', v) in tmpl:
            para, s = tmpl[(k, 'parameter', v)], tmpl[(k, 'phrase', v)]
        elif (k, 'parameter') in tmpl:
            para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
        else:
            if r == '==':
                s = 'the value of ' + k + ' is \'' + v + '\''
            elif r == '!=':
                s = 'the value of ' + k + ' is not \'' + v + '\''
            else:
                if r == '<=':
                    s = 'the value of ' + k + ' is less equal to ' + str(round(v, 3))
                else:
                    s = 'the value of ' + k + ' is greater than ' + str(round(v, 3))
            return s
        if len(para) < 2:
            s = s.replace('@(' + para[0] + ')', 'X')
            if r == '!=':
                s = ' not ' + s
            return s
        else:
            if r == '==':
                s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', str(v))
                return s
            s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', 'N' + str(i))
            if r == '!=':
                s = s + ' where N' + str(i) + ' is not ' + v
            elif r == '<=':
                s = s + ' where N' + str(i) + ' is less equal to ' + str(round(v, 3))
            else:
                s = s + ' where N' + str(i) + ' is greater than ' + str(round(v, 3))
            return s

    def _f2(i, r, v):
        k = attrs[i].lower().replace(' ', '_')
        v = x[i]
        if isinstance(v, str):
            v = v.lower().replace(' ', '_')
            v = 'null' if len(v) == 0 else v
        if isinstance(v, str) and (k, 'parameter', v) in tmpl:
            para, s = tmpl[(k, 'parameter', v)], tmpl[(k, 'phrase', v)]
        elif (k, 'parameter') in tmpl:
            para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
        else:
            if isinstance(v, str):
                s = 'the value of ' + k + ' is \'' + str(v) + '\''
            else:
                s = 'the value of ' + k + ' is \'' + str(round(v, 3)) + '\''
            return s
        if len(para) < 2:
            s = s.replace('@(' + para[0] + ')', 'X')
            return s
        else:
            if isinstance(v, str):
                s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', str(v))
            else:
                s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', str(round(v, 3)))
            return s

    def _f4(it):
        prefix = ' DOES HOLD because ' if evaluate(it, x) else ' DOES NOT HOLD because '
        if isinstance(it, tuple) and len(it) == 3:
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -i - 2
                r = nr[r]
            return prefix + _f2(i, r, v)
        elif it == -1:
            heads = attrs[-1].split('(')
            k, paras = heads[0].lower().replace(' ', '_'), heads[1].replace('\'', '').strip(')').split(',')
            if len(paras) < 2 and (k, 'parameter') in tmpl:
                para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
                s = s.replace('@(' + para[0] + ')', 'X')
            else:
                v = paras[1]
                v = x[it[0]]
                if (k, 'parameter', v) in tmpl:
                    para, s = tmpl[(k, 'parameter', v)], tmpl[(k, 'phrase', v)]
                    s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', str(v))
                else:
                    s = attrs[-1]
            return prefix + s
        else:
            k = 'ab' + str(abs(it) - 1)
            if (k, 'parameter') in tmpl:
                para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
                s = s.replace('@(' + para[0] + ')', 'X')
            else:
                s = 'exception ab' + str(abs(it) - 1)
            return prefix + s

    def _f1(it):
        if isinstance(it, tuple) and len(it) == 3:
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -i - 2
                r = nr[r]
            return '' + _f0(i, r, v)
        elif it == -1:
            suffix = ' DOES HOLD ' if justify_one(rules, x, it)[0] else ' DOES NOT HOLD '
            heads = attrs[-1].split('(')
            k, paras = heads[0].lower().replace(' ', '_'), heads[1].replace('\'', '').strip(')').split(',')
            if len(paras) < 2 and (k, 'parameter') in tmpl:
                para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
                s = s.replace('@(' + para[0] + ')', 'X')
            else:
                v = paras[1]
                if (k, 'parameter', v) in tmpl:
                    para, s = tmpl[(k, 'parameter', v)], tmpl[(k, 'phrase', v)]
                    s = s.replace('@(' + para[0] + ')', 'X').replace('@(' + para[1] + ')', str(v))
                else:
                    s = attrs[-1]
            return s + suffix
        else:
            if it not in [r[0] for r in rules]:
                suffix = ''
            else:
                suffix = ' DOES HOLD ' if justify_one(rules, x, it)[0] else ' DOES NOT HOLD '

            k = 'ab' + str(abs(it) - 1)
            if (k, 'parameter') in tmpl:
                para, s = tmpl[(k, 'parameter')], tmpl[(k, 'phrase')]
                s = s.replace('@(' + para[0] + ')', 'X')
            else:
                s = 'exception ab' + str(abs(it) - 1)
            return s + suffix

    def _f3(rule, indent=0):
        head = '\t' * indent + _f1(rule[0]) + 'because \n'
        body = ''
        for i in list(rule[1]):
            body = body + '\t' * (indent + 1) + _f1(i) + '' + _f4(i) + ' and\n'
        tail = ''
        for i in list(rule[2]):
            for r in rules:
                if i == r[0]:
                    tail = tail + _f3(r, indent + 1)
        _ret = head + body + tail
        chars = list(_ret)
        _ret = ''.join(chars)
        if _ret.endswith('and\n'):
            _ret = _ret[:-4] + '\n'
        return _ret

    for _r in rules:
        if _r[0] == -1:
            ret.append(_f3(_r))
    return ret


def translate_rules(model, file=None):
    if file:
        tmpl = load_template(file)
        text = translate(model.frs, model.attrs, tmpl)
    else:
        text = translate(model.frs, model.attrs)
    ret = ''
    for t in text:
        ret = ret + t
    return ret


def translate_proof(model, x, all_flag=False, file=None):
    ret = ''
    model.asp()
    all_pos = justify(model.frs, x, all_flag=all_flag)
    k = 1
    tmpl = load_template(file) if file else {}
    if len(all_pos) == 0:
        all_neg = rebut(model.frs, x)
        for rs in all_neg:
            ret += 'rebuttal ' + str(k) + ':\n'
            for r in proof_trans(rs, attrs=model.attrs, x=x, tmpl=tmpl):
                ret += r
            ret += str(justify_data(rs, x, attrs=model.attrs)) + '\n'
            k += 1
    else:
        for rs in all_pos:
            ret += 'answer ' + str(k) + ':\n'
            for r in proof_trans(rs, attrs=model.attrs, x=x, tmpl=tmpl):
                ret += r
            ret += str(justify_data(rs, x, attrs=model.attrs)) + '\n'
            k += 1
    return ret


def titanic_test():
    model, data_train, data_test = titanic()
    X_train, Y_train = split_xy(data_train)
    X_test, Y_test = split_xy(data_test)

    model.fit(X_train, Y_train, ratio=0.5)
    Y_test_hat = model.predict(X_test)
    model.print_asp()
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4), '\n')
    print(translate_rules(model, file='data/titanic/template.txt'))
    # exit()
    k = 1
    for i in range(len(X_test)):
        print('Proof Trees for example number', k, ':')
        print(translate_proof(model, X_test[i], file='data/titanic/template.txt'))
        k += 1


def main():
    # model, data = acute()
    # model, data = autism()
    model, data = breastw()
    # model, data = cars()
    # model, data = credit()
    # model, data = heart()
    # model, data = kidney()
    # model, data = krkp()
    # model, data = mushroom()
    # model, data = sonar()
    # model, data = voting()
    # model, data = ecoli()
    # model, data = ionosphere()
    # model, data = wine()
    # model, data = adult()
    # model, data = credit_card()
    # model, data = rain()
    # model, data = heloc()

    data_train, data_test = split_data(data, ratio=0.8, rand=True)

    X_train, Y_train = split_xy(data_train)
    X_test,  Y_test = split_xy(data_test)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.6)
    end = timer()

    model.print_asp()
    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')
    print(translate_rules(model))
    # exit()
    k = 1
    for i in range(len(X_test)):
        print('Proof Trees for example number', k, ':')
        print(translate_proof(model, X_test[i]))
        k += 1


if __name__ == '__main__':
    # titanic_test()
    main()
