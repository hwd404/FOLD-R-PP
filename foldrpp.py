
def load_data(file_name, str_attrs, num_attrs, label, pos_val, amount=-1):
    data_file = open(file_name, 'r')
    str_attr_idx, num_attr_idx, lbl_idx = [], [], -1
    ret, ln = [], 0
    for line in data_file.readlines():
        line = line.strip('\n').split(',')
        if ln == 0:
            str_attr_idx = [i for i, s in enumerate(line) if s in str_attrs]
            num_attr_idx = [i for i, n in enumerate(line) if n in num_attrs]
            str_attrs = [attr.lower().replace(' ', '_') for attr in str_attrs]
            num_attrs = [attr.lower().replace(' ', '_') for attr in num_attrs]
            lbl_idx = line.index(label)
        else:
            line_dict = {}
            for i, s in enumerate(line):
                if len(s) == 0:
                    continue
                if i in num_attr_idx:
                    key = num_attrs[num_attr_idx.index(i)]
                    try:
                        line_dict[key] = float(s)
                    except ValueError:
                        line_dict[key] = s
                elif i in str_attr_idx:
                    key = str_attrs[str_attr_idx.index(i)]
                    line_dict[key] = s
            y = 1 if line[lbl_idx] == pos_val else 0
            line_dict['label'] = y
            ret.append(line_dict)
        ln += 1
        amount -= 1
        if amount == 0:
            break
    return ret


def split_index_by_label(data):
    pos = [i for i, x in enumerate(data) if x['label']]
    neg = [i for i, x in enumerate(data) if not x['label']]
    return pos, neg


def split_data(data, ratio=0.8, rand=True):
    if rand:
        import random
        random.shuffle(data)
    num = int(len(data) * ratio)
    train, test = data[: num], data[num:]
    return train, test


def eval_item(item, x):
    """ item: tuple(attr, op, val) """
    attr, op, val = item
    x_val = x[attr] if attr in x else ''
    if attr is None:
        return False
    if isinstance(val, str):
        if op == '==':
            return x_val == val
        elif op == '!=':
            return x_val != val
        else:
            return False 
    elif isinstance(x_val, str):
        return False 
    elif op == '=<':
        return x_val <= val
    elif op == '>':
        return x_val > val
    else:
        return False 


def evaluate(rule, x):
    """ rule: tuple(head, main_items, ab_items) """
    _, main_items, ab_items = rule
    if not main_items and not ab_items:
        return False 
    if main_items and not all((eval_item(i, x) for i in main_items)):
        return False 
    if ab_items and any((evaluate(ab, x) for ab in ab_items)):
        return False
    return True 


def classify(rules, x):
    return any((evaluate(r, x) for r in rules))


def predict(rules, data):
    return [classify(rules, x) for x in data]


def heuristic(tp, fn, tn, fp):
    import math
    if tp + tn < fp + fn:
        return float('-inf')
    ret = 0
    tot_p, tot_n = float(tp + fp), float(tn + fn)
    tot = float(tot_p + tot_n)
    ret += tp / tot * math.log(tp / tot_p) if tp > 0 else 0
    ret += fp / tot * math.log(fp / tot_p) if fp > 0 else 0
    ret += tn / tot * math.log(tn / tot_n) if tn > 0 else 0
    ret += fn / tot * math.log(fn / tot_n) if fn > 0 else 0
    return ret


def best_item_on_attr(data, pos_idx, neg_idx, attr, used_items):
    """ item: tuple(attr, op, val) """
    num_pos, num_neg, str_pos, str_neg = 0, 0, 0, 0
    pos_cnt, neg_cnt = {}, {}
    nums, strs = set(), set()
    # noinspection DuplicatedCode
    for i in pos_idx:
        attr_val = data[i][attr] if attr in data[i] else ''
        if attr_val not in pos_cnt:
            pos_cnt[attr_val], neg_cnt[attr_val] = 0, 0
        pos_cnt[attr_val] += 1.0
        if isinstance(attr_val, str):
            strs.add(attr_val)
            str_pos += 1.0
        else:
            nums.add(attr_val)
            num_pos += 1.0
    # noinspection DuplicatedCode
    for i in neg_idx:
        attr_val = data[i][attr] if attr in data[i] else ''
        if attr_val not in neg_cnt:
            pos_cnt[attr_val], neg_cnt[attr_val] = 0, 0
        neg_cnt[attr_val] += 1.0
        if isinstance(attr_val, str):
            strs.add(attr_val)
            str_neg += 1.0
        else:
            nums.add(attr_val)
            num_neg += 1.0
    nums = sorted(list(nums))
    for i in range(1, len(nums)):
        pos_cnt[nums[i]] += pos_cnt[nums[i - 1]]
        neg_cnt[nums[i]] += neg_cnt[nums[i - 1]]
    best_score, op, val = float('-inf'), None, None
    for n in nums:
        pos_cnt_n, neg_cnt_n = pos_cnt[n], neg_cnt[n]
        if (attr, '=<', n) not in used_items and (attr, '>', n) not in used_items:
            score = heuristic(pos_cnt_n,
                              num_pos - pos_cnt_n + str_pos,
                              num_neg - neg_cnt_n + str_neg,
                              neg_cnt_n)
            if best_score < score:
                best_score, op, val = score, '=<', n
            score = heuristic(num_pos - pos_cnt_n,
                              pos_cnt_n + str_pos,
                              neg_cnt_n + str_neg,
                              num_neg - neg_cnt_n)
            if best_score < score:
                best_score, op, val = score, '>', n
    for s in strs:
        pos_cnt_s, neg_cnt_s = pos_cnt[s], neg_cnt[s]
        if (attr, '==', s) not in used_items and (attr, '!=', s) not in used_items:
            score = heuristic(pos_cnt_s,
                              str_pos - pos_cnt_s + num_pos,
                              str_neg - neg_cnt_s + num_neg,
                              neg_cnt_s)
            if best_score < score:
                best_score, op, val = score, '==', s
            score = heuristic(str_pos - pos_cnt_s + num_pos,
                              pos_cnt_s,
                              neg_cnt_s,
                              str_neg - neg_cnt_s + num_neg)
            if best_score < score:
                best_score, op, val = score, '!=', s
    return best_score, (attr, op, val)


def best_item(data, pos_idx, neg_idx, attrs, used_items):
    ret = None, None, None
    if not pos_idx + neg_idx:
        return ret
    best_score = float('-inf')
    for attr in attrs:
        score, item = best_item_on_attr(data, pos_idx, neg_idx, attr, used_items)
        if best_score < score:
            best_score, ret = score, item
    return ret


def learn_rule_set(data, pos_idx, neg_idx, attrs, used_items, ratio=0.5):
    ret = []
    while pos_idx:
        rule = learn_rule(data, pos_idx, neg_idx, attrs, used_items, ratio)
        fn_idx = [i for i in pos_idx if not evaluate(rule, data[i])]
        if len(pos_idx) == len(fn_idx):
            break
        pos_idx = fn_idx
        ret.append(rule)
    return ret


def learn_rule(data, pos_idx, neg_idx, attrs, used_items, ratio=0.5):
    """ item: tuple(attr, op, val) """
    """ rule: tuple(head, main_items, ab_items) """
    items = []
    while True:
        item = best_item(data, pos_idx, neg_idx, attrs, used_items + items)
        items.append(item)
        rule = -1, items, []
        pos_idx = [i for i in pos_idx if evaluate(rule, data[i])]
        neg_idx = [i for i in neg_idx if evaluate(rule, data[i])]
        if not item[0] or len(neg_idx) <= len(pos_idx) * ratio:
            if not item[0]:
                rule = -1, items[: -1], []
            if neg_idx and item[0]:
                ab_rules = learn_rule_set(data, neg_idx, pos_idx, attrs, used_items + items, ratio)
                if ab_rules:
                    rule = -1, rule[1], ab_rules
            break
    return rule


def unite_like_items(rule):
    """ item: tuple(attr, op, val) """
    head, main_items, ab_items = rule
    new_main, tab = [], {}
    for item in main_items:
        attr, op, val = item
        if isinstance(val, str):
            new_main.append(item)
            continue
        if attr not in tab:
            tab[attr] = []
        if op == '=<':
            tab[attr].append((float('-inf'), val, False))
        else:
            tab[attr].append((val, float('inf'), False))
    for attr in tab:
        left, right = float('inf'), float('-inf')
        for t in tab[attr]:
            if t[0] == float('-inf'):
                left = min(left, t[1])
            else:
                right = max(right, t[0])
        if left == float('inf'):
            new_main.append((attr, '>', right))
        elif right == float('-inf'):
            new_main.append((attr, '=<', left))
        else:
            new_main.extend([(attr, '>', right), (attr, '=<', left)])
    return head, new_main, ab_items


def unite_atom_ab(rule):
    """ item: tuple(attr, op, val) """
    """ rule: tuple(head, main_items, ab_items) """
    def _is_atom(_rule):
        if len(_rule[1]) == 1 and len(_rule[2]) == 0:
            return True, _rule[1][0]
        return False, None

    def _neg_item(_item):
        neg_tab = {'==': '!=', '!=': '==', '=<': '>', '>': '=<'}
        attr, op, val = _item
        op = neg_tab[op]
        return attr, op, val

    head, main_items, ab_items = rule
    if len(ab_items) == 0:
        return unite_like_items(rule)

    ab_items = [unite_atom_ab(_ab) for _ab in ab_items]
    new_ab_items = []
    for ab in ab_items:
        res, item = _is_atom(ab)
        if res:
            main_items.append(_neg_item(item))
        else:
            new_ab_items.append(ab)
    ret = head, main_items, new_ab_items
    return unite_like_items(ret)


def flatten_rules(rules, rule_head):
    """ item: tuple(attr, op, val) """
    """ rule: tuple(head, main_items, ab_items) """
    def_rules, ab_rules = [], []
    rule_map = {}
    flatten_rules.ab_idx = 1

    def _flattened_idx(rule, is_root=False):
        key = tuple(rule[1]), tuple([_flattened_idx(_r) for _r in rule[2]])
        if key not in rule_map:
            rule_map[key] = rule[0] if is_root else flatten_rules.ab_idx
            if rule[0] == -1:
                rule_map[key] = rule_head if is_root else ('ab' + str(flatten_rules.ab_idx), '==', 'True')
            head = rule_map[key]
            if is_root:
                def_rules.append((head, rule[1], list(key[1])))
            else:
                ab_rules.append((head, rule[1], list(key[1])))
                flatten_rules.ab_idx += 1
        elif is_root:
            def_rules.append((rule[0], rule[1], list(key[1])))
        return rule_map[key]

    for r in [unite_atom_ab(_r) for _r in rules]:
        _flattened_idx(r, True)
    return def_rules + ab_rules


# coverage_counter = {}


def justify_one(flat_rules, x, res, rule_head, start=0):
    for i in range(start, len(flat_rules)):
        head, main_items, ab_items = flat_rules[i]
        if tuple(head) != tuple(rule_head):
            continue
        if not all((eval_item(j, x) for j in main_items)):
            continue
        if ab_items:
            continue_flag = False
            for j in ab_items:
                if rebut_one(flat_rules, x, res=res, rule_head=j) < 0:
                    continue_flag = True
                    break
            if continue_flag:
                continue
        res.append(flat_rules[i])
        # if i not in coverage_counter:
        #     coverage_counter[i] = 0
        # coverage_counter[i] += 1
        return i
    return -1


def rebut_one(flat_rules, x, res, rule_head, start=0):
    for i in range(start, len(flat_rules)):
        head, main_items, ab_items = flat_rules[i]
        if tuple(head) != tuple(rule_head):
            continue
        if not all((eval_item(j, x) for j in main_items)):
            res.append(flat_rules[i])
            return i
        if ab_items:
            for j in ab_items:
                if justify_one(flat_rules, x, res=res, rule_head=j) > -1:
                    res.append(flat_rules[i])
                    return i
    return -1


def explain(flat_rules, x, rule_head, get_all=False):
    ret, i = [], 0
    y = True if justify_one(flat_rules, x, res=[], rule_head=rule_head) > -1 else False
    f = justify_one if y else rebut_one
    get_all = get_all if y else True
    while i < len(flat_rules):
        res = []
        idx = f(flat_rules, x, res=res, rule_head=rule_head, start=i)
        if idx > -1:
            ret.append(res)
            i = idx + 1
            if not get_all:
                break
        else:
            break
    return ret


def explain_data(flat_rules, x):
    ret = {}
    for rule in flat_rules:
        for item in rule[1]:
            attr, _, _ = item
            ret[attr] = x[attr]
    return ret


def decode_rules(rules, x=None):
    """ item: tuple(attr, op, val) """
    """ rule: tuple(head, main_items, ab_items) """
    attr_map, decode_rules.attr_idx = {}, 1

    def _attr_idx(attr):
        if attr not in attr_map:
            attr_map[attr] = decode_rules.attr_idx
            decode_rules.attr_idx += 1
        return str(attr_map[attr])

    def _item_prefix(item):
        if x is None:
            prefix = ''
        else:
            prefix = '[T]' if eval_item(item, x) else '[F]'
        return prefix

    def _rule_prefix(rule_idx):
        if x is None:
            prefix = ''
        elif rule_idx in [r[0] for r in rules]:
            prefix = '[T]' if justify_one(rules, x, res=[], rule_head=rule_idx) > -1 else '[F]'
        else:
            prefix = '[U]'
        return prefix

    def _decode_item(item):
        _neg_map = {'[T]': '[F]', '[F]': '[T]', '[U]': '[U]', '': ''}
        attr, op, val = item
        if isinstance(val, str):
            val = '\'' + val + '\''
        if op == '==':
            return _item_prefix(item) + attr + '(X,' + val + ')'
        elif op == '!=':
            return 'not ' + _neg_map[_item_prefix(item)] + attr + '(X,' + val + ')'
        else:
            return attr + '(X,N' + _attr_idx(attr) + '), ' + _item_prefix(item) + 'N' + _attr_idx(attr) + op + str(
                round(val, 3))

    def _decode_head(head):
        return _rule_prefix(head) + head[0] + '(X,\'' + head[2] + '\')'

    def _uniq_list(seq):
        seen = set()
        seen_add = seen.add
        return [s for s in seq if not (s in seen or seen_add(s))]

    def _decode_rule(rule):
        head = _decode_head(rule[0])
        body = ', '.join([_decode_item(i) for i in rule[1]])
        item_strs = body.split(', ')
        body = ', '.join(_uniq_list(item_strs))
        tail = ', '.join(['not ' + _decode_head(i) for i in rule[2]])
        return head + ' :- ' + body + (', ' + tail if len(tail) > 0 else '') + '.'

    return [_decode_rule(r) for r in rules]


def proof_tree(rules, x, rule_head):
    """ item: tuple(attr, op, val) """
    """ rule: tuple(head, main_items, ab_items) """
    attr_map, decode_rules.attr_idx = {}, 1

    def _item_suffix(item):
        return ' does hold' if eval_item(item, x) else ' does not hold'

    def _rule_suffix(_rule_head):
        return ' does hold' if justify_one(rules, x, rule_head=_rule_head, res=[]) > -1 else ' does not hold'

    def _strify(_val):
        if isinstance(_val, str):
            return '\'' + _val + '\''
        else:
            return str(round(_val, 3))

    def _decode_item(item):
        attr, op, val = item
        if op == '==':
            return 'the value of ' + attr + ' is ' + (
                _strify(x[attr]) if attr in x else 'null') + ' which should equal ' + _strify(val) + _item_suffix(
                item)
        elif op == '!=':
            return 'the value of ' + attr + ' is ' + (
                _strify(x[attr]) if attr in x else 'null') + ' which should not equal ' + _strify(
                val) + _item_suffix(item)
        elif op == '=<':
            return 'the value of ' + attr + ' is ' + (
                _strify(x[attr]) if attr in x else 'null') + ' which should be less equal to ' + _strify(
                val) + _item_suffix(item)
        elif op == '>':
            return 'the value of ' + attr + ' is ' + (
                _strify(x[attr]) if attr in x else 'null') + ' which should be greater than ' + _strify(
                val) + _item_suffix(item)
        else:
            return str(item)

    def _decode_head(head):
        prefix = '' if head == rule_head else 'exception '
        return prefix + head[0] + '(X,\'' + head[2] + '\')' + _rule_suffix(head) + ' because'

    def _decode_rule(rule, indent=0):
        head = '\t' * indent + _decode_head(rule[0]) + '\n'
        body = ''.join(['\t' * (indent + 1) + _decode_item(i) + ' and\n' for i in rule[1]])
        tail = ''.join([_decode_rule(rules[i], indent + 1) for i in range(len(rules)) if rules[i][0] in rule[2]])
        return head + body[:-4] + '\n' + tail

    return [_decode_rule(r) for r in rules if r[0] == rule_head]


def get_scores(predictions, labels):
    n = len(labels)
    if n == 0:
        return 0, 0, 0, 0
        
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(n):
        tp = tp + 1.0 if labels[i] and predictions[i] else tp
        tn = tn + 1.0 if not labels[i] and not predictions[i] else tn
        fn = fn + 1.0 if labels[i] and not predictions[i] else fn
        fp = fp + 1.0 if not labels[i] and predictions[i] else fp

    def _scores(_tp, _tn, _fp, _fn):
        if _tp < 1:
            p = 0 if _fp < 1 else _tp / (_tp + _fp)
            r = 0 if _fn < 1 else _tp / (_tp + _fn)
        else:
            p, r = _tp / (_tp + _fp), _tp / (_tp + _fn)
        f1 = 0 if r * p == 0 else 2 * r * p / (r + p)
        return (_tp + _tn) / n, p, r, f1

    return _scores(tp, tn, fp, fn)
    # p_scores, n_scores = _scores(tp, tn, fp, fn), _scores(tn, tp, fn, fp)
    # return [(p_scores[i] * (tp + fn) + n_scores[i] * (tn + fp)) / n for i in range(4)]


def num_predicates(rules):
    def _n_pred(rule):
        return len(rule[1])

    ret = 0
    for r in rules:
        ret += _n_pred(r)
    return ret


class Foldrpp:
    def __init__(self, str_attrs=None, num_attrs=None, label=None, pos_val=None):
        self.str_attrs = str_attrs
        self.num_attrs = num_attrs
        self.label = label
        self.pos_val = pos_val
        self.attrs = [attr.lower().replace(' ', '_') for attr in self.str_attrs + self.num_attrs]
        self.rule_head = (label.lower().replace(' ', '_'), '==', pos_val)
        self.rules = None
        self.flat_rules = None
        self._asp = None

    def reset(self):
        self.rules = None
        self.flat_rules = None
        self._asp = None

    def load_data(self, file_name, amount=-1):
        return load_data(file_name, self.str_attrs, self.num_attrs, self.label, self.pos_val, amount)

    def fit(self, data, ratio=0.5):
        pos_idx, neg_idx = split_index_by_label(data)
        self.rules = learn_rule_set(data, pos_idx, neg_idx, self.attrs, used_items=[], ratio=ratio)

    def classify(self, x):
        if self.rules is not None:
            return classify(self.rules, x)
        return justify_one(self.flat_rules, x, [], self.rule_head) >= 0

    def predict(self, data):
        ret = []
        for x in data:
            ret.append(self.classify(x))
        # print(coverage_counter)
        return ret

    def asp(self):
        if self.flat_rules is None:
            self.flat_rules = flatten_rules(self.rules, self.rule_head)
        if self._asp is None:
            self._asp = decode_rules(self.flat_rules)
        return self._asp

    def proof_rules(self, x, get_all=False):
        self.asp()
        ret = [str(explain_data(self.flat_rules, x))]
        for rule_set in explain(self.flat_rules, x, self.rule_head, get_all):
            ret.extend(decode_rules(rule_set, x))
        return ret

    def proof_trees(self, x, get_all=False):
        self.asp()
        ret = [str(explain_data(self.flat_rules, x))]
        for rule_set in explain(self.flat_rules, x, self.rule_head, get_all):
            ret.extend(proof_tree(rule_set, x, self.rule_head))
        return ret


def nicer_json_string(s, indent_level=3, indent_width=2):
    ret, indent = '', 0
    for c in s.replace(', ', ',').replace(': ', ':'):
        if c in {'(', '[', '{'}:
            ret += c
            if indent_level > 0:
                indent += 1
                ret += '\n' + ' ' * indent * indent_width
            indent_level -= 1
        elif c in {')', ']', '}'}:
            if indent_level >= 0:
                indent -= 1
                ret += '\n' + ' ' * indent * indent_width
            ret += c
            indent_level += 1
        elif c == ',' and indent_level >= 0:
            ret += c + '\n' + ' ' * indent * indent_width
        else:
            ret += c
    ret = ret.replace(',', ', ').replace(':', ': ')
    return ret


def save_model_to_file(model, file_name):
    import json

    def _rule_to_map(_rule):
        return {'head': _rule[0], 'main_items': _rule[1], 'ab_items': _rule[2]}

    model_tab = {'str_attrs': model.str_attrs, 'num_attrs': model.num_attrs,
                 'flat_rules': [_rule_to_map(r) for r in model.flat_rules], 'rule_head': model.rule_head,
                 'label': model.label, 'pos_val': model.pos_val,
                 }
    model_json = json.dumps(model_tab)
    model_json = nicer_json_string(model_json)
    with open(file_name, 'w') as f:
        f.write(model_json + '\n')


def load_model_from_file(file_name):
    import json

    def _norm_item(_item):
        head, op, val = _item
        return [head.lower().replace(' ', '_'), op, val]

    def _map_to_rule(_map):
        _ret = _norm_item(_map['head']), [_norm_item(it) for it in _map['main_items']], _map['ab_items']
        return _ret

    with open(file_name, 'r') as f:
        model_json = f.read()
    model_tab = json.loads(model_json)
    str_attrs, num_attrs = model_tab['str_attrs'], model_tab['num_attrs']
    label, pos_val = model_tab['label'], model_tab['pos_val']
    ret = Foldrpp(str_attrs, num_attrs, label, pos_val)
    ret.flat_rules = [_map_to_rule(mp) for mp in model_tab['flat_rules']]
    ret.rule_head = _norm_item(model_tab['rule_head'])
    return ret
