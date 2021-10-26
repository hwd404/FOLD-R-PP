from foldrpp import *


def main():
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
             'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='ckd')
    X, Y = model.load_data('data/kidney/kidney.csv')
    X_train, Y_train, X_test, Y_test = split_data(X, Y, ratio=0.8, rand=True)

    model.fit(X_train, Y_train, ratio=0.5)

    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)

    model.print_asp()
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))


if __name__ == '__main__':
    main()
