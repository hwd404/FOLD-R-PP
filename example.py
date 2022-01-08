from foldrpp import *
from timeit import default_timer as timer
from datetime import timedelta
from datasets import *


def example():
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
             'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='ckd')

    data = model.load_data('data/kidney/kidney.csv')
    data_train, data_test = split_data(data, ratio=0.8, rand=True)
    X_train, Y_train = split_xy(data_train)
    X_test,  Y_test = split_xy(data_test)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.5)
    end = timer()

    save_model_to_file(model, 'example.model')
    print('% load model from file then print asp rules')
    model2 = load_model_from_file('example.model')

    Y_test_hat = model2.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)

    model2.print_asp()
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')


def titanic_test():
    model, data_train, data_test = titanic()
    X_train, Y_train = split_xy(data_train)
    X_test, Y_test = split_xy(data_test)

    model.fit(X_train, Y_train, ratio=0.5)
    Y_test_hat = model.predict(X_test)
    model.print_asp()
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4), '\n')

    k = 1
    for i in range(len(X_test)):
        print('Explanation for example number', k, ':')
        print(model.explain(X_test[i], all_flag=True))
        print('Proof Trees for example number', k, ':')
        print(model.proof(X_test[i], all_flag=False))
        k += 1


if __name__ == '__main__':
    example()
    # titanic_test()
