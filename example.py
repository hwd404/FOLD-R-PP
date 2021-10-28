from foldrpp import *
from datasets import acute, adult, autism, breastw, cars, credit, heart, \
    kidney, krkp, mushroom, sonar, voting, ecoli, ionosphere, wine, credit_card
from scasp_utils import load_data_pred, load_translation, scasp_query, save_asp_to_file
from timeit import default_timer as timer
from datetime import timedelta


def main():
    model, X, Y = acute()
    # model, X, Y = autism()
    # model, X, Y = breastw()
    # model, X, Y = cars()
    # model, X, Y = credit()
    # model, X, Y = heart()
    # model, X, Y = kidney()
    # model, X, Y = krkp()
    # model, X, Y = mushroom()
    # model, X, Y = sonar()
    # model, X, Y = voting()
    # model, X, Y = ecoli()
    # model, X, Y = ionosphere()
    # model, X, Y = wine()
    # model, X, Y = adult()
    # model, X, Y = credit_card()

    X_train, Y_train, X_test, Y_test = split_data(X, Y, ratio=0.8, rand=True)
    start = timer()
    model.fit(X_train, Y_train, ratio=0.5)
    end = timer()
    Y_test_hat = model.predict(X_test)
    model.print_asp()
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')


def example():
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
             'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='ckd')
    X, Y = model.load_data('data/kidney/kidney.csv')
    X_train, Y_train, X_test, Y_test = split_data(X, Y, ratio=0.8, rand=True)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.5)
    end = timer()

    save_asp_to_file(model, 'example.lp')
    model.save_model_to_file('example.model')

    print('% load model from file then print asp rules')
    model2 = load_model_from_file('example.model')

    Y_test_hat = model2.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)

    model2.print_asp()
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')


def titanic():
    attrs = ['Sex', 'Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare', 'Class', 'Embarked']
    nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']
    model = Classifier(attrs=attrs, numeric=nums, label='Survived', pos='0')

    X_train, Y_train = model.load_data('data/titanic/train.csv')
    X_test, Y_test = model.load_data('data/titanic/test.csv')
    X_pred = load_data_pred('data/titanic/test.csv', numerics=nums)

    model.fit(X_train, Y_train, ratio=0.5)
    Y_test_hat = model.predict(X_test)
    model.print_asp()
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4), '\n')

    load_translation(model, 'data/titanic/template.txt')

    for i in range(len(X_test)):
        print(model.classify(X_test[i]))
        res = scasp_query(model, X_pred[i], pred=True)
        print(res)
        res = scasp_query(model, X_test[i])
        print(res)


if __name__ == '__main__':
    # main()
    # example()
    titanic()
