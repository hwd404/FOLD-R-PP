from foldrpp import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta


def main():
    # model, X, Y = acute()
    model, X, Y = autism()
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

    model.print_asp()
    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')

    for i in range(len(X_test)):
        print(model.classify(X_test[i]))
        model.explain(X_test[i], all_flag=True)


if __name__ == '__main__':
    main()
