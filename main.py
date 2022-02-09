from foldrpp import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta


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

    # model, data_train, data_test = titanic()

    X_train, Y_train = split_xy(data_train)
    X_test,  Y_test = split_xy(data_test)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.5)
    end = timer()

    model.print_asp(simple=True)
    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')

    # k = 1
    # for i in range(len(X_test)):
    #     print('Explanation for example number', k, ':')
    #     print(model.explain(X_test[i], all_flag=False))
    #     print('Proof Trees for example number', k, ':')
    #     print(model.proof(X_test[i], all_flag=False))
    #     k += 1


if __name__ == '__main__':
    main()
