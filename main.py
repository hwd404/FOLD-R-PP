from foldrpp import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta


def main():
    # model, data = acute()
    # model, data = autism()
    # model, data = breastw()
    model, data = cars()
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
    # model, data = parkison()

    data_train, data_test = split_data(data, ratio=0.8, rand=True)  
    # line 28: 80% as training data, 20% as test data. shuffle data first when rand is True

    # model, data_train, data_test = titanic()
    # model, data_train, data_test = avila()
    # model, data_train, data_test = anneal()

    X_train, Y_train = split_xy(data_train)  # split data into features and label
    X_test,  Y_test = split_xy(data_test)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.5)  
    # line 39: ratio means # of exception examples / # of default examples a rule can imply = 0.5  
    end = timer()

    model.print_asp(simple=True)
    # line 43: output simplified rules when simple is True, default value is False
    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')

    # k = 1
    # for i in range(10):
    #     print('Explanation for example number', k, ':')
    #     print(model.explain(X_test[i], all_flag=False))
    #     print('Proof Trees for example number', k, ':')
    #     print(model.proof(X_test[i], all_flag=False))
    #     k += 1


if __name__ == '__main__':
    main()
