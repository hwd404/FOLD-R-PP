
# python3 -m pip install foldrpp

from foldrpp import *
from timeit import default_timer as timer 
from datetime import timedelta


def main():
    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    nums = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='<=50K')
    data = model.load_data('data/adult/adult.csv')
    print('\n% adult dataset', len(data), len(data[0]))

    data_train, data_test = split_data(data, ratio=0.8, rand=True)

    X_train, Y_train = split_xy(data_train)
    X_test,  Y_test = split_xy(data_test)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.5)
    end = timer()

    save_model_to_file(model, 'example.model')
    # model.print_asp(simple=True)

    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')

    model2 = load_model_from_file('example.model')
    model2.print_asp(simple=True)

    # k = 1
    # for i in range(10):
    #     print('Explanation for example number', k, ':')
    #     print(model.explain(X_test[i], all_flag=False))
    #     print('Proof Trees for example number', k, ':')
    #     print(model.proof(X_test[i], all_flag=False))
    #     k += 1


if __name__ == '__main__':
    main()
