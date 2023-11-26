from foldrpp import split_data, get_scores, num_predicates
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta


def main():
    load_start = timer()

    # model, data = acute()
    # model, data = autism()
    # model, data = breastw()
    # model, data = cars()
    # model, data = credit()
    # model, data = heart()
    model, data = kidney()
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

    load_end = timer()
    print('% load data costs: ', timedelta(seconds=load_end - load_start), '\n')

    data_train, data_test = split_data(data, ratio=0.8, rand=True)

    start = timer()
    model.fit(data_train)
    end = timer()

    for r in model.asp():
        print(r)

    ys_test_hat = model.predict(data_test)
    ys_test = [x['label'] for x in data_test]
    acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
    print('% acc', round(acc, 3), 'p', round(p, 3), 'r', round(r, 3), 'f1', round(f1, 3))
    n_rules, n_preds = len(model.flat_rules), num_predicates(model.flat_rules)
    print('% #rules', n_rules, '#preds', n_preds)
    print('% foldrpp costs: ', timedelta(seconds=end - start), '\n')

    for x in data_test[:10]:
        for r in model.proof_rules(x):
            print(r)
        for r in model.proof_trees(x):
            print(r)

    from foldrpp import save_model_to_file, load_model_from_file
    save_model_to_file(model, 'model.txt')
    saved_model = load_model_from_file('model.txt')
    
    ys_test_hat = saved_model.predict(data_test)
    ys_test = [x['label'] for x in data_test]
    acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
    print('% acc', round(acc, 3), 'p', round(p, 3), 'r', round(r, 3), 'f1', round(f1, 3))

    # for x in data_test[:10]:
    #     for r in saved_model.proof_rules(x):
    #         print(r)
    #     for r in saved_model.proof_trees(x):
    #         print(r)


if __name__ == '__main__':
    main()
