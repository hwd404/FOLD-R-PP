# FOLD-R-PP
This is a Python implementation of FOLD-R++ algorithm which is built for binary classification tasks. FOLD-R++ algorithm learns default rules that are represented as an [answer set program](https://en.wikipedia.org/wiki/Answer_set_programming) which is a [logic program](https://en.wikipedia.org/wiki/Logic_programming) that include [negation of predicates](https://en.wikipedia.org/wiki/Negation) and follow the [stable model semantics](https://en.wikipedia.org/wiki/Stable_model_semantics) for interpretation. [Default logic](https://en.wikipedia.org/wiki/Default_logic) (with exceptions) closely models human thinking.

The previous version that is compatible with s(CASP) can be found in **old_version** directory.

## Installation
Only function library:
``` 
python3 -m pip install foldrpp
```

With the dataset examples:
``` 
git clone https://github.com/hwd404/FOLD-R-PP.git
```

### Prerequisites
This FOLD-R++ implementation is developed with only python3. No library is needed.

## Usage

### Data preparation

1. FOLD-R++ takes tabular data as input, the first line of the input data file should be the feature names of each column.
    
    |  id   |  age  |  bp   |  ...  |  ane  |  label  |
    |:-----:|:-----:|:-----:|:-----:|:-----:|:-------:|
    |   1   |  48   |  80   |  ...  |  no   |   ckd   |
    |   2   |   7   |  50   |  ...  |  no   |   ckd   |
    |  ...  |  ...  |  ...  |  ...  |  ...  |   ...   |
2. FOLD-R++ does not have to encode data like one-hot or integer encoding. It can deal with numerical, categorical, and even mixed-type features (one feature contains categorical and numerical values at same time) directly. The types of features should be identified before loading data:
   + numerical features will be dealt as mixed-type features in this implementation. Group them together as numerical features, they will have in/equality (=, !=) and numerical comparison (<=, >) literals as candidates.
   + categorical features will only have in/equality literals as candidates (only literals with = and != would be generated).

Many UCI dataset are included as examples in **data** directory. Their data preparation are listed in **datasets.py**. For example the UCI kidney dataset can be loaded with the following configuration:

``` python
str_attrs = ['al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
num_attrs = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
label, pos_val = 'label', 'ckd'
model = Foldrpp(str_attrs, num_attrs, label, pos_val)
data = model.load_data('data/kidney/kidney.csv')
```

- *str_attrs* lists categorical features, 
- *num_attrs* lists numerical and mixed-type features, 
- *label* is the name of classification label, 
- *pos_val* indicates the positive value of the label, 
- *model* is an initialized classifier object with the configuration of kidney dataset. 
- **Note**: For binary classification tasks, the label value with more examples should be selected as the label's positive value.

### Training
FOLD-R++ generates an explainable model that is represented by an answer set program for classification tasks. Here's a training example for kidney dataset:
``` python 
data_train, data_test = split_data(data, ratio=0.8, rand=True)
model.fit(data_train, ratio=0.5)
```
Note that the hyperparameter **ratio** in **fit** function ranges between 0 and 1. The default value is 0.5. It represents the threshold ratio of false positive (exception) examples to true positive examples that a rule can imply.

The generated rules are stored in the model object, to print out the rules:
``` python 
for r in model.asp():
    print(r)
```
output:
``` python
label(X,'ckd') :- sc(X,N1), N1>1.2.
label(X,'ckd') :- sg(X,N2), N2=<1.015.
label(X,'ckd') :- hemo(X,N3), N3=<12.7.
label(X,'ckd') :- not al(X,'0'), not al(X,'?').
```

### Testing
Given **data_test**, a list of test data samples, the **predict** function will predict the classification outcome for each of these data samples. 

``` python 
ys_test_hat = model.predict(data_test)
```

The original label of test samples:
``` python
ys_test = [x['label'] for x in data_test]
```

Accuracy, precision, recall and F1 score:
``` python
acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
```

The code of the above examples can be found in **main.py**. 

### Save model and Load model

``` python

from foldrpp import save_model_to_file, load_model_from_file
    save_model_to_file(model, 'model.txt')
    saved_model = load_model_from_file('model.txt')
```

A trained model can be saved to a json file with **save_model_to_file** function. **load_model_from_file** function helps load model from the json file.
``` json 
{
  "str_attrs": [
    "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
  ], 
  "num_attrs": [
    "age", "bp", "sg", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
  ], 
  "flat_rules": [
    {
      "head": ["label", "==", "ckd"], 
      "main_items": [["sc", ">", 1.2]], 
      "ab_items": []
    }, 
    {
      "head": ["label", "==", "ckd"], 
      "main_items": [["sg", "=<", 1.015]], 
      "ab_items": []
    }, 
    {
      "head": ["label", "==", "ckd"], 
      "main_items": [["hemo", "=<", 12.7]], 
      "ab_items": []
    }, 
    {
      "head": ["label", "==", "ckd"], 
      "main_items": [["al", "!=", "0"], ["al", "!=", "?"]], 
      "ab_items": []
    }
  ], 
  "rule_head": [
    "label", 
    "==", 
    "ckd"
  ], 
  "label": "label", 
  "pos_val": "ckd"
}

```
The generated rules in json file can also be fine-tuned as needed.
	
### Explanation and Proof Tree

FOLD-R++ provides simple format proof for predictions with **proof_rules** function, the parameter **get_all** means whether or not to list all the answer sets (all rules that predict the test example as positive).

``` python
for r in saved_model.proof_rules(x, get_all=True):
    print(r)
```

Here is an example for an instance from kidney dataset. The generated answer set program is :

``` 
label(X,'ckd') :- sc(X,N1), N1>1.2.
label(X,'ckd') :- sg(X,N2), N2=<1.015.
label(X,'ckd') :- hemo(X,N3), N3=<12.7.
label(X,'ckd') :- not al(X,'0'), not al(X,'?').
```

And the generated justification for an instance predicted as positive:

``` 
{'sc': 0.9, 'sg': 1.01, 'hemo': 12.4, 'al': '0'}
[T]label(X,'ckd') :- sg(X,N1), [T]N1=<1.015.
[T]label(X,'ckd') :- hemo(X,N1), [T]N1=<12.7.
```

There are 2 answers have been generated for the current instance, because **get_all** has been set as True when calling **proof_rules** function. Only 1 answer will be generated if **get_all** is False. In the generated answers, each literal has been tagged with a label. **[T]** means True, **[F]** means False, and **[U]** means unnecessary to evaluate. 

FOLD-R++ also provide proof tree for predictions with **proof_trees** function. 
``` python
for r in saved_model.proof_trees(x, get_all=False):
    print(r)
```
	
And generate proof tree for the instance above:
``` 
{'sc': 1.0, 'sg': 1.025, 'hemo': 16.1, 'al': '0'}
label(X,'ckd') does not hold because
	the value of sc is 1.0 which should be greater than 1.2 does not hold 

label(X,'ckd') does not hold because
	the value of sg is 1.025 which should be less equal to 1.015 does not hold 

label(X,'ckd') does not hold because
	the value of hemo is 16.1 which should be less equal to 12.7 does not hold 

label(X,'ckd') does not hold because
	the value of al is '0' which should not equal '0' does not hold and
	the value of al is '0' which should not equal '?' does hold 
```

For an instance predicted as negative, there's no answer set. Instead, the explanation has to list the predictions of all rules, and the parameter **get_all** will be ignored:

``` 
{'sc': 0.9, 'sg': '?', 'hemo': 15.0, 'al': '?'}
[F]label(X,'ckd') :- sc(X,N1), [F]N1>1.2.
[F]label(X,'ckd') :- sg(X,N1), [F]N1=<1.015.
[F]label(X,'ckd') :- hemo(X,N1), [F]N1=<12.7.
[F]label(X,'ckd') :- not [F]al(X,'0'), not [T]al(X,'?').
```

```
{'sc': 0.9, 'sg': '?', 'hemo': 15.0, 'al': '?'}
label(X,'ckd') does not hold because
	the value of sc is 0.9 which should be greater than 1.2 does not hold 

label(X,'ckd') does not hold because
	the value of sg is '?' which should be less equal to 1.015 does not hold 

label(X,'ckd') does not hold because
	the value of hemo is 15.0 which should be less equal to 12.7 does not hold 

label(X,'ckd') does not hold because
	the value of al is '?' which should not equal '0' does hold and
	the value of al is '?' which should not equal '?' does not hold 
```

## Citation

```
@inproceedings{foldrpp,
    author = {Wang, Huaduo and Gupta, Gopal},
    title = {{FOLD-R++}: A Scalable Toolset for Automated Inductive Learning of Default Theories from Mixed Data},
    year = {2022},
    isbn = {978-3-030-99460-0},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    url = {https://doi.org/10.1007/978-3-030-99461-7_13},
    doi = {10.1007/978-3-030-99461-7_13},
    booktitle = {Functional and Logic Programming: 16th International Symposium, FLOPS 2022},
    pages = {224–242},
    numpages = {19},
    location = {Kyoto, Japan}
}
```

``` 
@article{foldrm, 
    title = {{FOLD-RM}: A Scalable, Efficient, and Explainable Inductive Learning Algorithm for Multi-Category Classification of Mixed Data},
    DOI={10.1017/S1471068422000205}, 
    journal = {Theory and Practice of Logic Programming}, 
    publisher={Cambridge University Press}, 
    author={Wang, Huaduo and Shakerin, Farhad and Gupta, Gopal}, 
    year={2022},
    pages={1--20}
}
```

``` 
@article{DBLP:journals/corr/abs-1804-11162,
    author={Joaqu{\'{\i}}n Arias and Manuel Carro and Elmer Salazar and Kyle Marple and Gopal Gupta},
    title={Constraint Answer Set Programming without Grounding},
    journal={CoRR},
    volume={abs/1804.11162},
    year={2018},
    url={http://arxiv.org/abs/1804.11162}
}
```

## Acknowledgement
	
Authors gratefully acknowledge support from NSF grants IIS 1718945, IIS 1910131, IIP 1916206, and from Amazon Corp, Atos Corp and US DoD.
