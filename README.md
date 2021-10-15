# FOLD-R-PP
The implementation of FOLD-R++ algorithm. The target of FOLD-R++ algorithm is to learn an answer set program for a classification task.

## Installation
### Prerequisites
FOLD-R++ is developed with only python3. Numpy is the only dependency:

<code>
	python3 -m pip install numpy
	
</code>

## Instruction
### Data preparation

The FOLD-R++ algorithm takes tabular data as input, the first line for the tabular data should be the feature name for each column.
The FOLD-R++ does not need encoding for training data. It can deal with numeric, categorical, and even mixed features (one column contains categorical and numeric values) directly.
But, the numeric features should be specified before loading data, otherwise they would be dealt like categorical features (only equality literals would be generated).

There are many UCI datasets can be found in the **data** directory, and the data preparation code pieces should be added to datasets.py.


For example, the UCI breast-w dataset can be loaded with the following code.

<code>
	
    columns = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    nums = columns
    data, num_idx, columns = load_data('data/breastw/breastw.csv', attrs=columns, label=['label'], numerics=nums, pos='benign')
	
</code>

**columns** lists all the features needed, **nums** lists all the numeric features, **label** implies the feature name of the label, **pos** indicates the positive value of the label.

### Training
The FOLD-R++ algorithm generates an explainable model that is represented with an answer set program for classification tasks. Here's an example for breast-w dataset:

<code>
	
    X_train, Y_train = split_xy(data_train)
    X_pos, X_neg = split_X_by_Y(X_train, Y_train)
    rules1 = foldrpp(X_pos, X_neg, [])
	
</code>

We got a rule set **rules1** with intermediate representation, then: 

<code>
	
    fr1 = flatten(rules1)
    rule_set = decode_rules(fr1, attrs)
    for r in rule_set:
        print(r)
	
</code>

The training process can be started with: 
<code>
	python3 main.py

</code>

An answer set program that is compatible with s(CASP) is listed as below.

<code>
	
	% breastw dataset (699, 10).
	% the answer set program generated by foldr++:
	
	label(X,'benign'):- bare_nuclei(X,'?').
	label(X,'benign'):- bland_chromatin(X,N6), N6=<4.0,
						clump_thickness(X,N0), N0=<6.0,  
	                	bare_nuclei(X,N5), N5=<1.0, not ab7(X).   
	label(X,'benign'):- cell_size_uniformity(X,N1), N1=<2.0,
						not ab3(X), not ab5(X), not ab6(X).  
	label(X,'benign'):- cell_size_uniformity(X,N1), N1=<4.0,
						bare_nuclei(X,N5), N5=<3.0,
						clump_thickness(X,N0), N0=<3.0, not ab8(X).  
	ab2(X):- clump_thickness(X,N0), N0=<1.0.  
	ab3(X):- bare_nuclei(X,N5), N5>5.0, not ab2(X).  
	ab4(X):- cell_shape_uniformity(X,N2), N2=<1.0.  
	ab5(X):- clump_thickness(X,N0), N0>7.0, not ab4(X).  
	ab6(X):- bare_nuclei(X,N5), N5>4.0, single_epi_cell_size(X,N4), N4=<1.0.  
	ab7(X):- marginal_adhesion(X,N3), N3>4.0.  
	ab8(X):- marginal_adhesion(X,N3), N3>6.0.  
	
	% foldr++ costs:  0:00:00.027710  post: 0:00:00.000127
	% acc 0.95 p 0.96 r 0.9697 f1 0.9648 
	
</code>

### Testing in Python
The testing data can be predicted with the **predict** function in Python. 

<code>
	Y_test_hat = predict(rules1, X_test)

</code>

The **classify** function can also be used for a single data.
	
<code>
	y_test_hat = classify(rules1, x_test)

</code>

### Justification by using s(CASP)
Classification and justification can be conducted with s(CASP), but the data also need to be converted into predicate format.
The **decode_test_data** function can be used for generating predicates for testing data.

<code>
	
	data_pred = decode_test_data(data_test, attrs)
	for p in data_pred:
	    print(p)
</code>

Here is an example of testing data predicates along with the answer set program for acute dataset:

<code>
	
	% acute dataset (120, 7) 
	% the answer set program generated by foldr++:

	ab2(X):-a5(X,'no'),a1(X,N0),N0>37.9.
	label(X,'yes'):-not a4(X,'no'),not ab2(X).

	% foldr++ costs:  0:00:00.001990  post: 0:00:00.000040
	% acc 1.0 p 1.0 r 1.0 f1 1.0 

	id(1).
	a1(1,37.2).
	a2(1,'no').
	a3(1,'yes').
	a4(1,'no').
	a5(1,'no').
	a6(1,'no').

	id(2).
	a1(2,38.1).
	a2(2,'no').
	a3(2,'yes').
	a4(2,'yes').
	a5(2,'no').
	a6(2,'yes').

	id(3).
	a1(3,37.5).
	a2(3,'no').
	a3(3,'no').
	a4(3,'yes').
	a5(3,'yes').
	a6(3,'yes').

</code>

### s(CASP)

All the resources of s(CASP) can be found at https://gitlab.software.imdea.org/ciao-lang/sCASP.
