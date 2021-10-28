# FOLD-R-PP
The implementation details of FOLD-R++ algorithm and how to use it are described here. The target of FOLD-R++ algorithm is to learn an answer set program for a classification task. Answer set programs are logic programs that permit negation of predicates and follow the stable model semantics for interpretation. The rules generated are essentially default rules. Default rules (with exceptions) closely model human thinking.

## Installation
### Prerequisites
The FOLD-R++ algorithm is developed with only python3. Numpy is the only dependency:

<code>
	
	python3 -m pip install numpy
	
</code>

## Instruction
### Data preparation

The FOLD-R++ algorithm takes tabular data as input, the first line for the tabular data should be the feature names of each column.
The FOLD-R++ algorithm does not have to encode the data for training. It can deal with numerical, categorical, and even mixed type features (one column contains both categorical and numerical values) directly.
However, the numerical features should be identified before loading the data, otherwise they would be dealt like categorical features (only literals with = and != would be generated).

There are many UCI example datasets that have been used to pre-populate the **data** directory. Code for preparing these datasets has already been added to datasets.py.


For example, the UCI kidney dataset can be loaded with the following code:

<code>
	
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
             'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='ckd')
    X, Y = model.load_data('data/kidney/kidney.csv')
    X_train, Y_train, X_test, Y_test = split_data(X, Y, ratio=0.8, rand=True)
	
</code>

**attrs** lists all the features needed, **nums** lists all the numerical features, **label** is the name of the output classification label, **pos** indicates the positive value of the label, **model** is an initialized classifier object with the configuration of kidney dataset. **Note: For binary classification tasks, the label value with more examples should be selected as the label's positive value**.

### Training
The FOLD-R++ algorithm generates an explainable model that is represented by an answer set program for classification tasks. Here's a training example for kidney dataset:

<code>
	
    model.fit(X_train, Y_train, ratio=0.5)
	
</code>

Note that the hyperparameter **ratio** in **fit** function can be set by the user, and ranges between 0 and 1. Default value is 0.5. This hyperparameter represents the ratio of training examples that are part of the exception to the examples implied by only the default conclusion part of the rule. We recommend that the user experiment with this hyperparameter by trying different values to produce a ruleset with the best F1 score. A range between 0.2 and 0.5 is recommended for experimentation.

The rules generated by foldrpp will be stored in the model object. These rules are organized in a nested intermediate representation. The nested rules will be automatically flattened and decoded to conform to the syntax of answer set programs by calling **print_asp** function: 

<code>
	
    model.print_asp()
	
</code>

An answer set program, compatible with the s(CASP) answer set programming system, is printed as shown below. The s(CASP) system is a system for direclty executing predicate answer set programs in a query-driven manner.

<code>

	% the answer set program generated by foldr++:
	label(X,'ckd'):-hemo(X,N14),N14=<12.0.
	label(X,'ckd'):-pcv(X,'?'),not pc(X,'normal').
	label(X,'ckd'):-sc(X,N11),N11>1.2.
	label(X,'ckd'):-sg(X,N2),N2=<1.015.
	% acc 0.95 p 1.0 r 0.9149 f1 0.9556
	
</code>

### Testing in Python
Given **X_test**, a list of test data samples, the Python **predict** function will predict the classification outcome for each of these data samples. 

<code>
	
	Y_test_hat = model.predict(X_test)

</code>

The **classify** function can also be used to classify a single data sample.
	
<code>
	
	y_test_hat = model.classify(x_test)

</code>

#### The code of the above examples can be found in **main.py**. The examples below with more datasets and more functions can be found in **example.py**

### Save model and Load model

<code>
	
    model.save_model_to_file('example.model')
    model2 = load_model_from_file('example.model')
    save_asp_to_file(model2, 'example.lp')

</code>

A trained model can be saved to a file with **save_model_to_file** function. **load_model_from_file** function helps load model from file.
The generated ASP program can be saved to a file with **save_asp_to_file** function.

### Justification by using s(CASP)
**The installation of s(CASP) system is necessary for justification, and only for justification. The above examples do not need the s(CASP) system.**

Classification and its justification can be conducted with the s(CASP) system. However, each data sample needs to be converted into predicate format that the s(CASP) system expects. The **load_data_pred** function can be used for this conversion; it returns the data predicates string list. The parameter **numerics** lists all the numerical features.

<code>
	
	nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']
	X_pred = load_data_pred('data/titanic/test.csv', numerics=nums)

</code>

Here is an example of the answer set program generated for the titanic dataset by FOLD-R++, along with a test data sample converted into the predicate format.

<code>

	survived(X,'0'):-class(X,'3'),not sex(X,'male'),fare(X,N4),N4>23.25,not ab7(X),not ab8(X).
	survived(X,'0'):-sex(X,'male'),not ab2(X),not ab4(X),not ab6(X).
	... ...
	ab7(X):-number_of_parents_children(X,N3),N3=<0.0.
	ab8(X):-fare(X,N4),N4>31.275,fare(X,N4),N4=<31.387.
	... ...
	
	id(1).
	sex(1,'male').
	age(1,34.5).
	number_of_siblings_spouses(1,0.0).
	number_of_parents_children(1,0.0).
	fare(1,7.8292).
	class(1,'3').
</code> 

An easier way to get justification from the s(CASP) system is to call **scasp_query** function. It will send the generated ASP rules, converted data and a query to the s(CASP) system for justification. A previously specified natural language **translation template** can make the justification easier to understand, but is **not necessary**. The template indicates the English string corresponding to a given predicate that models a feature. Here is a (self-explanatory) example of a translation template:

<code>
	
	#pred sex(X,Y) :: 'person @(X) is @(Y)'.
	#pred age(X,Y) :: 'person @(X) is of age @(Y)'.
	#pred number_of_sibling_spouses(X,Y) :: 'person @(X) had @(Y) siblings or spouses'.
	... ...
	#pred ab2(X) :: 'abnormal case 2 holds for @(X)'.
	#pred ab3(X) :: 'abnormal case 3 holds for @(X)'.
	... ...
	
</code>

The template file can be loaded to the model object with **load_translation** function. Then, the justification is generated by calling **scasp_query**. If the input data is in predicate format, the parameter **pred** needs to be set as True.

<code>
	
	load_translation(model, 'data/titanic/template.txt')
	print(scasp_query(model, x, pred=False))
	
</code>

Here is the justification for a passenger in the titanic example above (note that survived(1,0) means that passenger with id 1 perished (denoted by 0):

<code>

	% QUERY:I would like to know if
	     'goal' holds (for 1).

		ANSWER:	1 (in 2.401 ms)

	JUSTIFICATION_TREE:
	'goal' holds (for 1), because
	    'survived' holds (for 1, and 0), because
		person 1 is male, and
		there is no evidence that abnormal case 2 holds for 1, because
		    there is no evidence that person 1 paid Var0 not equal 7.8292 for the ticket, and
		    person 1 paid 7.8292 for the ticket.
		there is no evidence that abnormal case 4 holds for 1, because
		    there is no evidence that 'class' holds (for 1, and 1).
		there is no evidence that abnormal case 6 holds for 1, because
		    there is no evidence that person 1 is of age Var1 not equal 34.5, and
		    person 1 is of age 34.5.

	MODEL:
	{ goal(1),  survived(1,0),  sex(1,male),  not ab2(1),  not fare(1,Var0 | {Var0 \= 7.8292}),  fare(1,7.8292),  not ab4(1),  not class(1,1),  not ab6(1),  not age(1,Var1 | {Var1 \= 34.5}),  age(1,34.5) }

</code>


### s(CASP)

All the resources of s(CASP) can be found at https://gitlab.software.imdea.org/ciao-lang/sCASP.

## Citation

<code>
	
	@misc{wang2021foldr,
	      title={FOLD-R++: A Toolset for Automated Inductive Learning of Default Theories from Mixed Data}, 
	      author={Huaduo Wang and Gopal Gupta},
	      year={2021},
	      eprint={2110.07843},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}
	}
	
</code>
<code>

	@article{DBLP:journals/corr/abs-1804-11162,
		author={Joaqu{\'{\i}}n Arias and Manuel Carro and Elmer Salazar and Kyle Marple and Gopal Gupta},
		title={Constraint Answer Set Programming without Grounding},
		journal={CoRR},
		volume={abs/1804.11162},
		year={2018},
		url={http://arxiv.org/abs/1804.11162}
	}

</code>

	
