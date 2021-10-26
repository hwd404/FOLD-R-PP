from foldrpp import *
import numpy as np


def acute():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    nums = ['a1']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='yes')
    X, Y = model.load_data('data/acute/acute.csv')
    print('\n% acute dataset', np.shape(X))
    return model, X, Y


def adult():
    attrs = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship',
    'race','sex','capital_gain','capital_loss','hours_per_week','native_country']
    nums = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='<=50K')
    X, Y = model.load_data('data/adult/adult.csv')
    print('\n% adult dataset', np.shape(X))
    return model, X, Y


def autism():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'age', 'gender', 'ethnicity', 'jaundice',
             'pdd', 'used_app_before', 'relation']
    nums = ['age']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='NO')
    X, Y = model.load_data('data/autism/autism.csv')
    print('\n% autism dataset', np.shape(X))
    return model, X, Y


def breastw():
    attrs = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='benign')
    X, Y = model.load_data('data/breastw/breastw.csv')
    print('\n% breastw dataset', np.shape(X))
    return model, X, Y


def cars():
    attrs = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='negative')
    X, Y = model.load_data('data/cars/cars.csv')
    print('\n% cars dataset', np.shape(X))
    return model, X, Y


def credit():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    nums = ['a2', 'a3', 'a8', 'a11', 'a14', 'a15']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='-')
    X, Y = model.load_data('data/credit/credit.csv')
    print('\n% credit dataset', np.shape(X))
    return model, X, Y


def heart():
    attrs = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
    'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',
    'slope', 'major_vessels', 'thal']
    nums = ['age', 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='absent')
    X, Y = model.load_data('data/heart/heart.csv')
    print('\n% heart dataset', np.shape(X))
    return model, X, Y


def kidney():
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
    'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='ckd')
    X, Y = model.load_data('data/kidney/kidney.csv')
    print('\n% kidney dataset', np.shape(X))
    return model, X, Y


def krkp():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='won')
    X, Y = model.load_data('data/krkp/krkp.csv')
    print('\n% krkp dataset', np.shape(X))
    return model, X, Y


def mushroom():
    attrs = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
    'spore_print_color', 'population', 'habitat']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='p')
    X, Y = model.load_data('data/mushroom/mushroom.csv')
    print('\n% mushroom dataset', np.shape(X))
    return model, X, Y


def sonar():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48',
    'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='Mine')
    X, Y = model.load_data('data/sonar/sonar.csv')
    print('\n% sonar dataset', np.shape(X))
    return model, X, Y


def voting():
    attrs = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
    'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
    'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue',
    'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='republican')
    X, Y = model.load_data('data/voting/voting.csv')
    print('\n% voting dataset', np.shape(X))
    return model, X, Y


def ecoli():
    attrs = ['sn','mcg','gvh','lip','chg','aac','alm1','alm2']
    nums = ['mcg','gvh','lip','chg','aac','alm1','alm2']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='cp')
    X, Y = model.load_data('data/ecoli/ecoli.csv')
    print('\n% ecoli dataset', np.shape(X))
    return model, X, Y


def ionosphere():
    attrs = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19',
    'c20','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30','c31','c32','c33','c34']
    model = Classifier(attrs=attrs, numeric=attrs, label='label', pos='g')
    X, Y = model.load_data('data/ionosphere/ionosphere.csv')
    print('\n% ionosphere dataset', np.shape(X))
    return model, X, Y


def wine():
    attrs = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','tot_phenols','flavanoids',
    'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','OD_of_diluted','proline']
    model = Classifier(attrs=attrs, numeric=attrs, label='label', pos='3')
    X, Y = model.load_data('data/wine/wine.csv')
    print('\n% wine dataset', np.shape(X))
    return model, X, Y


def credit_card():
    attrs = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
    'PAY_AMT5','PAY_AMT6']
    nums = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
    'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    model = Classifier(attrs=attrs, numeric=nums, label='DEFAULT_PAYMENT', pos='0')
    X, Y = model.load_data('data/credit_card/credit_card.csv')
    print('\n% credit card dataset', np.shape(X))
    return model, X, Y
