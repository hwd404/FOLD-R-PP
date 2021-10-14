from utils import *


def acute():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    nums = ['a1']
    data, num_idx, columns = load_data('data/acute/acute.csv', attrs=columns, label=['label'], numerics=nums, pos='yes')
    print('\n% acute dataset', np.shape(data), '\n')
    return columns, data, num_idx


def adult():
    columns = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship',
    'race','sex','capital_gain','capital_loss','hours_per_week','native_country']
    nums = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
    data, num_idx, columns = load_data('data/adult/adult.csv', attrs=columns, label=['label'], numerics=nums, pos='<=50K')
    print('\n% adult dataset', np.shape(data), '\n')
    return columns, data, num_idx


def autism():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'age', 'gender', 'ethnicity', 'jaundice',
    'pdd', 'used_app_before', 'relation']
    nums = ['age']
    data, num_idx, columns = load_data('data/autism/autism.csv', attrs=columns, label=['label'], numerics=nums, pos='NO')
    print('\n% autism dataset', np.shape(data), '\n')
    return columns, data, num_idx


def breastw():
    columns = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    nums = columns
    data, num_idx, columns = load_data('data/breastw/breastw.csv', attrs=columns, label=['label'], numerics=nums, pos='benign')
    print('\n% breastw dataset', np.shape(data), '\n')
    return columns, data, num_idx


def cars():
    columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    data, num_idx, columns = load_data('data/cars/cars.csv', attrs=columns, label=['label'], numerics=[], pos='negative')
    print('\n% cars dataset', np.shape(data), '\n')
    return columns, data, num_idx


def credit():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    nums = ['a2', 'a3', 'a8', 'a11', 'a14', 'a15']
    data, num_idx, columns = load_data('data/credit/credit.csv', attrs=columns, label=['label'], numerics=nums, pos='-')
    print('\n% credit dataset', np.shape(data), '\n')
    return columns, data, num_idx


def heart():
    columns = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
    'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',
    'slope', 'major_vessels', 'thal']
    nums = ['age', 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak']
    data, num_idx, columns = load_data('data/heart/heart.csv', attrs=columns, label=['label'], numerics=nums, pos='absent')
    print('\n% heart dataset', np.shape(data), '\n')
    return columns, data, num_idx


def kidney():
    columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
    'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    data, num_idx, columns = load_data('data/kidney/kidney.csv', attrs=columns, label=['label'], numerics=nums, pos='ckd')
    print('\n% kidney dataset', np.shape(data), '\n')
    return columns, data, num_idx


def krkp():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36']
    data, num_idx, columns = load_data('data/krkp/krkp.csv', attrs=columns, label=['label'], numerics=[], pos='won')
    print('\n% krkp dataset', np.shape(data), '\n')
    return columns, data, num_idx


def mushroom():
    columns = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
    'spore_print_color', 'population', 'habitat']
    data, num_idx, columns = load_data('data/mushroom/mushroom.csv', attrs=columns, label=['label'], numerics=[], pos='p')
    print('\n% mushroom dataset', np.shape(data), '\n')
    return columns, data, num_idx


def sonar():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48',
    'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']
    nums = columns
    data, num_idx, columns = load_data('data/sonar/sonar.csv', attrs=columns, label=['label'], numerics=nums, pos='Mine')
    print('\n% sonar dataset', np.shape(data), '\n')
    return columns, data, num_idx


def voting():
    columns = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
    'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
    'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue',
    'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    data, num_idx, columns = load_data('data/voting/voting.csv', attrs=columns, label=['label'], numerics=[], pos='republican')
    print('\n% voting dataset', np.shape(data), '\n')
    return columns, data, num_idx


def ecoli():
    columns = ['sn','mcg','gvh','lip','chg','aac','alm1','alm2']
    nums = ['mcg','gvh','lip','chg','aac','alm1','alm2']
    data, num_idx, columns = load_data('data/ecoli/ecoli.csv', attrs=columns, label=['label'], numerics=nums, pos='cp')
    print('\n% ecoli dataset', np.shape(data), '\n')
    return columns, data, num_idx


def ionosphere():
    columns = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19',
    'c20','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30','c31','c32','c33','c34']
    data, num_idx, columns = load_data('data/ionosphere/ionosphere.csv', attrs=columns, label=['label'], numerics=columns, pos='g')
    print('\n% ionosphere dataset', np.shape(data), '\n')
    return columns, data, num_idx


def wine():
    columns = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','tot_phenols','flavanoids',
    'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','OD_of_diluted','proline']
    data, num_idx = load_data('data/wine/wine.csv', attrs=columns, label=['label'], numerics=columns, pos='3')
    print('\n% wine dataset', np.shape(data), '\n')
    return columns, data, num_idx


def credit_card():
    columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
    'PAY_AMT5','PAY_AMT6']
    nums = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
    'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    data, num_idx, columns = load_data('data/credit_card/credit_card.csv', attrs=columns, label=['DEFAULT_PAYMENT'], numerics=nums,
                                       pos='0')
    print('\n% credit card dataset', np.shape(data), '\n')
    return columns, data, num_idx


def titanic_train():
    columns = ['Sex', 'Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare', 'Class', 'Embarked']
    nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']
    data, num_idx, columns = load_data('data/titanic/train.csv', attrs=columns, label=['Survived'], numerics=nums, pos='0')
    print('\n% titanic train dataset', np.shape(data), '\n')
    return columns, data, num_idx


def titanic_test():
    columns = ['Sex', 'Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare', 'Class', 'Embarked']
    nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']
    data, num_idx, columns = load_data('data/titanic/test.csv', attrs=columns, label=['Survived'], numerics=nums, pos='0')
    print('% titanic test dataset', np.shape(data), '\n')
    return columns, data, num_idx