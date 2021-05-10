import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import feature_importance_permutation
import csv
import random as rd
import random
from datetime import datetime

# Program variables
step_parameter = 20  # number of prediction on every iteration
new_data_length = 1000  # target new dataset size
rand_state = 44   # random state
fold_ = 5    # k fold validation
fac = 0.2 # accuracy range of filter
mlFilAcc = 0.8      # minimum validation accuracy
ts = 0.2   # validation size

start = datetime.now()

#feature importance calculation
data = pd.read_csv("standard.csv")
X = data.loc[:,'stress':'intention']  #independent columns
y = data['result']    #target column i.e price range

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts, random_state = rand_state, stratify=y)

scaler = StandardScaler()
scaler.fit(X)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

lb = LabelBinarizer()
lb.fit(y)
Y_scaled_train = lb.transform(y_train)
Y_scaled_test = lb.transform(y_test)


k = StratifiedKFold(fold_)
param_c = 10. ** np.arange(-5, 5)

param_grid = {
    'C': param_c,
    'gamma': ['auto', 'scale']
}

svm_model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid, cv=k)
svm_model.fit(X_scaled_train, Y_scaled_train.ravel())
filter_accuracy = svm_model.score(X_scaled_test, Y_scaled_test)
fistd, imp_all = feature_importance_permutation(
    predict_method=svm_model.predict,
    X=X_scaled_train,
    y=Y_scaled_train.ravel(),
    metric='accuracy',
    num_rounds=10,
    seed=rand_state)

std = np.std(imp_all, axis=1)
file = open('fi_standard.csv', 'a')
file.write(str(fistd))
file.write('\n')
file.write( str(std))
file.write('\n')
file.close()

def filter(p1, p2, p3, p4, p5, p6, p7):
    data_import = pd.read_csv('./SSG_augment.csv')
    total_input = len(data_import)
    features = ['stress', 'hapiness', 'attitude', 'sub_norm', 'control', 'intention']
    label = ['result']
    X = data_import.loc[:, features].values
    Y = data_import.loc[:, label].values
    x1 = (p1, p2, p3, p4, p5, p6)
    X = np.vstack((X, x1))
    y_ = p7
    Y = np.vstack((Y, y_))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ts, stratify=Y)
    scaler_f = StandardScaler()
    scaler_f.fit(X)
    X_scaled_train = scaler_f.transform(X_train)
    X_scaled_test = scaler_f.transform(X_test)
    lb_f = LabelBinarizer()
    lb_f.fit(Y)
    Y_scaled_train = lb_f.transform(Y_train)
    Y_scaled_test = lb_f.transform(Y_test)

    k = StratifiedKFold(fold_)
    param_c = 10. ** np.arange(-5, 5)

    param_grid = {
        'C': param_c,
        'gamma': ['auto', 'scale']
    }

    svm_model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid, cv=k)
    svm_model.fit(X_scaled_train, Y_scaled_train.ravel())
    fil_ValAcc = svm_model.score(X_scaled_test, Y_scaled_test)
    imp_vals, _ = feature_importance_permutation(
        predict_method=svm_model.predict,
        X=X_scaled_train,
        y=Y_scaled_train.ravel(),
        metric='accuracy',
        num_rounds=10,
        seed=rand_state)
    #print(imp_vals)

    f1 = imp_vals[0]
    f2 = imp_vals[1]
    f3 = imp_vals[2]
    f4 = imp_vals[3]
    f5 = imp_vals[4]
    f6 = imp_vals[5]


    file = open('filter_accuracy.csv', 'a')
    file.write('{}'.format(fil_ValAcc))
    file.write('\n')
    file.close()

    return(f1, f2, f3, f4, f5, f6, fil_ValAcc)

total_input = 0
count = 0
while total_input < new_data_length:
    data_import = pd.read_csv('./SSG_augment.csv')
    total_input = len(data_import)
    count += 1
    print('new dataset: ', total_input)
    features = ['stress', 'hapiness', 'attitude', 'sub_norm', 'control', 'intention']
    label = ['result']
    X = data_import.loc[:, features].values
    Y = data_import.loc[:, label].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ts, random_state=rand_state, stratify=Y)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    lb = LabelBinarizer()
    lb.fit(Y)
    Y_scaled_train = lb.transform(Y_train)
    Y_scaled_test = lb.transform(Y_test)

    ##### hyperparameter optimization #################
    k = StratifiedKFold(fold_)
    param_c = 10. ** np.arange(-5, 5)
    gamma_range = 10. ** np.arange(-5, 5)

    param_grid = {
        'C': param_c,
        'gamma': ['auto', 'scale']
    }

    svm_model = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'), param_grid=param_grid, cv=k)
    svm_model.fit(X_scaled_train, Y_scaled_train.ravel())
    svm_accuracy = svm_model.score(X_scaled_test, Y_scaled_test)
    file = open('accuracy.csv', 'a')
    file.write('{}'.format(svm_accuracy))
    file.write('\n')
    file.close()

    test_scores_mean = svm_model.cv_results_["mean_test_score"]


    with open('hyperparameter_optimization_gnuplot.csv', 'a') as opti:
        fieldColumns = ['count', 'param_c', 'test_scores_mean']
        writer = csv.DictWriter(opti, fieldnames=fieldColumns, delimiter=',', lineterminator='\n')
        writer.writeheader()
        for x in range(param_c.shape[0]): writer.writerow(
            {'count': count, 'param_c': param_c[x], 'test_scores_mean': test_scores_mean[x]})
        opti.close()

    import pickle

    filename = 'material_model.sav'
    pickle.dump(svm_model, open(filename, 'wb'))
    # Define parameter spaces
    p1_max = int(np.array(max(X[:, 0])))
    p1_min = int(np.array(min(X[:, 0])))

    p2_max = int(np.array(max(X[:,1])))
    p2_min =int(np.array(min(X[:, 1])))

    p3_max = int(np.array(max(X[:, 2])))
    p3_min = int(np.array(min(X[:, 2])))

    p4_max = int(np.array(max(X[:, 3])))
    p4_min = int(np.array(min(X[:, 3])))

    p5_max = int(np.array(max(X[:, 4])))
    p5_min = int(np.array(min(X[:, 4])))

    p6_max = int(np.array(max(X[:, 5])))
    p6_min = int(np.array(min(X[:, 5])))

    new_result = []
    for i in range(step_parameter):
        p1 = rd.randrange(p1_min, p1_max)
        p2 = rd.randrange(p2_min, p2_max)
        p3 = rd.randrange(p3_min, p3_max)
        p4 = rd.randrange(p4_min, p4_max)
        p5 = rd.randrange(p5_min, p5_max)
        p6 = rd.randrange(p6_min, p6_max)
        new_param_ = np.hstack((p1, p2, p3, p4, p5, p6)).reshape(1, -1)

        Pred_Y = svm_model.predict(scaler.transform(new_param_))

        pred_y = lb.inverse_transform(Pred_Y)
        pred_y = " ".join(map(str, pred_y)).replace('[', ' ').replace(']', ' ').replace("'", ' ')
        new_result_ = p1, p2, p3, p4, p5, p6, pred_y

        f1, f2, f3, f4, f5, f6, fil_ValAcc = filter(p1, p2, p3, p4, p5, p6, pred_y)

        if (fistd[0]*(1-fac)<=f1<=fistd[0]*(1+fac) and fistd[1]*(1-fac)<=f2<=fistd[1]*(1+fac) and \
            fistd[2]*(1-fac)<=f3<=fistd[2]*(1+fac) and fistd[3]*(1-fac)<=f4<=fistd[3]*(1+fac) and \
            fistd[4]*(1-fac)<=f5<=fistd[4]*(1+fac) and fistd[5]*(1-fac)<=f6<=fistd[5]*(1+fac) and \
            fil_ValAcc > mlFilAcc):
            print('pass')
            with open('SSG_augment.csv', 'a', newline='') as csvfile:
                fieldnames = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'pred_y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writerow({'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, 'pred_y': pred_y})
                csvfile.close()

end = datetime.now()
time = end-start
print('Calculation time (hh:mm:ss.ms): {}'.format(time))
with open('./calculation_time.dat', 'a') as result4:
    result4.writelines("Calculation time (hh:mm:ss.ms): %s\n" % format(time))
