import numpy as np
from kohonen import kohonen_nn

def params_normalization(X):
    count_featrue = X.shape[-1]
    f_maxs = []
    f_mins = []

    for i in range(count_featrue):
        f_max = np.max(X[:,[i]])
        f_min = np.min(X[:,[i]])
        f_maxs.append(f_max)
        f_mins.append(f_min)
    for i in range(len(X)):
        x_normalizatoion = np.array([0.0]*18)
        for j in range(len(X[i])):
            f_max = f_maxs[j]
            f_min = f_mins[j]
            x_normalizatoion[j] = (X[i][j] - f_min)/(f_max - f_min)
        X[i] = x_normalizatoion
    return X

def extract_XY(data):
    X,Y = [],[]
    for d in data:
        d = d.split(',')
        X.append(d[1:])
        Y.append(d[0])
    Y = [int(y) for y in Y]
    X = np.array([np.array([float(f) for f in x]) for x in X])      
    return X,Y
    
if __name__ == "__main__":
    with open('kohonen_wta/datasets/vehicle_clean.data','r',encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    X,Y = extract_XY(data)
    X = params_normalization(X)
    FEATURE_COUNT = len(X[0])
    VEHICLE_COUNT = len(list(set(Y)))
    print(f'FEATURE COUNT: {FEATURE_COUNT}')
    print(f'VEHICLE COUNT: {VEHICLE_COUNT}')

    kohonen = kohonen_nn(12, 12, wta=False, count_side=True) 
    kohonen.fit(X, Y, 8000, save_e=True, interval=100)
    kohonen.plot_error_history(filename='kohonen_wta/images/kohonen_error.png')
    print(f'Error: {kohonen.error}')

    for i in range(VEHICLE_COUNT):        
        kohonen.plot_class_density(X, Y, t=i, name='Vehicle %d'%(i+1), filename='kohonen_wta/images/vehicle_%d.png'%(i+1))
