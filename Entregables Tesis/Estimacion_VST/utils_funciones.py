import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import gamma, norm

# train, car
# beta = (beta_0_train,beta_time_train,beta_time_car,beta_cost_car)

def prob_1(X_cost,X_time,beta,beta_cost):

    if beta.ndim == 1:
        beta = beta.tolist()
    else:
        beta = beta.tolist()[0]
    beta_0 = [beta[0]]
    beta_0.append(0)
    beta_time = [beta[1],beta[2]]
    beta_cost = [0,beta_cost]

    numeradors = [[np.exp(b0 + beta_cost_m * xi_cost_m + beta_time_m * xi_time_m) for b0,beta_time_m,beta_cost_m\
                   ,xi_cost_m,xi_time_m in\
                   zip(beta_0,beta_time,beta_cost,xi_cost,xi_time)] \
                      for xi_cost,xi_time in zip(X_cost,X_time)]

    numeradors = np.array(numeradors)

    denominadors = numeradors.sum(axis = 1)

    return numeradors / (denominadors.reshape((-1,1)))

def log_likelihood_1(beta, data):
        X_cost = np.concatenate([np.zeros((data.shape[0],1)),np.array(data.car_cost).reshape((-1,1))],axis = 1)
        X_time = np.array(data[["train_time","car_time"]])
        beta_0_time = beta[:-1]
        beta_cost = beta[-1]
        P = prob_1(X_cost,X_time,beta_0_time,beta_cost)
        log_P = np.log(P)
        y = np.array(data[["y_train","y_car"]])
        return - np.sum(y * log_P)
    
def log_likelihood_igualdad(beta, data,E_target , lambda_ = 1e6):
        X_cost = np.concatenate([np.zeros((data.shape[0],1)),np.array(data.car_cost).reshape((-1,1))],axis = 1)
        X_time = np.array(data[["train_time","car_time"]])
        beta_0_time = beta[:-1]
        beta_cost = beta[-1]
        P = prob_1(X_cost,X_time,beta_0_time,beta_cost)
        log_P = np.log(P)
        y = np.array(data[["y_train","y_car"]])
        x_cost_car = X_cost[:,1]
        P_car = P[:,1]
        #E_real = beta[-1] * np.mean((1 - P_car) * x_cost_car)
        E_real = beta[-1] * np.mean((1 - y[:,1]) * x_cost_car)
        return - np.sum(y * log_P) + lambda_ * (E_real - E_target) ** 2
    
def log_likelihood_desigualdad(beta, data, E_max, lambda_ = 1e6):
    """
    Función de verosimilitud con penalización de desigualdad.
    Si la elasticidad real es menor o igual a E_max(zona factible), penalización = 0.
    Si la elasticidad real es mayor a E_max (violación), penalización = rho * (diferencia)^2.
    """
    # 1. Preparación de datos
    X_cost = np.concatenate([np.zeros((data.shape[0],1)), np.array(data.car_cost).reshape((-1,1))], axis=1)
    X_time = np.array(data[["train_time","car_time"]])
    beta_0_time = beta[:-1]
    beta_cost = beta[-1]

    # 2. Cálculo de Probabilidades Logit
    P = prob_1(X_cost, X_time, beta_0_time, beta_cost)
    log_P = np.log(P)
    y = np.array(data[["y_train","y_car"]])

    # 3. Cálculo de la Log-Verosimilitud estándar (Ajuste a los datos)
    ll_estandar = - np.sum(y * log_P)

    # 4. Cálculo de la Elasticidad Real del modelo actual
    x_cost_car = X_cost[:,1]
    E_real = beta[-1] * np.mean((1 - y[:,1]) * x_cost_car)

    # 5. APLICACIÓN DE LA DESIGUALDAD
    # diferencia = η_real - E_target
    diferencia = E_real - E_max

    # Solo penalizamos si la diferencia es positiva (violación de la restricción)
    # Si la diferencia es <= 0, max(0, diferencia) devuelve 0.
    penalizacion = lambda_ * (max(0, diferencia) ** 2)

    return ll_estandar + penalizacion


def max_x_0_approx(x,eps = 0.01):

        return x + np.sqrt(x ** 2 + eps)

def log_likelihood_ridge_1(beta,data,lambda_):

        X_cost = np.concatenate([np.zeros((data.shape[0],1)),np.array(data.car_cost).reshape((-1,1))],axis = 1)
        X_time = np.array(data[["train_time","car_time"]])
        beta_0_time = beta[:-1]
        beta_cost = beta[-1]
        P = prob_1(X_cost,X_time,beta_0_time,beta_cost)
        log_P = np.log(P)
        y = np.array(data[["y_train","y_car"]])
        return - np.sum(y * log_P) + lambda_ * (beta_0_time[-1] ** 2 + beta_cost ** 2 )

def cv_ridge_1(lambda_vals, data, init_beta, k=5):
        kf = KFold(n_splits=k)
        best_lambda = lambda_vals[0]
        best_score = -np.inf
        for lam in lambda_vals:
            scores = []
            for train_idx, test_idx in kf.split(data):
                train = data.iloc[train_idx]
                test = data.iloc[test_idx]
                res = minimize(log_likelihood_ridge_1, init_beta, args=(train, lam), method="BFGS")
                res = minimize(log_likelihood_ridge_1, res.x, args=(train, lam),method= "Newton-CG",\
                               jac=jac_log_likelihood_Ridge_1,hess=hess_log_likelihood_Ridge_1)
                score = -log_likelihood_ridge_1(res.x, test,0)
                scores.append(score)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_lambda = lam
                best_score = mean_score
                VST = (res.x[2] / res.x[3]) * 60
        return best_lambda,VST

def jac_log_likelihood_1(beta, data):

        X_cost = np.concatenate([np.zeros((data.shape[0],1)),np.array(data.car_cost).reshape((-1,1))],axis = 1)
        X_time = np.array(data[["train_time","car_time"]])
        beta_0_time = beta[:-1]
        beta_cost = beta[-1]
        P = prob_1(X_cost,X_time,beta_0_time,beta_cost)
        P_train = P[:,0]
        P_car = P[:,1]
        y_train = np.array(data["y_train"])
        y_car = np.array(data["y_car"])
        dLdbeta_0 = np.sum(P_train - y_train)
        dLdbeta_time_train = np.sum((P_train - y_train) * X_time[:,0])
        dLdbeta_time_car = np.sum((P_car - y_car) * X_time[:,1])
        dLdbeta_cost = np.sum((P_car - y_car) * X_cost[:,1])

        return np.array([dLdbeta_0,dLdbeta_time_train,dLdbeta_time_car,dLdbeta_cost])


        return grad_log_likelihood_standard + lambda_ * np.array([0,0,0,grad_pen * grad_E_real])
    
def jac_log_likelihood_igualdad(beta, data, E_target,lambda_ = 1e6):

        X_cost = np.concatenate([np.zeros((data.shape[0],1)),np.array(data.car_cost).reshape((-1,1))],axis = 1)
        y = np.array(data[["y_train","y_car"]])
        x_cost_car = X_cost[:,1]
        E_real = beta[-1] * np.mean((1 - y[:,1]) * x_cost_car)
        grad_pen = 2 * (E_real - E_target)
        grad_E_real = np.mean((1 - y[:,1]) * x_cost_car)
        grad_log_likelihood_standard = jac_log_likelihood_1(beta, data)
        return grad_log_likelihood_standard + lambda_ * np.array([0,0,0,grad_pen * grad_E_real])


def jac_log_likelihood_desigualdad(beta, data, E_max, rho = 1e6):

    # 1. Gradiente del Logit estándar
    grad_estandar = jac_log_likelihood_1(beta, data)

    # 2. Cálculo de la elasticidad actual
    X_cost = np.concatenate([np.zeros((data.shape[0],1)), np.array(data.car_cost).reshape((-1,1))], axis=1)
    y = np.array(data[["y_train","y_car"]])
    x_cost_car = X_cost[:,1]

    E_real = beta[-1] * np.mean((1 - y[:,1]) * x_cost_car)
    diferencia = E_real - E_max

    # 3. Derivada de la penalización:
    # Solo se aplica si diferencia > 0
    if diferencia > 0:
        # La derivada de rho * (E_real - E_max)^2 respecto a beta_cost es:
        # 2 * rho * (E_real - E_max) * (derivada de E_real respecto a beta_cost)
        grad_E_real = np.mean((1 - y[:,1]) * x_cost_car)
        grad_penalizacion = 2 * rho * diferencia * grad_E_real
    else:
        grad_penalizacion = 0

    # El gradiente solo afecta al último parámetro (beta_cost)
    return grad_estandar + np.array([0, 0, 0, grad_penalizacion])

def jac_log_likelihood_Ridge_1(beta, data, lambda_):

        beta_time = beta[-2]
        beta_cost = beta[-1]
        grad_log_likelihood_standard = jac_log_likelihood_1(beta, data)
        return grad_log_likelihood_standard + 2 * lambda_ * np.array([0,0,beta_time,beta_cost])

def hess_log_likelihood_1(beta, data):

        X_cost = np.concatenate([np.zeros((data.shape[0],1)),np.array(data.car_cost).reshape((-1,1))],axis = 1)
        X_time = np.array(data[["train_time","car_time"]])
        beta_0_time = beta[:-1]
        beta_cost = beta[-1]
        P = prob_1(X_cost,X_time,beta_0_time,beta_cost)
        P_train = P[:,0]
        P_car = P[:,1]
        y_train = np.array(data["y_train"])
        y_car = np.array(data["y_car"])
        ddLddbeta_0 = - np.sum(P_train * (1 - P_train))
        ddLdbeta_0dbeta_train_time = - np.sum(P_train * (1 - P_train) * X_time[:,0])
        ddLdbeta_0dbeta_car_time =  np.sum(P_train * (1 - P_train) * X_time[:,1])
        ddLdbeta_0dbeta_car_cost =  np.sum(P_train * P_car * X_cost[:,1])

        ddLddbeta_car_cost = - np.sum((1 - P_car) * P_car * X_cost[:,1] ** 2)
        ddLdbeta_car_costdbeta_car_time = - np.sum((1 - P_car) * P_car * X_cost[:,1] *  X_time[:,1])
        ddLdbeta_car_costdbeta_train_time =  np.sum(P_train * P_car * X_cost[:,1] *  X_time[:,0])

        ddLddbeta_car_time = - np.sum((1 - P_car) * P_car * X_time[:,1] ** 2)
        ddLdbeta_car_timedbeta_train_time =  np.sum(P_train * P_car * X_time[:,1] * X_time[:,0])

        ddLddbeta_train_time = - np.sum((1 - P_train) * P_train * X_time[:,0] ** 2)

        diag = np.diag([ddLddbeta_0 , ddLddbeta_train_time , ddLddbeta_car_time , ddLddbeta_car_cost])

        sobre_diag = np.array([[0 , ddLdbeta_0dbeta_train_time , ddLdbeta_0dbeta_car_time , ddLdbeta_0dbeta_car_cost],\
                              [0 , 0 , ddLdbeta_car_timedbeta_train_time , ddLdbeta_car_costdbeta_train_time],\
                             [0 , 0 , 0 , ddLdbeta_car_costdbeta_car_time] , \
                             [0 ] * 4])

        result = diag + sobre_diag + sobre_diag.T

        return - result
    
def hess_log_likelihood_igualdad(beta, data, E_target,lambda_ = 1e6):

    X_cost = np.concatenate([np.zeros((data.shape[0],1)),np.array(data.car_cost).reshape((-1,1))],axis = 1)
    y = np.array(data[["y_train","y_car"]])
    x_cost_car = X_cost[:,1]

    hess_pen = 2 * np.mean((1 - y[:,1]) * x_cost_car) ** 2
    hess_log_likelihood_standard = hess_log_likelihood_1(beta, data)

    return hess_log_likelihood_standard + lambda_ *np.diag([0,0,0,hess_pen])

def hess_log_likelihood_desigualdad(beta, data, E_max, rho = 10**6):

    # 1. Hessiana estándar
    hess_estandar = hess_log_likelihood_1(beta, data)

    # 2. Cálculo de la violación
    X_cost = np.concatenate([np.zeros((data.shape[0],1)), np.array(data.car_cost).reshape((-1,1))], axis=1)
    y = np.array(data[["y_train","y_car"]])
    x_cost_car = X_cost[:,1]

    E_real = beta[-1] * np.mean((1 - y[:,1]) * x_cost_car)

    # 3. Segunda derivada de la penalización
    if E_real > E_max:
        # La segunda derivada de la penalización respecto a beta_cost es:
        # 2 * rho * (grad_E_real)^2
        grad_E_real = np.mean((1 - y[:,1]) * x_cost_car)
        hess_penalizacion = 2 * rho * (grad_E_real ** 2)
    else:
        hess_penalizacion = 0

    # Añadimos la penalización solo a la posición (3,3) que corresponde a beta_cost
    hess_final = hess_estandar.copy()
    hess_final[3, 3] += hess_penalizacion

    return hess_final


def hess_log_likelihood_Ridge_1(beta, data, lambda_):


        hess_log_likelihood_standard = hess_log_likelihood_1(beta, data)
        return hess_log_likelihood_standard + 2 * lambda_ * np.diag([0,0,1,1])
