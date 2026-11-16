import numpy as np
import pandas as pd

def regressao_linear_multipla(arquivo):
    df = pd.read_csv(arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    n = len(preco)

    X = np.column_stack((np.ones(n), area, quartos))
    y = np.array(preco)

    Xt = np.transpose(X)
    XtX = np.matmul(Xt, X)
    XtX_inv = np.linalg.inv(XtX)
    Xty = np.matmul(Xt, y)
    beta = np.matmul(XtX_inv, Xty)

    return [float(beta[0]), float(beta[1]), float(beta[2])]