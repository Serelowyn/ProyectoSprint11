# ------------------ Importaciones

import pandas as pd

# ------------------ Importaciones


# 1. Descarga y prepara los datos.  Explica el procedimiento.

# data = pd.read_csv("/datasets/churn.csv") para tripleten
data = pd.read_csv(r"churn.csv")

"""se descarga el archivo desde la plataforma (churns.csv) y se carga con el nombre de variable data"""

# a continuacion se agregan nuevas impresiones para verificar la estructura del dataset que acabo de cargar y para verificar si debo hacer alguna limpieza de los datos

"""para una vista general y particular con los 3 primeros casos"""
print(data.head(3))
print(data.info)

"""para verufucar el tipo de dato de cada columna"""
print(data.dtypes)

"""para revisar la estructura general"""
print(data.shape)

"""para revisar los NaNs que existan en las columnas"""
print(data.isna().sum())

# se encuentran en orden los tipos de datos. Existen algunos strings con caracteres que no son comunes pero se optan por dejar ahi por el momento ya que sirven como una forma de clasificar o filtrar los datos. Solo la columna tenure tiene datos faltantes, para esto, se opta por usar la mediana y no el mean para rellenar los campos faltantes, pero ambos datos son exactamente iguales, ambos valores son 5, entoncces no afecta lo que se use. Si no fuera asi, usar la mediana seria optimo en este caso al serr variable discreta

data["Tenure"] = data["Tenure"].fillna(data["Tenure"].median())

"""esta parte sirve para poder procesar los datos de la columna en variables que el modelo de machine learning pueda procesar"""
data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe["Exited"]
features = data_ohe.drop("Exited", axis=1)

print(features)