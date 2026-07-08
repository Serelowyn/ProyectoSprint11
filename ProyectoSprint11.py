# ------------------ Importaciones

import pandas as pd
from sklearn.model_selection import train_test_split

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

# 2. Examina el equilibrio de clases. Entrena el modelo sin tener en cuenta el desequilibrio. Describe brevemente tus hallazgos.

"""se hace el conteo de las veces que aparece cada valor en la columna exited"""
print(target.value_counts(normalize=True))

"""se separan los datos donde el 25 por ciento va al ocnjunto de validacion y el resto al entrenamiento"""
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

"""se haze una lista de las columnas numericas y se convierten a float ya que habian tipos de datos int64 y si no lo cambio la prueba arrojara un error, se hace eso en train y en valid"""
numeric = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
features_train = features_train.astype({c: "float64" for c in numeric})
features_valid = features_valid.astype({c: "float64" for c in numeric})

