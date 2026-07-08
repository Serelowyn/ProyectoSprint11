# ------------------ Importaciones

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score, recall_score

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

"""esto me falto hacer antes del get dummies, ya que estas no se estandarizan"""
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

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

"""se crea una instancia para poder estandarizar los datos, el punro es restarle la media y dividir entre la desv stdr. aqui se realiza el calculo de la media y la desv stdr pero solo usando los datos del entrenamiento"""
scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train.loc[:, numeric] = scaler.transform(features_train[numeric])
features_valid.loc[:, numeric] = scaler.transform(features_valid[numeric])

"""se crea un modelo de regresion logistica, se especificar el solver para evitar algun error con el que se ajuste por default"""
#modelo de regresion lineal
model_lr = LogisticRegression(random_state=12345, solver="liblinear")

model_lr.fit(features_train, target_train) 
pred_lr = model_lr.predict(features_valid) #entrenamiento del modelo

"""se comparan los modelos"""
print("regresion logistica f1:", f1_score(target_valid, pred_lr), "recall:", recall_score(target_valid, pred_lr), "precision:", precision_score(target_valid, pred_lr))

"""se crea el modelo de arbol de decisiones"""
# modelo arbol de decisiones
model_tree = DecisionTreeClassifier(random_state=12345)

model_tree.fit(features_train, target_train)
pred_tree = model_tree.predict(features_valid)
print("arbol de decision f1:", f1_score(target_valid, pred_tree), "recall:", recall_score(target_valid, pred_tree), "precision:", precision_score(target_valid, pred_tree))

"""se crea el modelo de bosque aleatorio"""
# modelo bosque aletaroio

model_rf = RandomForestClassifier(random_state=12345, n_estimators=100) # 100 "arboles" de decisiones distintos
model_rf.fit(features_train, target_train)
pred_rf = model_rf.predict(features_valid)
print("bosque aleatorio f1:", f1_score(target_valid, pred_rf), "recall:", recall_score(target_valid, pred_rf), "precision:", precision_score(target_valid, pred_rf))

"""se puede verificar con los resultados, que sin correccion de desequilibro, no exitse ni un solo modelo que llegue de manera fiable al f1 de 0.59, aunque el bosque aleatroio se acerca no sirve porque no llega, y la regresion logistica es la mas afectadad de todas."""