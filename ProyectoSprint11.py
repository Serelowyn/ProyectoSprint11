# ------------------ Importaciones

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# ------------------ Importaciones


# 1. Descarga y prepara los datos.  Explica el procedimiento.

# data = pd.read_csv("/datasets/churn.csv") para tripleten
data = pd.read_csv(r"churn.csv")

"""se descarga el archivo desde la plataforma (churns.csv) y se carga con el nombre de variable data"""

# a continuacion se agregan nuevas impresiones para verificar la estructura del dataset que acabo de cargar y para verificar si debo hacer alguna limpieza de los datos

"""para una vista general y particular con los 3 primeros casos"""
print(data.head(3))
print(data.info())

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
#modelo de regresion logistica
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

# 3. Mejora la calidad del modelo. Asegúrate de utilizar al menos dos enfoques para corregir el desequilibrio de clases. Utiliza conjuntos de entrenamiento y validación para encontrar el mejor modelo y el mejor conjunto de parámetros. Entrena diferentes modelos en los conjuntos de entrenamiento y validación. Encuentra el mejor. Describe brevemente tus hallazgos.

"""primero se utiliza el argumento (class_weigvht="balanced") para que le de peso a la clase minoritaria durante el entrenamiento """
# ----------- 1: class_weight balanced"
model_lr_bal = LogisticRegression(random_state=12345, solver="liblinear", class_weight="balanced")
model_lr_bal.fit(features_train, target_train)
print("regresion logistica con nuevo f1:", f1_score(target_valid, model_lr_bal.predict(features_valid)))

model_rf_bal = RandomForestClassifier(random_state=12345, n_estimators=100, class_weight="balanced")
model_rf_bal.fit(features_train, target_train)
print("bosque aleatorio con nuevo f1:", f1_score(target_valid, model_rf_bal.predict(features_valid)))

model_tree_bal = DecisionTreeClassifier(random_state=12345, class_weight="balanced")
model_tree_bal.fit(features_train, target_train)
pred_tree_bal = model_tree_bal.predict(features_valid)
print("arbol de decisiones con nuevo f1:", f1_score(target_valid, pred_tree_bal))

# ----------- 2: sobremuestreo"
def upsample(features, target, repeat):
    features_zeros = features[target == 0] #tabla de features  que busca los 0
    features_ones = features[target == 1] #tabla de features que busca los 1
    target_zeros = target[target == 0] #tabla de target que busca los 0
    target_ones = target[target == 1] #tabla de target que busca los 1
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    return shuffle(features_upsampled, target_upsampled, random_state=12345)

for repeat in [2, 3, 4]:
    f_up, t_up = upsample(features_train, target_train, repeat)
    model = RandomForestClassifier(random_state=12345, n_estimators=100)
    model.fit(f_up, t_up)
    print(f"bosque aleatorio upsample repeat={repeat} f1:",
        f1_score(target_valid, model.predict(features_valid)))

"""para buscar el mejor rank de f1 buscamos el mejor resultado y los mejores parametros de dicho resultado"""
best_f1, best_params = 0, None
for est in [50, 100, 200]:
    for depth in [6, 8, 10, 12, None]:
        model = RandomForestClassifier(
            n_estimators=est, max_depth=depth,
            class_weight="balanced", random_state=12345
        )
        model.fit(features_train, target_train)
        f1 = f1_score(target_valid, model.predict(features_valid))
        if f1 > best_f1:
            best_f1, best_params = f1, (est, depth)

print("best configuracion balanceada:", best_params, "f1:", best_f1)

"""se compararon las dos tecnicas de correccion de balance: class_weight balanced (mejor f1 = 0.648 con n_estimators=200, max_depth=8), sobremuestreo con bosque aleatorio (mejor f1 = 0.62 con repeat=4). se eligio class_weight balanced junto con busqueda de hiperparametros porque dio el f1 mas alto al final."""

# 4. Realiza la prueba final.

"""separo el 20% de los datos para el conjunto de test"""
features_tv, features_test, target_tv, target_test = train_test_split(features, target, test_size=0.2, random_state=12345)

"""separo el 80% de los datos para el conjunto de validacion"""
features_train, features_valid, target_train, target_valid = train_test_split(features_tv, target_tv, test_size=0.25, random_state=12345)

"""bucle que recorre los 3 datasets y convierte los datos a float64"""
for df in [features_train, features_valid, features_test]:
    for col in numeric:
        df[col] = df[col].astype("float64")

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train.loc[:, numeric] = scaler.transform(features_train[numeric])
features_valid.loc[:, numeric] = scaler.transform(features_valid[numeric])
features_test.loc[:, numeric] = scaler.transform(features_test[numeric])

best_f1, best_params = 0, None
for est in [50, 100, 200]:
    for depth in [6, 8, 10, 12, None]:
        model = RandomForestClassifier(
            n_estimators=est, max_depth=depth,
            class_weight="balanced", random_state=12345
        )
        model.fit(features_train, target_train)
        f1 = f1_score(target_valid, model.predict(features_valid))
        if f1 > best_f1:
            best_f1, best_params = f1, (est, depth)

print("configuracion definitiva:", best_params, "f1 valid:", best_f1)

features_final = pd.concat([features_train, features_valid])
target_final = pd.concat([target_train, target_valid])

final_model = RandomForestClassifier(
    n_estimators=best_params[0], max_depth=best_params[1],
    class_weight="balanced", random_state=12345
)
final_model.fit(features_final, target_final)

predicted_test = final_model.predict(features_test)
probabilities_test = final_model.predict_proba(features_test)[:, 1]

f1_test = f1_score(target_test, predicted_test)
auc_roc_test = roc_auc_score(target_test, probabilities_test)

"""resultados finales"""
print("f1:", f1_test)
# el f1 supera el minimo pedido de 0.59

print("precision:", precision_score(target_test, predicted_test))
print("recall:", recall_score(target_test, predicted_test))
print("auc-roc:", auc_roc_test)
print("matriz de confusion:", confusion_matrix(target_test, predicted_test))

"""la metrica auc-roc fue bastante mas superior, muy probablemente porque el f1 tiene por default su umbral de clasificacion en 0.5  mientras que el otro modelo tiene la capacidad de separar ambas clases en la cantiadd de umbrales posibles. Podria traducirse sobre que la metreica auc roc es la que mejor diferencia los filtros anteriormente hechos de 1 y 0 para features y target"""