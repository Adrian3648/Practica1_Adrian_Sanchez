# Práctica 1 — Detección de Impago en LendingClub

Pipeline alternativo de Machine Learning aplicado a la predicción de impago de préstamos (LendingClub dataset, 2007-2017). El objetivo es construir un pipeline propio con decisiones justificadas en preprocesamiento, filtrado y modelado, y comparar el rendimiento de tres familias de modelos distintos contra un modelo base de referencia.

## Estructura del proyecto

```
├── practica1_notebook.ipynb               # Notebook principal (ejecutado)
├── src/
│   ├── preprocessing/
│   │   └── practica1_preprocessing.py     # Clase Practica1Preprocess (práctica)
│   └── filtering/
│       └── practica1_filtering.py         # Clase Practica1Filtering (práctica)
├── data/
│   ├── df_train_small.csv                 # Data de entrenamiento con 80.000 filas
│   ├── df_test_small.csv                  # Data de prueba con 20.000 filas
│   └── variables_withExperts.xlsx         # Diccionario de variables con expertos
```


## Flujo de trabajo

### 1. Preprocesamiento (`Practica1Preprocess`)

Pipeline alternativo a `BasePreprocess` con las siguientes diferencias:

**Variables:**
- Se usa `variables_withExperts.xlsx`, incorporando variables de evaluación de expertos
  (`grade`, `sub_grade`, `int_rate`, `fico_range_low`, `fico_range_high`, etc.)
  excluidas en el pipeline de referencia.

**Tratamiento de nulos:**
- **>98% nulos** → se elimina la variable.
- **Numéricas** → `SimpleImputer(strategy='median')`: robusto a outliers frecuentes en datos financieros.
- **Categóricas** → `SimpleImputer(strategy='constant', fill_value='DESCONOCIDO')`: preserva
  la señal de ausencia y permite a los encoders aprender el comportamiento de los valores faltantes.

**Variables categóricas — 4 estrategias según tipo:**
- **Ordinales** (`grade`, `sub_grade`, `emp_length`) -> `OrdinalEncoder` con orden natural explícito.
- **Binarias** (`term`, `application_type`) -> mapeo directo 0/1.
- **Texto libre** (`emp_title`, `desc`) -> `TextEncoder` (e5-small-v2, 20 componentes).
- **Nominales restantes** (`home_ownership`, `purpose`, etc.) -> `TargetEncoder`.

**Variables numéricas:**
- `RobustScaler`: usa mediana e IQR, insensible a outliers frecuentes en datos crediticios.

**Nuevas features (ratios financieros):**
- `fico_medio`, `installment_income_ratio`, `revol_limit_ratio`, `debt_income_ratio`, `income_bin`, `credit_age`.

### 2. Filtrado (`Practica1Filtering`)

Pipeline alternativo a `BaseFiltering` con 3 etapas secuenciales:

1. **VarianceThreshold** — elimina features con varianza <= umbral pequeño (casi constantes).
2. **SelectKBest** (`mutual_info_classif`) — selecciona las k features con mayor información mutua respecto al target, capturando relaciones no lineales.
3. **SelectFromModel** (Random Forest) — elimina features cuya importancia esté por debajo de la importancia media del modelo.

### 3. Modelos entrenados

Se entrenan tres modelos, uno de cada familia requerida:

| Modelo | Familia | Parámetros clave |
|---|---|---|
| `RandomForestClassifier` | Ensemble de árboles | n_estimators=100, max_depth=10, class_weight='balanced' |
| `SVC(kernel='rbf')` | SVM | C=20.0, probability=True, class_weight='balanced' |
| `MLPClassifier` | Red neuronal | hidden_layers=(100,100), relu, adam, early_stopping=True, learning_rate_init=0.0001, batch_size=32, max_iter=100, sample_weight |

### 4. Evaluación y comparación

Se calculan **Accuracy, Precision, Recall, F1-score y PR-AUC** para los tres modelos en el conjunto de test y se comparan contra el modelo base de referencia.

- **PR-AUC** es la métrica principal: con clases desbalanceadas (~80/20), es más informativa que ROC-AUC porque se centra en el rendimiento sobre la clase minoritaria (impago).
- **Recall** es la métrica más crítica en banca: un impago no detectado implica pérdida directa del capital prestado.

## Resultados

| Modelo | Accuracy | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Red Neuronal | 0.663 | 0.323 | 0.624 | 0.365 |
| Random Forest | 0.654 | 0.316 | 0.628 | 0.361 |
| SVM | 0.681 | 0.322 | 0.541 | 0.344 |
| Modelo Base | 0.717 | 0.265 | 0.235 | 0.159 |


Los tres modelos superan al modelo base en Recall y PR-AUC, triplicando la detección de impagos reales. Red Neuronal y Random Forest son los modelos más adecuados para este problema, con una mejora significativa respecto al modelo base.

## Dependencias principales

- **pandas** -- Manipulacion de datos.
- **scikit-learn** -- Modelos, metricas, transformadores.
- **feature-engine** -- Seleccion de variables (constantes, correlaciones, probes).
- **skrub** -- TextEncoder (embeddings), SquashingScaler.
- **sentence-transformers** -- Modelo de embeddings `e5-small-v2`.
- **matplotlib / seaborn** -- Visualizaciones.