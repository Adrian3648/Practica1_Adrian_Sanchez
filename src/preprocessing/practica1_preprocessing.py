import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from skrub import TextEncoder

class Practica1Preprocess:

    def __init__(self, var_to_process, target, use_text_vars=False, random_state=42):
        self.raw_predictors_vars = pd.read_excel(var_to_process)
        self.raw_predictors_vars = ( self.raw_predictors_vars
                                    .query("posible_predictora == 'si'")
                                    .variable
                                    .tolist())
        self.target_var = target
        self.random_state = random_state
        self.use_text_vars = use_text_vars


    def __add_features(self, X):
        """
        Generación de variables derivadas basadas en conocimiento del dominio financiero.

        En lugar de generar interacciones polinómicas (poco interpretables),
        se construyen ratios relevantes en modelos de riesgo crediticio.
        """
        X_new = X.copy()

        # Promedio del rango FICO: reduce redundancia y captura score central
        X_new["fico_medio"] = (X_new["fico_range_low"] + X_new["fico_range_high"])/2

        # Ratio de cuota mensual sobre ingreso mensual -> indicador clave de riesgo
        X_new["installment_income_ratio"] = X_new["installment"] / ((X_new["annual_inc"]/12) + 1)

        # Ratio de balance revolving sobre límite de crédito alto total revolving
        X_new["revol_limit_ratio"] = X_new["revol_bal"] / (X_new["total_rev_hi_lim"] + 1)

        # Ratio de prestamo realizado (deuda) sobre el ingreso anual
        X_new["debt_income_ratio"] = X_new["loan_amnt"] / (X_new["annual_inc"] + 1)

        # Discretización de ingresos -> captura relaciones no lineales
        X_new["income_bin"] = pd.qcut(X_new["annual_inc"].clip(lower=0), q=5, labels=False)

        # Antiguedad del historial crediticio (la data se encuentra entre 2007-2017)
        X_new['credit_age'] = 2017 - X_new['earliest_cr_line_year']
        
        return X_new


    def fit(self, data):
        # Leemos dataframe
        df = pd.read_csv(data)

        # Separar variables predictoras y target
        self.train_X_data = df[self.raw_predictors_vars].copy()
        self.train_y_data = df[self.target_var]

        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################
        self.train_X_data['earliest_cr_line'] = pd.to_datetime(self.train_X_data['earliest_cr_line'])
        self.train_X_data['earliest_cr_line_year'] = self.train_X_data['earliest_cr_line'].dt.year
        self.train_X_data['earliest_cr_line_month'] = self.train_X_data['earliest_cr_line'].dt.month.astype(str)
        
        ###########################################
        # Generación de nuevas features
        ###########################################
        self.train_X_data = self.__add_features(self.train_X_data)

        ###########################################
        # Tratamiento de nulls
        ###########################################
        size = self.train_X_data.shape[0]
        self.nulls_vars = ( (self.train_X_data.isnull().sum()/size)
                      .sort_values(ascending=False)
                      .to_frame(name="nulls_perc")
                      .reset_index() )
        
        # descartamos aquellas vars cuyos nulls sean mayor al 98 %
        self.var_with_most_nulls = ( self.nulls_vars
                               .query("nulls_perc > 0.98")["index"]
                               .tolist() )

        ###########################################
        # Selección de variables
        ###########################################
        self.categoric_vars = ( self.train_X_data
                               .loc[:, ~self.train_X_data.columns.isin(self.var_with_most_nulls)]
                               .select_dtypes(include="object")
                               .columns.tolist() )

        self.numeric_vars = ( self.train_X_data
                             .loc[:, ~self.train_X_data.columns.isin(self.var_with_most_nulls)]
                             .select_dtypes(include='number')
                              .columns.tolist() )

        ###########################################
        # Imputación de valores nulos (missings)
        ###########################################
        # Numéricas -> mediana (robusto a outliers)
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_imputer.fit(self.train_X_data[self.numeric_vars])

        # Categóricas -> categoría explícita
        # Permite capturar señal de missing en el model. Valor DESCONOCIDO por defecto.
        self.categ_imputer = SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")
        self.categ_imputer.fit(self.train_X_data[self.categoric_vars])

        ###########################################
        # Procesamiento de variables categóricas
        ###########################################
        # Variables ordinales con orden natural
        grades_ordered = ["G", "F", "E", "D", "C", "B", "A"]
        #G5, G4, G3, G2, G1, F5, F4...
        subgrades_ordered = [grade + str(i) for grade in grades_ordered for i in range(5, 0, -1)]
        emp_length_ordered = ["< 1 year", "1 year"] + [ str(i) + " years" for i in range(2, 10)] + ["10+ years"]
        ordinal_map = {
            "grade"     : grades_ordered,
            "sub_grade" : subgrades_ordered,
            "emp_length": emp_length_ordered
        }

        # Variables categoricas: binarias, ordinales, texto, variables restantes
        self.binary_vars = ["term", "application_type"]
        self.ordinal_vars = list(ordinal_map.keys())
        self.text_vars = ["emp_title", "desc"]
        non_target_vars = self.binary_vars + self.ordinal_vars + self.text_vars
        self.target_vars = [var for var in self.categoric_vars if var not in non_target_vars]

        # OrdinalEncoder para variables con orden natural
        self.ordinal_encoders = {}
        for var, order in ordinal_map.items():
            self.ordinal_encoders[var] = OrdinalEncoder(
                categories     = [order],
                handle_unknown = 'use_encoded_value',
                unknown_value  = -1
            )
            self.ordinal_encoders[var].fit(self.train_X_data[[var]])

        # Variables binarias se convierten en 0 y 1
        self.binary_map = {}
        for var in self.binary_vars:
            valores = sorted(self.train_X_data[var].dropna().unique())
            self.binary_map[var] = {
                valores[0] : 0, valores[1] : 1
            }

        # TargetEncoder para categorias nominales que no sean ordinales y ni binarias
        self.target_encoder = TargetEncoder(random_state=self.random_state)
        self.target_encoder.fit(self.train_X_data[self.target_vars], self.train_y_data != "Fully Paid")

        ###########################################
        # Procesamiento de variables numéricas
        ###########################################
        self.robust_scaler = RobustScaler()
        self.robust_scaler.fit(self.train_X_data[self.numeric_vars])

        ###########################################
        # Variables de texto (opcional)
        ###########################################
        if self.use_text_vars:
            # Limpieza inicial
            self.train_X_data["emp_title"] = self.train_X_data["emp_title"].fillna("DESCONOCIDO").astype(str)
            self.train_X_data["desc"] = self.train_X_data["desc"].fillna("DESCONOCIDO").astype(str)

            # Encoder basado en embeddings
            self.text_enc_title = TextEncoder(model_name='intfloat/e5-small-v2', n_components=20)
            self.text_enc_title.fit(self.train_X_data["emp_title"])

            # Limpieza de HTML en descripcion
            self.train_X_data['desc_formated'] = np.where(
                self.train_X_data['desc'] == 'DESCONOCIDO',
                'DESCONOCIDO',
                self.train_X_data['desc'].str.split('> ').str[1].str.split('<br>').str[0]
            )

            self.text_enc_desc = TextEncoder(model_name='intfloat/e5-small-v2', n_components=20)
            self.text_enc_desc.fit(self.train_X_data['desc_formated'])


    def transform(self, data):
        df = pd.read_csv(data)
        X_data = df[self.raw_predictors_vars].copy()
        y_data = df[self.target_var]

        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################
        X_data['earliest_cr_line'] = pd.to_datetime(X_data['earliest_cr_line'])
        X_data['earliest_cr_line_year'] = X_data['earliest_cr_line'].dt.year
        X_data['earliest_cr_line_month'] = X_data['earliest_cr_line'].dt.month.astype(str)

        ############################################
        # Generación de nuevas features
        ############################################
        X_data = self.__add_features(X_data)

        ###########################################
        # Eliminar variables con muchos nulos
        ###########################################
        X_data = X_data.drop(columns=self.var_with_most_nulls)

        ###########################################
        # Imputación de valores nulos (missings)
        ###########################################
        X_data_numeric = X_data[self.numeric_vars]
        X_data_categ   = X_data[self.categoric_vars]

        X_num_imputed = pd.DataFrame(
            self.numeric_imputer.transform(X_data_numeric),
            columns = self.numeric_vars
        )

        X_cat_imputed = pd.DataFrame(
            self.categ_imputer.transform(X_data_categ),
            columns = self.categoric_vars
        )

        ###########################################
        # Procesamiento de variables categóricas
        ###########################################
        # Variables ordinales
        X_ord_data = X_cat_imputed[self.ordinal_vars].copy()
        for var in self.ordinal_vars:
            X_ord_data[[var]] = self.ordinal_encoders[var].transform(X_ord_data[[var]])

        # Variables binarias
        X_bin_data = X_cat_imputed[self.binary_vars].copy()
        for var in self.binary_vars:
            X_bin_data[var] = X_bin_data[var].map(self.binary_map[var])

        # Variables nominales para target encoder
        X_target_nom_data = X_cat_imputed[self.target_vars].copy()
        X_target_nom_data[self.target_vars] = self.target_encoder.transform(X_target_nom_data)

        ###########################################
        # Variables de texto (opcional)
        ###########################################
        if self.use_text_vars:
            X_text_title = self.text_enc_title.transform(X_cat_imputed["emp_title"])
            X_cat_imputed['desc_formated'] = np.where(
                X_cat_imputed['desc'] == 'DESCONOCIDO',
                'DESCONOCIDO',
                X_cat_imputed['desc'].str.split('> ').str[1].str.split('<br>').str[0]
                )

            X_text_desc = self.text_enc_desc.transform(X_cat_imputed["desc_formated"])

        ###########################################
        # Procesamiento de variables numéricas
        ###########################################
        X_num_scaled = X_num_imputed.copy()
        X_num_scaled[self.numeric_vars] = self.robust_scaler.transform(X_num_scaled)

        ###########################################
        # Concatenación final
        ###########################################
        data_to_concat = [
            X_num_scaled,
            X_ord_data,
            X_bin_data,
            X_target_nom_data
        ]

        if self.use_text_vars:
            data_to_concat.extend([X_text_title, X_text_desc])

        X_data_output = pd.concat(data_to_concat, axis=1)

        # transformar y_data
        y_data_out = y_data != 'Fully Paid'
        return X_data_output, y_data_out
