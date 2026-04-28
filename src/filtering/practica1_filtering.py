from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
import pandas as pd

class Practica1Filtering:
    """
    Clase que encapsula el pipeline de seleccion de features en 3 etapas:

    1. Eliminación de variables con baja varianza:
       - Variables casi constantes no aportan información al modelo.

    2. Selección univariada (SelectKBest con mutual information):
       - Se seleccionan las variables más relevantes respecto al target
         de forma individual.

    3. Selección basada en modelo (Random Forest):
       - Se eliminan variables con baja importancia según un modelo
         no lineal que captura interacciones.

    Este enfoque combina filtros estadísticos y métodos embebidos,
    mejorando la robustez del proceso de selección.
    
    Sigue el patron fit/transform para poder ajustar en train y aplicar en test
    sin data leakage.
    """

    def __init__(self,
                 variance_threshold=0.01,
                 k_best=10,
                 rf_threshold="mean",
                 rf_n_estimators=100,
                 rf_max_depth=None,
                 random_state=42):

        # Paso 1: eliminar variables con baja varianza
        self.var_selector = VarianceThreshold(threshold=variance_threshold)

        # Paso 2: selección univariada basada en información mutua
        self.kbest_selector = SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state), 
            k=k_best
        )

        # Paso 3: selección basada en importancia de Random Forest
        self.rf_selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            threshold=rf_threshold
        )

    def fit(self, X_data, y_data):
        """
        Ajusta los 3 filtros secuencialmente sobre los datos de entrenamiento.
        Cada filtro aprende que features eliminar y guarda esa informacion
        para aplicarla luego en transform().
        """
        # ===============================
        # Paso 1: Baja varianza
        # ===============================
        # fit + transform para que el paso 2 reciba datos ya filtrados
        self.var_selector.fit(X_data)
        X_no_low_var = self.var_selector.transform(X_data)

        # Guardar nombres de variables seleccionadas
        self.cols_after_var = X_data.columns[self.var_selector.get_support()].tolist()
        X_no_low_var_df = X_data[self.cols_after_var]

        self.n_dropped_low_var = X_data.shape[1] - X_no_low_var.shape[1]

        # ===============================
        # Paso 2: SelectKBest
        # ===============================
        # fit + transform para que el paso 3 reciba datos ya filtrados
        self.kbest_selector.fit(X_no_low_var, y_data)
        X_no_kbest = self.kbest_selector.transform(X_no_low_var)

        # Recuperar nombres de columnsa seleccionadas
        self.cols_after_kbest = X_no_low_var_df.columns[self.kbest_selector.get_support()].tolist()
        X_no_kbest_df = X_no_low_var_df[self.cols_after_kbest]

        self.n_dropped_kbest = X_no_low_var.shape[1] - X_no_kbest.shape[1]

        # ===============================
        # Paso 3: Random Forest
        # ===============================
        # fit (el transform se hara cuando el usuario lo pida)
        self.rf_selector.fit(X_no_kbest, y_data)

        # Guardamos info para el resumen
        X_final = self.rf_selector.transform(X_no_kbest)

        # Recuperar nombres de columnsa seleccionadas
        self.cols_after_rf = X_no_kbest_df.columns[self.rf_selector.get_support()].tolist()
        X_final_df = X_no_kbest_df[self.cols_after_rf]

        self.n_dropped_rf = X_no_kbest.shape[1] - X_final.shape[1]
        self.n_features_initial = X_data.shape[1]
        self.n_features_final = X_final.shape[1]

        self.selected_features = X_final_df.columns.tolist()

    def transform(self, X_data):
        """
        Aplica los 3 filtros secuencialmente.
        Usa los parametros aprendidos en fit(), NO re-aprende nada.
        """
        X_out = self.var_selector.transform(X_data)
        X_out = self.kbest_selector.transform(X_out)
        X_out = self.rf_selector.transform(X_out)

        return pd.DataFrame(X_out, columns=self.selected_features, index=X_data.index)

    def print_summary(self):
        """Imprime un resumen del pipeline de filtrado."""
        print("=" * 60)
        print("RESUMEN DEL PIPELINE DE FILTRADO")
        print("=" * 60)
        print(f"  Features iniciales:              {self.n_features_initial}")
        print(f"  Eliminadas por baja varianza:     -{self.n_dropped_low_var}")
        print(f"  Eliminadas por SelectkBest:       -{self.n_dropped_kbest}")
        print(f"  Eliminadas por importancia de RF:      -{self.n_dropped_rf}")
        print(f"  Features seleccionadas finales:  {self.n_features_final}")
        print("=" * 60)
