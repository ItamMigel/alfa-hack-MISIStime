import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
import time 
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.utils import resample
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import os
import gc
from sklearn.neighbors import KNeighborsClassifier


def downcast_floats(df):
    # Select columns with float64 data type
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        # Convert column to float32
        converted = df[col].astype('float32')
        # Check if the conversion is safe
        if np.isclose(df[col], converted, equal_nan=True).all():
            df[col] = converted
    return df
def feature_engineering(train_data, test_data, target, poly_degree=2, pca_components=2, alpha=0.1, top_n=15, stratify_frac=0.1):
    """
    Полный процесс отбора и генерации признаков:
    1. Отбор топ-N признаков по важности (RandomForest, Mutual Information).
    2. Генерация новых признаков:
        - Полиномиальные признаки
        - Взаимодействия признаков
        - Логарифмическое преобразование
        - PCA (снижение размерности)
    Применяется для обучающих и тестовых данных.
    """
    print("Начало feature engineering...")
    
    # Разделяем на признаки и целевую переменную
    X_train = train_data
    y_train = target

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Стратифицированное разделение для использования только 10% данных
    X_train_stratified, _, y_train_stratified, _ = train_test_split(
        X_train_scaled, y_train, test_size=1-stratify_frac, random_state=42, stratify=y_train
    )

    print("rf обуч")

    # 2. Важность признаков на основе RandomForest
    rf = RandomForestClassifier(n_jobs = -1,n_estimators=50, random_state=42)
    rf.fit(X_train_stratified, y_train_stratified)
    rf_feature_importance = rf.feature_importances_

    print("mi обуч")
    # 3. Важность признаков на основе Mutual Information 
    mi = mutual_info_classif(X_train_stratified, y_train_stratified)

    # Собираем важность признаков
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'rf': rf_feature_importance,
        'mi': mi
    })

    # Выбираем топ-N признаков
    feature_importance['total_importance'] = feature_importance[['rf', 'mi']].sum(axis=1)
    top_important_features = feature_importance.nlargest(top_n, 'total_importance')['feature'].tolist()

    print("Топ 15 отобранных признаков:", top_important_features)

    # Отбираем только важные признаки
    X_train_top = X_train[top_important_features]
    X_test_top = test_data[top_important_features]

    print("Генерация взаимодействий между признаками...")
    # Генерация взаимодействий между признаками
    interaction_features_train = pd.DataFrame(index=X_train.index)
    interaction_features_test = pd.DataFrame(index=test_data.index)

    for i in range(len(top_important_features)):
        for j in range(i + 1, len(top_important_features)):
            feature1 = top_important_features[i]
            feature2 = top_important_features[j]

            # Умножение
            interaction_features_train[f'{feature1}_x_{feature2}_mul'] = X_train[feature1] * X_train[feature2]
            interaction_features_test[f'{feature1}_x_{feature2}_mul'] = test_data[feature1] * test_data[feature2]
            
            # Деление
            interaction_features_train[f'{feature1}_x_{feature2}_div'] = X_train[feature1] / (X_train[feature2] + 1e-6)
            interaction_features_test[f'{feature1}_x_{feature2}_div'] = test_data[feature1] / (test_data[feature2] + 1e-6)
            
            # Синус
            interaction_features_train[f'{feature1}_sin'] = np.sin(X_train[feature1])
            interaction_features_test[f'{feature1}_sin'] = np.sin(test_data[feature1])
            
            # Квадрат
            interaction_features_train[f'{feature1}_sq'] = X_train[feature1] ** 2
            interaction_features_test[f'{feature1}_sq'] = test_data[feature1] ** 2
            
            # Корень
            sqrt_train = np.sqrt(X_train[feature1])
            sqrt_test = np.sqrt(test_data[feature1])
            if not np.isnan(sqrt_train).any() and not np.isnan(sqrt_test).any():
                interaction_features_train[f'{feature1}_sqrt'] = sqrt_train
                interaction_features_test[f'{feature1}_sqrt'] = sqrt_test
            
            # Полиномиальные признаки    
            # Логарифм

            log_train = np.log(X_train[feature1] + 1e-6)
            log_test = np.log(test_data[feature1] + 1e-6)
            if not np.isnan(log_train).any() and not np.isnan(log_test).any():
                interaction_features_train[f'{feature1}_log'] = log_train
                interaction_features_test[f'{feature1}_log'] = log_test

    print("Снижение размерности с PCA...")
    # Снижение размерности с PCA
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_top)
    X_test_pca = pca.transform(X_test_top)

    pca_train_df = pd.DataFrame(X_train_pca, columns=[f'pca_{i+1}' for i in range(pca_components)], index=X_train.index)
    pca_test_df = pd.DataFrame(X_test_pca, columns=[f'pca_{i+1}' for i in range(pca_components)], index=test_data.index)

    print("Объединение всех признаков...")
    # Объединяем все признаки
    X_train_final = pd.concat([X_train, interaction_features_train, pca_train_df], axis=1)
    X_test_final = pd.concat([test_data, interaction_features_test, pca_test_df], axis=1)

    print("Feature engineering завершён.")
    return X_train_final, X_test_final, y_train

def feature_selection_with_boruta(X_train, y_train, max_iter=3, n_estimators=20):
    print("Отбор фичей")
    
    # Create a stratified sample of 10% of the data
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=0.1, stratify=y_train, random_state=42
    )
    
    # Инициализация случайного леса для Boruta
    rf = RandomForestClassifier(max_depth=4, n_estimators=n_estimators)
    
    # Инициализация Boruta с заданным количеством итераций
    boruta = BorutaPy(rf, n_estimators=n_estimators, max_iter=max_iter, random_state=42)
    
    # Обучение Boruta на стратифицированном подвыборке
    boruta.fit(X_sample.values, y_sample.values)
    
    # Получаем выбранные признаки
    green_area = X_sample.columns[boruta.support_].to_list()
    blue_area = X_sample.columns[boruta.support_weak_].to_list()
    selected_features = list(set(green_area + blue_area))
    print(f"Отобранные признаки: {selected_features}")
    
    return selected_features


def fitting(path, max_iter=3, n_estimators=20, num_workers=8):
    """
    Основная функция для загрузки данных, выполнения отбора признаков, обучения модели.
    """
    try:
        # Получим список файлов по выданному пути к папке
        current_data = os.listdir(path)
    except Exception:
        return "Не папка"
    
    train_data_file = [data for data in current_data if data.endswith('train.parquet')][0]
    test_data_file = [data for data in current_data if data.endswith('test.parquet')][0]
    
    train_data = pd.read_parquet(os.path.join(path, train_data_file))
    test_data = pd.read_parquet(os.path.join(path, test_data_file))
    

    # Удаляем ненужный столбец 'smpl', если он есть
    if 'smpl' in train_data.columns:
        train_data = train_data.drop(columns=['smpl'])
        test_data = test_data.drop(columns=['smpl'])
    
    # Разделяем на признаки и целевую переменную
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    
    # Оптимизация памяти
    X_train = downcast_floats(X_train)
    X_test = downcast_floats(test_data)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    # Применяем feature engineering
    X_train, X_test, y_train = feature_engineering(X_train, X_test, y_train)
    
    # Опять оптимизация памяти после feature engineering
    X_train = downcast_floats(X_train)

    X_test = downcast_floats(X_test)
    

    # Выполняем отбор признаков с Boruta
    selected_features = feature_selection_with_boruta(X_train, y_train, max_iter, n_estimators)
    
    # Применяем отбор признаков
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    X_train_selected['target'] = y_train
    
        # Обучение модели с AutoGluon, оптимизация по ROC AUC
    predictor = TabularPredictor(
        label='target',         # Целевая переменная
        problem_type='binary',  # Задача: бинарная классификация
        eval_metric='roc_auc'   # Метрика для оптимизации
    ).fit(
        train_data=X_train_selected,
        time_limit=500,         # Лимит времени на обучение (10 минут)
        presets='experimental_quality',
        num_cpus=8, # Оптимизация наилучшего качества
        hyperparameters={
            'GBM': {
                'num_boost_round': 500,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
            },
            'CAT': {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'grow_policy': 'SymmetricTree',
            },
            'XGB': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
            }
        }
    )

    autogluon_predictions = predictor.predict_proba(X_test_selected)
    autogluon_probs = autogluon_predictions[1]  # Вероятности для класса '1'
    autogluon_probs = autogluon_probs.to_numpy().flatten()

    target_column = 'target'

    roles = {'target': target_column}

    task = Task('binary', metric='auc')

    automl = TabularAutoML(
        task=task,
        timeout=500,
        general_params={
            'use_algos': [['lgb_tuned', 'xgb_tuned', 'catboost']]
        },
        reader_params={'n_jobs': 8}
    )

    oof_pred = automl.fit_predict(X_train_selected, roles=roles)

    lightautoml_predictions = automl.predict(X_test_selected)
    lightautoml_probs = lightautoml_predictions.data  # Вероятности для класса '1'
    lightautoml_probs = lightautoml_probs.flatten()  

    weights = np.arange(0.1, 1.1, 0.05)
    best_weights = None
    best_roc_auc = 0
    X_train_selected, weights_data = train_test_split(
        X_train_selected, 
        test_size=0.1,  # Размер подвыборки (10000 строк)
        stratify=X_train_selected['target'],      # Стратификация по целевой переменной
        random_state=42                           # Фиксируем случайность
    )
    y_test = weights_data['target']
    weights_data = weights_data.drop(columns=['target'])
    test_lightautoml_predictions = automl.predict(weights_data)
    test_lightautoml_probs = test_lightautoml_predictions.data  # Вероятности для класса '1'
    test_lightautoml_probs = test_lightautoml_probs.flatten() 
    test_autogluon_predictions = predictor.predict_proba(weights_data)
    test_autogluon_probs = test_autogluon_predictions[1]  # Вероятности для класса '1'
    test_autogluon_probs = test_autogluon_probs.to_numpy().flatten() 
    # Перебор весов для двух моделей
    for w1 in weights:
        for w2 in weights:
            # Проверяем, что сумма коэффициентов равна 1
            if not np.isclose(w1 + w2, 1.0):
                continue
            
            # Расчёт блендированных предсказаний
            blended_predictions = w1 * test_lightautoml_probs + w2 * test_autogluon_probs
            
            # Вычисление ROC AUC
            roc_auc = roc_auc_score(y_test, blended_predictions)
            
            # Обновление наилучших весов
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_weights = (w1, w2)
        


    test_data['target'] = best_weights[0]*lightautoml_probs + best_weights[1]*autogluon_probs
    prediction = test_data[['id', 'target']].sort_values(by='id', ascending=True)
    
    return prediction


def model():
    # Пропишем путь к файлам данных
    data = 'data'
    # Запишем список датасетов в папке:
    folders = os.listdir(data)
    # Создадим цикл для прохождения по каждому файлу и генерации предсказания
    for fold in folders:
        # Запишем новый путь к данным
        data_path = data + f'/{fold}'
        # Вызовем функцию, передав в нее путь к папке для обучения
        prediction = fitting(path=data_path)
        # Сохраним полученное предсказание
        if type(prediction) is not str:
            # Сохраняем предсказание
            prediction.to_csv(f"predictions/{fold}.csv", index=False)
            print("Предсказание создано!")
        else:
            print("Невозможно создать предсказание!")

        gc.collect()
# Обозначаем действия при запуске кода
if __name__ == "__main__":
    # Запускаем модель
    model()