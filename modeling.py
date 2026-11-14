import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

BASE = ['CreditScore', 'Gender', 'Age']
OPTIONAL = ['Geography', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

def prepare(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError('Nama kolom duplikat ditemukan.')
    missing = set(BASE) - set(df.columns)
    if missing:
        raise ValueError('Kolom wajib belum tersedia: ' + ', '.join(sorted(missing)))
    features = BASE + [c for c in OPTIONAL if c in df]
    for c in [c for c in features if c not in ['Gender', 'Geography']]:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if df[c].notna().sum() == 0:
            raise ValueError(f'Kolom {c} harus memiliki nilai numerik.')
    for c in ['Gender', 'Geography']:
        if c in df:
            df[c] = df[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
            if df[c].notna().sum() == 0:
                raise ValueError(f'Kolom {c} seluruhnya kosong.')
    if 'Exited' in df:
        y = pd.to_numeric(df['Exited'], errors='coerce')
        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise ValueError('Exited harus terisi 0 (bertahan) atau 1 (keluar) pada seluruh baris.')
        df['Exited'] = y.astype(int)
    return df, features

def train_models(df, features, test_size=0.25):
    if 'Exited' not in df:
        raise ValueError('Pelatihan membutuhkan kolom Exited.')
    counts = df.Exited.value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError('Diperlukan dua kelas Exited, masing-masing minimal 5 baris.')
    # CustomerId tidak masuk fitur; cegah nasabah sama muncul di train dan test.
    if 'CustomerId' in df and df.CustomerId.dropna().duplicated().any():
        raise ValueError('CustomerId berulang. Siapkan satu baris per nasabah sebelum melatih model.')
    xtrain, xtest, ytrain, ytest = train_test_split(df[features], df.Exited, test_size=test_size, stratify=df.Exited, random_state=42)
    cat = [c for c in features if c in ['Gender', 'Geography']]
    num = [c for c in features if c not in cat]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
    ])
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=min(5, len(xtrain)))
    }
    rows, results = [], {}
    for name, estimator in models.items():
        pipeline = Pipeline([('prepare', preprocess), ('model', estimator)])
        # Setiap model memerlukan transformer sendiri.
        from sklearn.base import clone
        pipeline = clone(pipeline)
        pipeline.fit(xtrain, ytrain)
        prob = pipeline.predict_proba(xtest)[:, 1]
        pred = (prob >= .5).astype(int)
        rows.append(dict(Model=name, Accuracy=accuracy_score(ytest,pred), Precision=precision_score(ytest,pred,zero_division=0), Recall=recall_score(ytest,pred,zero_division=0), F1=f1_score(ytest,pred,zero_division=0), ROC_AUC=roc_auc_score(ytest,prob)))
        results[name] = dict(pipeline=pipeline, confusion=confusion_matrix(ytest,pred,labels=[0,1]), roc=roc_curve(ytest,prob), test=pd.DataFrame({'Aktual':ytest,'Probabilitas_Churn':prob,'Prediksi':pred}))
    return pd.DataFrame(rows), results, len(xtrain), len(xtest)
