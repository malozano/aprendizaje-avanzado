# Práctica 1: Aprendizaje Supervisado

## Objetivo

Aplicar los principales algoritmos de aprendizaje supervisado vistos en teoría sobre un mismo problema de clasificación, comparando su rendimiento y analizando el efecto de sus hiperparámetros.

Se asume que ya dominas el preprocesamiento básico (tratamiento de valores faltantes, escalado, encoding y división train/test) visto en la Práctica 0. Todo el preprocesamiento se da por supuesto y no es necesario detallarlo de nuevo en el informe.

## Dataset

Puedes elegir:

- **Opción A:** Reutilizar el dataset de la Práctica 0 (recomendado para comparar directamente con los resultados anteriores).
- **Opción B:** Seleccionar un nuevo dataset de clasificación con los mismos requerimientos que la Práctica 0.

El mismo dataset debe usarse en todas las partes de la práctica.

---

## Parte 1: Support Vector Machines

### 1.1 SVM con Diferentes Kernels

Entrena modelos SVM con los cuatro kernels principales y compara su rendimiento con los hiperparámetros por defecto.

```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# IMPORTANTE: SVM requiere escalado obligatorio
kernels = {
    'Linear':     SVC(kernel='linear',  random_state=42),
    'Poly (d=3)': SVC(kernel='poly',    degree=3, random_state=42),
    'RBF':        SVC(kernel='rbf',     random_state=42),
    'Sigmoid':    SVC(kernel='sigmoid', random_state=42)
}

for name, svc in kernels.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', svc)])
    pipe.fit(X_train, y_train)
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    n_sv = pipe.named_steps['svm'].n_support_
    print(f"{name}: CV={scores.mean():.4f} (+/-{scores.std():.4f}), "
          f"SV={n_sv.sum()} ({n_sv.sum()/len(X_train)*100:.1f}%)")
```

**Reporta en una tabla:**

| Kernel | CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | N° SV | % Training |
|--------|-------------|---------------|-----------|--------|----------|-------|------------|
| Linear | ... | ... | ... | ... | ... | ... | ... |
| Poly (d=3) | ... | ... | ... | ... | ... | ... | ... |
| RBF | ... | ... | ... | ... | ... | ... | ... |
| Sigmoid | ... | ... | ... | ... | ... | ... | ... |

### 1.2 Optimización del Kernel RBF

El kernel RBF suele ser el más efectivo. Explora el efecto de los hiperparámetros `C` y `gamma`.

- **C**: controla el trade-off entre maximizar el margen y permitir errores. C alto → margen pequeño, menos errores (riesgo de overfitting). C bajo → margen grande, más errores permitidos (riesgo de underfitting).
- **gamma**: controla la influencia de cada muestra en la frontera. gamma alto → influencia local, frontera compleja. gamma bajo → influencia amplia, frontera suave.

```python
from sklearn.model_selection import GridSearchCV

pipe_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42))
])

param_grid = {
    'svm__C':     [0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
}

grid_rbf = GridSearchCV(pipe_rbf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_rbf.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_rbf.best_params_}")
print(f"Mejor CV score:     {grid_rbf.best_score_:.4f}")
print(f"Test score:         {grid_rbf.best_estimator_.score(X_test, y_test):.4f}")
```

**Reporta en una tabla (selecciona al menos 10 combinaciones representativas):**

| C | gamma | CV Accuracy | Test Accuracy | N° SV | % Training | Tiempo (s) |
|---|-------|-------------|---------------|-------|------------|------------|
| 0.1 | 0.001 | ... | ... | ... | ... | ... |
| 1 | scale | ... | ... | ... | ... | ... |
| 10 | 0.1 | ... | ... | ... | ... | ... |
| 100 | 1 | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
| **Mejor** | **...** | **...** | **...** | **...** | **...** | **...** |

### 1.3 Optimización del Kernel Polinomial

```python
pipe_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='poly', random_state=42))
])

param_grid = {
    'svm__degree': [2, 3, 4],
    'svm__C':      [0.1, 1, 10],
    'svm__gamma':  ['scale', 'auto', 0.1]
}

grid_poly = GridSearchCV(pipe_poly, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_poly.fit(X_train, y_train)
```

**Reporta en una tabla:**

| degree | C | gamma | CV Accuracy | Test Accuracy | N° SV |
|--------|---|-------|-------------|---------------|-------|
| 2 | 1 | scale | ... | ... | ... |
| 3 | 1 | scale | ... | ... | ... |
| 4 | 1 | scale | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |
| **Mejor** | **...** | **...** | **...** | **...** | **...** |

### 1.4 Comparación de Implementaciones (Kernel Lineal)

Scikit-learn ofrece tres implementaciones de SVM lineal con distinta formulación y coste computacional:

| Implementación | Formulación | Cuándo usar |
|---------------|-------------|-------------|
| `SVC(kernel='linear')` | Dual (libsvm) | n_samples < 10,000 |
| `LinearSVC` | Primal (liblinear) | Kernel lineal, datasets medianos |
| `SGDClassifier(loss='hinge')` | SGD con hinge loss | Datasets grandes (> 100,000) |

```python
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from time import time

models = {
    'SVC(linear)':   Pipeline([('scaler', StandardScaler()),
                                ('svm', SVC(kernel='linear', random_state=42))]),
    'LinearSVC':     Pipeline([('scaler', StandardScaler()),
                                ('svm', LinearSVC(random_state=42, max_iter=10000))]),
    'SGDClassifier': Pipeline([('scaler', StandardScaler()),
                                ('svm', SGDClassifier(loss='hinge', random_state=42))])
}

for name, model in models.items():
    t0 = time()
    model.fit(X_train, y_train)
    t1 = time()
    cv = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: CV={cv.mean():.4f}, Test={model.score(X_test, y_test):.4f}, "
          f"Tiempo={t1-t0:.3f}s")
```

**Reporta en una tabla:**

| Implementación | CV Accuracy | Test Accuracy | N° SV | Tiempo (s) |
|---------------|-------------|---------------|-------|------------|
| SVC(linear) | ... | ... | ... | ... |
| LinearSVC | ... | ... | N/A | ... |
| SGDClassifier | ... | ... | N/A | ... |

### 1.5 Análisis de Vectores de Soporte

```python
model_svc = SVC(kernel='rbf', C=..., gamma=..., random_state=42)
model_svc.fit(X_train_scaled, y_train)

print(f"Total de vectores de soporte: {model_svc.n_support_.sum()}")
print(f"Por clase:                    {model_svc.n_support_}")
print(f"Porcentaje del training set:  {model_svc.n_support_.sum()/len(X_train)*100:.2f}%")
```

**Reporta en una tabla comparando distintas configuraciones:**

| Configuración | N° SV | % Training Set |
|--------------|-------|----------------|
| Linear (C=1) | ... | ... |
| RBF (C=1, γ=scale) | ... | ... |
| RBF (C=10, γ=0.1) | ... | ... |
| RBF (C=100, γ=1) | ... | ... |
| Poly (d=3, C=1) | ... | ... |

### 1.6 Evaluación Final (Parte 1)

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

best_svm = grid_rbf.best_estimator_
best_svm.fit(X_train, y_train)
y_pred = best_svm.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.savefig('confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
```

**Reporta en una tabla resumen de la Parte 1:**

| Modelo | CV Accuracy | Test Accuracy | F1-Score | N° SV | % Training |
|--------|-------------|---------------|----------|-------|------------|
| SVM Linear (mejor C) | ... | ... | ... | ... | ... |
| SVM Poly (mejor config) | ... | ... | ... | ... | ... |
| SVM RBF (mejor config) | ... | ... | ... | ... | ... |
| SVM Sigmoid (mejor config) | ... | ... | ... | ... | ... |

---

## Parte 2: Árboles de Decisión

### 2.1 Árbol sin Restricciones (Baseline)

Entrena un árbol sin limitar su crecimiento para observar el comportamiento de overfitting.

```python
from sklearn.tree import DecisionTreeClassifier

tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)

print(f"Train accuracy: {tree_full.score(X_train, y_train):.4f}")
print(f"Test  accuracy: {tree_full.score(X_test,  y_test):.4f}")
print(f"Profundidad:    {tree_full.get_depth()}")
print(f"N° hojas:       {tree_full.get_n_leaves()}")
```

**Reporta en una tabla:**

| Configuración | Train Accuracy | Test Accuracy | Profundidad | N° Hojas |
|---------------|----------------|---------------|-------------|----------|
| Sin restricciones | ... | ... | ... | ... |

### 2.2 Efecto de la Profundidad Máxima

```python
depths = [1, 2, 3, 5, 7, 10, 15, None]

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    cv   = cross_val_score(tree, X_train, y_train, cv=5)
    tree.fit(X_train, y_train)
    print(f"max_depth={str(depth):>4}: train={tree.score(X_train, y_train):.4f}, "
          f"cv={cv.mean():.4f}, test={tree.score(X_test, y_test):.4f}, "
          f"hojas={tree.get_n_leaves()}")
```

**Reporta en una tabla:**

| max_depth | Train Acc | CV Acc | Test Acc | Profundidad real | N° Hojas |
|-----------|-----------|--------|----------|-----------------|----------|
| 1 | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... | ... |
| 10 | ... | ... | ... | ... | ... |
| None | ... | ... | ... | ... | ... |

### 2.3 Criterios de División

Compara los criterios de impureza disponibles. Para clasificación: Gini (usado en CART), entropía (usado en ID3/C4.5) y log_loss.

```python
criterios = ['gini', 'entropy', 'log_loss']

for criterio in criterios:
    tree = DecisionTreeClassifier(criterion=criterio, random_state=42)
    cv   = cross_val_score(tree, X_train, y_train, cv=5)
    tree.fit(X_train, y_train)
    print(f"{criterio}: CV={cv.mean():.4f} (+/-{cv.std():.4f}), "
          f"Test={tree.score(X_test, y_test):.4f}, hojas={tree.get_n_leaves()}")
```

**Reporta en una tabla:**

| Criterio | CV Accuracy | Test Accuracy | N° Hojas |
|----------|-------------|---------------|----------|
| Gini | ... | ... | ... |
| Entropy | ... | ... | ... |
| Log Loss | ... | ... | ... |

### 2.4 Poda Previa: Grid Search de Hiperparámetros

Además de `max_depth`, explora el efecto del resto de parámetros de poda previa.

```python
param_grid = {
    'max_depth':         [3, 5, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5, 10],
    'max_leaf_nodes':    [None, 10, 20, 50]
}

grid_tree = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_tree.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_tree.best_params_}")
print(f"Mejor CV score:     {grid_tree.best_score_:.4f}")
print(f"Test score:         {grid_tree.best_estimator_.score(X_test, y_test):.4f}")
```

**Reporta en una tabla con las mejores configuraciones encontradas:**

| max_depth | min_samples_split | min_samples_leaf | max_leaf_nodes | CV Acc | Test Acc |
|-----------|-------------------|------------------|----------------|--------|----------|
| ... | ... | ... | ... | ... | ... |
| **Mejor** | **...** | **...** | **...** | **...** | **...** |

### 2.5 Poda Posterior: Cost-Complexity Pruning

CART implementa poda posterior mediante `ccp_alpha`. A mayor valor de alpha, mayor poda.

```python
# Obtener la secuencia de alphas generada por CART
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # El último genera solo el nodo raíz

for alpha in ccp_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    cv   = cross_val_score(tree, X_train, y_train, cv=5)
    tree.fit(X_train, y_train)
    print(f"alpha={alpha:.5f}: CV={cv.mean():.4f}, "
          f"Test={tree.score(X_test, y_test):.4f}, hojas={tree.get_n_leaves()}")
```

**Reporta en una tabla (selecciona valores representativos del rango):**

| ccp_alpha | CV Accuracy | Test Accuracy | Profundidad | N° Hojas |
|-----------|-------------|---------------|-------------|----------|
| 0.0 (sin poda) | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| **Mejor** | **...** | **...** | **...** | **...** |
| alpha grande | ... | ... | ... | ... |

### 2.6 Importancia de Características

```python
best_tree = grid_tree.best_estimator_
importances  = best_tree.feature_importances_
feature_names = X.columns.tolist()

indices = importances.argsort()[::-1]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
```

**Reporta en una tabla:**

| Ranking | Feature | Importancia (Gini) |
|---------|---------|--------------------|
| 1 | ... | ... |
| 2 | ... | ... |
| ... | ... | ... |

### 2.7 Visualización del Árbol

```python
from sklearn.tree import plot_tree, export_text

plt.figure(figsize=(20, 10))
plot_tree(best_tree, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=10, max_depth=4)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')

# Representación en texto
print(export_text(best_tree, feature_names=feature_names, max_depth=4))
```

### 2.8 Evaluación Final (Parte 2)

**Reporta en una tabla comparativa:**

| Modelo | CV Accuracy | Test Accuracy | F1-Score | Profundidad | N° Hojas |
|--------|-------------|---------------|----------|-------------|----------|
| Sin restricciones | ... | ... | ... | ... | ... |
| Mejor poda previa | ... | ... | ... | ... | ... |
| Mejor poda posterior | ... | ... | ... | ... | ... |

---

## Parte 3: Random Forest

### 3.1 Random Forest con Configuración por Defecto

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
rf.fit(X_train, y_train)

print(f"OOB Score:  {rf.oob_score_:.4f}")
print(f"Test Score: {rf.score(X_test, y_test):.4f}")
```

**Reporta en una tabla:**

| Configuración | OOB Score | Test Accuracy | F1-Score |
|---------------|-----------|---------------|----------|
| RF por defecto (n=100) | ... | ... | ... |

### 3.2 Efecto del Número de Estimadores

```python
for n in [10, 25, 50, 100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_train, y_train)
    print(f"n_estimators={n:>3}: OOB={rf.oob_score_:.4f}, "
          f"Test={rf.score(X_test, y_test):.4f}")
```

**Reporta en una tabla:**

| n_estimators | OOB Score | Test Accuracy | Tiempo (s) |
|--------------|-----------|---------------|------------|
| 10 | ... | ... | ... |
| 25 | ... | ... | ... |
| 50 | ... | ... | ... |
| 100 | ... | ... | ... |
| 200 | ... | ... | ... |
| 500 | ... | ... | ... |

### 3.3 Efecto de max_features

`max_features` controla cuántas features se consideran en cada split. Reduce la correlación entre árboles, aumentando la diversidad del ensemble.

```python
for mf in ['sqrt', 'log2', 0.5, None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf,
                                random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_train, y_train)
    print(f"max_features={str(mf):>6}: OOB={rf.oob_score_:.4f}, "
          f"Test={rf.score(X_test, y_test):.4f}")
```

**Reporta en una tabla:**

| max_features | OOB Score | Test Accuracy |
|--------------|-----------|---------------|
| sqrt | ... | ... |
| log2 | ... | ... |
| 0.5 | ... | ... |
| None (todas) | ... | ... |

### 3.4 Grid Search

```python
param_grid = {
    'n_estimators':     [100, 200],
    'max_features':     ['sqrt', 'log2', None],
    'max_depth':        [None, 10, 20],
    'min_samples_leaf': [1, 2, 5]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_rf.best_params_}")
print(f"Mejor CV score:     {grid_rf.best_score_:.4f}")
print(f"Test score:         {grid_rf.best_estimator_.score(X_test, y_test):.4f}")
```

### 3.5 Importancia de Características

```python
import numpy as np
from sklearn.inspection import permutation_importance

best_rf = grid_rf.best_estimator_

# Gini Importance (MDI)
importances_gini = best_rf.feature_importances_

# Permutation Importance (más fiable con features correlacionadas)
perm = permutation_importance(best_rf, X_test, y_test, n_repeats=10, random_state=42)

indices = importances_gini.argsort()[::-1]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}: "
          f"Gini={importances_gini[idx]:.4f}, "
          f"Perm={perm.importances_mean[idx]:.4f} (+/-{perm.importances_std[idx]:.4f})")
```

**Reporta en una tabla:**

| Ranking | Feature | Gini Importance | Permutation Importance |
|---------|---------|-----------------|----------------------|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |
| ... | ... | ... | ... |

### 3.6 Extra Trees

Extra Trees introduce aún más aleatoriedad: en lugar de buscar el mejor split, lo elige de forma aleatoria entre las features seleccionadas.

```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
cv_et = cross_val_score(et, X_train, y_train, cv=5)

print(f"Extra Trees: CV={cv_et.mean():.4f} (+/-{cv_et.std():.4f}), "
      f"Test={et.score(X_test, y_test):.4f}")
```

### 3.7 Evaluación Final (Parte 3)

**Reporta en una tabla comparativa:**

| Modelo | CV Accuracy | Test Accuracy | F1-Score | OOB Score | Tiempo (s) |
|--------|-------------|---------------|----------|-----------|------------|
| Decision Tree (baseline) | ... | ... | ... | N/A | ... |
| Random Forest por defecto | ... | ... | ... | ... | ... |
| Random Forest mejor config | ... | ... | ... | ... | ... |
| Extra Trees | ... | ... | ... | N/A | ... |

---

## Parte 4: Gradient Boosting

### 4.1 Gradient Boosting con Configuración por Defecto

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
cv_gb = cross_val_score(gb, X_train, y_train, cv=5)

print(f"CV={cv_gb.mean():.4f} (+/-{cv_gb.std():.4f}), "
      f"Test={gb.score(X_test, y_test):.4f}")
```

### 4.2 Efecto del Número de Estimadores y Learning Rate

En Gradient Boosting el número de estimadores y el learning rate están fuertemente relacionados: cuantos más estimadores, menor debe ser el learning rate para no sobreajustar.

```python
configs = [
    (50,  0.1), (100, 0.1), (200, 0.1),
    (50,  0.5), (100, 0.05), (200, 0.01)
]

for n_est, lr in configs:
    gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
    cv = cross_val_score(gb, X_train, y_train, cv=5)
    gb.fit(X_train, y_train)
    print(f"n={n_est}, lr={lr}: CV={cv.mean():.4f}, Test={gb.score(X_test, y_test):.4f}")
```

**Reporta en una tabla:**

| n_estimators | learning_rate | CV Accuracy | Test Accuracy |
|--------------|---------------|-------------|---------------|
| 50 | 0.1 | ... | ... |
| 100 | 0.1 | ... | ... |
| 200 | 0.1 | ... | ... |
| 100 | 0.05 | ... | ... |
| 200 | 0.01 | ... | ... |

### 4.3 Efecto de max_depth

En Gradient Boosting los árboles base son deliberadamente simples (stumps o árboles poco profundos).

```python
for depth in [1, 2, 3, 5]:
    gb = GradientBoostingClassifier(max_depth=depth, n_estimators=100,
                                    learning_rate=0.1, random_state=42)
    cv = cross_val_score(gb, X_train, y_train, cv=5)
    gb.fit(X_train, y_train)
    print(f"max_depth={depth}: CV={cv.mean():.4f}, "
          f"train={gb.score(X_train, y_train):.4f}, test={gb.score(X_test, y_test):.4f}")
```

**Reporta en una tabla:**

| max_depth | Train Acc | CV Accuracy | Test Accuracy |
|-----------|-----------|-------------|---------------|
| 1 (stump) | ... | ... | ... |
| 2 | ... | ... | ... |
| 3 | ... | ... | ... |
| 5 | ... | ... | ... |

### 4.4 Grid Search

```python
param_grid = {
    'n_estimators':  [100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'max_depth':     [1, 2, 3],
    'subsample':     [0.8, 1.0]
}

grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_gb.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_gb.best_params_}")
print(f"Mejor CV score:     {grid_gb.best_score_:.4f}")
print(f"Test score:         {grid_gb.best_estimator_.score(X_test, y_test):.4f}")
```

### 4.5 AdaBoost

AdaBoost es otro método de boosting. En lugar de ajustar residuos, pondera las muestras mal clasificadas para que los siguientes estimadores les presten más atención.

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost con árbol de decisión de profundidad 1 (stump) como estimador base
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)
cv_ada = cross_val_score(ada, X_train, y_train, cv=5)

print(f"AdaBoost: CV={cv_ada.mean():.4f} (+/-{cv_ada.std():.4f}), "
      f"Test={ada.score(X_test, y_test):.4f}")
```

**Reporta en una tabla explorando n_estimators y learning_rate:**

| n_estimators | learning_rate | CV Accuracy | Test Accuracy |
|--------------|---------------|-------------|---------------|
| 50 | 1.0 | ... | ... |
| 100 | 1.0 | ... | ... |
| 100 | 0.5 | ... | ... |
| 200 | 0.1 | ... | ... |

### 4.6 Evaluación Final (Parte 4)

**Reporta en una tabla comparativa:**

| Modelo | CV Accuracy | Test Accuracy | F1-Score | Tiempo (s) |
|--------|-------------|---------------|----------|------------|
| Random Forest (mejor config) | ... | ... | ... | ... |
| Gradient Boosting por defecto | ... | ... | ... | ... |
| Gradient Boosting mejor config | ... | ... | ... | ... |
| AdaBoost | ... | ... | ... | ... |

---

## Parte Final: Comparación y Evaluación Global

Reúne los mejores modelos de cada parte en una tabla comparativa final.

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

best_models = {
    'SVM (mejor config)':   ...,
    'Decision Tree':        ...,
    'Random Forest':        ...,
    'Gradient Boosting':    ...,
    'AdaBoost':             ...
}

for name, model in best_models.items():
    model.fit(X_train, y_train)
    cv  = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    acc = model.score(X_test, y_test)
    print(f"{name}: CV={cv.mean():.4f} (+/-{cv.std():.4f}), Test={acc:.4f}")
```

**Tabla comparativa final (mejor configuración de cada método):**

| Modelo | CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | Tiempo (s) |
|--------|-------------|---------------|-----------|--------|----------|------------|
| SVM | ... | ... | ... | ... | ... | ... |
| Decision Tree | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... |
| Gradient Boosting | ... | ... | ... | ... | ... | ... |
| AdaBoost | ... | ... | ... | ... | ... | ... |

**Incluye la matriz de confusión del mejor modelo global.**

```python
best_model = ...
y_pred = best_model.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.savefig('confusion_matrix_final.png', dpi=300, bbox_inches='tight')
```

---

## Entregas

### 1. Memoria en LaTeX (PDF)
- Usa la plantilla proporcionada.
- Incluye todas las tablas con resultados completos de cada parte.
- Matriz de confusión del mejor modelo de cada parte y de la comparativa final.
- Máximo 20 páginas.

### 2. Notebook de Python (.ipynb)
- Código completo y ejecutable.
- Comentado apropiadamente.
- Organizado en secciones que se correspondan con las partes del enunciado.

## Recursos

- SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- LinearSVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
- SGDClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
- DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- ExtraTreesClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
- GradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- AdaBoostClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html