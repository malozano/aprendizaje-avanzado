# Métodos de ensemble

La idea tras los modelos de _ensemble_ [@sagi2018ensemble] consiste en combinar diferentes modelos base sencillos para construir un modelo más robusto y preciso que cualquiera de los modelos individuales. 

Hemos visto que modelos diferentes cometen errores diferentes. Buscamos combinarlos de forma que podamos eliminar o reducir esos errores individuales. Para ello necesitaremos combinar un conjunto diverso de modelos, y cada uno de estos modelos individuales debería proporcionar por si mismo una precisión que sea superior al azar.

El aprendizaje con métodos de _ensemble_ puede descomponerse en dos tareas principales:

- Aprender un conjunto de modelos base a partir de los datos de entrenamiento.
- Combinarlos para construir el predictor conjunto.



Una de las principales ventajas de los _ensembles_ es que pueden mejorar el compromiso entre **sesgo y varianza**. Recordemos que el error esperado se puede descomponer como: 

$$E[(y - \hat{y})^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Existen cuatro principales tipos de enfoques para abordar la combinación de modelos: _Voting_, _Stacking_, _Bagging_ y _Boosting_. Diferentes enfoques abordarán el problema de forma distinta. Algunos tipos de métodos de _ensemble_ se centran en reducir principalmente la varianza, como por ejemplo los métodos de _Bagging_, mientras que otros se centran fundamentalmente en el sesgo, como sería el caso del _Boosting_. 

Cada categoría de métodos de _ensemble_ tiene un enfoque diferente para generar y combinar modelos, tal como se resume en la siguiente tabla:

| Categoría | Modelos | Datos de entrenamiento | Combinación | Entrenamiento |
|-----------|---------|----------------------|-------------|---------------|
| **Voting** | Heterogéneos | Iguales | Fija (votos/promedio) | Independiente |
| **Stacking** | Heterogéneos | Iguales | Aprendida (meta-modelo) | Dos niveles |
| **Bagging** | Homogéneos | Bootstrap (diferentes) | Fija (promedio) | Independiente |
| **Boosting** | Homogéneos | Ponderados/residuos | Ponderada (aprendida) | Secuencial |

A continuación estudiaremos en detalle cada una de estas categorías, y los principales métodos que existen dentro de cada una de ellas.


## Voting

La idea tras los modelos de _Voting_ [@kittler1998combining] es la de entrenar múltiples modelos independientes y combinar sus predicciones mediante votación (en el caso de clasifiación) o mediante promediado (en el caso de regresión). 

En este caso los modelos se entrenan de forma independiente con el **mismo conjunto de datos**, y no hay dependencia entre modelos, por lo que pueden entrenarse en paralelo. Además, podemos **combinar diferentes tipos de modelos**. 

Por ejemplo, podríamos combinar un modelo de Regresión Logística, con KNN y SVM, obtener la predicción que devuelve cada uno de ellos, y devolver aquella que obtenga más votos. 

Encontramos diferentes formas de abordar la votación, que podremos aplicar según se trate de un problema de clasificación o de regresión.

Vamos a considerar que combinamos $M$ clasificadores $\{h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_M(\mathbf{x}) \}$. A continuación veremos cómo combinar sus predicciones con cada enfoque de votación.

### Hard Voting

Se trata de un enfoque dirigido al problema de clasificación. Con el enfoque _Hard Voting_, la clase predicha por el clasificador será la que reciba más votos por parte de los clasificadores individuales (es decir, la moda del conjunto de predicciones). 

$$\hat{y} = \text{mode}(h_1(\mathbf{x}), h_2(\mathbf{x}), ..., h_M(\mathbf{x}))$$

**Ejemplo:**
```
Problema: Clasificar un email como spam o no-spam
Modelo 1 (Logistic Regression): spam
Modelo 2 (Decision Tree):       spam  
Modelo 3 (SVM):                 no-spam
Modelo 4 (KNN):                 spam

Resultado final: spam (3 votos contra 1)
```

### Soft Voting

A diferencia del caso anterior, con el enfoque _Soft Voting_ lo que tendremos en cuenta es la suma de probabilidades de predicción de cada clasificador individual. Aquella clase $k$ cuya suma de probabilidades de predicción sea mayor, será la seleccionada como predicción del modelo combinado:


$$\hat{y} = \arg\max_k \frac{1}{M} \sum_{i=1}^{M} P_i(y = k | \mathbf{x})$$

**Ejemplo:**
```
Modelo 1: P(spam) = 0.9, P(no-spam) = 0.1
Modelo 2: P(spam) = 0.6, P(no-spam) = 0.4
Modelo 3: P(spam) = 0.4, P(no-spam) = 0.6

Promedio: P(spam) = 0.633, P(no-spam) = 0.367
Resultado: spam
```

Es importante destacar que para poder utilizar este enfoque, los modelos individuales deben poder proporcionarnos la probabilidad de la predicción. Este enfoque tiene la ventaja de que considera la confianza de cada modelo en la predicción realizada.

### Soft Voting ponderado

Se trata de un caso similar al anterior, pero dando un peso diferente a cada predictor individual en la suma:


$$\hat{y} = \arg\max_k \sum_{i=1}^{M} w_i \cdot P_i(y = k | \mathbf{x})$$

Donde $\sum w_i = 1$ y $w_i$ nos permite dar mayor peso a los modelos en los que tengamos mayor confianza. Estos valores se pueden determinar a partir de la precisión de los modelos individuales en la validación, de métricas como F1-score o el inverso del error cometido.

**Ejemplo:**
```
Modelo 1 (accuracy=0.85): w1 = 0.4
Modelo 2 (accuracy=0.80): w2 = 0.35  
Modelo 3 (accuracy=0.75): w3 = 0.25

La votación ponderada da más peso al mejor modelo
```

### Promediado

Este enfoque, a diferencia de los anteriores, está dirigido al problema de regresión. En caso de regresión las predicciones de promedian de la siguiente forma:

$$\hat{y} = \frac{1}{M} \sum_{i=1}^{M} h_i(\mathbf{x})$$

También es posible asignar un peso a cada predictor, igual que en el caso del enfoque anterior:

$$\hat{y} = \sum_{i=1}^{M} w_i  h_i(\mathbf{x})$$

### Implementación

En sklearn tenemos las clases [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier) y [VotingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor), para problemas de clasificación y regresión respectivamente.

Para utilizar estas clases deberemos pasar como parámetro `estimator` la lista de predictores individuales que queramos combinar, y nos permitirán aplicar cualquiera de los enfoques anteriores.

En el caso de la clasificación, podemos elegir el tipo de votación con el parámetro `voting`, que podrá tomar como valores `hard` o `soft`, y en el caso del segundo tipo con `weights` podemos especificar pesos específicos para cada predictor.

En el caso de la regresión, también contamos con un parámetro `weights` para especificar los pesos de cada predicción.

### Consideraciones finales sobre _Voting_

Se trata de un método muy fácil de entender y sencillo de implementar, que se basa en modelos existentes que no es necesario modificar. Todos los modelos base se entrenan independientemente y podría hacerse en paralelo. 

Debemos tener en cuenta que para que sea efectivo los diferentes modelos base deben ser diversos, ya que si todos son similares no obtendremos apenas ganancia combinándolos. Funcionará mejor cuando los modelos base capturen aspectos distintos de los datos y cometan errores diferentes. 

Este método también ser verá beneficiado cuando los modelos tengan un rendimiento similar. Si uno de los modelos base fuera muy superior, sería recomendable utilizar únicamente dicho modelo, mientras que si tenemos un modelo peor que el azar, ese modelo perjudicará al _ensemble_ y convendría eliminarlo.

Voting es la forma más básica de modelo de _ensemble_, en la que se utiliza una combinación fija de modelos (votos o promedios con pesos predefinidos). Esto nos lleva a preguntarnos si podríamos aprender la mejor forma de combinar los modelos. Esto lo abordaremos con los métodos de _Stacking_ que veremos a continuación.



## Stacking

En el caso de _Stacking_ [@wolpert1992stacked], al igual que en _Voting_ tenemos una serie de modelos base heterogéneos que entrenamos de forma independiente, pero a diferencia del caso anterior, no tendremos una combinación fija, sino que utilizaremos un **meta-modelo** para aprender la forma de combinar los diferentes modelos base. 

De esta forma, se podrán capturar relaciones complejas entre las diferentes predicciones. Tendremos una arquitectura en dos niveles, en la que en el nivel inferior tendremos los $M$ diferentes modelos base, cada uno de los cuales producirá una predicción $p_i$ y en el nivel superior tendremos el meta-modelo, que recibirá como entrada las diferentes predicciones $\{ p_1, p_2, \ldots, p_M \}$ y producirá como salida la predicción del _ensemble_.

```
NIVEL 0 (Modelos base):
├─ Modelo 1 (ej: DT)               → predicción p1
├─ Modelo 2 (ej: SVM)              → predicción p2
├─ Modelo 3 (ej: Logistic Reg)     → predicción p3
└─ Modelo 4 (ej: KNN)              → predicción p4

NIVEL 1 (Meta-modelo):
└─ Recibe [p1, p2, p3, p4] como features
   → aprende la mejor combinación
   → predicción final
```

### Algoritmo de entrenamiento

El entrenamiento de un _ensemble_ de tipo _Stacking_ se hará en varios pasos, ya que en primer lugar deberemos obtener una serie de predicciones de los modelos base para entrenar con ellas el meta-modelo. Para reducir el _overfitting_, utilizaremos _cross-validation_ a la hora de generar estas predicciones, siguiendo el siguiente proceso:

**Paso 1: Obtención de predicciones para el meta-modelo (Nivel 0)**

Supongamos que contamos con $M$ modelos base $m_j$, con $j=1, 2, \ldots, M$. Entrenaremos cada uno de estos modelos utilizando _cross-validation_, y para cada _fold_:

- Dividimos el _dataset_ en datos de entrenamiento y datos de validación. 
- Cada modelo base se entrena con el conjunto de entrenamiento del _fold_
- Se generan predicciones para los datos del conjunto de validación. Estas son predicciones **out-of-fold** (OOF), ya que han sido generadas para cada observación utilizando un modelo que no ha sido entrenado con dicha observación.
- Guardamos las predicciones OOF generadas.

De esta forma, obtendremos un nuevo conjunto de datos compuesto por las predicciones OOF generadas por cada modelo base para cada ejemplo de entrada. 

Considerando que tenemos $N$ ejemplos de entrada en nuestro _dataset_ y $M$ modelos, tendremos una matriz como la siguiente:

$$
Z = 
\begin{bmatrix}
\hat{y}_1^{m_1} & \hat{y}_1^{m_2} & \ldots & \hat{y}_1^{m_M} \\
\hat{y}_2^{m_1} & \hat{y}_2^{m_2} & \ldots & \hat{y}_2^{m_M} \\
\vdots & \vdots &   & \vdots \\
\hat{y}_N^{m_1} & \hat{y}_N^{m_2} & \ldots & \hat{y}_N^{m_M} \\
\end{bmatrix} 
$$


Esta matriz $Z$ constituirá los datos de entrada para el entrenamiento del meta-modelo. Es importante haber utilizado las predicciones OOF para construirla, ya que si hubiéramos obtenido predicciones de los modelos base obtenidas a partir de datos vistos durante el entrenamiento, habríamos tenido un caso demasiado optimista y habríamos favorecido el _overfitting_.

**Paso 2: Entrenar el meta-modelo (Nivel 1)**

En este punto utilizamos la matriz $Z$ de predicciones OOF generada en el paso anterior como entrada del meta-modelo. Considerando que $\mathbf{z}_i$ (fila $i$-ésima de la matriz $Z$) es una tupla con las predicciones OOF generadas por cada uno de los $M$ modelos para el ejemplo de entrada $\mathbf{x}_i$, al entrenar el meta-modelo la salida esperada será la etiqueta original $y_i$. 

Es decir, utilizaremos para entrenar el meta-modelo un conjunto de pares $(\mathbf{z}_i, y_i)$, con $i=1, 2, \ldots, N$. 

**Paso 3: Entrenamiento definitivo de los modelos base (Nivel 0)**

Dado que queremos tener el mejor modelo posible, entrenaremos ahora de nuevo todos los modelos base pero utilizando el _dataset_ completo. De esta forma, guardaremos estos modelos base reentrenados junto con el meta-modelo entrenado en el paso anterior, y con esto tendremos el modelo completo.

**Predicción**

Una vez entrenados de forma definitiva modelos base y meta-modelo, a la hora de obtener una predicción con un nuevo dato $\mathbf{x}$ seguiremos el siguiente proceso

1. Cada modelo base $m_j$, con $j=1,2,\ldots,M$, produce una predicción $z_j$ para $\mathbf{x}$.
2. El meta-modelo recibe como entrada la tupla $(z_1, z_2, \ldots, z_M)$ de predicciones realizadas por los modelos base.
3. El meta-modelo genera predicción final $y$.


### Implementación

En la librería _sklearn_ contamos con las clases [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) y [StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) para implementar _Stacking_ tanto en problemas de clasificación como de regresión. 

A la hora de construir este _ensemble_, deberemos proporcionar tanto un conjunto de modelos base, en el parámetro `estimators`, como un meta-modelo, en `final_estimator`. A continuación, mostramos un ejemplo de código con 3 modelos base, y con regresión logística como meta-modelo:

```python
# Definir modelos base (nivel 0)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svm', SVC(probability=True))
]

# Definir meta-modelo (nivel 1)
meta_model = LogisticRegression()

# Crear stacking ensemble
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# Entrenar
stacking.fit(X_train, y_train)

# Predecir
predictions = stacking.predict(X_test)
```

### Variantes de _Stacking_

Podemos encontrar diferentes variantes de _Stacking_. Vamos a continuación a describir algunas de ellas.

Una de las variantes es la conocida como **Blending**. Se diferencia de _Stacking_ básicamente en que en lugar de utilizar un _K-Fold_ para generar predicciones OOF para el entrenamiento del meta-modelo, utiliza un único particionamiento en conjunto de entrenamiento y conjunto de validación. Por ejemplo, podemos particionar con un 80% de los datos para entrenamiento y 20% para validación, con lo cual, sólo se estarían generando predicciones con el 20% de los datos. _Blending_ resulta más sencillo y rápido, pero estaremos entrenando el meta-modelo con menos datos. Con un _dataset_ pequeño podemos perder rendimiento y tendremos mayor varianza, pero en caso de tener un _dataset_ grande y buscar reducir el coste computacional _blending_ puede ser una opción adecuada.

Otra variante a considerar es el **Stacking multi-nivel**. Este método consiste en apilar múltiples niveles de meta-modelos:
```
NIVEL 0: Modelos base 
    ↓ (predicciones)
NIVEL 1: Primer grupo de meta-modelos
    ↓ (predicciones)
NIVEL 2: Meta-meta-modelo final
    ↓
Predicción final
```

Esta técnica tiene como ventaja que puede ser capaz de capturar relaciones muy complejas, pero tiene mayor riesgo de _overfitting_ y resulta difícil de interpretar. 

Según el **tipo de meta-modelo**, podríamos agrupar los tipos de _Stacking_ en tres grandes categorías:

- **Linear Stacking:** El meta-modelo es un modelo lineal, típicamente Regresión Lineal o Regresión Logística.
- **Tree Stacking:** El meta-modelo es un modelo basado en árboles, como Árboles de Decisión o Random Forest.
- **Neural Stacking:** El meta-modelo es una Red Neuronal.

También podemos considerar varias variantes según el **tipo de meta-_features_** (_features_ generadas para el meta-modelo):

- **Predicciones solo:** Tenemos como meta-_features_ únicamente las predicciones de los modelos base. Por ejemplo, tendríamos únicamente $0$ o $1$ en caso de clasificación binaria. Si queremos que _sklearn_ utilice este tipo de predicciones, podemos proporcionar el parámetro `stack_method='predict'`. 
- **Probabilidad:** Como meta-_features_ tendríamos las probabilidades de pertenencia a cada clase. Para que _sklearn_ utilice este tipo de características, todos los modelos base deben contar con el método `predict_proba`. Como por defecto tenemos `stack_method='auto'`, si todos los modelos base cuentan con `predict_proba` entonces utilizará de forma preferente estas probabilidades.
- **Predicciones y _features_ originales:** Concatenemos las _features_ originales del ejemplo de entrada con las predicciones realizadas por los modelos base. En _sklearn_ esto lo conseguiremos proporcionando el parámetro `passthrough=True`.

### Consideraciones finales sobre _Stacking_

_Stacking_ lleva la idea de ensemble un paso más allá que _Voting_, ya que en lugar de usar combinaciones fijas, **aprende la mejor forma de combinar modelos**. Resulta muy **flexible**, pudiendo utilizar cualquier tipo de modelo en cualquier nivel y normalmente ofrecerá **mejor rendimiento** que _Voting_.

Sin embargo, es más **complejo**, ya que debemos ajustar un gran número de hiper-parámetros en sus diferentes niveles. Tiene un **alto coste computacional**, ya que tenemos que entrenar tanto modelos base como meta-modelo, y es **difícil de interpretar**.

Además, _Stacking_ es propenso al **_overfitting_** y para mitigarlo es importante, tal como hemos comentado, utilizar _cross-validation_ para generar las meta-_features_. Otras estrategias que pueden ayudar a mitigar este problema es utilizar un meta-modelo sencillo, como puede ser Regresión Logística, y utilizar siempre regularización en el meta-modelo. 

_Stacking_ será un método adecuado cuando contemos con **modelos buenos y diversos**, y un **conjunto de datos suficientemente grande** como para evitar el _overfitting_. Sin embargo, con conjuntos de datos pequeños tendremos un alto riesgo de _overfitting_. 

Hasta ahora hemos visto métodos que combinan modelos **heterogéneos** entrenados en los **mismos datos**, pero, ¿y si en lugar de esto buscamos la diversidad entrenando un mismo modelo con diferentes muestras de datos?. Esta pregunta nos lleva a explorar el siguiente tipo de métodos de _ensemble_, el conocido como **Bagging**.



## Bagging (Bootstrap Aggregating)

A diferencia de _Voting_ y _Stacking_, los métodos de _Bagging_ [@breiman1996bagging] se basan en entrenar múltiples modelos **homogéneos** en **diferentes muestras** _bootstrap_ (con reemplazo) del conjunto de entrenamiento, promediando sus predicciones.

Con esta técnica se busca principalmente **reducir la varianza**, con lo que va a ser efectiva principalmente cuando se aplique a modelos con alta varianza, como es el caso de los árboles de decisión. Los modelos pueden entrenarse en paralelo, cada uno de ellos con una muestra diferente de datos. Utiliza **muestreo con reemplazo** (_bootstrap sampling_) para obtener la muestra con la que se entrenará cada modelo. Es también importante destacar que **se le da la misma importancia a todas las predicciones**. Estas se combinarán mediante promedio en caso de regresión, y mediante votación en caso de clasificación.

### Bootstrap Sampling

El término _Bootstrap_ se refiere a una técnica estadística de remuestreo con reemplazo. A partir de un _dataset_ original $\mathcal{D}$ con $N$ ejemplos obtenemos $B$ _bootstrap samples_ $\mathcal{D}_b$, con $b=1, 2, \ldots, B$. Cada _bootstrap sample_ tendrá también $N$ ejemplos, pero podrá haber ejemplos del conjunto original repetidos o ausentes (_out-of-bag_). 

Por ejemplo, considerando $N=10$, podríamos tener:

$$
\begin{align*}
\mathcal{D} &= \{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10\} \\
\Downarrow \\
\mathcal{D}_1 &= [1, 3, 3, 5, 7, 8, 8, 9, 10, 10] \\ 
\mathcal{D}_2 &= [1, 2, 2, 4, 4, 5, 6, 8, 9, 9] \\
 & \ldots \\
\mathcal{D}_B &= [2, 3, 4, 5, 5, 6, 7, 7, 8, 10]
\end{align*}
$$

Podemos calcular la probabilidad de que un ejemplo sea _out-of-bag_ como $(1-\frac{1}{N})^N$, y con un valor alto de $N$ está probabilidad converge a $0.368$. Por lo tanto, aproximadamente el $37 \%$ de los ejemplos quedarán fuera de cada muestra.


### Algoritmo de Bagging

A continuación se muestra el algoritmo de entrenamiento de _Bagging_:

$$
\begin{align*}
& \text{Entrada: } \mathcal{D} \\
& \text{Para } b=1 \text{ hasta } B:\\
& \quad \mathcal{D}_b \leftarrow \text{Seleccionar, con reemplazo } N \text{ ejemplos del conjunto } \mathcal{D} \\
& \quad h_b \leftarrow \text{Entrenar un modelo con }\mathcal{D}_b \\
& \text{Devuelve: } H = \{ h_1, h_2, \ldots, h_B \}
\end{align*}
$$

Una vez entrenado el _ensemble_ $H$, la **predicción en caso de regresión** se calculará como el promedio de las predicciones de cada modelo del _ensemble_:

$$
\hat{y} = \frac{1}{B} \sum_{b=1}^B h_b(\mathbf{x})
$$

La **predicción en caso de clasificación** se obtendrá mediante votación (moda del conjunto de predicciones):

$$
\hat{y} = \text{moda}(h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_B(\mathbf{x}))
$$

Si cada modelo individual $h_b$ proporciona una probabilidad $P_b(y=k|\mathbf{x})$ de que el ejemplo de entrada $\mathbf{x}$ pertenezca a la clase $k$, entonces también podríamos calcular la probabilidad global del _ensemble_ mediante promediado:

$$
P(y=k|\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y=k|\mathbf{x})
$$


### Out-of-Bag (OOB) Error

Como hemos comentado anteriormente, aproximadamente el 37% de las muestras no aparecen en cada _bootstrap sample_. Estas muestras **out-of-bag** se pueden usar para validación sin necesidad de un conjunto separado.

A continuación se muestra el algoritmo para el **cálculo del OOB error:**

$$
\begin{align*}
& \text{Entrada: } \mathcal{D} \\
& \text{Para cada } (\mathbf{x}_i, y_i) \in \mathcal{D}\\
& \quad \mathcal{B}_i \leftarrow \{b: (\mathbf{x}_i, y_i) \notin \mathcal{D}_b  \} \quad \text{(Modelos que no usaron } \mathbf({x}_i, y_i) \text{ en el entrenamiento)} \\
& \quad \text{Si } |\mathcal{B}_i| > 0: \quad \text{(Calcula predicciones para estos modelos) }\\
& \quad\quad  \hat{y}_i^{\text{OOB}} =
\begin{cases}
    \text{moda}\{h_b(\mathbf{x}_i) : b \in \mathcal{B}_i\} \quad \text{(Clasificación)} \\
    \frac{1}{|\mathcal{B}_i|} \sum_{b \in \mathcal{B}_i} h_b(\mathbf{x}_i) \quad \text{(Regresión)}
\end{cases}
 \\
& \mathcal{I}_{OOB} \leftarrow \{ i: |\mathcal{B}_i| > 0 \} \quad \text{(Índices de las muestras que fueron OOB para al menos un modelo)} \\ 
&OOBError \leftarrow \begin{cases}
    \frac{1}{|\mathcal{I}_{OOB}|} \sum_{i \in \mathcal{I}_{OOB}} \mathbb{1}(\hat{y}_i^{\text{OOB}} \neq y_i)  \quad \text{(Clasificación)} \\
    \frac{1}{|\mathcal{I}_{OOB}|} \sum_{i \in \mathcal{I}_{OOB}} (y_i - \hat{y}_i^{\text{OOB}})^2 \quad \text{(Regresión)}
\end{cases}


\end{align*}
$$

De esta forma podemos estimar el error de generalización sin necesitar un conjunto de validación separado y sin suponer un coste computacional adicional

### Análisis del método

Vamos a analizar a continuación los motivos por los que la técnica de _Bagging_ ayuda a reducir la varianza. Vamos a empezar haciendo un estudio a nivel teórico.

Supongamos que tenemos B modelos independientes (sin correlación entre ellos) con varianza $\sigma^2$ cada uno. Si los promediamos:

$$\text{Var}(\text{promedio}) = \text{Var}\left(\frac{1}{B}\sum_{i=1}^B h_i\right) = \frac{1}{B^2} \sum_{i=1}^B \text{Var}(h_i) = \frac{\sigma^2}{B}$$

La varianza se reduce por un factor de B. Sin embargo, en la práctica los modelos no son completamente independientes, ya que están entrenados con muestras correlacionadas, pero aún así la reducción de varianza es significativa. Si consideramos que tenemos una correlación $\rho$ entre modelos, entonces tendríamos:

$$\text{Var}(\text{ensemble}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Si los modelos predicen exactamente lo mismo entonces tendremos correlación $\rho=1$, y no obtendremos ninguna ganancia con el _ensemble_. Sin embargo, conforme consigamos reducir la correlación podremos reducir la varianza del modelo. Vemos que el primer término de la función anterior solo se puede reducir reduciendo la correlación, mientras que el segundo término se podría reducir aumentando el número de modelos. El caso ideal teórico sería conseguir $\rho=0$, pero es difícil conseguirlo en la práctica. Lo principal a tener en cuenta es que a mayor correlación, la ganancia será menor, y por ello es importante la **diversidad** de los modelos.


### Implementación de Bagging Genérico

La librería _sklearn_ incluye las clase [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) y [`BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html) que nos permiten utilizar de forma genérica está técnica, tanto para problemas de clasificación como de regresión.

Deberemos proporcionar el modelo base que queremos entrenar en el parámetro `estimator`. Por defecto utilizará como clasificador base Árboles de Decisión. Podemos también indicar la cantidad $B$ de clasificadores a entrenar en el parámetro `num_estimators`. 

Es interesante observar que esta implementación nos da gran flexibilidad para la implementación del método. Por ejemplo, con `max_samples` podemos indicar el número de ejemplos que tendrá cada muestra generada (por defecto tendrá $N$ ejemplos, tantos como el conjunto de entrada original), y también podemos indicar con el parámetro `bootstrap` si queremos que muestree con reemplazo o sin reemplazo. Es recomendable que este parámetro tenga siempre valor `True`, que es la opción por defecto. También tenemos el parámetro `oob_score` que nos permite indicar si queremos utilizar los ejemplos _out-of-bag_ para estimar el error de generalización (esto solo es posible si se utiliza muestre con reemplazo).

Además, no solo nos permite muestrear con reemplazo los ejemplos de entrada, sino que también nos permite hacer lo mismo con las _features_ (esto como veremos a continuación es algo que incorporan los _Randon Forest_). Con `max_features` y `bootstrap_feature` podemos indicar el número de _features_ que queremos seleccionar en cada muestra y si queremos que se puedan seleccionar con reemplazo, respectivamente. En este caso por defecto está establecido que no se realice muestreo con reemplazo de las _features_. 


```python
# Bagging con árboles de decisión profundos
bagging_tree = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),  # Árbol sin restricciones
    n_estimators=100,           # Número de modelos
    max_samples=1.0,            # Usar 100% de muestras (por defecto)
    max_features=1.0,           # Usar 100% de features (por defecto)
    bootstrap=True,             # Con reemplazo (por defecto)
    oob_score=True              # Calcular OOB error

)

bagging_tree.fit(X_train, y_train)
print(f"OOB Score: {bagging_tree.oob_score_:.4f}")
```


### Consideraciones finales sobre _Bagging_

Las técnicas de _Bagging_ están enfocadas a la **reducción de la varianza**, y por lo tanto son muy efectivas con modelos con alta varianza, como es el caso de los Árboles de Decisión profundos. El _ensemble_ es más **robusto** frente a _outliers_ que un modelo individual.

Es **paralelizable**, ya que todos los modelos se pueden entrenar independientemente, y resulta **sencillo** de implementar y entender. Además, nos permite validar sin coste adicional mediante la evaluación **OOB**. 

Sin embargo, entre sus limitaciones encontramos que **no reduce sesgo**. Además, es **menos interpretable** que un modelo invidual y computacionalmente más cosoto. 

Será interesante utilizar _Bagging_ con modelos con **alta varianza**, si queremos **reducir el _overfitting_** y si el modelo base es rápido de entrenar. Por estos motivos, podemos encontrar un mayor beneficio al aplicar _Bagging_ con modelos como Árboles de Decisión sin poda o KNN con $K$ pequeño, o en general cualquier modelo con alta varianza y bajo sesgo. Sin embargo, otros modelos como Regresión Logística que ya tienen baja varianza obtendrán normalmente poco beneficio.

Centrándonos en el caso de _Bagging_ con Árboles de Decisión, como hemos comentado, _Bagging_ reduce efectivamente la varianza al promediar múltiples árboles entrenados con diferentes muestras _bootstrap_. Sin embargo, los árboles resultantes tienden a ser similares entre sí. Dado que todos consideran las mismas características en cada división, suelen seleccionar las variables más informativas en los nodos superiores, generando estructuras correlacionadas. Esta correlación limita la reducción de varianza que puede conseguir el _ensemble_. Random Forest introduce una modificación sencilla pero efectiva: en cada división del árbol, en lugar de considerar todas las características disponibles, selecciona un subconjunto aleatorio de ellas. Esta aleatoriedad adicional reduce sustancialmente la correlación entre árboles, permitiendo una mayor reducción de varianza y mejorando significativamente el rendimiento del _ensemble_.



## Random Forest

Random Forest [@breiman2001random] es el método de _Bagging_ más popular, específicamente diseñado y optimizado para árboles de decisión. La innovación fundamental que introducen es combinar _Bagging_ con selección aleatoria de características. 

Para aumentar la diversidad entre los árboles, Random Forest introduce **dos fuentes de aletoriedad**:

1. **Bootstrap sampling (como Bagging estándar):**: Cada árbol se entrena en una muestra _bootstrap_ diferente, con lo que aproximadamente el 37% de ejemplos quedan _out-of-bag_ en cada árbol.

2. **Random feature selection:**: En el _split_ de cada nodo solo se considera un subconjunto aleatorio de $m$ _features_, ayudando a reducir la correlación entre árboles.

### Algoritmo de Random Forest

Detallamos a continuación el algoritmo para el **entrenamiento** de un modelo de tipo _Random Forest_:

$$
\begin{align*}
& \text{Entrada: Dataset } \mathcal{D} \text{, número de árboles } B \text{, número de feature } m \\
& \text{Para } b=1 \text{ hasta } B:\\
& \quad \mathcal{D}_b \leftarrow \text{Seleccionar, con reemplazo } N \text{ ejemplos del conjunto } \mathcal{D} \\
& \quad T_b \leftarrow \text{Crear árbol} \\
& \quad \text{Para cada nodo en } T_b: \\
& \quad \quad \text{Seleccionar } m \text{ features al azar del total } d  \\
& \quad \quad \text{Encontrar el mejor split utilizando solo esas features} \\
& \quad \quad \text{Dividir el nodo con el mejor split} \\
& \quad \quad \text{Hacer crecer el árbol hasta profundidad máxima (sin poda)} \\
& \text{Devuelve: } \text{Random Forest} \{ T_1, T_2, \ldots, T_B \}
\end{align*}
$$

Podemos observar que la principal diferencia con _Bagging_ puro de árboles es que con _Bagging_ en cada _split_ se estarían considerando siempre todas las $d$ _features_,  mientras que con _Random Forest_ en cada _split_ se considera un conjunto aleatorio de _features_, siendo $m \ll d$.


### Hiperparámetros 

En sklearn contamos con las implementaciones [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) y [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) de este modelo. A continuación enumeramos los principales hiperparámetros a tener en cuenta:

- **`n_estimators`** (número de árboles $B$): Obtendremos mayor rendimiento cuantos más árboles utilicemos (hasta la convergencia), sin llegar a causar _overfitting_, pero cuanto mayor sea este número mayor coste computacional tendremos. Típicamente toma valores entre $100$ (por defecto) y $500$.

- **`max_features`** (_features_ por split $m$): El valor por defecto cambia según estemos en un problema de clasificación ($m = \sqrt{d}$) o de regresión ($m=d$). Valores menores de este parámetro reducirán la correlación, pero pueden aumentar el sesgo.

- **`bootstrap`** (usar _bootstrap sampling_): Por defecto toma valor `True`. En caso de cambiar a `False` utilizaría siempre el _dataset_ completo.

- **`oob_score`** (calcular error OOB): Por defecto es `False`. El cálculo del error OOB solo está disponible si `bootstrap=True`. 

Los siguientes parámetros nos permiten ajustar cómo se construyen los árboles individuales:

- **`max_depth`** (profundidad máxima de cada árbol). Por defecto es `None` (sin límite), con lo que los árboles crecen completamente. Sin introducimos valores menores tendremos menos _overfitting_, pero mayor sesgo.

- **`min_samples_split`** (mínimo de muestras para dividir un nodo): Por defecto toma el valor $2$. Con valores mayores tendremos árboles más simples. 

- **`min_samples_leaf`** (mínimo de muestras en una hoja). Por defecto toma el valor $1$. Solo se dividirá un nodo si en cada una de las hojas resultantes hay al menos este número de muestras. Con valores más altos tendremos una regularización más fuerte que producirá modelos más suaves, especialmente en el caso de regresión.

- **`max_leaf_nodes`** (máximo número de hojas). Por defecto toma valor `None` (sin límite). Nos permite controlar la complejidad del árbol. 

### _Feature Importance_

Una vez entrenado el modelo de _Random Forest_, podemos calcular la importancia de cada característica. Esta información nos será de utilidad para:

- **Interpretabilidad del modelo**: Nos permite identificar qué características son más relevantes para las predicciones. Esta información nos ayudará a explicar el comportamiento del modelo y validar que se estén utilizando características con sentido. 

- **Selección de características**: Nos permite simplificar el modelo eliminando características no relevantes. Esto reducirá el coste computacional, puede mejorar la generalización, eliminando ruido, y facilitará el despliegue del modelo al requerir menos datos de entrada. 

Podemos calcular la importancia de cada _feature_ de dos formas diferentes.

#### Gini Importance

Esta primera forma de medir la importancia, también conocida como _Mean Decrease in Impurity_ (MDI) está basada en la mejora promedio de la impureza cuando se usa esa feature:

$$\text{Importance}(x_j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} \Delta I_t \cdot \mathbb{1}(v_t = j)$$

Donde $\Delta I_t$ es la reducción en impureza en el nodo $t$ y $v_t$ es la _feature_ usada en el _split_ del nodo $t$. Esta medida suma toda la reducción de impureza de los nodos en los que se utiliza $j$ como _feature_ para dividir, y promedia sobre todos los árboles $B$. 

A continuación podemos ver cómo obtener este valor de impureza con sklearn:

```python
# Calcular importancia
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_

# Ordena por importancia y muestra un ranking de características
indices = np.argsort(importances)[::-1]

print("Ranking de caracteristicas:")
for i in range(X.shape[1]):
    print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
```

Esta forma de obtener la importancia es muy rápida de calcular, ya que se obtiene a partir de las medidas de impureza obtenidas a partir del entrenamiento. 

Sin embargo, tiene un sesgo hacia características de alta cardinalidad, es decir, características con muchos valores únicos, como por ejemplo un ID que es único para cada ejemplo de entrada, ya que dividen artificialmente el espacio, pero no generalizan. 

También es problemático con características correlacionadas, ya que la importancia se diluye entre ellas, y no se sabe cuál es realmente importante.


#### Permutation Importance

Esta segunda forma mide la degradación del rendimiento cuando se permutan los valores de una _feature_. 

Para ello, primer evaluaremos en modelo con un _dataset_ $\mathcal{D}$, obteniendo el error _baseline_. Una vez hecho esto, en el conjunto $\mathcal{D}$ permutamos los valores de la _feature_ (columna) $j$, corrompiendo así los datos para dicha _feature_, y calculamos el error del modelo con la _feature_ corrupta. Repetiremos la permutación varias veces y promediamos el error obtenido al corromper $j$. Con esto, podemos calcular la importancia de permutación de $j$ calculando la diferencia entre el error _baseline_ y el error promedio obtenido al corromper $j$.

A continuación podemos ver la implementación en sklearn:


```python
from sklearn.inspection import permutation_importance

# Calcular permutation importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10)

importances = result.importances_mean

# Mostramos ranking de caracteristicas
for i in np.argsort(importances)[::-1]:
    print(f"Feature {i}: {importances[i]:.4f} +/- {result.importances_std[i]:.4f}")
```

Este tipo de cálculo de la importancia es más fiable cuando contamos con características correlacionadas, y no existe sesgo por la cardinalidad, pero tiene un mayor coste computacional y requiere un conjunto de _test_ o validación.



### Out-of-Bag (OOB) Error en Random Forest

Como Random Forest usa _bootstrap_, hereda la estimación del error OOB. A continuación podemos ver cómo obtener este valor con sklearn:

```python
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True  # Activar OOB
)

rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"Test Score: {rf.score(X_test, y_test):.4f}")
```

El _OOB score_ es típicamente una buena aproximación del error de generalización.



### Extra Trees (Extremely Randomized Trees)

Los _Extra Trees_ son una variante de _Random Forest_ en la que se introduce aún más aleatoriedad. Si bien en _Random Forest_ en cada nodo de los árboles se elige el mejor _split_ entre las _features_ seleccionadas, en _Extra Trees_ se elige un _split_ de forma aleatoria para cada _feature_, y nos quedamos con aquella que proporciona una mayor ganancia.   

En la siguiente tabla se resumen las principales diferencias estre estos modelos:

| Aspecto | Random Forest | Extra Trees |
|---------|---------------|-------------|
| **Sampling** | _Bootstrap_ (por defecto con reemplazo) | Todo el _dataset_ (por defecto sin _bootstrap_) |
| **Splits** | Mejor _split_ entre $m$ features | _Split_ **aleatorio** entre $m$ features |
| **Varianza** | Bajo | Aún menor |
| **Sesgo** | Bajo | Ligeramente mayor |
| **Velocidad** | Más lento | Más rápido (_splits_ aleatorios) |

En sklearn tenemos las clase [ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) y [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html). A continuación vemos un ejemplo de implementación:

```python
from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees
et = ExtraTreesClassifier(
    n_estimators=100,
    max_features='sqrt',
    bootstrap=False  # No usa bootstrap (default)
)

et.fit(X_train, y_train)
```

Será conveniente utilizar _Extra Trees_ cuando busquemos reducir aún más la varianza, aunque sea con un ligero aumento del sesgo, y cuando necesitemos una mayor velocidad de entrenamiento. 



### Consideraciones finales

Random Forest es **uno de los mejores algoritmos _out-of-the-box_**. Es decir, es capaz de funcionar correctamente sin necesidad de configuración o ajustes adicionales, funcionando bien con los hiperparámetros por defecto.

Es un método **robusto**, que puede manejar bien datos con ruido, _outliers_ y _features_ irrelevantes. Puede manejar además valores faltantes (_missing values_). Mediante la evaluación OOB podemos validar el modelo sin requerir un conjunto de validación por separado. Una limitación que encontramos es que el modelo no extrapola, es decir, no puede predecir fuera del rango de entrenamiento.

Es también **versátil**, dando buenos resultados tanto en problemas de clasificación como de regresión. Permite también tener _features_ mixtas, numéricas y categóricas, y permite capturar relaciones complejas (no lineales) de los datos de entrada. También es invariante a la escala de las _features_, por lo que no necesita normalización. Sin embargo, con _datasets_ pequeños puede haber algo de _overfitting_, y puede tener un peor rendimiento cuando tenemos _features_ muy correlacionadas.

En cuanto a la **interpretabilidad**, tenemos la posibilidad de obtener una medida de la importancia de características, pudiendo destacar las más relevantes. Sin embargo, es menos interpretable que un árbol individual.

Respecto al **coste**, el entrenamiento es muy rápido y es paralelizable, pero cuando tenemos muchos árboles la predicción puede ser más lenta. Además, tiene un alto coste espacial en memoria, ya que debe almacenar muchos árboles profundos. El tamaño de los modelos puede ser grande. 

En resumen, _Random Forest_ es robusto y versátil, siendo el método de _Bagging_ más popular y uno de los algoritmos más importantes en _Machine Learning_. Combina:
- _Bootstrap sampling_ (como _Bagging_)
- _Random feature selection_
- Árboles profundos sin poda
- Promediado/votación

Normalmente _Random Forest_ funcionará mejor que _Bagging_ porque existe menos correlación entre los árboles y esta mayor diversidad produce una reducción de la varianza. Con _Random Forest_ podemos obtener de forma rápida un _baseline_ robusto para tratar datos tabulares. 

Sin embargo, en algunos casos deberíamos considerar otros modelos. Si necesitamos mayor rendimiento podríamos considerar métodos como XGBoost o LightGBM. Si buscamos una alta interpretabilidad será más adecuado utilizar árboles individuales o modelos lineales. Si tenemos datos de tipo imagen o texto, será más adecuado utilizar redes neuronales profundas. En caso de tener _datasets_ pequeños, con solo decenas o unos pocos cientos de ejemplos, deberíamos considerar modelos más simples. 

Hasta ahora hemos visto métodos que combinan modelos entrenados **independientemente** (en paralelo), ya sea en los mismos datos (_Voting_, _Stacking_) o en diferentes muestras (_Bagging_). La siguiente pregunta es: **¿y si los modelos se entrenan secuencialmente, aprendiendo cada uno de los errores del anterior?**. En esto se basarán los métodos de **Boosting**.



