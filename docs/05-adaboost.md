# Boosting

_Boosting_ es un meta-algoritmo de aprendizaje automático, que busca **reducir principalmente el sesgo**, aunque también algo la varianza. 

A diferencia de los métodos de _ensemble_ estudiados hasta el momento (_Voting_, _Stacking_, _Bagging_), en los que contábamos con modelos independientes que se entrenaban en paralelo, _Boosting_ entrena los modelos secuencialmente, de forma que cada nuevo modelo se centra en corregir los errores cometidos por el _ensemble_ actual.


## Combinación de clasificadores

La idea que buscamos con _Boosting_ es combinar una serie de **clasificadores débiles** para construir un **clasificador fuerte**. Los definimos de la siguiente forma:

- **Clasificador débil (weak learner):** Se trata de clasificadores simples que funcionan mejor que la clasificación aleatoria. Es decir, su _accuracy_ debe ser mayor que 50% (mejor que lanzar una moneda). Formalmente, para clasificación binaria, un clasificador débil debe cumplir que su tasa de error sea
$\epsilon < \frac{1}{2}$.

- **Clasificador fuerte (strong learner):** Se trata de un clasificador con alta precisión, capaz de aproximarse arbitrariamente bien a la función objetivo. Su tasa de error puede ser tan pequeña como se desee con suficientes datos y suficiente capacidad del modelo.

### Clasificadores débiles

Algunos de los ejemplos de clasificadores débiles típicos son los siguientes:

- **Decision stumps**: Árboles de decisión con una única división (profundidad $1$). Equivale a una regla simple, como por ejemplo, "si $x_1 > 5$ entonces clase positiva, si no, clase negativa".

- **Árboles poco profundos**: Árboles con profundidad máxima $2$ o $3$.

- **Clasificadores lineales** en problemas no lineales.

### Clasificadores fuertes

A continuación mostramos algunos ejemplos de clasificadores fuertes típicos:

- Redes neuronales profundas
- Árboles de decisión muy profundos
- SVMs con _kernels_ complejos
- Random Forests
- Gradient Boosting

## Base fundacional

En 1988 y 1989 Kearns y Valiant [@kearns1988cryptographic;@kearns1989cryptographic] plantearon la pregunta que constituye la fundación teórica del _Boosting_: **"¿Puede un conjunto de clasificadores débiles crear un único clasificador fuerte?"**. 

En 1990 Schapire [@schapire1990strength] demuestra que esto es posible, lo cual supone uno de los resultados más importantes en teoría de aprendizaje automático. Lo que nos dice el teorema que plantea es que si existe un clasificador débil que puede lograr error menor que $1/2 - \gamma$ (donde $\gamma > 0$), entonces existe un algoritmo de _Boosting_ que puede combinarlo para lograr un error arbitrariamente pequeño en el conjunto de entrenamiento.

Este teorema es importante porque:

1. Demuestra que la "debilidad" es suficiente para el aprendizaje.
2. Proporciona una garantía matemática sobre los métodos de _Boosting_.
3. Nos muestra cómo construir el clasificador fuerte.

En este punto, nos podemos plantear la pregunta "¿Por qué usar clasificadores débiles?". Aunque pueda parecer contraintuitivo usar modelos "débiles", hay varios motivos para hacerlo:

1. **Prevención del _overfitting_:**  Los modelos débiles tienen baja varianza, son menos propensos a aprender el ruido, y la combinación de modelos reduce el _overfitting_.

2. **Eficiencia computacional:** Modelos sencillos como los _decision stumps_ son extremadamente rápidos. Podemos entrenar cientos o miles de ellos de forma eficiente.

3. **Interpretabilidad:** Cada modelo débil es fácil de entender, y la combinación de modelos mantiene cierta trazabilidad.

4. **Teoría sólida:** Existen garantías matemáticas de convergencia y el error de generalización puede ser acotado.


Para que un clasificador débil sea útil en un _ensemble_ debe cumplir:

1. **Mejor que el azar:** Su error debe ser $\epsilon < 0.5$.
2. **Diversidad:** Debe cometer errores diferentes a los que cometen otros clasificadores.
3. **Eficiencia:** Debe ser rápido de entrenar.
4. **Estabilidad:** No debe ser extremadamente sensible a pequeños cambios en los datos.

Respecto a la relación con el problema **sesgo-varianza**, los clasificadores débiles tendrán un alto sesgo, pero baja varianza (deben ser estables ante cambios en los datos). Por el contrario, los clasificadores fuertes tendrán bajo sesgo, ya que pueden aprender patrones complejos, pero una alta varianza, ya que serán más sensibles a los datos de entrenamiento.

Un _ensemble_ de clasificadores débiles reducirá el sesgo, al combinar de forma secuencial los clasificadores, manteniendo la baja varianza de los modelos individuales, obteniendo así lo mejor de cada tipo de clasificador. 

## Muestreo y votos ponderados

Como hemos comentado, una de las principales diferencias de _Boosting_ con los métodos de _ensemble_ vistos anteriormente es que en lugar de entrenar los modelos en paralelo, _Boosting_ realiza el entrenamiento secuencialmente, buscando que los nuevos clasificadores débiles corrijan los principales errores del _ensemble_ actual. Para ello introduce ponderación a dos niveles: **en el muestreo de ejemplos de entrenamiento** y en **la importancia de cada clasificador**.

Si recordamos las características de los métodos de _Bagging_, tenemos:

- _Bagging_ realiza un muestreo aleatorio de los ejemplos de entrada para entrenar cada clasificador, pero todos estos ejemplos de entrada reciben el mismo peso.
- _Bagging_ da la misma importancia a todos los clasificadores. El voto de cada clasificador vale lo mismo.

A diferencia de esto, _Boosting_ introduce:

- **Muestreo ponderado**: No se da el mismo peso a todos los ejemplos de entrenamiento. El entrenamiento se concentrará en los ejemplos más difíciles. Intuitivamente, podríamos considerar que aquellos ejemplos cercanos a la frontera de decisión son más difíciles de clasificar, y deberían por lo tanto recibir pesos más altos. Podemos establecer una relación entre esto y los vectores de soporte en SVM, ya que en ambos casos buscamos basarnos en los ejemplos más difíciles de clasificar para obtener el clasificador. 

- **Votos ponderados**: Se da diferente peso a los diferentes clasificadores, que al combinarlos se obtiene un "voto ponderado". Esto, junto a la estrategia de muestreo anterior, ayudará a producir un clasificador más fuerte.

Considerando que contamos con un _dataset_ $\mathcal{D}$ con $N$ ejemplos de entrenamiento $(\mathbf{x}_i, y_i)$, con $i = 1, 2, \ldots, N$, y $T$ clasificadores débiles $h_1, h_2, \ldots, h_T$, definimos:

- $w_i^{(t)}$: Peso del ejemplo de entrenamiento $(\mathbf{x}_i, y_i)$ para entrenar el clasificador $h_t$.

- $\alpha_t$: Peso del clasificador $h_t$ en la votación del _ensemble_.

De esta forma, para el caso de clasificación binaria con $y_i \in \{-1, 1\}$, podemos definir el _ensemble_ como:

$$
F(\mathbf{x}) = \sum_{t=1}^T \alpha_t h_t(\mathbf{x})
$$

Es decir, la predicción de cada clasificador $h_t$ estará ponderada por el peso $\alpha_t$, pudiendo así dar mayor peso a los clasificadores con menor error. La predicción final vendría dada por el signo de la función anterior:

$$
H(\mathbf{x}) = \text{signo} \left( F(\mathbf{x}) \right)
$$

## AdaBoost (_Adaptive Boosting_)

AdaBoost [@freund1997decision] fue desarrollado por Freund y Schapire en 1997, y constituye el primer algoritmo de _Boosting_ exitoso.

La idea central de este algoritmo es:

1. Entrenar un **clasificador débil**.
2. **Aumentar el peso** de los ejemplos mal clasificados.
3. Entrenar el siguiente clasificador con los **datos ponderados**.
4. **Repetir** el proceso.
5. **Combinar** todos los clasificadores ajustando sus pesos según su _accuracy_.


Considerando el caso de clasificación binaria, donde $y_i \in \{ -1, 1\}$, el **algoritmo AdaBoost** buscará minimizar la **pérdida exponencial**, que se define como:

$$
L(y, F) = \sum_{i=1}^N e^{-y_i F(\mathbf{x}_i)} 
$$

Podemos observar que esta función de pérdida penaliza aquellos ejemplos en los que $y_i$ y $F(\mathbf{x}_i)$ tengan distinto signo, es decir, aquellos que están mal clasificados. Además, la pérdida no crece linealmente con el error, sino de forma exponencial. Un ejemplo muy mal clasificado contribuirá a la pérdida de forma desproporcionada.

A nivel general, el algoritmo AdaBoost realizará los siguientes pasos:

$$
\begin{align*}
& \text{Entrada: } \text{Conjunto de entrenamiento } \mathcal{D}= \{(\mathbf{x}_i, y_i)\}_{i=1}^N \text{ con } y_i \in \{-1, +1\} \\
& w_i^{(1)} \leftarrow \frac{1}{N} \quad \forall i \in 1, 2, \ldots, N \quad \text{(Inicializa todos los ejemplos con peso uniforme)} \\
& \text{Para  } t = 1, \ldots, T \\
& \quad h_t \leftarrow \text{Entrenar un clasificador débil con los pesos } \mathbf{w}^{(t)} \\
& \quad \alpha_t \leftarrow \text{Calcular el peso del clasificador } h_t \\
& \quad w_i^{(t+1)} \leftarrow \text{Actualiza los pesos de los ejemplos para el siguiente clasificador } \forall i \\
& \text{Devuelve: } H(\mathbf{x}) = \text{signo} \left( F(\mathbf{x}) \right) = \text{signo} \left( \sum_{t=1}^T \alpha_t h_t(\mathbf{x}) \right)
\end{align*}
$$

El algoritmo construye el _ensemble_ de forma voraz, iteración a iteración. En cada iteración $t$ añade al _ensemble_ un nuevo clasificador $h_t$. Para entrenarlo presta especial atención a los ejemplos de entrenamiento que tengan un mayor peso $w_i^{(t)}$ en dicha iteración.

Vamos a continuación a detallar cada uno de los pasos de este algoritmo.

> **Nota**: La derivación que presentamos aquí, basada en la minimización de la pérdida exponencial, corresponde a la reinterpretación estadística que realizan Friedman, Hastie y Tibshirani en el año 2000 [@friedman2000special]. La formulación original de Freund y Schapire [@freund1997decision] partía de un marco de teoría de juegos y aprendizaje PAC (_Probably Approximately Correct_), llegando al mismo algoritmo desde una perspectiva diferente.

### Actualización de los pesos

Como hemos comentado, al final de cada iteración deberán ajustarse los pesos de cada ejemplo de entrenamiento, de forma que se le dé mayor peso a los ejemplos más difíciles (los peor clasificados hasta el momento), para que el siguiente clasificador se centre en mejorar su clasificación. 

Siguiendo el proceso del algoritmo anterior, podemos considerar que hasta la iteración $t-1$ tendremos un **clasificador fuerte** $F_{t-1}$, que se obtiene como:

$$
F_{t-1}(\mathbf{x}) = \alpha_1 h_1(\mathbf{x}) + \alpha_2 h_2(\mathbf{x}) + \ldots + \alpha_{t-1} h_{t-1}(\mathbf{x})
$$

En la iteración $t$ buscaremos mejorarlo añadiendo un **nuevo clasificador débil** $h_t$:

$$
F_{t}(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \alpha_t h_t(\mathbf{x}) 
$$

Buscamos que el nuevo clasificador débil minimice la pérdida $L_t(y, F_t)$:

$$
\begin{align*}
L_t(y, F_t) &= \sum_{i=1}^N e^{-y_i F_t(\mathbf{x}_i)} = \\
&=  \sum_{i=1}^N e^{-y_i (F_{t-1}(\mathbf{x}_i) + \alpha_t h_t(\mathbf{x}_i))} = \\
&= \sum_{i=1}^N e^{-y_i F_{t-1}(\mathbf{x}_i)} e^{ - \alpha_t y_i h_t(\mathbf{x}_i)}
\end{align*}
$$

El primer factor $e^{-y_i F_{t-1}(\mathbf{x}_i)}$ no depende de lo que se está optimizando en este paso ($\alpha_t$ y $h_t$), sino que actúa como una constante multiplicativa para cada ejemplo de entrada $i$. De esta forma, consideramos esa constante como un peso $w_i^{(t)}$ que se define como:

$$
w_i^{(t)} = e^{-y_i F_{t-1}(\mathbf{x}_i)}
$$

Este peso será mayor cuanto peor esté clasificado el ejemplo $i$ en el _ensemble_ $t-1$.

Reemplazándolo en la función de pérdida anterior tendríamos:

$$
\begin{align*}
L_t(y, F_t) &=  \sum_{i=1}^N w_i^{(t)} e^{ - \alpha_t y_i h_t(\mathbf{x}_i)}
\end{align*}
$$

Es decir, en la función de pérdida se le dará mayor peso a los ejemplos peor clasificados por el _ensemble_ anterior $t-1$.

Vamos a ver ahora la forma de actualizar los pesos. El peso para la siguiente iteración $t+1$ sería:

$$
w_i^{(t+1)} = e^{-y_i F_{t}(\mathbf{x}_i)} =  e^{-y_i F_{t-1}(\mathbf{x}_i)} e^{ -  \alpha_t y_i h_t(\mathbf{x}_i)} = w_i^{(t)} e^{ - \alpha_t y_i h_t(\mathbf{x}_i)}
$$

Por lo tanto, tenemos la forma en la que se actualizarán los pesos tras cada iteración:

$$ \quad w_i^{(t+1)} = w_i^{(t)} e^{-\alpha_t y_i h_t(\mathbf{x}_i)}  \quad \forall i 
$$

A partir de esta forma de actualizar los pesos, podemos observar que los ejemplos mal clasificados tendrán signo positivo en la exponencial, por lo que aumentará su peso, mientras que los bien clasificados tendrán signo negativo y en ese caso disminuirá su peso.

Además, deberemos normalizar los pesos para posteriormente poder calcular correctamente el error del clasificador a partir de ellos:

$$
w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^N w_j^{(t+1)}}  \quad \forall i 
$$


> **Relación con la derivada**: Si tratamos $F(\mathbf{x}_i)$ como una variable independiente para cada ejemplo de entrada $\mathbf{x}_i$, y evaluamos cuánto afecta a la pérdida total calculando la derivada parcial, tenemos:
>
> $$ \frac{\partial L_t(y, F_{t})}{\partial F_{t}(\mathbf{x}_i)} = -y_i e^{-y_i F_{t}} $$
>
> Si atendemos al gradiente negativo, podemos identificar dos partes:
>
> - $y_i$ nos indica la dirección, es decir, hacia dónde hay que mover $F(\mathbf{x}_i)$ para reducir la pérdida.
> - $e^{-y_i F_{t}}$ nos indicará la magnitud, lo cual nos da una medida de cuánto importa el ejemplo $\mathbf{x}_i$ en este momento.
> Por este motivo, definiremos este segundo término como el peso que le daremos a cada ejemplo de entrada. 



### Entrenar un clasificador débil

Buscamos minimizar la función de pérdida exponencial $L(y, F_t)$, que a partir de la definición de los pesos expuesta en el apartado anterior, podemos escribir como:

$$
L(y, F_t)  = \sum_{i=1}^N w_i^{(t)} e^{- \alpha_t y_i h_t(\mathbf{x}_i)}
$$

Si en la función separamos los ejemplos que se clasifican correcta e incorrectamente, tendríamos:

$$
L(y, F_t) = \sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)} e^{-\alpha_t } + \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} e^{\alpha_t }
$$

De forma equivalente, podría expresarse como:

$$
L(y, F_t) = \sum_{i=1}^N w_i^{(t)} e^{-\alpha_t } + \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} (e^{\alpha_t } - e^{-\alpha_t })
$$

En la ecuación anterior, vemos que sólo el segundo término depende de $h_t$. Por lo tanto, el clasificador que minimizará el error será aquel que minimice $\sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)}$, es decir, la suma de los pesos de los ejemplos de entrada mal clasificados. Tendremos que entrenar los modelos base dando a cada ejemplo de entrada $(\mathbf{x}_i, y_i)$ su correspondiente peso $w_i^{(t)}$.

A nivel práctico, en sklearn, esto lo podremos hacer mediante el parámetro `sample_weight` de la función `fit`. Cada tipo de modelo tratará estos pesos de forma distinta. Por ejemplo, en el caso de Árboles de Decisión se tendrán en cuenta los pesos en el cálculo de la impureza de cada nodo, mientras que otros modelos como Regresión Logística o SVM incorporarán los pesos en su propia función de pérdida, haciendo que cada ejemplo contribuya a la pérdida con un factor proporcional a $w_i^{(t)}$. Esto tiene la implicación de que solo podremos usar como modelos base en AdaBoost aquellos estimadores que soporten ese parámetro. 

### Calcular el peso del clasificador

Deberemos buscar el valor del peso de cada clasificador que **minimice el error del _ensemble_**. Para hacer esto, en primer lugar derivaremos la función de pérdida respecto al peso $\alpha_t$ de cada clasificador:

$$
\begin{align*}
\frac{\partial L(y, F_t)}{\partial \alpha_t} &= \frac{\partial \left( \sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)} e^{-\alpha_t } + \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} e^{\alpha_t } \right)}{\partial \alpha_t} = 
\\
&= -\sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)} e^{-\alpha_t } + \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} e^{\alpha_t }
\end{align*}
$$

Teniendo en cuenta que la función de error es una función convexa, la igualaremos a $0$ para buscar el punto en la que es mínima:

$$
-\sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)} e^{-\alpha_t } + \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} e^{\alpha_t } = 0
$$

$$
\Downarrow
$$

$$
\sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)} e^{-\alpha_t } = \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} e^{\alpha_t }
$$

$$
\Downarrow
$$

$$
\frac{\sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)}}{\sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)}} = \frac{e^{\alpha_t }}{e^{-\alpha_t }} = e^{2\alpha_t}
$$

Por lo tanto, tenemos:

$$
\alpha_t = \frac{1}{2} \ln \frac{\sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)}}{\sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)}}
$$

Considerando que los pesos están normalizados, podemos definir el error $\epsilon_t$ del clasificador débil $h_t$, de la siguiente forma:

$$
\epsilon_t = \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)}
$$

Observamos que calculamos el error de $h_t$ como la suma de los pesos de los ejemplos mal clasificados por dicho clasificador débil. De esta forma, el clasificador tendrá error bajo cuando el peso total de los ejemplos mal clasificados sea bajo, y los ejemplos de mayor peso hayan sido correctamente clasificados. 

Teniendo esta definición en cuenta, y considerando que los pesos están normalizados y que por lo tanto $\sum_{i=1}^N w_i^{(t)} = 1$, tenemos:

$$
\begin{align*}
\alpha_t &= \frac{1}{2} \ln \frac{\sum_{y_i = h_t(\mathbf{x}_i)} w_i^{(t)}}{\sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)}} = \\
&= \frac{1}{2} \ln \frac{\sum_{i=1}^N w_i^{(t)} - \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)}}{\sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)}} = \\

&= \frac{1}{2} \ln \frac{1-\epsilon_t}{\epsilon_t}
\end{align*}
$$

Con esto podemos observar que $\alpha_t$ será mayor cuanto menor error $\epsilon_t$ tenga el clasificador. Lo peor que pueda ocurrir es que $\epsilon_t = 0.5$, ya que en ese caso el clasificador equivale a lanzar una moneda al aire, y en tal caso tendremos un peso $\alpha_t = 0$. Podemos observar también que si $\epsilon_t > 0.5$ (peor que el azar), el peso pasará a ser negativo, es decir, se invierte el clasificador para que así pase a ser algo mejor que el azar.  

### Algoritmo detallado

Con todo lo anterior, podemos escribir de forma completa el algoritmo AdaBoost detallando en cada paso la forma en la que se calculan los errores y los pesos:

$$
\begin{align*}
& \text{Entrada: } \text{Conjunto de entrenamiento } \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N \text{ con } y_i \in \{-1, +1\} \\
& w_i^{(1)} \leftarrow \frac{1}{N} \quad \forall i \in 1, 2, \ldots, N \quad \text{(Inicializa todos los ejemplos con peso uniforme)} \\
& \text{Para  } t = 1, \ldots, T \\
& \quad h_t \leftarrow \text{Entrenar un clasificador débil que minimice } \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} \\
& \quad \epsilon_t \leftarrow \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} \quad \text{(Calcular el error del clasificador débil)} \\
& \quad \alpha_t \leftarrow \frac{1}{2} \ln \left( \frac{1-\epsilon_t}{\epsilon_t} \right) \quad \text{(Calcular el peso del clasificador)}  \\
& \quad w_i^{(t+1)} \leftarrow w_i^{(t)} e^{-\alpha_t y_i h_t(\mathbf{x}_i)}  \quad \forall i \quad \text{(Actualiza los pesos de los ejemplos para el siguiente clasificador)} \\
& \quad w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_{j=1}^N w_j^{(t+1)}}  \quad \forall i \quad \text{(Normaliza los nuevos pesos)} \\
& \text{Devuelve: } H(\mathbf{x}) = \text{signo} \left( F(\mathbf{x}) \right) = \text{signo} \left( \sum_{t=1}^T \alpha_t h_t(\mathbf{x}) \right)
\end{align*}
$$

## Propiedades de AdaBoost

### Adaptatividad

Una propiedad importante de AdaBoost es su adaptatividad, lo cual le da el nombre (**Ada**ptative **Boost**ing). El término "adaptativo" tiene además dos significados:

- **Adaptación a los datos de entrada**: Los pesos $w_i$ se ajustan en función de los ejemplos que se clasifican mal. El algoritmo es capaz de identificar estos ejemplos a partir de los errores observados, y se concentra en ellos.

- **Adaptación a la calidad de los clasificadores base**: El peso $\alpha_t$ que recibe cada clasificador no se fija de antemano, sino que el algoritmo lo calcula de forma automática a partir del error $\epsilon_t$. A diferencia de propuestas previas de métodos de _Boosting_, no es necesario conocer de antemano la calidad de los clasificadores base ni su error, sino que se adaptará automáticamente a partir del error observado.

### Cota del error de entrenamiento

AdaBoost tiene una garantía teórica sobre el error de entrenamiento [@schapire2012boosting] tras $T$ iteraciones:

$$\frac{1}{N} \sum_{i=1}^N 1(H(\mathbf{x}_i) \neq y_i) \leq \prod_{t=1}^{T} 2 \sqrt{\epsilon_t(1-\epsilon_t)} \leq \exp\left(-2\sum_{t=1}^{T}\gamma_t^2\right)$$

Donde $\gamma_t = \frac{1}{2} - \epsilon_t$ es el margen o ventaja del clasificador débil $h_t$ sobre el azar. Podemos ver que si cada clasificador supera al azar, aunque sea por poco ($\gamma_t > 0$), el error de entrenamiento decrece.

Podemos simplificar la cota anterior, considerando que todos los clasificadores tienen al menos una ventaja mínima uniforme $\gamma_t \geq \gamma \gt 0$, de la siguiente forma:

$$\text{Training Error} \leq \exp\left(-2 T \gamma^2\right)$$

Aquí podemos ver más claramente la implicación de esta cota, y es que el error de entrenamiento **decrece exponencialmente con el número de iteraciones $T$**. Basta que cada clasificador sea ligeramente mejor que el azar para que el error de entrenamiento se reduzca a cero.






## Variantes de AdaBoost

El algoritmo AdaBoost original [@freund1997decision] está diseñado para clasificación binaria, pero desde su publicación se han propuesto numerosas variantes que amplían su aplicabilidad a clasificación multiclase, regresión, y escenarios con distintos requisitos de robustez. 

Encontramos diferentes variantes para clasificación binaria, como Real AdaBoost [@schapire1999improved], Gentle AdaBoost y LogitAdaBoost [@friedman2000special]. También tenemos extensiones para clasificación multiclase, como AdaBoost.M1 / AdaBoost.M2 [@freund1996experiments] y SAMME / SAMME.R [@zhu2009multi]. Por otro lado, también tenemos AdaBoost.R2 [@drucker1997improving], una adaptación al caso continuo enfocada a problemas de regresión.

Vamos a centrarnos en las más relevantes: SAMME para problemas de clasificación multiclase y AdaBoost.R2 para regresión. 


### Clasificación multiclase con SAMME

SAMME (_Stagewise Additive Modeling using a Multi-class Exponential loss_) [@zhu2009multi] es la generalización más comúnmente utilizada de AdaBoost al caso multiclase, y cuenta con una sólida base teórica. 

El problema fundamental al pasar a clasificación multiclase es que la condición de debilidad $\epsilon_t < 0.5$ es demasiado estricta cuando el número de clases aumenta. El azar uniforme entre $K$ clases corresponde a un error de $1 - \frac{1}{K}$, que para $K \geq 3$ supera $0.5$. Exigir $\epsilon_t < 0.5$ equivale entonces a pedir que el clasificador base supere al azar binario en un problema que no lo es, lo que resulta cada vez más difícil a medida que $K$ crece.

Si consideramos el caso del azar para cada $K$ clases, tendríamos $\epsilon_t = 1 - \frac{1}{K}$. En tal caso, si no introducimos ninguna corrección en el cálculo de los pesos quedaría lo siguiente:

$$
\begin{align*}
\alpha_t &= \ln\frac{1 - \epsilon_t}{\epsilon_t} = \ln\frac{1 - 1 + \frac{1}{K}}{1-\frac{1}{K}} = \ln\frac{\frac{1}{K}}{1-\frac{1}{K}} =\\
& = \ln\frac{1}{K-1} = -\ln (K-1)
\end{align*}
$$

Para un número de clases $K > 2$ nos estará dando un peso negativo del clasificador, cuando realmente debería darnos un peso $0$, ya que esta situación corresponde al azar. Por este motivo, la principal modificación clave que introduce SAMME está en modificar la forma de calcular el peso de cada clasificador como se muestra a continuación:

$$\alpha_t = \ln\frac{1 - \epsilon_t}{\epsilon_t} + \ln(K - 1)$$

El término adicional $\ln(K-1)$ tiene dos consecuencias importantes:

- En primer lugar, la condición de debilidad pasa a ser $\epsilon_t < 1 - \frac{1}{K}$, que corresponde a superar al azar uniforme entre $K$ clases, lo cual es la condición teóricamente correcta. Cuando un clasificador funcione igual que el azar, su peso será $0$.

- En segundo lugar, para $K = 2$ se tiene $\ln(K-1) = \ln 1 = 0$ y el algoritmo producirá las mismas predicciones que AdaBoost binario, confirmando que es una generalización consistente.

El algoritmo completo de entrenamiento es el siguiente:

$$
\begin{align*}
& \text{Entrada: } \text{Conjunto de entrenamiento } \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N \text{ con } y_i \in \{1, \ldots, K\} \\
& w_i^{(1)} \leftarrow \frac{1}{N} \quad \forall i \in 1, 2, \ldots, N \quad \text{(Inicializa todos los ejemplos con peso uniforme)} \\
& \text{Para  } t = 1, \ldots, T \\
& \quad h_t \leftarrow \text{Entrenar un clasificador débil que minimice } \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} \\
& \quad \epsilon_t \leftarrow \sum_{y_i \neq h_t(\mathbf{x}_i)} w_i^{(t)} \quad \text{(Calcular el error del clasificador débil)} \\
& \quad \text{Si } \epsilon_t \geq 1 - \frac{1}{K} \text{: detener} \\
& \quad \alpha_t \leftarrow  \ln \left( \frac{1-\epsilon_t}{\epsilon_t} \right) + \ln(K-1) \quad \text{(Calcular el peso del clasificador)}  \\
& \quad w_i^{(t+1)} \leftarrow w_i^{(t)} e^{\alpha_t \cdot \mathbb{1}(h_t(\mathbf{x}_i) \neq y_i)}  \quad \forall i \quad \text{(Actualiza los pesos de los ejemplos para el siguiente clasificador)} \\
& \quad w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_{j=1}^N w_j^{(t+1)}}  \quad \forall i \quad \text{(Normaliza los nuevos pesos)} \\
& \text{Devuelve: } H(\mathbf{x}) = \arg \max_k  \sum_{t: h_t(\mathbf{x})=k} \alpha_t 
\end{align*}
$$

La predicción final asigna a cada clase la suma de pesos de los clasificadores que la predicen, eligiendo la clase con mayor suma acumulada.


### AdaBoost.R2 para regresión

AdaBoost.R2 [@drucker1997improving] adapta AdaBoost al problema de regresión. La dificultad principal es que en regresión no existe un concepto de "azar" tan claro como en clasificación, y el error no puede definirse como una proporción de ejemplos mal clasificados.

Lo que plantea esta variante es normalizar el error de cada ejemplo al rango $[0,1]$, para así poder recuperar la condición de debilidad $\epsilon_t < 0.5$. Para cada ejemplo $i$ y cada iteración $t$, se define la **pérdida normalizada** de la siguiente forma:

$$L_i^{(t)} = \frac{|y_i - h_t(\mathbf{x}_i)|}{\max_j |y_j - h_t(\mathbf{x}_j)|}$$

Esta pérdida normalizada toma valores en $[0,1]$ y mide el error relativo al peor error cometido por el regresor base en esa iteración. El ejemplo con peor error tendrá $L_i^{(t)}=1$, y el que mejor se ajuste tendrá un valor cercano a $0$. Hay variantes que en lugar de utilizar error lineal utilizan error cuadrático o exponencial.  

A partir de la pérdida normalizada calculamos el **error ponderado del regresor base**:

$$\epsilon_t = \sum_{i=1}^{N} w_i^{(t)} L_i^{(t)}$$

Dado que la condición de debilidad es $\epsilon_t < 0.5$, si obtenemos $\epsilon_t \geq 0.5$ el algoritmo se detendrá. Podemos interpretar este error ponderado como una esperanza. Es decir, $\epsilon_t$ sería el valor esperado del error normalizado bajo la distribución de pesos actual. 

Como pesos para los estimadores $h_t$, en regresión en lugar de utilizar $\alpha_t$ utilizaremos $\beta_t$, que se calcula de la siguiente forma:

$$
\beta_t = \frac{\epsilon_t}{1 - \epsilon_t}
$$

El valor de este peso estará dentro del rango $\beta_t \in (0,1)$ cuando se cumpla la condición $\epsilon_t < 0.5$. Tendríamos $\beta_t = 1$ en caso de tener un regresor muy malo, justo en el límite $\epsilon_t = 0.5$, y el valor será menor conforme mejor sea el regresor, aproximándose a $0$ en los mejores casos.

La actualización de los pesos se hará de la siguiente forma:

$$
w_i^{(t+1)} = w_i^{(t)} \beta_t^{1-L_i^{(t)}}
$$

Interpretando el exponente de la actualización anterior tenemos:

- Si el error del ejemplo $i$ es pequeño ($L_i^{(t)}$ es cercano a $0$), entonces el exponente será cercano a $1$ y el peso se multiplica por $\beta_t < 1$, lo cual hará que se reduzca.

- Si por el contrario el error del ejemplo $i$ es máximo ($L_i^{(t)}$ es $1$), entonces el exponente será  $0$ y el peso se multiplica por $\beta_t^0 = 1$, manteniendo su valor.

Al igual que en el caso de clasificación, esto hará que el peso de los ejemplos difíciles crezca, mientras que los fáciles perderán peso.

AdaBoost.R2 difiere considerablemente respecto a AdaBoost en la forma de obtener la predicción final. En lugar de una suma ponderada de predicciones, se obtendrá mediante una mediana ponderada. La elección de la mediana en lugar de la media se debe a que la mediana será más robusta frente a predicciones extremas de regresores de baja calidad.

A continuación se muestra el algoritmo completo:

$$
\begin{align*}
& \text{Entrada: } \text{Conjunto de entrenamiento } \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N \\
& w_i^{(1)} \leftarrow \frac{1}{N} \quad \forall i \in 1, 2, \ldots, N \quad \text{(Inicializa todos los ejemplos con peso uniforme)} \\
& \text{Para  } t = 1, \ldots, T \\
& \quad h_t \leftarrow \text{Entrenar regresor base con pesos } w_i^{(t)} \\
& \quad L_i^{(t)} \leftarrow \frac{|y_i - h_t(\mathbf{x}_i)|}{\max_j |y_j - h_t(\mathbf{x}_j)|} \quad \forall i \quad \text{(Pérdida normalizada)}  \\
& \quad \epsilon_t \leftarrow \sum_{i=1}^{N} w_i^{(t)}  L_i^{(t)} \quad \text{(Error ponderado del regresor base)}  \\
& \quad \beta_t \leftarrow \frac{\epsilon_t}{1 - \epsilon_t} \quad \text{(Calcula el peso del regresor)}  \\
& \quad w_i^{(t+1)} \leftarrow w_i^{(t)}  \beta_t^{1 - L_i^{(t)}} \quad \forall i \quad \text{(Actualización de pesos de los ejemplos)}  \\
& \quad w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_{j=1}^N w_j^{(t+1)}}  \quad \forall i \quad \text{(Normaliza los nuevos pesos)} \\
& \text{Devuelve: } H(\mathbf{x}) = \text{mediana ponderada de } \{h_t(\mathbf{x})\}_{t=1}^T \text{ con pesos } \ln\frac{1}{\beta_t} 
\end{align*}
$$

Observamos que la actualización de pesos actúa de forma inversa a la clasificación: los ejemplos con **error pequeño** ($L_i^{(t)} \approx 0$) reducen su peso, ya que $\beta_t^{1-L_i^{(t)}} \approx \beta_t < 1$, mientras que los ejemplos difíciles conservan o aumentan su importancia relativa.

La predicción final es la **mediana ponderada** de los regresores base, donde el peso de cada regresor es $\ln(1/\beta_t)$: los regresores con menor error ponderado $\epsilon_t$ tienen mayor $\ln(1/\beta_t)$ y por tanto contribuyen más.

## Implementación

En sklearn contamos con las implementaciones [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) y [AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) para problemas de clasificación y regresión, respectivamente.

En este caso, `AdaBoostClassifier` implementa el algoritmo SAMME, mientras que `AdaBoostRegressor` implementa AdaBoost.R2. 

A continuación podemos ver un ejemplo de código que utiliza AdaBoost para clasificación con _decision stumps_, lo cual es la opción por defecto.

```python
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Decision stump
    n_estimators=50         # Número de clasificadores débiles
)

ada.fit(X_train, y_train)

# Inspeccionar clasificadores y pesos
print("Pesos de clasificadores:", ada.estimator_weights_)
print("Errores de clasificadores:", ada.estimator_errors_)
```

De igual forma, a continuación incluimos un ejemplo de regresión:

```python
reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=200,
    loss='linear'   
)
reg.fit(X_train, y_train)
```

En este caso contamos con el parámetro `loss` que permite elegir cómo se calcula la pérdida normalizada $L_i^{(t)}$. Con `'linear'` se usa el error absoluto normalizado (el descrito en el algoritmo anterior), con `'square'` el error cuadrático normalizado, y con `'exponential'` una pérdida exponencial normalizada. La opción `'linear'` es la correspondiente al algoritmo AdaBoost.R2 original y suele ser la más robusta.




## Consideraciones finales

AdaBoost tiene varias ventajas. Es fácil de entender e implementar, puede funcionar con cualquier clasificador débil y tiene pocos hiperparámetros. No necesitamos conocer los errores $\epsilon_t$, el algoritmo se adapta automáticamente. Puede alcanzar buenos resultados con clasificadores muy simples.

Sin embargo, también encontramos una serie de desventajas. No es paralelizable como otros tipos de _ensembles_ y puede ser costoso si contamos con clasificadores débiles lentos. Debemos tener en cuenta también que no funciona con clasificadores débiles peores que el azar.

La principal desventaja que deberemos tener en cuenta es que es muy **sensible al ruido y a los _outliers_**. Hemos visto que al utilizar la función de pérdida exponencial, se da un peso desmesurado a ejemplos muy mal clasificados, dominando así la optimización. Esto hará que AdaBoost acabará dedicando casi toda su capacidad a intentar clasificar correctamente un ejemplo imposible, degradando el rendimiento en el resto. 

Pero si AdaBoost equivale a minimizar la pérdida exponencial, ¿qué ocurriría si **cambiamos esa función de pérdida**? Si utilizamos pérdida logística tendremos la variante LogitBoost que será más robusta frente al ruido al crecer linealmente para errores grandes. Si generalizamos para cualquier función de pérdida diferenciable obtendremos **Gradient Boosting**, que estudiaremos en la siguiente sesión.

En el caso de **regresión** encontramos que AdaBoost.R2 presenta dos debilidades estructurales:

- La primera es que la normalización por el máximo error hace el algoritmo muy sensible a **outliers**, al igual que ocurre en el caso de la clasificación. Un único ejemplo con un error enorme hace que el denominador sea muy grande y que todos los demás errores normalizados sean casi cero, lo que distorsiona completamente el cálculo de $\epsilon_t$ y de los pesos.

- La segunda es que la elección de la función de pérdida no viene de un marco teórico coherente como en el caso de la clasificación. En clasificación, la actualización de pesos se deriva directamente de la pérdida exponencial. En AdaBoost.R2, la normalización y la actualización son heurísticas razonables pero no tienen una justificación como minimizadores de ninguna función de pérdida concreta.

**Gradient Boosting** resuelve ambos problemas: permite elegir explícitamente la función de pérdida en función de las necesidades del problema, y está teóricamente bien fundamentado. Por ello, en la práctica AdaBoost.R2 ha sido ampliamente superado por Gradient Boosting.

Tanto si tenemos un problema de regresión, como si en el caso de clasificación tenemos mucho ruido y _outliers_ o buscamos máximo rendimiento, convendrá considerar como alternativa métodos de _Gradient Boosting_.  







