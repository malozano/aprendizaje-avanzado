# Aprendizaje Automático en Problemas del Mundo Real


A lo largo de la asignatura hemos estudiado algoritmos de ML bajo el supuesto de que disponemos de un _dataset_ limpio, estático y representativo, lo entrenamos, lo evaluamos y reportamos métricas. Este modelo de trabajo es útil para aprender los fundamentos, pero es una simplificación que omite casi todo lo que hace difícil el ML en la práctica.

Cuando un sistema de ML se despliega en el mundo real, enfrenta al menos cinco problemas que no aparecen en nuestros _datasets_ de prueba:

- **Los datos son privados o sensibles** y no pueden centralizarse libremente.
- **Los datos reflejan sesgos históricos** que el modelo puede amplificar.
- **El modelo es una caja negra** y las decisiones que toma afectan a personas reales.
- **El mundo cambia** y la distribución de los datos en producción difiere de la del entrenamiento.
- **Los atacantes pueden manipular las entradas** para engañar al modelo.

En producción, el ML no es un proyecto con inicio y fin, sino un ciclo continuo, como se muestra en la [](#fig-ciclo). Cada paso en este diagrama es un punto donde algo puede salir mal, y en el que nos podremos encontrar problemas como los indicados anteriormente.

Figure: Ciclo de vida de un sistema de ML. {#fig-ciclo}

![](images/t13_ml_ciclo.svg)

Este tema revisa cada uno de estos problemas, las herramientas matemáticas y computacionales que la comunidad ha desarrollado para tratarlos, y el marco legal que en Europa obliga a tenerlos en cuenta. 

## Marco Legal y Ético

### El Reglamento General de Protección de Datos (RGPD)

El [RGPD](https://www.boe.es/doue/2016/119/L00001-00088.pdf) [@gdpr2016], en vigor desde 2018, es el marco legal principal que afecta al ML en Europa. Sus implicaciones para los sistemas de aprendizaje automático son directas y no opcionales.

En el Art. 5 se establecen los **principios relativos al tratamiento** que tienen un significativo impacto en su uso en ML:

- **Limitación de la finalidad:** Los datos recogidos para un propósito no pueden usarse para entrenar un modelo con un propósito distinto.
- **Minimización de datos:** Solo se pueden recopilar los datos estrictamente necesarios para el fin declarado. Esto limita la posibilidad de recopilar datos "por si acaso" para entrenar modelos.
- **Exactitud:** Los datos deben ser correctos y estar actualizados, lo que tiene implicaciones para el mantenimiento de los _datasets_ de entrenamiento.
- **Limitación del plazo de conservación:** Los datos no pueden guardarse indefinidamente, por lo que los modelos entrenados con datos que ya no pueden conservarse legalmente plantean preguntas sobre el "derecho al olvido" en ML.
- **Integridad y confidencialidad:** Los datos deben ser tratados de forma que se garantice su seguridad, lo cual tendrá implicaciones en la privacidad de los modelos entrenados. 

**El derecho a explicación (Art. 22):** Cuando una decisión automatizada afecta significativamente a una persona (concesión de crédito, contratación, diagnóstico médico), el interesado tiene derecho a "a no ser objeto de una decisión basada únicamente en el tratamiento
automatizado" y, según se indica en el Art. 13, a obtener "información significativa sobre la lógica aplicada". Esto hace que la explicabilidad no sea solo una buena práctica sino una obligación legal. Los modelos de caja negra como las redes neuronales profundas o los ensembles son problemáticos en estos contextos sin herramientas de XAI adicionales.

### El Reglamento de Inteligencia Artificial de la UE (AI Act)

El [AI Act](https://artificialintelligenceact.eu/es) [@aiact2024], publicado en 2024, es el primer marco regulatorio integral sobre IA del mundo. Clasifica los sistemas de IA en cuatro categorías de riesgo:

**Riesgo inaceptable (prohibidos)**

- Sistemas de puntuación social por parte de gobiernos.
- Manipulación subliminal del comportamiento.
- Reconocimiento facial en tiempo real en espacios públicos (con excepciones muy limitadas).

**Alto riesgo (regulados con obligaciones estrictas)**

Esta categoría incluye la mayoría de aplicaciones ML de impacto real: infraestructura crítica, educación, empleo, crédito, acceso a servicios públicos, justicia penal y gestión de fronteras. Los sistemas de alto riesgo deben cumplir requisitos de gestión de riesgos, calidad de datos (_datasets_ sin sesgos relevantes), transparencia y explicabilidad, supervisión humana, robustez y exactitud. Deben registrarse y someterse a evaluaciones de conformidad.

**Riesgo limitado** 

Sistemas con obligaciones de transparencia (como por ejemplo los _chatbots_, que deben identificarse como IA). La principal obligación en este grupo de riesgo es al menos informar al usuario.

**Riesgo mínimo** 

La mayoría de aplicaciones (como por ejemplo filtros de spam, recomendadores de contenido, videojuegos).

La **implicación práctica para el desarrollo de ML** es que cualquier sistema que se pretenda desplegar en categorías de alto riesgo debe diseñarse desde el principio con estos requisitos en mente, no como una capa añadida al final. Esto implica documentación exhaustiva de los datos de entrenamiento, trazabilidad de las decisiones del modelo, y capacidad de auditoría.

### Ética más allá de la ley

El cumplimiento legal es un mínimo, pero podemos ir más allá. La comunidad de ML ha desarrollado marcos éticos más amplios. Los más influyentes son los [**principios de IA de la OCDE**](https://datos.gob.es/es/blog/los-principios-de-inteligencia-artificial-de-la-ocde) (beneficio para las personas y el planeta, valores centrados en el ser humano, transparencia, robustez y rendición de cuentas) y las [**directrices de IA fiable de la Comisión Europea**](https://digital-strategy.ec.europa.eu/es/library/ethics-guidelines-trustworthy-ai) (agencia y supervisión humanas, solidez técnica y seguridad, privacidad y gobernanza de datos, transparencia, diversidad y no discriminación, bienestar social y medioambiental, y rendición de cuentas).

Estos marcos no tienen fuerza legal directa pero orientan la práctica del sector y están siendo progresivamente incorporados a la regulación.


## Privacidad

### El problema de la centralización de datos

El paradigma clásico del ML asume que todos los datos de entrenamiento están disponibles en un único servidor. Esto es incompatible con muchos dominios reales:

- **Salud:** Los historiales clínicos de diferentes hospitales no pueden compartirse por regulación (RGPD, HIPAA en EEUU). Sin embargo, un modelo entrenado sobre datos de múltiples hospitales sería mucho más potente que uno entrenado en un solo centro.
- **Banca y finanzas:** Los datos de transacciones son sensibles y los bancos no pueden compartirlos entre sí, aunque colaborar mejoraría los modelos de detección de fraude.
- **Dispositivos móviles:** Los teclados predictivos o asistentes de voz aprenden de lo que el usuario escribe o dice, datos que son extremadamente privados.

Existen dos grandes enfoques complementarios para entrenar modelos con garantías de privacidad: el **aprendizaje federado** y la **privacidad diferencial**.

### Federated Learning (FL)

#### Motivación 

El aprendizaje federado fue introducido por Google en 2017 en el contexto de Gboard [@hard2018gboard], su teclado predictivo para móvil, donde el texto escrito por cada usuario es demasiado sensible para enviarse a servidores centrales. Este enfoque invierte el paradigma clásico, y en lugar de llevar los datos al modelo, llevamos el modelo a los datos.

En FL, existe un **servidor central** que coordina el entrenamiento y $K$ **clientes** (hospitales, bancos, dispositivos móviles) que tienen datos locales que nunca salen de su entorno (ver [](#fig-fl)). El proceso es iterativo:

1. El servidor envía el modelo global actual $w^t$ a un subconjunto de clientes.
2. Cada cliente $k$ entrena el modelo localmente sobre sus datos $\mathcal{D}_k$ durante $E$ épocas locales, obteniendo una actualización $w_k^{t+1}$.
3. Los clientes envían sus actualizaciones (gradientes o pesos) al servidor.
4. El servidor agrega las actualizaciones para obtener el nuevo modelo global $w^{t+1}$.

Figure: Arquitectura de un sistema de Federated Learning. El servidor comparte un modelo centrar con cada cliente. Cada cliente entrena el modelo con sus propios datos de forma local y devuelve el modelo actualizado al servidor. El servidor promedia los modelos recibidos ponderando por el número de ejemplos de cada cliente. {#fig-fl}

![](images/t13_federated_learning.svg)


#### FedAvg

El algoritmo **FedAvg** [@mcmahan2017fedavg] es el punto de partida de la mayoría de métodos de FL. La agregación en el servidor se realiza como una media ponderada por el tamaño del _dataset_ local:

$$w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{t+1}$$

donde $n_k = |\mathcal{D}_k|$ es el número de ejemplos del cliente $k$ y $n = \sum_k n_k$ el total.

La actualización local de cada cliente minimiza su pérdida local mediante SGD, con la siguiente actualización en cada paso del entrenamiento:

$$w_k^{t+1} = w^t - \eta \nabla \mathcal{L}_k(w^t)$$

donde $\mathcal{L}_k$ es la función de pérdida evaluada sobre $\mathcal{D}_k$.

#### El problema de los datos no-IID

El supuesto implícito de FedAvg es que los datos locales son representativos de la distribución global. En la práctica, los datos de cada cliente son **heterogéneos** (non-IID). Por ejemplo, el hospital A puede tener principalmente pacientes diabéticos mientras que el B tiene principalmente cardíacos. Esta heterogeneidad causa **client drift**: el modelo local de cada cliente se aleja del óptimo global durante el entrenamiento local, lo que degrada la convergencia.

**FedProx** [@li2020fedprox] aborda esto añadiendo un término de regularización proximal a la pérdida local que penaliza el alejamiento del modelo global:

$$\mathcal{L}_k^{\text{prox}}(w) = \mathcal{L}_k(w) + \frac{\mu}{2} \|w - w^t\|^2$$

El hiperparámetro $\mu \geq 0$ controla el _trade-off_ entre adaptación local y coherencia global. Cuando $\mu = 0$ recuperamos FedAvg.

#### Limitaciones del FL

El FL no garantiza privacidad perfecta por sí solo. Los gradientes que los clientes envían al servidor pueden filtrar información sobre los datos locales. Ataques como el **gradient inversion** [@zhu2019deep] demuestran que es posible reconstruir las imágenes de entrenamiento a partir de los gradientes con alta fidelidad. Para garantías formales de privacidad es necesario combinar FL con **privacidad diferencial**.

### Privacidad Diferencial (DP)

#### Definición formal

La privacidad diferencial [@dwork2006dp] proporciona una garantía matemática formal sobre cuánta información sobre un individuo puede inferirse a partir de la salida de un algoritmo.

En DP hablamos de **mecanismo aleatorio** $\mathcal{M}: \mathcal{D} \to \mathcal{R}$ para referirnos a un proceso que toma como entrada unos datos $D \in \mathcal{D}$  y produce un resultado aleatorizado $\mathcal{R}$ (nuestro modelo entrenado). En el contexto del entrenamiento de modelos, $\mathcal{M}$ sería el propio proceso de entrenamiento: toma el _dataset_ $D$ como entrada y produce un modelo entrenado $w$ como salida. La aleatoriedad es esencial, ya que un algoritmo determinista siempre produciría la misma salida para el mismo _dataset_, lo que permitiría a un adversario detectar diferencias entre _datasets_ vecinos con certeza absoluta.

**Definición ($\varepsilon, \delta$-DP):** Un mecanismo aleatorio $\mathcal{M}: \mathcal{D} \to \mathcal{R}$ satisface $(\varepsilon, \delta)$-privacidad diferencial si para todo par de _datasets_ vecinos $D, D'$ que difieren en exactamente un individuo, y para todo subconjunto de salidas $S \subseteq \mathcal{R}$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\varepsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$


La interpretación intuitiva de esta expresión  es que la presencia o ausencia de un individuo en el _dataset_ cambia la distribución de salidas del mecanismo a lo sumo por un factor $e^\varepsilon$ (más un término aditivo $\delta$). Un $\varepsilon$ pequeño (en torno a 1) implica privacidad fuerte, mientras que $\varepsilon > 10$ comienza a ser una garantía débil.

El parámetro $\varepsilon$ se denomina presupuesto de privacidad y cuantifica la garantía de privacidad que ofrece el sistema. Un $\varepsilon$ pequeño significa que es muy difícil inferir si un individuo concreto estaba en el _dataset_, mientras que un $\varepsilon$ grande significa que esa garantía se ha debilitado. En DP-SGD, cada paso de gradiente es un mecanismo DP que añade ruido, pero también revela una pequeña cantidad de información sobre los datos. La combinación de muchos pasos permite a un adversario acumular evidencia progresivamente, por lo que el $\varepsilon$ acumulado crece con cada paso de entrenamiento ($\varepsilon_{\text{total}} = \sum_i \varepsilon_i$​). Esto implica que entrenar durante más épocas mejora la utilidad del modelo pero debilita la garantía de privacidad, por lo que el número de pasos de entrenamiento no es un hiperparámetro libre en DP-SGD.

#### El mecanismo gaussiano

El mecanismo más utilizado en ML consiste en añadir ruido gaussiano calibrado a la sensibilidad $\Delta f$ de la función que se desea privatizar:

$$\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 \cdot \mathbf{I})$$

donde la sensibilidad global $\Delta f = \max_{D, D'} \|f(D) - f(D')\|_2$ mide cuánto puede cambiar la salida de $f$ al cambiar un individuo. Para lograr $(\varepsilon, \delta)$-DP se puede demostrar que es suficiente tomar $\sigma = \frac{\Delta f \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$.

#### DP-SGD: entrenamiento diferencialment privado de redes neuronales

El algoritmo **DP-SGD** [@abadi2016dpsgd], implementado en la librería **Opacus** de Meta, adapta SGD para satisfacer DP:

1. Por cada _batch_, calcular el gradiente por cada ejemplo individual: $g_i = \nabla_w \mathcal{L}(w; x_i, y_i)$.
2. **Clipping:** Limitar la norma de cada gradiente a $C$: $\tilde{g}_i = g_i / \max(1, \|g_i\|_2 / C)$. Esto acota la sensibilidad.
3. **Perturbación:** Añadir ruido gaussiano al gradiente agregado: $\tilde{g} = \frac{1}{B}\left(\sum_i \tilde{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})\right)$.
4. Actualizar los pesos con el gradiente privatizado.

El coste de la privacidad diferencial es la **pérdida de utilidad**, ya que el modelo entrenado con DP-SGD suele rendir peor que sin DP. El _trade-off_ privacidad-utilidad es uno de los problemas de investigación activa más importantes del área.

Destacamos en este área **herramientas como** [Opacus](https://opacus.ai/) (PyTorch), [TensorFlow Privacy](https://github.com/tensorflow/privacy), y [diffprivlib](https://diffprivlib.readthedocs.io/) (scikit-learn compatible).

### Ataques de inferencia de pertenencia (Membership Inference Attacks)

#### Motivación

Hasta ahora hemos tratado la privacidad como un problema sobre los **datos**, viendo cómo evitar que los datos brutos salgan del entorno donde residen. Pero el modelo entrenado en sí mismo puede filtrar información sobre los datos de entrenamiento, incluso cuando esos datos nunca se comparten directamente. Los **ataques de inferencia de pertenencia** (MIA) [@shokri2017mia] demuestran que, dado acceso a un modelo entrenado, un adversario puede determinar con notable precisión si un registro concreto formó parte del conjunto de entrenamiento.

Esto tiene consecuencias legales directas bajo el RGPD, ya que si un hospital entrena un modelo sobre historiales clínicos y lo publica, un adversario podría inferir qué pacientes están en el _dataset_, violando su privacidad incluso sin acceder a ningún dato directamente.

#### Mecanismo del ataque

El ataque explota una propiedad bien conocida de los modelos sobreajustados, que consiste en que los modelos tienen mayor confianza (probabilidades de salida más altas y más concentradas) sobre los ejemplos que han visto durante el entrenamiento que sobre los que no han visto.

Formalmente, dado un modelo $f_\theta$ y una instancia $(x, y)$, el objetivo del adversario es construir un clasificador binario $\mathcal{A}$ que prediga:

$$\mathcal{A}(f_\theta(x), y) = \begin{cases} 1 & \text{si } (x, y) \in \mathcal{D}_{\text{train}} \\ 0 & \text{en caso contrario} \end{cases}$$

El ataque de Shokri et al. se basa en **modelos sombra** (*shadow models*). Esto es, el adversario entrena $k$ modelos $\{f_{\theta_i}^{\text{shadow}}\}$ sobre datos con distribución similar a la del modelo objetivo, con la diferencia de que sí conoce qué instancias forman parte del entrenamiento de cada modelo sombra. Con estos modelos genera un _dataset_ de entrenamiento para $\mathcal{A}$:

- Ejemplos positivos ($\text{miembro} = 1$): pares $(f_{\theta_i}^{\text{shadow}}(x), y)$ para $(x, y) \in \mathcal{D}_{\text{train}}^{\text{shadow}_i}$.
- Ejemplos negativos ($\text{miembro} = 0$): pares $(f_{\theta_i}^{\text{shadow}}(x), y)$ para $(x, y) \notin \mathcal{D}_{\text{train}}^{\text{shadow}_i}$.

El clasificador de ataque $\mathcal{A}$ aprende a distinguir los patrones en el vector de probabilidades de salida $f_\theta(x)$ que caracterizan a los miembros del entrenamiento. En la práctica, el vector de salida de los miembros tiende a tener una probabilidad mucho más alta en la clase correcta y una entropía más baja que el de los no miembros (ver [](#fig-mia)).

Figure: Vulnerabilidad frente a un ataque MIA. Se muestra las distribuciones de la función de pérdida del modelo en dos casos. A la izquierda tenemos un modelo con overfitting, donde los ejemplos de entrenamiento tienen de forma general una pérdida menor que los de test. A la derecha vemos un modelo regularizado en el que apenas existe brecha entre la distribución de train y la de test, siendo de esta forma menos vulnerable a ataques MIA. {#fig-mia}

![](images/t13_mia_loss.svg)



#### Variantes sin modelos sombra

Yeom et al. [@yeom2018mia] propusieron un ataque más simple que no requiere modelos sombra, y que consiste en clasificar $(x, y)$ como miembro si y solo si la pérdida del modelo sobre ese ejemplo es inferior a la pérdida media sobre el conjunto de entrenamiento:

$$\mathcal{A}(x, y) = \mathbf{1}\left[\mathcal{L}(f_\theta(x), y) < \tau\right]$$

Este ataque es menos preciso pero revela con claridad que la causa raíz del problema es la **brecha de generalización** $\mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}}$. Cuanto mayor es el sobreajuste, más fácil es el ataque. Un modelo perfectamente generalizado (sin sobreajuste) sería trivialmente resistente a este ataque.


#### Evaluación de la vulnerabilidad

La vulnerabilidad de un modelo ante MIA se cuantifica habitualmente con dos métricas:

- **Ventaja del adversario:** $\text{Adv}(\mathcal{A}) = |P(\mathcal{A}(x,y)=1 \mid \text{miembro}) - P(\mathcal{A}(x,y)=1 \mid \text{no miembro})|$. Un valor de 0 indica que el ataque no es mejor que el azar, mientras que un valor cercano a 1 indica alta vulnerabilidad.
- **AUC de la curva ROC** del clasificador de ataque $\mathcal{A}$. Un AUC de 0.5 equivale a azar, mientras que un AUC alto indica que el modelo es muy vulnerable.

#### Conexión con la privacidad diferencial

La privacidad diferencial ofrece una garantía formal contra los MIA. Bajo $(\varepsilon, \delta)$-DP, puede demostrarse que la ventaja máxima de cualquier adversario está acotada:

$$\text{Adv}(\mathcal{A}) \leq \frac{e^\varepsilon - 1}{e^\varepsilon + 1} + \frac{2\delta}{e^\varepsilon + 1}$$

Para $\varepsilon$ pequeño, esta cota es pequeña, limitando formalmente cuánto puede aprender el adversario sobre la pertenencia de cualquier individuo al conjunto de entrenamiento. Esto refuerza la utilidad de DP-SGD, ya que el ruido añadido durante el entrenamiento no solo protege los datos individuales en el gradiente, sino que también limita la capacidad de un adversario de hacer inferencias sobre el _dataset_ a partir del modelo publicado.

Como **herramientas** destacadas en este campo tenemos la herramienta de auditoría de privacidad basada en MIA [ML Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter), y [TensorFlow Privacy](https://github.com/tensorflow/privacy), que incluye utilidades de evaluación de MIA.

Respecto a FL, [Flower](https://flower.ai) es uno de los principales _frameworks_ de referencia. 

## Sesgo y Fairness

### Orígenes del sesgo en sistemas de ML

El sesgo en ML no surge de la nada, sino que los modelos aprenden de datos históricos que reflejan desigualdades y discriminaciones del mundo real. Si históricamente ciertos grupos han tenido menos acceso al crédito, los datos de impago reflejarán esa historia, y un modelo entrenado sobre ellos perpetuará esa desigualdad, incluso si no tiene acceso explícito al atributo protegido (raza, género, etc.).

Los orígenes del sesgo pueden clasificarse en tres categorías:

- **Sesgo de representación:** El _dataset_ de entrenamiento no es representativo de la población sobre la que el modelo se aplicará. Por ejemplo, _datasets_ médicos construidos mayoritariamente sobre población masculina caucásica.
- **Sesgo de medición:** Los _proxies_ que se usan como variables objetivo son imperfectos y pueden introducir sesgo. La tasa de reincidencia criminal, por ejemplo, depende de quién es arrestado, no de quién comete delitos.
- **Sesgo de retroalimentación:** Cuando las predicciones del modelo influyen sobre el comportamiento del mundo real, que a su vez genera los próximos datos de entrenamiento. Los modelos de vigilancia policial predictiva son el ejemplo más estudiado.

### Métricas de fairness

El primer problema al que nos enfrentamos es que "justicia" es un concepto ambiguo con múltiples definiciones matemáticamente precisas que son **mutuamente incompatibles** (teorema de imposibilidad de Chouldechova [@chouldechova2017fairness] y Kleinberg et al. [@kleinberg2017fairness]). Definimos las métricas sobre un clasificador binario con predicción $\hat{Y} \in \{0,1\}$, variable objetivo real $Y \in \{0,1\}$ y atributo protegido $A \in \{a, b\}$.

**Paridad demográfica (_Independence_):**

$$P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = b)$$

Evalúa si el modelo predice la clase positiva con la misma tasa para todos los grupos. No tiene en cuenta si las tasas base de $Y=1$ difieren entre grupos.

**Igualdad de oportunidades (_Equal Opportunity_ [@hardt2016equality]) :** 

$$P(\hat{Y} = 1 \mid Y = 1, A = a) = P(\hat{Y} = 1 \mid Y = 1, A = b)$$

Evalúa si la tasa de verdaderos positivos (sensibilidad o _recall_) es igual entre grupos. En el contexto de crédito, por ejemplo, podríamos evaluar si la tasa de aprobación entre quienes realmente devolverían el préstamo es la misma independientemente del grupo.

**Posibilidades igualadas (_Equalized Odds_):**

$$P(\hat{Y} = 1 \mid Y = y, A = a) = P(\hat{Y} = 1 \mid Y = y, A = b) \quad \forall y \in \{0, 1\}$$

Evalúe si tanto la tasa de verdaderos positivos como la de falsos positivos son iguales entre grupos. Es una condición más fuerte que la igualdad de oportunidades.

**Calibración:**

$$P(Y = 1 \mid \hat{p} = p, A = a) = P(Y = 1 \mid \hat{p} = p, A = b) = p$$

Evalúa si la probabilidad predicha tiene el mismo significado independientemente del grupo. Si el modelo predice 0.7 de riesgo, eso debe significar lo mismo para ambos grupos.

#### El teorema de imposibilidad

Chouldechova [@chouldechova2017fairness] demostró que cuando las tasas base (proporciones naturales de la clase positiva en cada grupo) difieren entre grupos, no es posible satisfacer simultáneamente calibración y posibilidades igualadas, y Kleinberg et al. [@kleinberg2017fairness] añade otros resultados complementarios. Esto significa que cualquier decisión de diseño de un sistema de ML que afecte a grupos con tasas base distintas conlleva una decisión ética implícita, ya que elegir una métrica de _fairness_ implica sacrificar otra.

Este resultado tiene consecuencias profundas, no existe una solución técnica "objetiva" al problema del sesgo. La elección de qué métrica priorizar es una decisión política y ética que debe tomarse explícitamente, no delegarse al algoritmo.

### Técnicas de mitigación

Las técnicas de mitigación se clasifican según en qué fase del _pipeline_ actúan:

**Pre-procesamiento (sobre los datos):**

- *Reweighing:* Asignar pesos a los ejemplos de entrenamiento para que la combinación de grupo y etiqueta esté balanceada. Los grupos subrepresentados o con etiquetas históricamente sesgadas reciben mayor peso.
- *Resampling:* Sobremuestrear grupos desfavorecidos o submuestrear los favorecidos.
- *Aprendizaje de representaciones justas:* Entrenar un _encoder_ que elimine la información del atributo protegido de la representación latente antes del entrenamiento del clasificador.

**In-procesamiento (durante el entrenamiento):**

Añadir una restricción de _fairness_ al problema de optimización:

$$\min_w \mathcal{L}(w) \quad \text{sujeto a} \quad |P(\hat{Y}=1|A=a) - P(\hat{Y}=1|A=b)| \leq \gamma_{\text{fair}}$$

Esta restricción puede incorporarse mediante multiplicadores de Lagrange, convirtiendo el problema restringido en uno sin restricciones:

$$\min_w \max_\lambda \; \mathcal{L}(w) + \lambda \cdot g_{\text{fair}}(w)$$

donde $g_{\text{fair}}(w)$ mide la violación de la restricción de fairness.

**Post-procesamiento (sobre las predicciones):**

Una vez entrenado el modelo, se ajustan los umbrales de decisión de forma diferenciada por grupo para satisfacer la métrica de _fairness_ deseada. Es la técnica más sencilla de implementar pero requiere acceso al atributo protegido en tiempo de inferencia.

Como **herramientas** destacadas encontramos [Fairlearn](https://fairlearn.org/) (Microsoft), [AI Fairness 360](https://aif360.readthedocs.io/) (IBM), [What-If Tool](https://pair-code.github.io/what-if-tool/) (Google).

## Explicabilidad (XAI)

### Motivación

Aunque no siempre real, en ocasiones parece existir un conflicto entre la precisión de un modelo y su interpretabilidad. Los modelos lineales y los árboles de decisión son fácilmente interpretables, pero en muchos problemas complejos los modelos de caja negra (redes neuronales, _ensembles_) obtienen mejores resultados. La XAI busca abrir esas cajas negras sin sacrificar su potencia.

Distinguimos dos tipos de explicaciones:

- **Explicaciones globales:** Describen el comportamiento del modelo en general. ¿Qué variables son más importantes para el modelo en conjunto?
- **Explicaciones locales:** Describen por qué el modelo tomó una decisión particular para una instancia concreta. ¿Por qué se denegó *este* crédito?

Y dos tipos de métodos según su relación con el modelo:

- **Métodos agnósticos al modelo (model-agnostic):** Funcionan para cualquier modelo tratándolo como caja negra. SHAP y LIME son los ejemplos principales.
- **Métodos específicos del modelo:** Aprovechan la estructura interna del modelo. Grad-CAM para redes convolucionales, por ejemplo.

### SHAP (SHapley Additive exPlanations)

SHAP [@lundberg2017shap] es el método de explicabilidad más utilizado en la práctica. Se basa en los **valores de Shapley** de la teoría de juegos cooperativos.

#### Valores de Shapley

En un juego cooperativo con $d$ jugadores (_features_), el valor de Shapley $\phi_j$ del jugador $j$ mide su contribución marginal promedio sobre todas las posibles coaliciones (subconjuntos) de jugadores:

$$\phi_j(f, x) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(d - |S| - 1)!}{d!} \left[ f_{S \cup \{j\}}(x_{S \cup \{j\}}) - f_S(x_S) \right]$$

donde $\mathcal{F}$ es el conjunto de todas las features, $S$ es una coalición que no incluye a $j$, y $f_S(x_S)$ es la predicción del modelo usando solo las features en $S$ (marginalizando el resto). Intuitivamente, podemos observar que lo que estamos midiendo es cuánto mejor incluir $j$ en cada una de las posibles combinaciones sin $j$, para así estimar la importancia de la _feature_ $j$.

Los valores de Shapley satisfacen cuatro propiedades axiomáticas deseables: eficiencia ($\sum_j \phi_j = f(x) - \mathbb{E}[f(x)]$, las contribuciones suman la diferencia entre la predicción y la media), simetría, nulidad (_features_ que no afectan al modelo tienen $\phi_j = 0$), y aditividad.


#### Variantes de SHAP según el tipo de modelo

El cálculo exacto de los valores de Shapley requiere $O(2^d)$ evaluaciones del modelo, lo que es inviable para $d$ grande. Sin embargo, existen variantes enfocadas a diferentes familias de modelos, que permiten optimizar el cálculo de estos valores.

**Tree SHAP** se aplica a modelos basados en árboles (Random Forest, XGBoost, etc.). Lundberg et al. (2018) [@lundberg2018treeshap] desarrollaron un algoritmo exacto en $O(TLD^2)$ donde $T$ es el número de árboles, $L$ el número de hojas y $D$ la profundidad máxima, polinomial en lugar de exponencial. De esta forma, los modelos basados en árboles tienen la gran ventaja de poder aplicar SHAP sobre ellos de forma eficiente.

**Linear SHAP** se aplica a modelos lineales. En este caso los valores de Shapley se obtienen analíticamente a partir de los coeficientes del modelo y la covarianza entre _features_, sin necesidad de muestreo. Es exacto y prácticamente instantáneo, aunque asume que las _features_ son independientes entre sí. Cuando hay correlaciones fuertes, la atribución puede distorsionarse.

**Deep SHAP** está diseñado para redes neuronales. Aproxima los valores de Shapley propagando hacia atrás las diferencias en la activación de cada neurona respecto a una instancia de referencia (el vector de medias del _dataset_, habitualmente), aprovechando la estructura del grafo computacional de la red para hacerlo eficiente [@shrikumar2017deeplift]. Es más rápido que el cálculo exacto, pero la aproximación se degrada en arquitecturas con no-linealidades complejas.

**Kernel SHAP** es la variante agnóstica al modelo, es decir, funciona con cualquier modelo tratándolo como caja negra. Estima los valores de Shapley ajustando un modelo lineal ponderado sobre coaliciones de _features_ muestreadas aleatoriamente, usando un núcleo de pesos determinado por la teoría de juegos (no un hiperparámetro del usuario). Esto lo hace más estable que LIME (que veremos a continuación), pero la calidad de la estimación depende del número de coaliciones muestreadas y el coste crece con $d$.

La tabla siguiente resume cuándo usar cada variante:

| Variante | Modelo objetivo | Exactitud | Coste |
|---|---|---|---|
| **TreeSHAP** | Árboles, Random Forest, XGBoost | Exacto | $O(TLD^2)$ |
| **Linear SHAP** | Modelos lineales | Exacto | $O(d)$ |
| **Deep SHAP** | Redes neuronales | Aproximado | $O(d \cdot \text{backprop})$ |
| **Kernel SHAP** | Cualquier modelo | Aproximado | $O(M \cdot d)$ |

La recomendación práctica es usar la variante específica del modelo siempre que esté disponible, y recurrir a Kernel SHAP únicamente cuando el modelo no encaje en ninguna de las otras categorías o se necesite un método verdaderamente agnóstico.

#### Interpretación de los valores SHAP

Para una instancia $x$, la predicción del modelo se descompone como:

$$f(x) = \mathbb{E}[f(X)] + \sum_{j=1}^{d} \phi_j$$

donde $\mathbb{E}[f(X)]$ es la predicción base (media sobre el _dataset_) y $\phi_j$ es la contribución de la feature $j$, que será positiva si empuja la predicción hacia arriba, negativa si la empuja hacia abajo.

El **summary plot** de SHAP muestra, para cada feature, la distribución de valores de Shapley sobre todo el _dataset_, y permite identificar qué features son globalmente más importantes y cómo su valor se relaciona con su contribución (ver [](#fig-shap)).

Figure: Summary Plot de SHAP aplicado a un dataset sintético basado en la estructura de Adult Income. Vemos un resumen sobre cómo contribuye cada característica de cada ejemplo de entrada en la decisión, ordenado de mayor a menor importancia.  {#fig-shap}

![](images/t13_shap_summary.svg)

### LIME (Local Interpretable Model-agnostic Explanations)

LIME [@ribeiro2016lime] genera explicaciones locales construyendo un modelo interpretable (habitualmente lineal) que aproxima el modelo de caja negra en el entorno de la instancia a explicar.

El proceso para explicar la predicción $f(x)$ sobre una instancia $x$ es el siguiente:

1. Generar una muestra de instancias $\{z_i\}$ perturbando $x$ (activando/desactivando features o añadiendo ruido). Con ello, generamos un conjunto de datos en la vecindad del punto $x$ a analizar. 
2. Obtener las predicciones $f(z_i)$ del modelo de caja negra sobre esas instancias.
3. Ponderar cada $z_i$ según su proximidad a $x$: $\pi_x(z_i) = \exp(-d(x, z_i)^2 / \sigma^2)$.
4. Ajustar un modelo lineal $g$ minimizando la pérdida local ponderada con una penalización de complejidad $\Omega(g)$:

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

Los coeficientes del modelo lineal $g$ son la explicación local, y nos indican qué _features_ contribuyeron positiva o negativamente a la predicción en el entorno de $x$.

Es importante destacar que aunque los datos a escala global no sean linealmente separables, a escala local podríamos aproximar la frontera mediante un hiperplano (ver [](#fig-lime)), y en este modelo local podemos interpretar de forma directa cómo contribuye cada _feature_.

Figure: Ejemplo de aplicación de LIME sobre un conjunto de datos con forma de doble luna (izquierda). La instancia a explicar está marcada con forma de estrella. Se generan perturbaciones de esta instancia (derecha), y se evalúan con el modelo de "caja negra" (la salida generada por el modelo se muestra en una escala de color de azul a rojo, según si pertenece a la clase 0 o a la clase 1). Cada una recibe un peso en función de su distancia al ejemplo a explicar (se representan con mayor radio las de mayor peso), y se entrena un modelo interpretable sobre este dataset local. Se muestra en verde el hiperplano de separación obtenido. Los coeficientes del hiperplano nos dan la importancia de cada variable. {#fig-lime}

![](images/t13_lime.svg)


Como **limitación crítica** encontramos la inestabilidad de LIME. Pequeñas variaciones en el muestreo de perturbaciones o en el ancho de banda $\sigma$ pueden producir explicaciones muy diferentes para la misma instancia. Esto es un problema grave en contextos de alto impacto donde la consistencia de las explicaciones es necesaria.

### Explicaciones contrafactuales con DiCE

Las explicaciones contrafactuales responden a la pregunta **"¿qué tendría que cambiar en mi situación para obtener una decisión diferente?"**. En el contexto de un crédito denegado, por ejemplo, podríamos indicar "si sus ingresos fueran 5.000€ en lugar de 3.000€, el crédito habría sido aprobado".

**DiCE** (Diverse Counterfactual Explanations) [@mothilal2020dice] genera múltiples explicaciones contrafactuales diversas $c_1, c_2, \ldots, c_k$, que son versiones modificadas del punto original $x$. Por ejemplo, si $x$ representa a una persona con ingresos de 2.000€, edad 25 y deuda 10.000€, entonces $c_1$ podría ser esa misma persona pero con ingresos 3.500€ y deuda 8.000€, $c_2$​ con ingresos 4.000€ y edad 30, etc. Buscamos minimizar el siguiente objetivo:

$$\min_{c_1, \ldots, c_k} \frac{1}{k} \sum_{i=1}^k \text{yloss}(f(c_i), \hat{y}) + \frac{\lambda_1}{k} \sum_{i=1}^k \text{dist}(c_i, x) - \frac{\lambda_2}{k^2} \sum_{i,j} \text{dist}(c_i, c_j)$$

Donde $\hat{y}$ se refiere a la **clase deseada** (esto es, la alternativa que se desea obtener, por ejemplo, "crédito aprobado"), $\lambda_1$ y $\lambda_2$ son hiperparámetros que controlan el equilibrio de los tres términos. El primer término obliga a que los contrafactuales tengan la clase deseada ($\text{yloss}$ mide cuánto se aleja la predicción de cada contrafactual de la clase deseada), el segundo que sean cercanos al punto original $x$ (cambios mínimos), y el tercero que sean diversos entre sí (no todos el mismo cambio).

**Herramientas** clave en este campo son [SHAP](https://shap.readthedocs.io/), [LIME](https://github.com/marcotcr/lime), [DiCE](https://github.com/interpretml/DiCE) y [Captum](https://captum.ai/) (para redes neuronales, basado en gradiente).

