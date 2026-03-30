# Proyecto final: Aprendizaje Automático en Problemas del Mundo Real

| | |
|---|---|
| **Propuesta** | Martes 31 de marzo, antes de las 23:59 |
| **Entrega** | Martes 19 de mayo, antes de las 23:59 |
| **Presentación oral** | Martes 19 de mayo, de 9:00 a 15:00 (15-20 minutos por grupo) |
| **Grupos** | 3-4 estudiantes |
| **Extensión** | 10–15 páginas (sin contar referencias y anexos) |
| **Formato** | PDF del informe (plantilla disponible en Moodle) + enlace a repositorio GitHub |
| | |


## 1. Descripción y Objetivos

Este trabajo tiene como objetivo aplicar de forma integrada los conocimientos adquiridos a lo largo de la asignatura sobre algoritmos de aprendizaje automático, evaluación de modelos y preprocesamiento de datos, pero yendo un paso más allá: explorando las problemáticas reales asociadas al despliegue de sistemas de ML en el mundo real, como la privacidad, el sesgo, la interpretabilidad o la robustez.

Cada grupo elegirá un **_dataset_** y definirá un **problema del mundo real** a tratar con él. A partir de ese punto de partida, deberá identificar y justificar qué problemáticas de las descritas en la [Sección 3](#3-dataset-problema-y-problematicas) son relevantes para su problema concreto, documentarse sobre las técnicas y herramientas del estado del arte para tratarlas, aplicarlas, y evaluar los resultados obtenidos. El trabajo seguirá el formato de un artículo científico.

Tras finalizar el trabajo buscamos ser capaces de:

- Identificar y analizar problemas éticos, legales y sociales asociados a sistemas de ML reales.
- Aplicar herramientas especializadas del estado del arte más allá de scikit-learn.
- Comunicar resultados técnicos y reflexiones críticas en formato científico.
- Evaluar modelos con métricas apropiadas al dominio del problema.
- Trabajar de forma colaborativa en un proyecto realista.

## 2. Hitos y entregas

Antes del comienzo del trabajo, deberá realizarse una **propuesta de grupo** y del **_dataset_** y el **problema** sobre el cual tratará el proyecto. 

- Los **grupos** podrán estar formados por un máximo de **4 personas**.
- En la siguiente sección se proponen una serie de **_datasets_** sobre los cuales realizar el proyecto, aunque se puede proponer cualquier otro, siempre que tenga una complejidad y envergadura suficiente. 

Esta información deberá enviarse a través de una entrega de Moodle **antes del martes 31 de marzo a las 23:59**.

La **entrega del proyecto** constará de un **informe** en PDF y del **código fuente** utilizado, tal como se detalla en la [Sección 4](#4-formato-del-informe) y en la [Sección 5](#5-codigo-fuente), respectivamente.  Se enviará a través de una entrega de Moodle **antes del martes 19 de mayo a las 23:59**.

Además de los entregables, se hará una **defensa oral** del proyecto el día **19 de mayo** en el horario de clase. Las defensas se distribuirán entre los turnos de teoría y de prácticas, asignando una de estas franjas a cada grupo. Cada grupo tendrá entre 15 y 20 minutos para realizar la defensa.


## 3. _Dataset_, Problema y Problemáticas

### 3.1. Elección del _dataset_ y el problema

El punto de partida del trabajo es la elección de un **_dataset_** y la definición de un **problema del mundo real** concreto a tratar con él. El _dataset_ debe ser tabular, suficientemente rico para que el análisis sea significativo (al menos varios miles de instancias y un número razonable de _features_), y el problema debe tener una motivación real clara: ¿quién usaría este sistema? ¿Sobre quién toma decisiones? ¿Qué consecuencias tiene una predicción errónea?

Los siguientes _datasets_ son sugerencias orientativas. Los grupos pueden proponer uno distinto, que deberá acordarse con el profesorado de la asignatura.

| _Dataset_ | Tarea | Fuente | Problemáticas naturales |
|---|---|---|---|
| **Adult Income** | Clasificación (ingresos >50k) | [UCI](https://archive.ics.uci.edu/dataset/2/adult) | Sesgo (género, raza), fairness, XAI |
| **COMPAS Recidivism** | Clasificación (reincidencia) | [ProPublica](https://github.com/propublica/compas-analysis) | Sesgo racial, fairness, XAI, impacto legal |
| **German Credit** | Clasificación (riesgo crediticio) | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | Sesgo, fairness, XAI, derecho a explicación |
| **Home Credit Default Risk** | Clasificación (impago) | [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) | XAI, fairness, datos escasos |
| **Diabetes 130-US Hospitals** | Clasificación (reingreso) | [UCI](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | Privacidad (FL), XAI, desbalanceo |
| **UCI Heart Disease** | Clasificación (enfermedad cardíaca) | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) | Privacidad (FL), XAI, datos escasos |
| **Credit Card Fraud** | Clasificación (fraude) | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Desbalanceo, datos sintéticos, privacidad |
| **ELEC2** | Clasificación (precio electricidad) | [OpenML](https://www.openml.org/d/151) | Deriva del modelo (concept drift) |
| **NYC Taxi Trips** | Regresión / clasificación | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Deriva temporal, fairness geográfica |
| **HAR (Human Activity Recognition)** | Clasificación | [UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | Privacidad (FL), datos sintéticos, desbalanceo |

### 3.2. Identificación de problemáticas

Una vez elegido el _dataset_ y definido el problema, el grupo debe identificar y justificar qué problemáticas de la siguiente tabla son relevantes para su caso concreto, y seleccionar aquellas sobre las que profundizar en el trabajo.

La identificación de las problemáticas debe estar argumentada en el informe en función del contexto de uso real del sistema (¿quién usa el modelo?, ¿sobre quién toma decisiones?, ¿en qué entorno se desplegaría?) y de las características del _dataset_ (¿hay atributos sensibles?, ¿está desbalanceado?, ¿tiene dimensión temporal?).

| Problemática | Descripción | Herramientas clave |
|---|---|---|
| **Privacidad y Federated Learning** | Los datos son sensibles y no pueden centralizarse. El modelo se entrena de forma distribuida. | [Flower (flwr)](https://flower.ai), [TensorFlow Federated](https://www.tensorflow.org/federated) |
| **Privacidad diferencial** | Garantías formales de que el modelo no revela información sobre individuos del training set. Trade-off privacidad–utilidad. | [Opacus](https://opacus.ai/) (Meta), [diffprivlib](https://diffprivlib.readthedocs.io/) (IBM) |
| **Sesgo y Fairness** | El modelo discrimina sistemáticamente a grupos vulnerables en decisiones de alto impacto. | [Fairlearn](https://fairlearn.org), [AI Fairness 360](https://aif360.readthedocs.io/) (IBM) |
| **Explicabilidad (XAI)** | Las decisiones del modelo afectan a personas y deben poder justificarse (RGPD Art. 22). | [SHAP](https://shap.readthedocs.io/), [LIME](https://github.com/marcotcr/lime), [DiCE](https://github.com/interpretml/DiCE) |
| **Robustez y ataques adversariales** | El modelo es vulnerable a manipulaciones de las entradas en el momento de la inferencia. | [Adversarial Robustness Toolbox](https://adversarial-robustness-toolbox.readthedocs.io/) (IBM), [Foolbox](https://foolbox.jonasrauber.de/) |
| **Deriva del modelo (drift)** | La distribución de los datos cambia con el tiempo y el modelo se degrada en producción. | [Evidently AI](https://www.evidentlyai.com/), [WhyLogs](https://whylabs.ai/), [River](https://riverml.xyz/) |
| **Datos escasos y sintéticos** | Los datos etiquetados son escasos o la clase de interés está muy subrepresentada. | [CTGAN/SDV](https://sdv.dev/), [Imbalanced-learn](https://imbalanced-learn.org/) (SMOTE) |

### 3.3. Esquema metodológico recomendado

Con independencia de las problemáticas elegidas, se recomienda que todos los trabajos sigan este esquema: 

1. Definir el problema y justificar su relevancia real.
2. Analizar el _dataset_ identificando las problemáticas presentes.
3. Implementar un modelo _baseline_ estándar sin tratar las problemáticas.
4. Aplicar las técnicas específicas para cada problemática identificada.
5. Comparar los resultados antes y después de cada intervención. 
6. Reflexionar críticamente sobre las implicaciones de los resultados obtenidos.


## 4. Formato del informe

El informe debe seguir la estructura de un **artículo científico**. Se recomienda usar la plantilla disponible en Moodle, aunque se acepta cualquier formato de artículo estándar siempre que incluya todos los apartados indicados.

### 4.1. Estructura 

El artículo deberá estructurarse en las secciones que se indican a continuación:

#### Título, autores y resumen (_Abstract_)
El título debe ser informativo y reflejar tanto el problema como el enfoque. El _abstract_ debe condensar en **150–250 palabras** el problema abordado, los datos utilizados, los métodos aplicados y los resultados principales. Es lo último que se escribe pero lo primero que se lee, y debe poder leerse de forma independiente.

#### 1. Introducción
Debe responder a cuatro preguntas: ¿Cuál es el problema y por qué es relevante? ¿Qué implicaciones éticas, sociales o legales tiene? ¿Qué se propone hacer este trabajo? ¿Cómo se organiza el resto del documento? No es un resumen del trabajo, sino una motivación y contextualización.

#### 2. Trabajo relacionado (Related Work)
Revisión crítica de la literatura relevante: artículos que abordan problemáticas relacionadas, herramientas de referencia y trabajos previos sobre el mismo _dataset_ si los hay. No se trata de hacer una lista de resúmenes, sino una síntesis que muestre cómo el trabajo se sitúa respecto al estado del arte. Se recomienda citar al menos **5–6 referencias de calidad** (artículos de revistas o conferencias científicas, no solo documentación de librerías).

#### 3. Descripción del problema y los datos
Definición formal de la tarea (clasificación, regresión, etc) con su función objetivo. Descripción del _dataset_ (origen, tamaño, _features_, variable objetivo, licencia y consideraciones éticas sobre su uso). Análisis exploratorio relevante con visualizaciones informativas. Identificación de problemas en los datos (desbalanceo, valores ausentes, sesgos observables).

#### 4. Metodología
Descripción del _pipeline_ completo: preprocesamiento, ingeniería de _features_, modelo _baseline_, técnicas específicas aplicadas (este sería el núcleo del trabajo), y protocolo de evaluación (validación cruzada, métricas elegidas y justificación). Esta sección debe ser suficientemente detallada para que el trabajo sea **reproducible**.

#### 5. Experimentos y resultados
Presentación clara de los resultados en tablas y gráficas comparativas. Comparativa entre el _baseline_ y los métodos propuestos. Visualizaciones que apoyen las conclusiones. Se limita a presentar resultados sin interpretarlos.

#### 6. Discusión
Interpretación de los resultados en el contexto del problema real. Análisis crítico de las limitaciones. Implicaciones éticas, sociales o legales de los resultados. 

#### 7. Conclusiones
Síntesis de las contribuciones en 3–4 párrafos. Lecciones aprendidas. Líneas de trabajo futuro concretas y bien motivadas.

#### Referencias
Formato APA o IEEE. Deben incluirse artículos de revistas o conferencias científicas, no solo páginas web.

#### Anexos (opcionales)
Enlace al repositorio GitHub. Tablas o figuras adicionales. Tabla de contribución individual de cada miembro del grupo.

### 4.2. Calidad del informe 

Se valorará especialmente la precisión del lenguaje técnico, la coherencia entre metodología y resultados, y la profundidad de la discusión crítica. Evitar afirmaciones sin respaldo ("el modelo funciona bien") y sustituirlas por cifras concretas con el contexto adecuado ("el modelo alcanza un F1-score de 0.87 en la clase minoritaria, mejorando en 12 puntos porcentuales al baseline").

### 4.3. Fuentes de referencia recomendadas

- **Artículos científicos:** [Google Scholar](https://scholar.google.com), [Semantic Scholar](https://www.semanticscholar.org), [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) / [cs.AI](https://arxiv.org/list/cs.AI/recent) / [stat.ML](https://arxiv.org/list/stat.ML/recent), [Papers with Code](https://paperswithcode.com).
- **Datasets:** [UCI ML Repository](https://archive.ics.uci.edu), [Kaggle](https://www.kaggle.com/datasets), [HuggingFace Datasets](https://huggingface.co/datasets), [OpenML](https://www.openml.org), [PhysioNet](https://physionet.org) (datos clínicos).

## 5. Código fuente

Junto al informe se entregará el código fuente utilizado para la implementación del método propuesto y para su validación experimental, así como todo el código de apoyo para la elaboración del trabajo (EDA, comparación con métodos alternativos del estado del arte, etc).


### 5.1. Repositorio de código

El código debe alojarse en un repositorio público GitHub e incluir:

- `README.md` con descripción del proyecto, instrucciones de instalación y ejecución.
- Archivo `requirements.txt` o `environment.yml` con las dependencias exactas.
- Semillas aleatorias fijadas (`random_state`, `np.random.seed`) para garantizar la reproducibilidad.
- Cuadernos Jupyter o scripts Python bien comentados y organizados en carpetas lógicas.

### 5.2. Calidad del código

Se valorará especialmente la corrección y la reproducibilidad del código, garantizando que con las instrucciones indicadas se pueda ejecutar en un nuevo entorno y obtener los resultados reportados. 

También se valorará que el código sea limpio, esté correctamente estructurado y documentado, y sea fácilmente entendible. 

Se permite la asistencia mediante IA generativa para la creación del código, siempre que el código sea perfectamente comprendido por todos los miembros del grupo, y que se adapte de forma adecuada a los requerimientos del trabajo.


## 6. Rúbrica de Evaluación

La nota final del trabajo se obtiene de los siguientes criterios ponderados. Cada criterio se evalúa en cuatro niveles: **Sobresaliente (A), Notable (B), Aprobado (C) y Suspenso (D)**. Los criterios 3, 4 y 7 tienen mayor peso, los dos primeros porque representan el núcleo técnico y reflexivo del trabajo, y el séptimo porque la defensa oral es un elemento clave de la evaluación.

### 6.1. Tabla de puntuaciones por criterio

| Criterio | Peso | Sobresaliente (A) | Notable (B) | Aprobado (C) | Suspenso (D) |
|---|:---:|:---:|:---:|:---:|:---:|
| 1. Planteamiento y contexto | 10 % | 9-10 | 7-8 | 5–6 | 0–4 |
| 2. Análisis de datos (EDA) | 10 %  | 9-10 | 7-8 | 5–6 | 0–4 |
| 3. Metodología y correctitud técnica | 20 % | 18-20 | 14-17 | 10-13 | 0–9 |
| 4. Análisis crítico y discusión | 20 % | 18-20 | 14-17 | 10-13 | 0–9 |
| 5. Escritura y formato científico | 10 %  | 9-10 | 7-8 | 5–6 | 0–4 |
| 6. Código reproducible | 10 % | 9-10 | 7-8 | 5–6 | 0–4 |
| 7. Defensa oral y contribución del grupo | 20 % | 18-20 | 14-17 | 10-13 | 0–9 |


### 6.2. Descriptores de cada criterio

#### Criterio 1 — Planteamiento y contexto (10 %)

| Nivel | Descripción |
|---|---|
| **Sobresaliente (A)** | Problema definido con precisión formal. Motivación sólida con implicaciones reales/éticas. Revisión de literatura crítica, estructurada temáticamente e identificando relaciones entre trabajos y vacíos (incluyendo al menos 5 referencias de calidad). |
| **Notable (B)** | Problema bien definido. Motivación clara. Revisión de literatura adecuada pero basada en una secuencia de descripciones con conexiones superficiales entre trabajos. |
| **Aprobado (C)** | Problema identificable pero impreciso. Motivación genérica. Literatura escasa o limitada a documentación de librerías. |
| **Suspenso (D)** | Problema mal delimitado. Ausencia o irrelevancia del trabajo relacionado. |

#### Criterio 2 — Análisis de datos (EDA) (10 %)

| Nivel | Descripción |
|---|---|
| **Sobresaliente (A)** | EDA profundo. Se identifican y discuten sesgos, desbalanceo o calidad de etiquetas con implicaciones claras. Visualizaciones rigurosas e informativas. |
| **Notable (B)** | EDA correcto. Problemas del _dataset_ mencionados sin profundizar. Visualizaciones adecuadas. |
| **Aprobado (C)** | EDA básico (estadísticos estándar). Problemas del _dataset_ apenas señalados. |
| **Suspenso (D)** | EDA mínimo o ausente. No se reflexiona sobre la naturaleza de los datos. |

#### Criterio 3 — Metodología y correctitud técnica (20 %)

| Nivel | Descripción |
|---|---|
| **Sobresaliente (A)** | _Pipeline_ bien diseñado y justificado. Sin data _leakage_. Métricas apropiadas al problema. Las técnicas aplicadas para tratar las problemáticas identificadas se usan con comprensión real de sus fundamentos (no como caja negra). _Baseline_ adecuado. |
| **Notable (B)** | Metodología correcta con algún error menor. Técnicas bien aplicadas con justificaciones parciales. _Baseline_ presente. |
| **Aprobado (C)** | Errores conceptuales menores. Técnicas usadas sin comprender bien sus fundamentos. Protocolo de evaluación con algún problema. |
| **Suspenso (D)** | Errores graves de metodología. Métricas o protocolo incorrectos. Técnicas mal aplicadas o sin justificación. |

#### Criterio 4 — Análisis crítico y discusión (20 %)

| Nivel | Descripción |
|---|---|
| **Sobresaliente (A)** | Discusión que interpreta los resultados en el contexto real del problema. Limitaciones no triviales analizadas con rigor. Implicaciones éticas/sociales fundamentadas y específicas. |
| **Notable (B)** | Discusión sólida pero superficial en implicaciones reales o éticas. Limitaciones mencionadas sin profundidad. |
| **Aprobado (C)** | Discusión básica (los resultados son buenos/malos). Implicaciones sociales y éticas genéricas o ausentes. |
| **Suspenso (D)** | Sin discusión real. Solo resultados sin interpretación. Cuestiones sociales y éticas ignoradas. |

#### Criterio 5 — Escritura y formato científico (10 %)

| Nivel | Descripción |
|---|---|
| **Sobresaliente (A)** | Formato de artículo seguido fielmente. Escritura clara y precisa. La notación es clara y consistente. Figuras/tablas con leyendas y referenciadas en el texto. Referencias completas y consistentes siguiendo formato estándar. |
| **Notable (B)** | Formato correcto con pequeñas desviaciones. Escritura clara con alguna ambigüedad. Algunos elementos tienen fallos de redacción o notación confusa que pueden hacer al lector necesitar releer alguna frase. Referencias con alguna inconsistencia. |
| **Aprobado (C)** | Formato aproximado con desviaciones notables (redacción coloquial, uso de primera persona sin consistencia, o prosa excesivamente verbosa que dificulta la lectura). Las ideas se expresan pero con imprecisión terminológica o rodeos innecesarios. Figuras sin leyendas o referencias incompletas. |
| **Suspenso (D)** | Sin formato científico. Escritura confusa. Referencias ausentes o incorrectas. |

#### Criterio 6 — Código reproducible (10 %)

| Nivel | Descripción |
|---|---|
| **Sobresaliente (A)** | Repositorio de código organizado con README claro. Código limpio, modular y comentado. Experimentos completamente reproducibles (semillas fijas, entorno documentado). |
| **Notable (B)** | Código funcional y razonablemente organizado. Reproducible con algún esfuerzo adicional. |
| **Aprobado (C)** | Código funcional pero desorganizado o sin comentarios. Reproducibilidad parcial. |
| **Suspenso (D)** | Código ausente, con errores o irreproducible. |

#### Criterio 7 — Defensa oral y contribución del grupo (20 %)

| Nivel | Descripción |
|---|---|
| **Sobresaliente (A)** | Todos los miembros demuestran conocimiento del trabajo completo. Contribuciones equilibradas y documentadas. La exposición es clara, bien estructurada, y todos los miembros responden adecuadamente a las preguntas realizadas. |
| **Notable (B)** | Contribución mayormente equilibrada con algún desajuste menor. La exposición muestra de forma adecuada gran parte del trabajo realizado, pero la estructuración es mejorable o se echa en falta detalles sobre algún punto relevante. |
| **Aprobado (C)** | Desequilibrios notables. Algún miembro no conoce partes del trabajo. La exposición da una idea del trabajo realizado, pero está mal organizada y se hace difícil de seguir. |
| **Suspenso (D)** | Un miembro o más no puede responder. Contribución claramente desigual. La presentación es confusa y no transmite el contenido real del trabajo. |

El **criterio 7** se evaluará mediante una breve defensa oral de 15-20 minutos por grupo, con preguntas cruzadas a los miembros. La ausencia injustificada a la defensa implica 0 en este criterio.

El **plagio** total o parcial de código o texto implica la calificación de $0$ en el trabajo, siguiendo el criterio establecido en el Reglamento para la Evaluación de los Aprendizajes de la Universidad de Alicante.


## Preguntas Frecuentes

**¿Puedo usar un dataset diferente a los sugeridos?**  
Sí, siempre que el _dataset_ sea tabular, suficientemente rico (al menos varios miles de instancias), y sea apropiado para identificar en él problemáticas del mundo real. Consúltalo con el profesorado antes de comenzar el trabajo.

**¿Puedo usar IA generativa para redactar el informe?**  
El uso de herramientas de IA generativa está permitido como apoyo a la redacción, pero no como sustituto del trabajo intelectual del grupo. El informe debe reflejar una comprensión genuina de los métodos aplicados. En la defensa oral se verificará que todos los miembros comprenden el contenido. Se recomienda indicar en los anexos si se han utilizado estas herramientas, y cómo se ha hecho.

**¿Qué pasa si nuestro modelo no obtiene buenos resultados?**  
La nota no depende de obtener un F1-score alto, sino de aplicar la metodología correctamente, entender por qué los resultados son los que son, y reflexionar críticamente sobre ello. Un trabajo que analiza honestamente por qué un método no funciona bien en un _dataset_ particular puede obtener la máxima nota.

**¿Cuántas referencias bibliográficas son suficientes?**  
Se recomienda un mínimo de 5–6 referencias de calidad (artículos de revistas o conferencias como NeurIPS, ICML, ICLR, KDD, etc). No se valoran las referencias a páginas de documentación de librerías como referencias principales, aunque sí se pueden incluir como referencias secundarias.

