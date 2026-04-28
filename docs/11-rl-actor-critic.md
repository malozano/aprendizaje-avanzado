# Aprendizaje por Refuerzo

En la sesión anterior estudiamos los fundamentos de los métodos de aprendizaje por refuerzo basados en gradiente de política, y de forma particular el algoritmo REINFORCE [@williams1992simple], que constituye el algoritmo más básico de esta familia de métodos. Vimos que su principal debilidad es su alta varianza. Vamos a ver en esta sesión tres propuestas orientadas a corregir las debilidades de REINFORCE básico: REINFORCE con _baseline_, Actor-Critic y PPO.

## Baseline y reducción de varianza

Una de las formas más sencillas de reducir la varianza consiste en introducir una estimación de la función valor y un término llamado "ventaja".

### La Función de Ventaja

Para introducir la función de ventaja, en primer lugar vamos a ver que el Teorema del Gradiente de Política seguirá siendo válido si restamos cualquier función $b(s)$ que dependa únicamente del estado (no de la acción) [@sutton2000policy]:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \gamma^t \left(G_t - b(s_t)\right) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]\,.
$$

**Demostración:** Necesitamos mostrar que $\mathbb{E}\left[ b(s) \nabla_\theta \log \pi_\theta(a|s) \right] = 0$. Como $b(s)$ no depende de $a$, es una constante dentro de la esperanza sobre $a$ y puede sacarse fuera de ella:

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\!\left[ b(s) \nabla_\theta \log \pi_\theta(a|s)\right] = b(s)\, \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\!\left[  \nabla_\theta \log \pi_\theta(a|s)\right].
$$

Dado que $\pi_\theta(a \mid s)$ es una distribución de probabilidad, la esperanza de su gradiente logarítmico será siempre 0. Lo podemos ver a continuación, aplicando el "truco del gradiente logarítimico" visto en la sesión anterior:

$$
\begin{align*}
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\!\left[\nabla_\theta \log \pi_\theta(a|s)\right]
&= \sum_a \pi_\theta(a|s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} \\
&= \nabla_\theta \sum_a \pi_\theta(a|s) \\
&= \nabla_\theta 1 = 0
\end{align*}
$$

Por lo tanto, el término completo es 0, y la _baseline_ no afecta a $\nabla_\theta J(\theta)$.

Vemos entonces que la _baseline_ no introduce sesgo en el gradiente, pero puede reducir drásticamente la varianza si se elige bien.

Una elección óptima para la **baseline** es:

$$
b^\ast(s_t) = V^\pi(s_t) = \mathbb{E}_{\pi}\left[G_t \mid s_t\right]\,,
$$

es decir, **la función de valor bajo la política actual**. Intuitivamente, $V^\pi(s_t)$ representa el retorno *promedio* desde el estado $s_t$, por lo que $(G_t - V^\pi(s_t))$ captura cuánto **mejor o peor** fue la acción $a_t$ respecto al promedio.

Esta cantidad con signo es la **función de ventaja** (*Advantage function*), que se define en términos de las funciones $Q^\pi$ y $V^\pi$:

$$
\boxed{A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)}
$$

donde $Q^\pi(s_t, a_t) = \mathbb{E}_\pi[G_t \mid s_t, a_t]$ es el valor esperado del retorno tras tomar la acción $a_t$ en el estado $s_t$ y seguir la política $\pi$ a partir de ahí. La función $V^\pi(s_t) = \mathbb{E}_{a \sim \pi}[Q^\pi(s_t, a)]$ es su promedio sobre todas las acciones posibles. La diferencia entre ambas mide el mérito **relativo** de la acción $a_t$ (será positivo si es mejor que el promedio y negativo si es peor).

En la práctica no conocemos $Q^\pi$ ni $V^\pi$ exactamente, por lo que usamos estimadores. En REINFORCE, el retorno observado $G_t$ es un estimador Monte Carlo no sesgado de $Q^\pi(s_t, a_t)$, de modo que:

$$
\hat{A}_t = G_t - V_\phi(s_t)
$$

es el estimador Monte Carlo de la ventaja que emplearemos en REINFORCE con baseline.

La actualización del gradiente queda entonces:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \gamma^t \hat{A}_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]\,.
$$

Intuitivamente, si comparamos dos actualizaciones en el mismo estado $s$ con y sin _baseline_, tenemos:

- Sin _baseline_, una acción que produce $G_t = 7$ siempre recibe un empuje positivo, incluso si todas las acciones desde $s$ producen $\approx 7$ de media (lo que significa que esta acción no es especial).
- Con baseline $b = V(s) = 7$, la ventaja es $\approx 0$ y la actualización del gradiente es prácticamente nula, reflejando correctamente que ninguna acción merece ser especialmente reforzada.


### REINFORCE con Baseline

Podemos estimar $V^\pi(s_t)$ a partir de los propios datos manteniendo una **red de valor** $V_\phi(s)$ con parámetros $\phi$. Tras cada episodio, ajustamos $V_\phi$ a los retornos observados minimizando el error cuadrático medio:

$$
\mathcal{L}(\phi) = \sum_{t=0}^{T-1} \left(G_t - V_\phi(s_t)\right)^2\,.
$$

La estimación de la ventaja se convierte entonces en $\hat{A}_t = G_t - V_\phi(s_t)$.

Es importante destacar que es beneficioso en este caso utilizar tasas de aprendizaje diferentes para la red de política ($\alpha_\theta$) y para la red de valor ($\alpha_\phi$), ya que su naturaleza es distinta. A continuación, mostramos el algoritmo de REINFORCE con baseline:

!!! algorithm "Algoritmo: REINFORCE con baseline"
      **Entradas** Tasas de aprendizaje $\alpha_\theta$, $\alpha_\phi$, descuento $\gamma$, número de episodios $K$

      **Salida** Parámetros de política $\theta$

      1. **Inicializar** $\theta$, $\phi$ aleatoriamente
      2. **Para** cada episodio $i = 1, \ldots, K$:
         1. Generar trayectoria $(s_0, a_0, r_1, \ldots, s_{T-1}, a_{T-1}, r_T)$ siguiendo $\pi_\theta$
         2. Calcular retornos hacia atrás: $G_t = r_{t+1} + \gamma G_{t+1}$, $G_T = 0$
         3. **Para** $t = 0, 1, \ldots, T-1$:
            1. $\hat{A}_t = G_t - V_\phi(s_t)$
            2. $\phi \leftarrow \phi - \alpha_\phi  \nabla_\phi\left(G_t - V_\phi(s_t)\right)^2$
            3. $\theta \leftarrow \theta + \alpha_\theta  \gamma^t  \hat{A}_t  \nabla_\theta \log \pi_\theta(a_t \mid s_t)$

      **devolver** $\theta$

En la práctica, REINFORCE con baseline converge notablemente más rápido que REINFORCE puro porque:

- Las actualizaciones están **centradas**, y la señal del gradiente ya no es uniformemente positiva o negativa.
- La varianza de $\hat{A}_t$ es mucho menor que la varianza de $G_t$.

No obstante, seguimos encontrando como limitación que **la actualización sigue siendo por episodios** (Monte Carlo). Debemos esperar al final de cada episodio para calcular $G_t$, lo que impide el aprendizaje en línea. Los métodos Actor-Critic abordan esta limitación.


## Actor-Critic

Como hemos comentado, una desventaja que comparten tanto la versión básica de REINFORCE como su versión con _baseline_ es que son algoritmos _offline_, en los que el agente debe esperar hasta el final del episodio para actualizar los parámetros de la política. La familia de algoritmos Actor-Critic, propuesta por Sutton y Barto [@sutton2018reinforcement] viene a resolver este problema.

En Actor-Critic, denominamos **actor** a la política aprendida, y **crítico** a la función _baseline_.

### Retornos de N pasos

Para entender el salto conceptual entre REINFORCE y Actor-Critic conviene introducir brevemente los **retornos de $N$ pasos**. En REINFORCE usamos el retorno completo del episodio:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{T-1-t} r_T\,,
$$

que corresponde a estimar $Q^\pi(s_t, a_t)$ con todas las recompensas futuras reales. El extremo opuesto es usar únicamente un paso real y sustituir el resto por la estimación del crítico:

$$
G_t^{(1)} = r_{t+1} + \gamma V_\phi(s_{t+1})\,.
$$

Entre ambos extremos existe una familia continua de estimadores de $n$ pasos:

$$
G_t^{(N)} = \left( \sum_{k=0}^{N-1} \gamma^k r_{t+k+1} \right) + \gamma^N V_\phi(s_{t+N})\,.
$$

Con $N \to \infty$ tenemos el retorno Monte Carlo, mientras que con $N = 1$ obtenemos el objetivo TD de un paso. Este planteamiento pone de manifiesto que REINFORCE y Actor-Critic son extremos del mismo espectro, y que la elección de $n$ controla el compromiso entre sesgo (pequeño con $N$ grande, ya que dependemos menos del crítico) y varianza (pequeña con $N$ pequeño, porque promediamos menos ruido estocástico del entorno). Actor-Critic corresponde al caso $N=1$, que veremos a continuación.

### Arquitectura de dos redes

Hemos visto dos grandes familias de métodos de RL: los métodos basados en valor y los métodos basados en gradiente de política. Los métodos **Actor-Critic** combinan lo mejor de ambos mundos:

- El **actor** es la política $\pi_\theta(a|s)$, actualizada mediante gradiente de política.
- El **crítico** (*critic*) es la función de valor $V_\phi(s)$, actualizada mediante Diferencia Temporal (TD).

El cambio fundamental respecto a REINFORCE con _baseline_ es que el **crítico utiliza estimaciones TD** en lugar de retornos Monte Carlo. Esto elimina la necesidad de esperar al final del episodio, permitiendo **actualizaciones en línea, paso a paso**.

La estimación TD de la ventaja en el instante $t$ es:

$$
\hat{A}_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\,,
$$

Esta expresión recibe el nombre de **error TD** y se denota habitualmente como $\delta_t$.

Esta cantidad es exactamente la diferencia entre el objetivo (_target_) TD $[r_{t+1} + \gamma V_\phi(s_{t+1})]$ y la estimación actual $V_\phi(s_t)$. La idea es análoga a la regla de actualización de SARSA y Q-Learning ya que en ambos casos se mide la diferencia entre una estimación del valor futuro y la estimación actual. La diferencia es que en los anteriores algoritmos se utilizaba para actualizar $Q(s,a)$, mientras que aquí se aplica como estimador de la ventaja $A^\pi(s_t, a_t)$. Para ver por qué es un estimador válido, debemos considerar que si el crítico estuviera perfectamente entrenado, $V_\phi(s_{t+1}) \approx V^\pi(s_{t+1})$, y entonces:

$$
\mathbb{E}[\delta_t \mid s_t, a_t] = \mathbb{E}[r_{t+1} + \gamma V^\pi(s_{t+1})] - V^\pi(s_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)\,.
$$

Es decir, si el crítico estuviera perfectamente entrenado, $\delta_t$ sería en promedio exactamente $A^\pi(s_t, a_t)$. En la práctica, el error de aproximación de la red introduce un pequeño sesgo. 

El error TD mide cuánto mejor o peor fue la acción tomada respecto a lo que el crítico esperaba en ese estado. En lugar de esperar al final del episodio para conocer el retorno real $G_t$, usamos la recompensa inmediata $r_{t+1}$ más la estimación del valor del siguiente estado $V_\phi(s_{t+1})$ como sustituto del retorno futuro.

A esta técnica de usar la propia estimación aprendida $V_\phi(s_{t+1})$ como sustituto del retorno futuro, en lugar de esperar al valor real observado al final del episodio, se le denomina **bootstrap**. La ventaja es que permite actualizar la política en cada paso de tiempo (o cada $N$ pasos, como hemos visto en el punto anterior) sin esperar al final del episodio, a costa de introducir un pequeño sesgo derivado del error de aproximación del crítico.

De forma análoga a SARSA, así como SARSA hace _bootstrap_ de los valores $Q$, el crítico hace _bootstrap_ de los valores $V$. La diferencia es que aquí la estimación con _bootstrap_ sirve para guiar al actor en lugar de actualizar una tabla $Q$.

<!-- TODO: Figura arquitectura -->

En implementaciones basadas en _deep learning_, el actor y el crítico suelen **compartir las capas inferiores** (extracción de características) y divergen únicamente en sus cabezas de salida, lo que mejora la eficiencia en el número de parámetros.

### Actualización TD del Crítico

El crítico se actualiza para minimizar el error TD. Dado que el objetivo TD es $r_{t+1} + \gamma V_\phi(s_{t+1})$, la pérdida del crítico es:

$$
\mathcal{L}(\phi) = \left(r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\right)^2 = \delta_t^2\,,
$$

y la actualización es un paso estándar de descenso por gradiente:

$$
\phi \leftarrow \phi - \alpha_\phi  \nabla_\phi \, \delta_t^2 = \phi + \alpha_\phi  \delta_t  \nabla_\phi V_\phi(s_t)\,.
$$

Destacamos que, como ocurre en Q-Learning, el objetivo TD $r_{t+1} + \gamma V_\phi(s_{t+1})$ se trata como una **constante** al calcular el gradiente (no se diferencia a través de él), ya que queremos mover $V_\phi(s_t)$ hacia un objetivo fijo y no mover ambos extremos simultáneamente.

La actualización del actor utiliza el error TD como estimación de la ventaja:

$$
\theta \leftarrow \theta + \alpha_\theta  \delta_t  \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,.
$$

El factor $\gamma^t$ que aparecía en REINFORCE ha desaparecido aquí. En el caso episódico con horizonte largo, este factor tiende a cero a medida que avanza el episodio, reduciendo las actualizaciones de los pasos tardíos de forma innecesaria. En la práctica, los métodos Actor-Critic paso a paso omiten $\gamma^t$ o lo tratan implícitamente mediante el descuento acumulado en el crítico, lo que simplifica la implementación sin afectar significativamente al rendimiento empírico.
