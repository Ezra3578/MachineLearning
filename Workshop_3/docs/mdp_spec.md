# MDP Specification — Workshop 3 (RL on Fashion-MNIST)

> Codigo-artefacto del punto 1 de la guia (`MDP Formulation Document`).
> El reporte PDF (`groupN_workshop3.pdf`) NO se genera aqui por peticion del usuario.

## 1. Problem Statement

El agente RL resuelve clasificacion de Fashion-MNIST (10 clases) como proceso de
decision secuencial: en cada paso observa una imagen y elige la etiqueta predicha.
Se usa RL como enfoque alternativo/complementario a los modelos supervisados de
Workshop 1 (Random Forest / XGBoost / HGB) y Workshop 2 (SmallCNN / ResNet-18).

RL es apropiado aqui porque: (a) la tarea admite formularse como decisiones
secuenciales donde cada prediccion recibe una recompensa inmediata verificable;
(b) disponemos de un simulador barato (muestreo i.i.d. del dataset); (c) permite
estudiar exploracion vs. explotacion, reward shaping y sample-efficiency frente al
aprendizaje supervisado. Se reporta honestamente que el entorno es un bandit
contextual (transicion independiente de la accion), por lo que el supervisado
sera mas sample-efficient: la comparacion es el punto cientifico del workshop.

## 2. State Space

- Observacion: imagen Fashion-MNIST 28x28 en escala de grises, normalizada con
  las estadisticas globales del Workshop 1/2: `mean = 0.2860`, `std = 0.3530`.
- Representacion para la red: tensor `float32` de forma `(1, 28, 28)` en `[0,1]`
  normalizado; `observation_space = Box(low=0, high=1, shape=(1,28,28))`.
- Sin apilado temporal ni historial: el estado es Markoviano por construccion
  (muestreo i.i.d.), `state_dim = 784` pixeles efectivos.
- Split usado: el mismo 70/15/15 estratificado con `seed=42` de Workshop 2, para
  metricas comparables.

## 3. Action Set

- Discreto, `action_space = Discrete(10)`, enumeracion identica a `CLASS_NAMES`:
  `0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt,`
  `7 Sneaker, 8 Bag, 9 Ankle boot`.
- Interpretacion: la accion `a_t` ES la etiqueta predicha para `s_t`.
- Por ser discreto y pequeno se justifica DQN (guia, seccion Agent Algorithm).

## 4. Transition Dynamics

- `s_{t+1} = f(s_t, a_t) = sample_uniform(split)`: la siguiente imagen se muestrea
  uniformemente del split activo, independientemente de la accion tomada.
- Determinista dado el indice muestreado; estocastica desde la perspectiva del
  agente (i.i.d.). No hay dinamica latente ni memoria entre pasos.
- Consecuencia: `gamma` tiene efecto limitado (recompensa inmediata domina); se
  barre `gamma in {0.90, 0.95, 0.99}` igualmente como pide la guia para verificarlo
  empiricamente.

## 5. Reward Function

- Por defecto (`reward_mode="plus_minus"`, densa y simetrica):
  `r_t = +1` si `a_t == y_t`, `-1` en caso contrario.
- Retorno de episodio con `N = max_steps = 100`:
  `G in [-100, 100]`; `accuracy = (G + N) / (2N)`.
- Modos alternos (`--reward-mode`): `zero_one` (`+1/0`, util para comparar contra
  accuracy directa) y `sparse` (0 en pasos intermedios, `+#aciertos` al final;
  solo para el analisis de sparse-rewards de la guia, converge lento).
- Horizonte relevante: retorno acumulado del episodio (largo plazo = suma de
  decisiones); cada paso tambien es interpretable (corto plazo = acierto/fallo).

## 6. Episode Termination

- `truncated = True` cuando `t == max_steps` (defecto 100). `terminated` nunca se
  activa (no hay estado absorbente de exito/fracaso).
- La terminacion equivale a "presupuesto de clasificaciones por episodio" y hace
  comparable el retorno entre episodios. El exito de la tarea se mide por retorno
  medio y accuracy, no por terminacion temprana.
