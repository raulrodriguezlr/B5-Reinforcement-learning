# CLAUDE.md: práctica B5 Aprendizaje por Refuerzo (taller trader)

Contexto y reglas de trabajo para esta práctica. Manda sobre el CLAUDE.md global solo donde lo contradiga de forma explícita; en todo lo demás valen las reglas globales (resumidas en la sección 8).

## 1. Contexto

- Asignatura: Aprendizaje por Refuerzo, MIAX-14, Bloque V. Profesor: Luis F. Lago Fernández (UAM), luis.lago@uam.es.
- Material de clase (solo lectura, no se copia al repo salvo lo indicado): `C:\Users\raulr\Documents\MIAX\Bloque V\aprendizaje por refuerzo\`
  - `slides1.pdf`: fundamentos, bandido multibrazo, MDP, ecuaciones de Bellman.
  - `slides2.pdf`: programación dinámica (policy evaluation, policy iteration, value iteration) y Monte Carlo (first-visit, ES, on-policy, off-policy con importance sampling).
  - `slides3.pdf`: TD(0), TD vs MC en batch (ejemplo A-B), Sarsa, Q-learning, Cliff Walking.
  - `scripts/*.ipynb`: gridworld, clase_20261023, gambler, frozen_lake, taxi, blackjack, taller_trader.
  - `transcripciones.txt`: VTT de las dos sesiones (23 y 24 de octubre de 2026).
- Enunciado oficial: `taller_trader.ipynb` (versión de Descargas = versión de `scripts/`, mismo texto). Contiene el entorno `SimpleTradingEnv`, los cuatro generadores de precios y el agente aleatorio de referencia.
- Lectura de referencia: Sutton & Barto, 2.ª ed., caps. 1 a 6. Complemento: curso de David Silver, clases 1 a 5.

## 2. Qué pide la práctica

Objetivo: agente de RL que aprenda a invertir en un entorno simulado con un único activo.

Mínimos (50 % de la nota):
1. Implementar al menos **Q-Learning** y **Sarsa** y probarlos con las **cuatro** funciones generadoras de precios.
   - Entrenamiento con muchos episodios; **cada episodio con una serie nueva** generada por la misma función.
   - Evaluación sobre series de **test nuevas** de la misma función, **promediando** sobre varias series.
2. Analizar la solución de cada generador y compararla con el **agente aleatorio** y con **buy & hold**. Razonar si la estrategia encontrada es la esperada.
3. Ajustar hiperparámetros: α, ε (con decaimiento), γ y número de episodios.
4. Tabla comparativa algoritmos + baselines por generador, con al menos **rendimiento** y **tiempo de convergencia**.

Resto de la nota: pruebas más allá de los mínimos 10 %, calidad de presentación de resultados 20 %, justificación de decisiones de diseño 10 %, análisis crítico 10 %.

Entrega:
- PDF de **máximo dos páginas** (resumen de pruebas, resultados y conclusiones). Puede ser documento o dos diapositivas. Por grupo de TFM.
- Antes del **martes 27/10/2026**: un representante del grupo escribe al profesor para organizar el orden de presentación.
- Antes de la clase del **jueves 29/10/2026**: primera versión del PDF por email a luis.lago@uam.es.
- En esa clase: presentación y discusión de **15 minutos** por grupo.
- Entrega final en Moodle hasta el **29/10/2026 a las 23:59**, incorporando el feedback de la clase.

## 3. Lo que dijo el profesor (transcripción, relevante para la práctica)

- Le importa más **entender y analizar** la solución que programar los algoritmos a mano; acepta código generado con ayuda, pero preguntará por convergencia, por qué un algoritmo converge antes que otro y qué estrategia ha encontrado el agente.
- "Si tuviese que usar un método tabular, me iría directamente a por Q-learning."
- La observación es continua: para métodos tabulares hay que **discretizar la tendencia** ("partís en 10 cachitos").
- Idealmente cada episodio usa una serie distinta; como reto mayor, entrenar mezclando funciones generadoras distintas para que el agente funcione en todas.
- Se pueden usar precios reales descargados (ojo al **sesgo de supervivencia**, lo mencionó explícitamente).
- Él entrenó un agente que supera al aleatorio y a buy & hold en el generador 1 (tendencia + ciclo).
- Pregunta abierta que dejó: con precios completamente aleatorios (browniano), ¿cuál sería la política óptima? Esperar algo como "da igual" o "no operar" y verificarlo con simulación.
- Pide comprobar empíricamente lo que dicen los libros (por ejemplo, tiempos de convergencia), no darlo por hecho. Para convergencia, contar iteraciones o episodios y medir tiempo.
- ε: empezar alto e ir bajando. En entornos no estacionarios (mercados) puede tener sentido no llevar ε a 0.
- Sarsa con ε fijo converge a la mejor política ε-greedy (más conservadora, ejemplo Cliff Walking); Q-learning converge a la óptima pero su rendimiento online puede ser peor. Con ε → 0 Sarsa también tiende a la óptima.
- Con datos finitos (batch, caso típico en finanzas) TD suele converger antes y generalizar mejor que MC.
- Diseñar la recompensa es lo más difícil del RL: el agente explota recompensas mal definidas.
- La propiedad de Markov en finanzas probablemente no se cumple; depende de cómo se defina el estado. Buen punto para el análisis crítico.
- Ideas que sugirió en clase como extensión: probar MC (on-policy, off-policy), variar penalizaciones y tamaño del problema, ε ajustable, más generadores o series reales.

## 4. El entorno `SimpleTradingEnv` (verificado leyendo el código)

- Acciones: 0 vender, 1 mantener, 2 comprar. Comprar solo actúa con posición 0; vender solo con posición 1.
- Observación: `[clip(tendencia, -1, 1), posición]`, `float32`. `tendencia = (p_t - media(ventana)) / media(ventana)` con ventana `prices[t-5 : t+1]` (6 precios; el comentario del código dice "5 pasos" pero son 6 observaciones, como dice el enunciado).
- Ejecución al precio `prices[t]` (precio ya observado); la recompensa usa el retorno `t -> t+1`. No hay mirada al futuro en la decisión.
- Recompensa: `-0.001` por cada operación ejecutada + retorno simple `(p_{t+1} - p_t) / p_t` si se tiene el activo tras actuar. Comisión 0.1 % sobre el capital al comprar y al vender.
- `balance` en `info`: efectivo + acciones a precio actual. Capital inicial 1000.
- Episodio: empieza en `current_step = 5`, termina cuando `current_step >= len(prices) - 1`. Con `n_steps = 500` son 494 pasos.
- `reset(seed=None)` no controla la serie de precios: los generadores usan el `np.random` global. Para reproducibilidad hay que fijar semillas al generar cada serie.
- El agente aleatorio usa `env.action_space.sample()`: hay que sembrar `env.action_space.seed(...)`.
- Buy & hold de referencia en el notebook: compra en `prices[5]` con comisión del 0.1 % y mantiene hasta el final.
- El entorno del enunciado se copia **tal cual** para los resultados principales. Cualquier variante (otra recompensa, otro estado, venta en corto) va aparte, como extensión, y se justifica.

Escala real de la tendencia (50 series de 500 pasos, semilla 0):

| Generador | p1 | p25 | p50 | p75 | p99 |
|---|---|---|---|---|---|
| 1. tendencia + ciclo | -0.31 | -0.08 | 0.02 | 0.13 | 0.37 |
| 2. browniano | -0.058 | -0.017 | 0.000 | 0.016 | 0.057 |
| 3. reversión a la media | -0.056 | -0.016 | 0.000 | 0.016 | 0.057 |
| 4. regímenes | -0.067 | -0.012 | 0.001 | 0.013 | 0.062 |

Consecuencia: el clip a ±1 nunca actúa y 10 tramos uniformes en [-1, 1] dejarían casi todo en uno o dos tramos. La discretización tiene que adaptarse a la escala (cuantiles calculados con series de entrenamiento, o límites por generador). Es una decisión de diseño que se justifica en el PDF. Los límites se calculan solo con datos de entrenamiento, nunca con test.

Otras observaciones del generador 1: crece de forma explosiva (drift `0.0005 * t` con `t` hasta 50), así que buy & hold será muy difícil de batir en capital final. Conviene reportar también métricas relativas (retorno, Sharpe, drawdown, número de operaciones), no solo balance final.

## 5. Cómo programa el profesor (para que el código encaje)

- Notebooks en Colab, Python con `numpy`, `matplotlib`, `gymnasium`, `tqdm`, `collections.defaultdict`, `seaborn` (blackjack).
- Funciones con docstring corto y comentarios en español; nombres de variables en inglés (`value_iteration`, `q_values`, `num_iters`, `epsilon_greedy`).
- Q como array `np.zeros((n_estados..., n_acciones))`; política ε-greedy como función `epsilon_greedy(Q, state, epsilon)` que devuelve acción aleatoria con probabilidad ε y `np.argmax` en otro caso.
- Decaimientos multiplicativos por episodio (`alpha *= 0.99`, `epsilon *= 0.99`) con un tope máximo de pasos.
- Devuelve contadores de iteraciones para analizar convergencia (`return V, policy, num_iters`).
- Gráficos: `plt.figure`, `plt.grid()`, ejes con etiqueta; curvas de recompensa por episodio; panel doble precio + balance como en el taller.

Para este repo se mantiene ese estilo pero cumpliendo las reglas de la sección 8 (comentarios breves, type hints, semillas, sin rutas absolutas).

## 6. Estructura propuesta del repo (a confirmar con el usuario antes de crearla)

```
src/rl_trader/
  env.py          # SimpleTradingEnv copiado tal cual del enunciado
  prices.py       # los cuatro generadores (+ extras), con semilla
  discretize.py   # discretización de la tendencia
  agents.py       # QLearning, Sarsa (y extras), política ε-greedy
  baselines.py    # aleatorio, buy & hold
  train.py        # bucle de entrenamiento, curvas, criterio de convergencia
  evaluate.py     # evaluación en series de test, métricas, tablas
notebooks/        # experimentos y figuras, uno por bloque de análisis
tests/            # pytest, sin red
results/          # tablas y figuras generadas (csv, png)
report/           # PDF de dos páginas y su fuente
config/           # hiperparámetros y semillas
```

## 7. Criterios de experimentación

- Semillas separadas para entrenamiento, validación (ajuste de hiperparámetros) y test. Ninguna serie de test se usa para ajustar nada.
- Resultados de test promediados sobre varias series y varias semillas de entrenamiento; reportar media y dispersión.
- Convergencia: episodios hasta estabilizar la recompensa media móvil y tiempo de reloj. Definir el criterio una vez y usarlo igual para todos los algoritmos.
- Todo hiperparámetro y umbral va en config o como argumento con valor por defecto explícito.
- Cada conclusión del PDF se apoya en un número o una figura generada por el código del repo.

## 8. Reglas de trabajo (del CLAUDE.md global, aplicadas aquí)

Cadena obligatoria cada vez que se toca código, salvo que el usuario diga "sin cadena":

| Paso | Quién | Qué hace |
|---|---|---|
| 0. Plan | Agente principal (Opus) con el usuario | Lee el código, acota la tarea, escribe el plan. Decide y orquesta. |
| 1. Código | `dev-coder` (Sonnet) | Implementa el plan y nada más. |
| 2. Auditoría | `dev-auditor` (Sonnet) | Regresiones, corrección, fuga de datos (train/test), alcance, malas prácticas. No edita. |
| 3. Corrección | `dev-fixer` (Sonnet) | Aplica solo los hallazgos del auditor. |
| 4. QA | `dev-qa` (Sonnet) | Ejecuta de punta a punta y prueba casos reales y bordes. No edita. |
| 5. Test | `dev-tester` (Sonnet) | Escribe y ejecuta tests; reporta la salida tal cual. |

- Subagentes en serie, siempre Sonnet, nunca en paralelo sobre los mismos ficheros.
- BLOQUEANTE del auditor: tras el fixer se repite la auditoría una vez.
- Fallos de QA o tester vuelven al fixer; tras dos vueltas sin arreglo se para y se avisa.
- Al final el agente principal revisa el diff y resume.
- Briefs e informes entre agentes en modo caveman; código, rutas, comandos y errores exactos.
- En esta práctica el auditor revisa en especial: fuga de información de test a entrenamiento, uso del precio futuro en la decisión, semillas, y que el entorno base no se haya modificado.

Commits y PR:
- Prohibido atribuir nada a Claude o Anthropic (ni `Co-Authored-By`, ni pie "Generated with Claude Code").
- Mensaje en imperativo, explicable en una línea. Si no cabe, son dos commits.
- No se hace commit ni push sin que el usuario lo pida.

Comentarios en el código:
- Breves, estilo telegrama, solo para orientar o aclarar lo no obvio. No se comenta lo que el código ya dice.
- Docstrings de una o dos frases, sin secciones `Parameters` / `Returns`; los tipos van en la firma.

Redacción (comentarios, markdown, notebooks, PDF):
- Prohibida la primera persona: se usa la forma impersonal con "se" ("se entrena", "se observa").
- Ninguna alusión a la IA.
- Sin relleno, sin listas de emojis, sin tono de folleto.
- Prohibidos el guion largo y el semilargo; se usan paréntesis, comas o dos puntos.

Buenas prácticas:
- Type hints en funciones públicas.
- Sin rutas absolutas en el código: todo relativo a la raíz del repo o desde config.
- Umbrales como parámetros con valor por defecto explícito.
- Semillas fijas en todo lo aleatorio.
- Los tests no tocan la red; lo que descargue datos reales se prueba una vez a mano.
- Refactorizar cuando haga falta, sin inflar cambios pequeños.
- No se instala una dependencia sin decirlo antes.

## 9. Entorno local

- Python 3.14 por defecto (hay también 3.12 y 3.13). Instalados: numpy 2.4, pandas 3.0, matplotlib 3.11, scipy 1.18, tqdm, pytest 9.
- `gymnasium` 1.4.0 instalado (26/10/2026) en Python 3.14. Verificado: `SimpleTradingEnv` del enunciado corre un episodio aleatorio completo (494 pasos con `n_steps = 500`).
