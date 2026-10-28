# B5 Aprendizaje por Refuerzo: taller trader

Agentes tabulares (Q-learning y Sarsa, más Monte Carlo como extensión) que aprenden a invertir en un activo simulado con `SimpleTradingEnv` del enunciado (`docs/taller_trader.ipynb`, no versionado). Resultados y análisis en `report/informe.pdf` (dos páginas).

## Instalación

Python 3.10 o superior y las dependencias de `requirements.txt`:

```bash
pip install -r requirements.txt
```

`gymnasium` 1.4 requiere Python 3.11 o superior; con 3.10 se usa `gymnasium` 1.2.x (fijado en `requirements.txt`).

## Estructura

```
src/rl_trader/   env, prices, discretize, agents, baselines, train, evaluate,
                 experiments, tuning, sweep, env_variants, figures
scripts/         tune.py, sweep.py, run_experiments.py, run_extensions.py, make_figures.py
config/          default.json (hiperparámetros, semillas, criterios de convergencia)
notebooks/       01_resultados.ipynb (tablas y figuras desde results/)
results/         tuning/ (validación), final/ (test), ext/ (extensiones)
report/          informe.html y informe.pdf, figures/
tests/           pytest, sin red
```

## Reproducir

Todos los comandos se lanzan desde la raíz del repo. Las series de entrenamiento, validación y test usan rangos de semillas disjuntos (`src/rl_trader/prices.py`); los hiperparámetros se eligen solo en validación y el test solo se usa para informar.

```bash
python scripts/tune.py
```

Rejilla de α, γ y decaimiento de ε (3000 episodios, 3 semillas). Escribe `results/tuning/best_params.json`.

```bash
python scripts/sweep.py
```

Barrido de `epsilon_min` y del número de episodios en torno al óptimo anterior. Escribe `results/tuning/sweep_best.json`.

```bash
python scripts/run_experiments.py --params results/tuning/sweep_best.json --out results/final
```

Entrenamiento final (5 semillas) y evaluación en 100 series de test, con los baselines aleatorio y buy and hold. El número de episodios de cada agente sale de `sweep_best.json`; `n_episodes` de `config/default.json` solo se usa si no se pasa `--params`.

```bash
python scripts/run_extensions.py bins --value 5
python scripts/run_extensions.py cost --value 0.005
python scripts/run_extensions.py mc
python scripts/run_extensions.py mixed
```

Extensiones: número de tramos de la discretización, comisión, Monte Carlo on-policy y off-policy, y entrenamiento mezclando generadores. Repetir `bins` con 20 y `cost` con 0 para los demás puntos de la sensibilidad. Salida en `results/ext/`.

```bash
python scripts/make_figures.py
```

Figuras y tablas en `report/figures/`.

## Tests

```bash
python -m pytest -q
```

## Notas

- El entorno base no se modifica. La variante con comisión configurable (`env_variants.py`) solo se usa en las extensiones.
- Los tiempos de entrenamiento dependen de la máquina y de cuántos procesos corran a la vez.
