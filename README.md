# Métodos Tabulares - Reinforcement Learning

Implementación de algoritmos de reinforcement learning tabulares aplicados a diversos dominios.

## Estructura del Proyecto

```
.
├── Agents/                          # Implementación de algoritmos de RL
│   ├── qlearning.py                 # Q-learning estándar
│   ├── sarsa.py                     # SARSA y n-step SARSA
│   ├── rmax.py                      # R-Max (model-based)
│   ├── dynaq.py                     # Dyna-Q (planning + learning)
│   ├── qlearning_multigoal.py       # Q-learning con múltiples objetivos
│   ├── sarsa_multigoal.py           # SARSA con múltiples objetivos
│   └── qlearning_multiagent.py      # Q-learning multi-agente (centralizado/descentralizado)
│
├── Environments/                    # Ambientes de prueba
│   ├── AbstractEnv.py               # Interfaz base para ambientes
│   ├── GridEnv.py                   # Grilla base
│   │
│   ├── SimpleEnvs/                  # Ambientes completamente observables
│   │   ├── CliffEnv.py              # Cliff Walking (high variance)
│   │   └── EscapeRoomEnv.py         # Escape Room (exploración)
│   │
│   ├── MultiGoalEnvs/               # Ambientes con múltiples objetivos
│   │   └── RoomEnv.py               # Room environment con metas aleatorias
│   │
│   ├── PartiallyObservableEnvs/     # Ambientes con observación parcial (POMDP)
│   │   └── InvisibleDoorEnv.py      # Puerta invisible que requiere memoria
│   │
│   └── MultiAgentEnvs/              # Ambientes multi-agente
│       ├── AbstractMultiAgentEnv.py # Interfaz base multi-agente
│       ├── CentralizedHunterEnv.py  # Hunter cooperativo centralizado
│       ├── HunterEnv.py             # Hunter cooperativo descentralizado
│       └── HunterAndPreyEnv.py      # Hunter vs Prey competitivo
│
├── MemoryWrappers/                  # Wrappers de memoria para POMDPs
│   ├── KOrderMemory.py              # Buffer de últimas k observaciones
│   ├── BinaryMemory.py              # n bits controlables por el agente
│   └── SelectiveMemory.py           # Buffer con save/ignore (implementado por usuario)
│
├── MainSimpleEnvs.py                # Experimentos en ambientes simples
├── MainMultiGoal.py                 # Experimentos con múltiples objetivos
├── MainPartiallyObservable.py       # Experimentos con observación parcial
└── MainMultiAgent.py                # Experimentos multi-agente
```

## Instalación

Requiere Python 3.x con:
```bash
pip install numpy matplotlib
```

## Experimentos

### 1. Ambientes Simples (MainSimpleEnvs.py)

**Ejecutar:**
```bash
python MainSimpleEnvs.py
```

**Opciones:**
- `1`: Jugar manualmente CliffEnv
- `2`: Comparar Q-learning, SARSA y 4-step SARSA en CliffEnv (100 runs, 500 episodios)
- `3`: Comparar R-Max vs Dyna-Q en EscapeRoomEnv (5 runs, 20 episodios)

**Experimento 2 - CliffEnv:**
- **Algoritmos:** Q-learning, SARSA, 4-step SARSA
- **Parámetros:** γ=1.0, α=0.1, ε=0.1
- **Output:** `cliffenv_comparacion.png`

**Experimento 3 - EscapeRoomEnv:**
- **Algoritmos:** R-Max vs Dyna-Q con [0, 1, 10, 100, 1000, 10000] pasos de planning
- **Parámetros:** γ=1.0, α=0.5, ε=0.1
- **Output:** `rmax_dynaq_comparacion.png`

---

### 2. Múltiples Objetivos (MainMultiGoal.py)

**Ejecutar:**
```bash
python MainMultiGoal.py
```

**Opciones:**
- `1`: Jugar manualmente RoomEnv
- `2`: Comparar algoritmos estándar vs multi-objetivo (100 runs, 500 episodios)

**Experimento - RoomEnv:**
- **Algoritmos:** Q-learning, SARSA, 8-step SARSA, Q-learning MultiGoal, SARSA MultiGoal
- **Parámetros:** γ=0.99, α=0.1, ε=0.1, Q_init=1.0
- **Output:** `roomenv_comparacion.png`

---

### 3. Observación Parcial (MainPartiallyObservable.py)

**Ejecutar:**
```bash
python MainPartiallyObservable.py
```

**Opciones:**
- `1`: Jugar manualmente InvisibleDoorEnv (sin memoria)
- `2`: Jugar con K-order memory
- `3`: Jugar con Binary memory
- `4`: Jugar con Selective memory
- `5`: Experimento POMDP sin memoria (30 runs, 1000 episodios)
- `6`: Experimento POMDP con 2-order memory
- `7`: Experimento POMDP con 1-bit binary memory
- `8`: Experimento POMDP con selective memory (buffer=1)

**Parámetros comunes:** γ=0.99, α=0.1, ε=0.01, Q_init=1.0

**Outputs:**
- `pomdp_comparacion.png` - Sin memoria
- `pomdp_2order_memory_comparacion.png` - Con 2-order memory
- `pomdp_binary_memory_comparacion.png` - Con binary memory (1 bit)
- `pomdp_selective_memory_comparacion.png` - Con selective memory

**Nota:** Los experimentos con memoria modifican el espacio de acciones:
- K-order memory: mantiene el espacio de acciones original
- Binary memory: acciones = (acción_ambiente, valor_memoria)
- Selective memory: acciones = (acción_ambiente, "save"/"ignore")

---

### 4. Multi-Agente (MainMultiAgent.py)

**Ejecutar:**
```bash
python MainMultiAgent.py
```

**Opciones:**
- `1`: Jugar manualmente HunterAndPreyEnv
- `2`: Experimento Centralized Cooperative (30 runs, 50000 episodios)
- `3`: Experimento Decentralized Cooperative (30 runs, 50000 episodios)
- `4`: Experimento Decentralized Competitive (30 runs, 50000 episodios)
- `5`: Ejecutar todos los experimentos multi-agente

**Parámetros comunes:** γ=0.95, α=0.1, ε=0.1, Q_init=1.0

**Output:** `multiagent_comparacion.png`

**Tipos de configuraciones:**
- **Centralized Cooperative:** Un agente controla ambos hunters, espacio de acciones conjunto
- **Decentralized Cooperative:** Cada hunter aprende independientemente, observación local
- **Decentralized Competitive:** Hunters vs Prey, todos aprenden simultáneamente

---

## Notas sobre Reproducibilidad

- Todos los experimentos usan múltiples runs (5-100 dependiendo del experimento) para reportar estadísticas robustas
- Los gráficos muestran promedios y desviaciones estándar cuando aplica
- Los prints durante ejecución muestran:
  - Progreso cada 10 runs
  - Estadísticas de últimos episodios (min, max, avg, std)
  - Porcentaje de episodios largos (>1000 pasos) que pueden indicar timeouts
  - Resumen final con tabla comparativa

## Cómo Agregar un Nuevo Ambiente

1. Heredar de `AbstractEnv` o `GridEnv`
2. Implementar `reset()`, `step(action)`, `show()`
3. Definir `action_space` como property
4. Crear un nuevo Main* o agregar al existente

## Cómo Usar Memory Wrappers

```python
from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MemoryWrappers.KOrderMemory import KOrderMemory

# Ejemplo: 2-order memory
env = InvisibleDoorEnv()
env = KOrderMemory(env, memory_size=2)

# El agente ve tuplas de observaciones: (obs_t-1, obs_t)
state = env.reset()  # tuple de longitud 2
```
