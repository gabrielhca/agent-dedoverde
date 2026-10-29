# Dedo Verde (`agent-dedoverde`)

**Controle autônomo de estufa inteligente via Aprendizagem por Reforço (DQN)**

Projeto da disciplina GBC063 - Inteligência Artificial (Faculdade de Computação, Universidade Federal de Uberlândia), ministrada pelo Prof. Marcelo Keese Albertini.

## Visão Geral

O Dedo Verde modela o controle climático e de irrigação de uma estufa como um problema de decisão sequencial sob incerteza. O agente deve manter temperatura e umidade na faixa ideal lidando com recursos escassos (tanque de água finito, recarregado apenas pela chuva) e consequências adiadas (gastar água hoje pode comprometer a planta amanhã se não chover).

## Modelagem (MDP)

O problema é formulado para ser resolvido com Deep Q-Network (DQN) e ações discretas.

| Elemento | Definição |
|---|---|
| Ambiente | Estufa simulada com dinâmica estocástica (clima e chuva) |
| Estado $s_t$ | Temperatura interna, umidade interna, nível do tanque, previsão de chuva, temperatura externa, hora do dia (seno e cosseno) |
| Ação $a_t$ | Irrigação, aquecimento e ventilação, cada um em {0%, 50%, 100%} (27 combinações) |
| Recompensa $r_t$ | +1,0 por passo na faixa ideal; penalidade crescente fora dela; custo de energia e de água (maior com o tanque baixo); terminal letal de −100 |
| Desconto $\gamma$ | 0,99 (horizonte efetivo de cerca de 100 passos) |

O estado foi projetado para satisfazer a Hipótese de Markov. Detalhes e decisões em aberto estão em [`docs/project.md`](docs/project.md).

## Roadmap

- [ ] Etapa 1 (atual): definição do ambiente e agente
- [ ] Etapa 2: implementação do simulador
- [ ] Etapa 3: treinamento do agente (DQN)

## Estrutura do Repositório

```text
agent-dedoverde/
├── docs/
│   └── project.md    # escopo, modelagem do MDP e roadmap
├── src/              # (futuro) simulador e agente
└── README.md
```
