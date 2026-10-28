# Dedo Verde (`agent-dedoverde`)

**Controle autônomo de estufa inteligente via Aprendizagem por Reforço**

Projeto da disciplina GBC063 - Inteligência Artificial (Faculdade de Computação, Universidade Federal de Uberlândia), ministrada pelo Prof. Marcelo Keese Albertini.

## Visão Geral

O Dedo Verde modela o controle climático e de irrigação de uma estufa como um problema de decisão sequencial sob incerteza. O agente deve manter temperatura e umidade na faixa ideal lidando com recursos escassos (tanque de água finito, recarregado apenas pela chuva) e consequências adiadas (gastar água hoje pode comprometer a planta amanhã se não chover).

## Modelagem (MDP)

O problema é formulado para ser resolvido com a arquitetura Ator-Crítico e ações contínuas.

| Elemento | Definição |
|---|---|
| Ambiente | Estufa simulada com dinâmica estocástica (clima e chuva) |
| Estado $s_t$ | Temperatura interna, umidade interna, nível do tanque, previsão de chuva, temperatura externa, hora do dia |
| Ação $a_t$ | Vetor contínuo em $[0,1]^3$: irrigação, aquecimento, exaustão/ventilação |
| Recompensa $r_t$ | Sem penalidade na faixa ideal; custo proporcional a energia e água (maior com o tanque baixo); penalidade terminal severa em extremos letais |

O estado foi projetado para satisfazer a Hipótese de Markov. Detalhes e decisões em aberto estão em [`docs/project.md`](docs/project.md).

## Roadmap

- [ ] Etapa 1 (atual): modelagem do MDP e apresentação em 3 slides
- [ ] Etapa 2: implementação do simulador
- [ ] Etapa 3: treinamento do agente (Ator-Crítico)

## Estrutura do Repositório

```text
agent-dedoverde/
├── docs/
│   └── project.md    # escopo, modelagem do MDP e roadmap
├── src/              # (futuro) simulador e agente
└── README.md
```