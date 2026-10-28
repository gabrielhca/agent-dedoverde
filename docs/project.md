# Dedo Verde — `agent-dedoverde`

---

## 1. Visão Geral

- Controle autônomo de uma estufa inteligente (clima + irrigação) sob incerteza climática.
- Água é o recurso escasso: tanque finito, recarregado só pela chuva.

---

## 2. Modelagem do MDP

### 2.1 Ambiente
- Estufa **simulada**, passo de tempo discreto.
- **Consequências adiadas:** gastar água hoje pode causar a morte da planta amanhã se não chover.
- Fatores exógenos (fora do controle do agente): temperatura externa, ciclo diário, chuva.

### 2.2 Estado Markoviano ($s_t$)

| Variável | Papel na Hipótese de Markov |
|---|---|
| Temperatura interna | Estado físico a controlar |
| Umidade interna | Estado físico a controlar |
| Nível do tanque | Torna a escassez de água visível ao agente |
| Sinal de previsão de chuva | Resume o que o passado diz sobre a recarga futura |
| Temperatura externa | Define a perda/ganho térmico |
| Hora do dia (seno/cosseno) | Captura o ciclo diário; mantém a dinâmica estacionária |

### 2.3 Ações contínuas ($a_t \in [0,1]^3$, i.e. 0–100%)

| Ação | Efeito | Custo |
|---|---|---|
| Irrigação | Sobe umidade, esvazia tanque | Água |
| Aquecimento | Sobe temperatura | Energia |
| Exaustão/Ventilação | Aproxima temperatura da externa | Energia |

- **Por que contínuas:** o Ator-Crítico gera a ação direto da política, sem `argmax` sobre Q a cada passo.

### 2.4 Política de Recompensas ($r_t$) — diretrizes anti-Goodhart

- **Recompensar estabilidade:** zero de penalidade dentro da faixa ideal de temperatura e umidade; penalidade crescente fora dela.
- **Penalizar energia:** custo proporcional à potência usada.
- **Penalizar água:** custo proporcional ao uso, **maior quando o tanque está baixo**.
- **Penalidade terminal severa:** extremos letais (morte da planta) encerram o episódio.
- **Guarda-corpos contra reward hacking:**
  - penalidade terminal grande o bastante para que "morrer cedo" nunca compense;
  - zona ideal ampla, para o agente não colar na borda da faixa;
  - tanque no estado e na recompensa, para não esvaziá-lo por conforto imediato.

---

## 3. Arquitetura do Agente

- **Ator** ($\pi_\theta$): política estocástica contínua que escolhe irrigação, aquecimento e ventilação a partir de $s_t$; atualizada pela **vantagem** (ações melhores que a média do estado ganham probabilidade).
- **Crítico** ($\hat V(s;w)$): avalia a qualidade do estado atual (retorno esperado) e fornece o sinal de vantagem ao Ator (erro TD / GAE).

---

## 4. Roadmap

| Passo | Entrega | Status |
|---|---|---|
| 1 | Estruturar os **3 slides** de apresentação do MDP | **Atual** |
| 2 | Implementação do simulador | Futuro |
| 3 | Treinamento do agente | Futuro |

---

## 5. Decisões em aberto

- Faixas ideais/críticas (depende da cultura).
- Pesos da recompensa e valor da penalidade terminal.
- Fator de desconto $\gamma$ e duração do episódio.