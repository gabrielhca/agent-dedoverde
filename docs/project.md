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
| Hora do dia (seno/cosseno) | Captura o ciclo diário; evita o salto entre 23h e 0h |

- Estado contínuo (7 valores numéricos): por isso a tabela Q não serve e usamos uma rede neural.

### 2.3 Ações discretas ($a_t$)

Cada atuador opera em três níveis: desligado (0%), médio (50%) ou máximo (100%). São 3 × 3 × 3 = 27 combinações.

| Ação | Efeito | Custo |
|---|---|---|
| Irrigação | Sobe umidade, esvazia tanque | Água |
| Aquecimento | Sobe temperatura | Energia |
| Ventilação | Aproxima temperatura da externa (troca térmica) | Energia |

- **Por que discretas:** o DQN escolhe a ação por `argmax` sobre Q, o que exige um conjunto finito de ações.

### 2.4 Política de Recompensas ($r_t$) — diretrizes anti-reward hacking

| Componente | Valor |
|---|---|
| Manutenção na faixa ideal | +1,0 por passo |
| Fora da faixa ideal | -0,2 a -2,0 (cresce com o afastamento) |
| Gasto energético | -0,1 a -0,5 (proporcional ao nível acionado) |
| Gasto hídrico | -0,1 a -1,5 (dinâmico: maior com o tanque baixo) |
| Estado terminal letal | -100 e fim do episódio |

- **Fator de desconto:** $\gamma = 0{,}99$, horizonte efetivo de cerca de 100 passos ($1/(1-\gamma)$).
- **Guarda-corpos contra reward hacking:**
  - penalidade terminal maior que a soma descontada dos custos de sobrevivência, para que "morrer cedo" nunca compense (exige custo por passo normalizado, em torno de 1 ou menos);
  - zona ideal ampla, para o agente não colar na borda da faixa;
  - tanque no estado e na recompensa, para não esvaziá-lo por conforto imediato.
- Os valores são ponto de partida e serão calibrados no simulador.

---

## 3. Arquitetura do Agente (DQN)

- **Rede neural** $\hat Q(s,a;w)$: recebe o estado e devolve um valor Q para cada uma das 27 ações.
- **Política:** $\epsilon$-gulosa sobre $\hat Q$ (argmax na maior parte das vezes, ação aleatória com probabilidade $\epsilon$).
- **Alvo:** $y = r + \gamma \max_{a'} \hat Q(s',a';w^-)$ (ou $y = r$ se o episódio terminou).
- **Repetição de experiência:** buffer de transições, treino com minibatches aleatórios.
- **Rede-alvo:** cópia congelada dos pesos ($w^-$), sincronizada a cada $C$ passos.

---

## 4. Roadmap

| Passo | Entrega | Status |
|---|---|---|
| 1 | Estruturar os **3 slides** de apresentação do MDP | **Atual** |
| 2 | Implementação do simulador | Futuro |
| 3 | Treinamento do agente (DQN) | Futuro |

---

## 5. Decisões em aberto

- Faixas ideais/críticas (depende da cultura).
- Calibração dos pesos da recompensa e do valor da penalidade terminal.
- Passo de tempo (sugestão: 1 hora) e duração do episódio (exemplo: 30 dias).
- Hiperparâmetros do DQN: decaimento de $\epsilon$, tamanho do buffer, frequência $C$ da rede-alvo.
