# Dedo Verde — `agent-dedoverde`

---

## 1. Visão Geral

- Controle autônomo de uma estufa em **lugar inóspito**, que produz comida de verdade, sob incerteza climática.
- **Dois canteiros**: **milho** (calorias; bebe muito, gosta de calor) e **feijão-caupi** (proteína; aguenta seca, não tolera encharcamento).
- O ar é compartilhado: a temperatura precisa ser um meio-termo (as faixas ideais só se cruzam entre 24 e 28 °C).
- Água é o recurso escasso: tanque único e finito, recarregado só pela chuva. Na seca, o agente faz **triagem** entre os canteiros.

---

## 2. Modelagem do MDP

### 2.1 Ambiente
- Estufa **simulada**; 1 passo = 1 hora; episódio = 30 dias (720 passos), a fase de floração.
- **Consequências adiadas:** gastar água hoje pode matar uma cultura amanhã se não chover.
- Fatores exógenos (fora do controle do agente): temperatura externa, ciclo diário, **luz solar**, nuvens e chuva. Estiagens de vários dias surgem de um regime seco/úmido com persistência.
- O sol aquece a estufa e seca o substrato; as nuvens fecham antes da chuva, a luz cai e a estufa esfria. Energia e água ficam amarradas.

### 2.2 Estado Markoviano ($s_t$) — 11 valores, normalizados em [−1, 1]

| Variável | Sensor | Papel na Hipótese de Markov |
|---|---|---|
| Temperatura interna | termômetro | Estado físico a controlar |
| Umidade do substrato (por canteiro) | sensor de umidade do solo | Estado físico a controlar |
| Nível do tanque | boia/ultrassônico | Torna a escassez de água visível |
| Previsão de chuva (48 h) | serviço meteorológico | Resume a recarga futura (com ruído) |
| Temperatura externa | termômetro externo | Define a perda/ganho térmico |
| Luminosidade | piranômetro | Ganho térmico e evaporação |
| Hora do dia (seno/cosseno) | relógio | Ciclo diário sem salto entre 23h e 0h |
| Cultura viva/morta (por canteiro) | inspeção | Indica quais canteiros ainda contam |

- Morte instantânea ao passar do limite crítico: não há contador escondido de "horas sob estresse".

### 2.3 Ações discretas ($a_t$) — 36

Irrigação liga/desliga em cada canteiro (2 × 2) × aquecimento {0, 50, 100%} × ventilação {0, 50, 100%}.

| Ação | Efeito | Custo |
|---|---|---|
| Irrigação (por canteiro) | Sobe umidade do canteiro, esvazia tanque | Água |
| Aquecimento (estufa toda) | Sobe temperatura | Energia |
| Ventilação (estufa toda) | Aproxima temperatura da externa | Energia |

- Com N culturas: 2ᴺ × 9 ações (3 culturas = 72; 4 = 144).

### 2.4 Política de Recompensas ($r_t$)

| Componente | Valor |
|---|---|
| Cultura viva na faixa ideal | +0,5 por cultura (+1 com as duas) |
| Cultura viva fora da faixa ideal | de +0,3 a 0, caindo até a borda crítica (nunca negativo) |
| Gasto energético | −0,3 × aquecimento − 0,2 × ventilação (até −0,5) |
| Gasto hídrico | −0,1 a −1,5 (dinâmico: maior com o tanque baixo) |
| Morte de uma cultura | −100 por cultura; o episódio acaba quando as duas morrem |

- **Fator de desconto:** $\gamma = 0{,}99$ (horizonte ~100 passos ≈ 4 dias).
- **Guarda-corpos contra reward hacking:**
  - cultura viva, mesmo sofrendo, nunca rende menos que morta, e cada morte custa −100: "deixar morrer para parar de pagar" nunca compensa;
  - zona ideal ampla;
  - tanque no estado e na recompensa;
  - evaporação menor com substrato seco: economizar água é possível, ao custo de sair da faixa ideal.

---

## 3. Arquitetura do Agente (DQN)

- **Rede neural** $\hat Q(s,a;w)$: recebe o estado (11) e devolve um valor Q para cada uma das 36 ações.
- **Política:** $\epsilon$-gulosa sobre $\hat Q$.
- **Alvo:** $y = r + \gamma \max_{a'} \hat Q(s',a';w^-)$ (ou $y = r$ se o episódio terminou).
- **Repetição de experiência** e **rede-alvo** sincronizada a cada $C$ passos.

---

## 4. Roadmap

| Passo | Entrega | Status |
|---|---|---|
| 1 | 3 slides do MDP | Feito |
| 2 | Simulador + linhas de base (aleatória, termostato) | Feito (falta rodar no Colab) |
| 3 | Treinamento do DQN e gráficos | **Atual** (12/11) |

---

## 5. Decisões em aberto

- Hiperparâmetros do DQN: decaimento de $\epsilon$, tamanho do buffer, frequência $C$ da rede-alvo.
- Calibração fina dos custos se o DQN não superar o termostato.
