# Dedo Verde (`agent-dedoverde`)

Uma estufa com dois canteiros (milho e feijão), simulada como ambiente Gymnasium, em que um agente DQN controla irrigação, aquecimento e ventilação com um tanque de água que só enche com chuva. Comparamos o DQN com um termostato (regra fixa) nos mesmos 100 climas de teste.

Trabalho da disciplina GBC063 – Inteligência Artificial (Faculdade de Computação, UFU).

**Página animada do episódio de seca:** https://gabrielhca.github.io/agent-dedoverde/estufa/

## Como rodar

Os notebooks rodam no **Google Colab**. A primeira célula de cada um clona este repositório e instala as bibliotecas. Abra-os nesta ordem:

1. `notebooks/01_modelagem.ipynb`: o simulador (parâmetros, clima, ambiente) e as linhas de base (aleatória e termostato).
2. `notebooks/02_dqn.ipynb`: treino do DQN, curva de aprendizado, comparação com as linhas de base e o episódio de seca. Salva e carrega o modelo no Google Drive (`MyDrive/dedoverde/1000k_s4242`). Se o modelo já estiver lá, ele é carregado em vez de treinar de novo.
3. `notebooks/03_visualizacao.ipynb`: gera o episódio de seca hora a hora e monta a página animada.

Para gerar o episódio e a página na sua máquina (em CPU, sem GPU):

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install gymnasium stable-baselines3 matplotlib
uv run python scripts/gerar_episodios.py   # -> resultados/episodios/episodio_1.json
uv run python scripts/montar_pagina.py     # -> docs/estufa/index.html
```

Para ver a página localmente, abra `docs/estufa/index.html` no navegador.

## Estrutura

```text
agent-dedoverde/
├── estufa.py                  # ambiente EstufaEnv, clima e políticas de base (importado pelos notebooks)
├── notebooks/
│   ├── 01_modelagem.ipynb
│   ├── 02_dqn.ipynb
│   └── 03_visualizacao.ipynb
├── scripts/
│   ├── gerar_episodios.py     # roda o episódio de seca com termostato e DQN e grava o JSON
│   └── montar_pagina.py       # põe o JSON dentro do modelo da página
├── resultados/
│   ├── 1000k_s4242/           # treino atual: 1 milhão de passos, semente 4242
│   └── episodios/             # episódio de seca hora a hora (JSON)
├── web/
│   └── estufa_modelo.html     # modelo da página, com o marcador __DADOS__
└── docs/
    ├── project.md             # escopo e modelagem do MDP
    ├── slide1.pdf             # 1ª apresentação
    └── estufa/index.html      # página animada (gerada)
```

## Resultados

Os resultados atuais estão em `resultados/1000k_s4242/`:

- `dqn_final.zip`: o modelo do fim do treino (stable-baselines3);
- `hiper.json`: hiperparâmetros, número de passos, semente e tempo de treino;
- `treino.monitor.csv` e `curva_aprendizado.png`: retorno e duração de cada episódio de treino;
- `comparacao.csv`: retorno, água e sobrevivência de cada política em cada um dos 100 climas de teste (sementes 0–99).

Resumo da comparação (100 climas de teste):

| Política | Retorno médio | Água (L) | Milho sobrevive | Feijão sobrevive |
|---|---|---|---|---|
| Aleatória | −203,0 | 224 | 0% | 0% |
| Termostato | 424,7 | 2704 | 68% | 75% |
| DQN | 329,6 | 2652 | 46% | 68% |

O DQN aprendeu (saiu do nível da política aleatória), mas não superou o termostato.

## Tecnologias

- Python, NumPy, pandas e Matplotlib
- Gymnasium (ambiente)
- stable-baselines3 e PyTorch (DQN)
- Google Colab e Google Drive (treino)
- HTML, CSS e JavaScript (página animada)

## O que aprendi

- Modelar um problema como MDP muda tudo o que vem depois. Na primeira versão, uma planta viva mas fora da faixa rendia menos que uma planta morta, então o agente podia "preferir" deixá-la morrer. Tive que repensar a recompensa para que viver sempre valesse mais.
- O estado precisa ter tudo o que o agente usa para decidir. Um contador de "horas sob estresse" escondido quebraria a hipótese de Markov, por isso a morte é instantânea.
- Treinar um DQN não garante resultado melhor que uma regra simples. O termostato continuou ganhando, e entender o porquê foi tão importante quanto treinar.
- Para comparar políticas de forma justa, todas precisam enfrentar os mesmos climas (mesmas sementes), e os climas de teste não podem ser usados no treino.
- Separar o código do simulador (`estufa.py`) dos notebooks facilitou reaproveitar o mesmo ambiente no treino, na avaliação e na geração da página.
