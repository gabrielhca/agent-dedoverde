"""Gera o episódio de seca (semente 1) hora a hora, para a página animada.

Roda o mesmo clima com duas políticas, o termostato e o DQN treinado, e grava
tudo em resultados/episodios/episodio_1.json. Para o DQN, guarda também os
36 valores Q que a rede calculou antes de cada decisão.

Uso (na raiz do repositório, em CPU):
    python scripts/gerar_episodios.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from estufa import EstufaEnv, politica_termostato  # noqa: E402

SEED_SECA = 1  # a mesma semente da seção 10 do notebook 02_dqn
ARQ_MODELO = RAIZ / "resultados" / "1000k_s4242" / "dqn_final.zip"
ARQ_SAIDA = RAIZ / "resultados" / "episodios" / f"episodio_{SEED_SECA}.json"

COLS = ["t", "T_in", "T_ext", "M_milho", "M_caupi", "tanque", "viva_milho", "viva_caupi",
        "sol", "chuva", "prev", "irr_milho", "irr_caupi", "aquec", "vent", "r", "G"]


def rodar_episodio(env, politica, modelo=None):
    """Roda um episódio na semente da seca e devolve as linhas hora a hora.

    Se `modelo` for dado, guarda também os 36 valores Q de cada decisão e
    confere que a ação escolhida é a de maior Q.
    """
    obs, info = env.reset(seed=SEED_SECA)
    linhas, qs, mortes = [], [], []
    G, done = 0.0, False
    while not done:
        a = politica(env, obs, info)
        if modelo is not None:
            with torch.no_grad():
                q = modelo.q_net(torch.as_tensor(obs).unsqueeze(0)).squeeze(0)
            assert int(q.argmax()) == a, f"t={info['t']}: argmax Q = {int(q.argmax())}, predict = {a}"
            qs.append([round(float(x), 3) for x in q])
        obs, r, term, trunc, info = env.step(a)
        G += r
        done = term or trunc
        irr, aquec, vent = env.decodificar(a)
        linhas.append([
            info["t"], round(info["T_in"], 2), round(info["T_ext"], 2),
            round(float(info["M"][0]), 2), round(float(info["M"][1]), 2), round(info["tanque"], 1),
            int(info["viva"][0]), int(info["viva"][1]),
            round(info["sol"], 3), round(info["chuva"], 2), round(float(info["prev"]), 3),
            irr[0], irr[1], aquec, vent, round(r, 4), round(G, 3),
        ])
        mortes += [f"{nome} no dia {info['t'] / 24:.1f}" for nome in info["mortes"]]
    return linhas, (qs if modelo is not None else None), dict(retorno=G, agua=info["agua_gasta"], mortes=mortes)


def main():
    env = EstufaEnv()
    modelo = DQN.load(ARQ_MODELO, device="cpu")

    def politica_dqn(env, obs, info):
        return int(modelo.predict(obs, deterministic=True)[0])

    politicas = [
        ("Termostato",
         "Regra fixa: mira 26 °C e rega quando a umidade chega perto do limite de baixo da faixa ideal. "
         "Não olha o tanque nem a previsão.",
         politica_termostato, None),
        ("DQN",
         "Modelo do fim do treino (1 milhão de passos). Escolhe sempre a ação de maior Q.",
         politica_dqn, modelo),
    ]

    eps = []
    for nome, desc, politica, mod in politicas:
        linhas, qs, res = rodar_episodio(env, politica, mod)
        eps.append(dict(nome=nome, desc=desc, h=linhas, q=qs))
        print(f"{nome}: retorno {res['retorno']:.1f} | água {res['agua']:.0f} L | "
              f"{len(linhas)} horas | mortes: {', '.join(res['mortes']) or 'nenhuma'}")

    dados = dict(
        episodio=SEED_SECA,
        culturas=[dict(nome=k["nome"], T_ideal=list(k["T_ideal"]), T_crit=list(k["T_crit"]),
                       M_ideal=list(k["M_ideal"]), M_crit=list(k["M_crit"])) for k in env.culturas],
        cols=COLS,
        eps=eps,
    )
    ARQ_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, separators=(",", ":"))
    print("gravado:", ARQ_SAIDA.relative_to(RAIZ))


if __name__ == "__main__":
    main()
