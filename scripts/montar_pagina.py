"""Monta a página animada da estufa a partir do modelo e do episódio gerado.

Lê web/estufa_modelo.html, troca o marcador __DADOS__ pelo conteúdo de
resultados/episodios/episodio_1.json e grava docs/estufa/index.html.
O resultado é um arquivo único, que abre direto no navegador.

Uso (na raiz do repositório, depois de scripts/gerar_episodios.py):
    python scripts/montar_pagina.py
"""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQ_MODELO = RAIZ / "web" / "estufa_modelo.html"
ARQ_DADOS = RAIZ / "resultados" / "episodios" / "episodio_1.json"
ARQ_SAIDA = RAIZ / "docs" / "estufa" / "index.html"
MARCADOR = "__DADOS__"


def main():
    modelo = ARQ_MODELO.read_text(encoding="utf-8")
    dados = ARQ_DADOS.read_text(encoding="utf-8").strip()
    json.loads(dados)  # só confere que o arquivo é um JSON válido
    if modelo.count(MARCADOR) != 1:
        raise SystemExit(
            f"esperava o marcador {MARCADOR} uma única vez em {ARQ_MODELO.name}"
        )
    ARQ_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    ARQ_SAIDA.write_text(modelo.replace(MARCADOR, dados), encoding="utf-8")
    print(
        f"gravado: {ARQ_SAIDA.relative_to(RAIZ)} ({ARQ_SAIDA.stat().st_size / 1024:.0f} KB)"
    )


if __name__ == "__main__":
    main()
