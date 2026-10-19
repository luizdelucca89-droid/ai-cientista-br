"""Biblioteca de exemplos few-shot para calibrar os agentes.

Portado do AI Scientist v1 (``ai_scientist/fewshot_examples/``), onde exemplos
reais de pareceres são injetados no prompt do revisor para calibrar o nível de
rigor e o formato da saída. Aqui os exemplos são escritos em português e no
estilo de um periódico brasileiro, e servem para:

* **revisão** -- mostrar ao revisor o nível de detalhe esperado;
* **hipóteses** -- mostrar o formato de uma hipótese testável bem formulada.

Os exemplos são texto fixo e curto: não entram no cache como dados variáveis e
não alteram o determinismo do pipeline.
"""

from __future__ import annotations

# Exemplo de parecer calibrado: aponta problemas verificáveis, com severidade e
# recomendação acionável, sem elogio vazio nem crítica genérica.
REVIEW_FEWSHOT = """EXEMPLO DE PARECER BEM CALIBRADO (use como referência de rigor e formato):

{
  "parecer": "revisões maiores",
  "pontos_fortes": [
    "O protocolo é reprodutível: semente fixa, versões de bibliotecas e artefatos em JSON.",
    "A comparação contra linha de base estatística é feita no mesmo particionamento."
  ],
  "pontos_fracos": [
    "A base é sintética, o que impede qualquer afirmação sobre o fenômeno real investigado.",
    "Não há análise de poder; com n pequeno, o intervalo de confiança do F1 é largo."
  ],
  "recomendacoes": [
    "Replicar o protocolo em uma base pública brasileira e reportar o mesmo conjunto de métricas.",
    "Adicionar validação cruzada temporal, já que as observações são autocorrelacionadas."
  ],
  "severidade": "alta",
  "erros_metodologicos": [
    "O vazamento entre treino e teste não é descartado: as features foram selecionadas no conjunto completo."
  ],
  "vieses_detectados": [
    "Viés de seleção na escolha das séries, que não é justificada por critério pré-registrado."
  ]
}

Note o padrão: cada crítica é específica, verificável no texto e acompanhada de uma ação.
Evite críticas genéricas como "o artigo precisa de mais experimentos"."""


HYPOTHESIS_FEWSHOT = """EXEMPLO DE HIPÓTESE BEM FORMULADA (use como referência de formato):

{
  "id": "H1",
  "enunciado": "Um modelo não linear supera a linha de base sazonal em AUC ao prever
  semanas epidêmicas, com janela de previsão de quatro semanas.",
  "justificativa": "Interações entre temperatura, precipitação e acumulado de casos
  produzem limiares não lineares que um modelo de média móvel não representa.",
  "testavel": true,
  "variaveis": {
    "independentes": ["temperatura_media", "precipitacao_acumulada", "casos_defasados"],
    "dependentes": ["semana_epidemica"]
  },
  "metrica": "auc_roc",
  "risco": "medio"
}

Note o padrão: o enunciado nomeia a comparação, a métrica e a condição de teste,
de modo que o experimento possa falsificá-lo."""


def review_fewshot() -> str:
    """Exemplos de calibração para o agente revisor."""
    return REVIEW_FEWSHOT


def hypothesis_fewshot() -> str:
    """Exemplos de calibração para o agente planejador."""
    return HYPOTHESIS_FEWSHOT
