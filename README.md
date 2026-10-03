# Análisis de resultados

Evaluamos la calidad de la traducción automática **qom→español**, una
lengua indígena de bajos recursos, sobre el corpus QomL'aqtaqa, contrastando la
evaluación humana con las métricas automáticas. El trabajo abarcó la anotación
manual de la calidad de las traducciones por los tres integrantes, la medición del
acuerdo entre anotadores (IAA, Alpha de Krippendorff), el diseño y refinamiento de
una guía de anotación propia, el cálculo de métricas automáticas (BLEU y
BERTScore) y el análisis de su correlación con los juicios humanos. El informe está
en `informe/` y el código que reproduce los números, en `scripts/`.

Reproduce el IAA, las métricas automáticas y la comparación humano–métrica del
conjunto de datasets anotados (`../datasets_anotados/`).

## Uso

```bash
pip install -r requirements.txt
python analisis_resultados.py   # IAA, BLEU, BERTScore, correlaciones
python figuras.py               # figuras del informe
```

- `analisis_resultados.py` escribe `../resultados/resultados.json` (fuente de
  verdad de todos los números del informe) y `../resultados/por_segmento_C.csv`.
- `figuras.py` escribe las figuras en `../informe/figuras/`.

## Notas de implementación

- **IAA**: Alpha de Krippendorff (librería `krippendorff`). Escala ordinal para
  todas las variables. 
- **BLEU**: sacreBLEU 2.6.0 (corpus, tokenizer `13a`; oración con smoothing `exp`).
- **BERTScore**: mBERT (`bert-base-multilingual-cased`, `lang=es`, reescalado
  con baseline oficial). Descarga pesos de HuggingFace en la primera ejecución.
- En Windows, ejecutar con `PYTHONIOENCODING=utf-8` para imprimir el texto qom.
