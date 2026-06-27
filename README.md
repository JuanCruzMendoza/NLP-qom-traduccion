# Análisis de resultados — TP Temas de NLP (Grupo 6)

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
  todas las variables; en C (Direct Assessment 0–20) se reporta además `interval`.
  Matrices de confusión por coincidencia de pares de anotadores.
- **BLEU**: sacreBLEU 2.6.0 (corpus, tokenizer `13a`; oración con smoothing `exp`).
- **BERTScore**: BETO (`dccuchile/bert-base-spanish-wwm-cased`, capa 9, sin
  reescalado). Descarga pesos de HuggingFace en la primera ejecución.
- En Windows, ejecutar con `PYTHONIOENCODING=utf-8` para imprimir el texto qom.
