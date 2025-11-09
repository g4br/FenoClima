# FenoClima

Os códigos apresentados aqui estão em metodologia GO HORSE, com caráter de playgroud e desenvolvimento continuado. No atual ponto não apresentam maturidade operacional. 

## Pipeline de fenologia e clima em nível municipal:
- Coleta de séries no Google Earth Engine/BigQuery,
- Extração de métricas fenológicas (adaptação do TIMESAT),
- Indicadores climáticos,
- Rede neural híbrida para prever produtividade (desvio da tendência tecnológica). 

## Estrutura do repositório
- big_query-gee.py — Cálculo e extração do NDVI em agregado municipal + máscara de soja (MapBiomas) + máscara de nuvens + tipo de solo no GEE (BigQuery). 
- timesat.py — Detecção das métricas fenológicas a partir da série temporal de NDVI
- clima_timesat.py — Calculo dos parâmetros climáticos de acordo com o estádio fenológico, datas e municipio
- FenoClima.py — Orquestra o fluxo de features e treina a rede de DL híbrida. 
- diagram.png — Diagrama do fluxo da RNA. 
- terminal-print.txt — Saída do console, ao final do treino.
- resultado-dados-test.csv — Resultado do modelo treinado para os dados no conjunto teste [ não conhecidos pela RNA ]

## [ Extra ] Code GEE
code: https://code.earthengine.google.com/c078670b4d00f5614b5bf91921ca15eb
app: https://ee-gabrielluanrodrigues.projects.earthengine.app/view/ndvi-mapbiomas-nuvens-mask
