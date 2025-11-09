# FenoClima

## Pipeline de fenologia e clima em nível municipal:
- Coleta de séries no Google Earth Engine/BigQuery,
- Extração de métricas fenológicas (adaptação do TIMESAT),
- Indicadores climáticos,
- Rede neural híbrida para prever produtividade (desvio da tendência tecnológica). 

## Estrutura do repositório
- big_query-gee.py — coleta/consulta no GEE/BigQuery. 
- timesat.py — funções de apoio para a etapa fenológica. 
- clima_timesat.py — métricas fenológicas e suavização estilo TIMESAT. 
- FenoClima.py — orquestra o fluxo de features e treina a rede de DL híbrida. 
- diagram.png — diagrama do fluxo da RNA. 
- terminal-print.txt — exemplo de saída em console. 
