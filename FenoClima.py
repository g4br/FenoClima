import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
import glob
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from sklearn.isotonic import IsotonicRegression
import math

pd.set_option('display.max_rows', 500)

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================
TARGET = 'prod'
ID_MUNICIPIO = 'cod'
ID_ANO = 'ano'
ANOS_TESTE = {2005, 2012, 2017, 2023, 2024}

# ==============================================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================

path_csv = '/home/admin2/datain/complete_timesat'
files = glob.glob(f'{path_csv}/*.csv')

gdf = gpd.read_file('/home/admin2/datain/complete_timesat/BR_Municipios_2024.zip')
gdf_coords = gdf.copy()
gdf_coords['lon'] = gdf_coords.geometry.centroid.x
gdf_coords['lat'] = gdf_coords.geometry.centroid.y

DF = []
for f in files:
    try:
        df = pd.read_csv(f, index_col=0)
        df['cod'] = Path(f).stem
        DF.append(df)
    except Exception as e:
        print(f"Erro ao ler {f}: {e}")

DF = pd.concat(DF, axis=0)
DF = DF[DF['ERROR'] == 0]
DF.drop(['ERROR', 'ER_DISCR'], axis=1, inplace=True, errors='ignore')
DF.set_index('ano', inplace=True)
DF = DF.sort_index()
DF.reset_index(inplace=True)

DF = DF.merge(
    gdf_coords[['CD_MUN', 'lon', 'lat']],
    left_on='cod',
    right_on='CD_MUN',
    how='left'
).drop('CD_MUN', axis=1)

DF['MOSL'] = (pd.to_datetime(DF['MOS_DT2']) - pd.to_datetime(DF['MOS_DT1'])).dt.days.astype(int)

# convertendo as datas para dayofyear
datas_cols = ['data_ini','data_max', 'data_fim','ROI_DT','ROD_DT','MOS_DT1', 'MOS_DT2']
DF[datas_cols] = DF[datas_cols].apply(lambda s: pd.to_datetime(s, errors='coerce').dt.dayofyear.astype('Int16'))

DF.drop('safra_temporada',axis=1,inplace=True, errors='ignore')
original_features = list(DF.columns)

# ==============================================================================
# APLICAR FILTROS SEQUENCIAIS
# ==============================================================================

print("🔍 ESTADO INICIAL DO DATAFRAME:")
print(f"    Total de registros: {len(DF)}")
print(f"    Municípios únicos: {DF['cod'].nunique()}")
print(f"    Período: {DF['ano'].min()} a {DF['ano'].max()}")

# 1. Primeiro filtro: remover duplicatas priorizando menor data_max
print(f"\n🎯 APLICANDO FILTRO 1: Remover registros duplicados (menor data_max)")
registros_antes_dup = len(DF)
duplicatas_antes = DF.duplicated(subset=['cod', 'ano']).sum()

# Ordenar por cod, ano e data_max (ascendente para pegar a menor data_max primeiro)
DF = DF.sort_values(['cod', 'ano', 'data_max'])

# Remover duplicatas, mantendo a primeira ocorrência (que terá a menor data_max)
DF = DF.drop_duplicates(subset=['cod', 'ano'], keep='first')
DF.reset_index(drop=True, inplace=True)

duplicatas_removidas = registros_antes_dup - len(DF)
print(f"    Registros duplicados encontrados: {duplicatas_antes}")
print(f"    Registros removidos: {duplicatas_removidas}")
print(f"    Após filtro 1 - Registros: {len(DF)}, Municípios: {DF['cod'].nunique()}")

# 2. Segundo filtro: remover municípios com menos de 3 registros
print(f"\n🎯 APLICANDO FILTRO 2: Remover municípios com < 5 registros (até 2022)")

# Contar registros por município
contagem_por_municipio = DF[DF['ano'] <= 2022].groupby('cod').size()
municipios_validos = contagem_por_municipio[contagem_por_municipio >= 5].index
municipios_insuficientes = contagem_por_municipio[contagem_por_municipio < 5]

print(f"    Municípios com dados insuficientes (<5 registros): {len(municipios_insuficientes)}")
print(f"    Municípios válidos (≥5 registros): {len(municipios_validos)}")

if len(municipios_insuficientes) > 0:
    print(f"    🗑️  Removendo municípios: {municipios_insuficientes.index.tolist()}")
    
    # Aplicar filtro
    registros_antes = len(DF)
    DF = DF[DF['cod'].isin(municipios_validos)].copy()
    DF.reset_index(drop=True, inplace=True)
    
    print(f"    Registros removidos: {registros_antes - len(DF)}")
    print(f"    Registros mantidos: {len(DF)}")
else:
    print(f"    ✅ Nenhum município com dados insuficientes encontrado")

print(f"    Após filtro 2 - Registros: {len(DF)}, Municípios: {DF['cod'].nunique()}")

# 3. Verificação final da estrutura
print(f"\n✅ ESTRUTURA FINAL APÓS FILTROS:")
print(f"    Total de registros: {len(DF)}")
print(f"    Municípios únicos: {DF['cod'].nunique()}")
print(f"    Período: {DF['ano'].min()} a {DF['ano'].max()}")

# Verificar distribuição por município
contagem_final = DF.groupby('cod').size()
print(f"    Média de registros por município: {contagem_final.mean():.1f}")
print(f"    Mínimo de registros por município: {contagem_final.min()}")
print(f"    Máximo de registros por município: {contagem_final.max()}")

# ==============================================================================
# ENGENHARIA DE FEATURES COM FATOR TECNOLÓGICO
# ==============================================================================

def criar_interacoes_estrategicas(df):
    def relu(x): 
        return np.maximum(0, x)

    # z-score por safra (ou global se preferir)
    def z(s): 
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    
    df = df.copy()
    eps = 1e-6

    # parâmetros fáceis de calibrar
    TH_PLANTIO_MM   = 40.0     # chuva mínima p/ estabelecimento
    CALOR_EXTREMO   = 2.0      # anomalia Tmax (°C)
    SPI_SECA_EXT    = -1.5     # limiar SPI
    PRECOCE_DIAS    = 95       # LOS < PRECOCE_DIAS => colheita_precoce
    JANELA_PRE_COL  = 15       # últimos dias mais sensíveis ao calor

    # DOY já numéricos
    ini_doy = pd.to_numeric(df['data_ini'], errors='coerce')
    pos_doy = pd.to_numeric(df['POS'], errors='coerce')
    fim_doy = pd.to_numeric(df['data_fim'], errors='coerce')

    dur_veg   = pd.to_numeric(df['MOS_DT1'], errors='coerce')  # SOS→POS
    dur_rep   = pd.to_numeric(df['MOS_DT2'], errors='coerce')  # POS→EOS
    dur_ciclo = pd.to_numeric(df['LOS'],     errors='coerce')  # SOS→EOS

    # -------------------------------------------------------------------------
    # POS_relativo por município: POS_ano(municipio) / POS_max(municipio)
    # chave pode ser 'cod' (preferida) ou 'municipio'
    chave_muni = 'cod' if 'cod' in df.columns else ('municipio' if 'municipio' in df.columns else None)
    if chave_muni is not None:
        pos_max_muni = pos_doy.groupby(df[chave_muni]).transform('max')
        df['POS_relativo'] = (pos_doy / (pos_max_muni + eps)).clip(0, 1)
    else:
        # se não houver chave municipal, usa o máximo global
        df['POS_relativo'] = (pos_doy / (pos_doy.max() + eps)).clip(0, 1)
    # -------------------------------------------------------------------------

    df['FENO_POS_MOS'] = df['POS'] + df['MOS']
    df['FORCA_CALOR_MAX'] = (df['tmax_anom'] + df['tmed_std'])* df['tmax_max']
    df['STRESS_CALOR_SECA'] = df['tmax_anom'] + (-df['spi'])
    df['FENO_POS_MOS_SPI10'] = df['POS'] + df['MOS'] + 0.1*df['spi']

    C = relu(z(df["tmax_anom"].astype(float)))
    S1 = relu(z(-df["prec_anom_safra"].astype(float)))
    S2 = relu(z(-df["spi"].astype(float)))
    V = relu(z(df["tmed_std"].astype(float)))
    X  = C * (S1 + S2)
    df["INTERACAO_CALOR_SECA"] = C + (S1+S2) + X + V

    # amplitude do dossel
    amp = df['AOS'] if 'AOS' in df.columns else df['MOS']
    amp = amp.fillna(df['MOS'])

    # ----- Água -----
    df['sazonalidade_chuva_plantio']  = df['prec_plantio']  / (df['prec_safra'] + eps)
    df['sazonalidade_chuva_colheita'] = df['prec_colheita'] / (df['prec_safra'] + eps)
    df['chuva_vs_roi'] = df['prec_plantio'] / (np.minimum(dur_veg, 30).clip(lower=1) + eps)
    df['chuva_vs_pos'] = df['anom_prec_colheita'] / (dur_rep.clip(lower=1) + eps)

    # ----- Calor/Frio -----
    df['amplitude_termica'] = df['tmax_max'] - df['tmin_min']
    df['estresse_termico_rel'] = np.maximum(df['tmax_anom'], 0) / (
        (df['tmax_mean'] - df['tmin_min']).clip(lower=0.5) + eps
    )
    df['calor_fase_sensivel_aprox'] = np.maximum(df['tmax_anom'], 0) * dur_rep.fillna(0)

    # ----- Fenologia -----
    df['assimetria_fases'] = (dur_veg + eps) / (dur_rep + eps)
    df['rod_roi_ratio']    = np.abs(df['ROD']) / (np.abs(df['ROI']) + eps)
    df['vigor_x_ciclo']    = amp * (dur_ciclo + 1)

    # ----- Geoclima -----
    if 'lat' in df.columns:
        df['gradiente_lat_tmax'] = df['lat'] * df['tmax_anom']
        df['gradiente_lat_spi']  = df['lat'] * df['spi']

    # ----- Indicadores térmicos e hídrico-térmicos (sem balanço hídrico) -----
    df['eficiencia_termica']       = df['gdd'] / (dur_ciclo + eps)
    df['estresse_termico_hidrico'] = np.maximum(df['tmax_anom'], 0) * (-df['spi'].clip(upper=0))
    df['produtividade_termica']    = (df['gdd'] / (dur_ciclo + eps)) * amp
    if 'tmin_anom' in df.columns and 'lat' in df.columns:
        df['lat_x_tmin'] = df['lat'] * df['tmin_anom']
    df['consistencia_fenologica']  = amp / (dur_ciclo + eps)

    # ----- Transformações úteis -----
    for col in ['prec_safra','prec_plantio','prec_colheita']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    # ----- Estresses por fase -----
    stress_def  = np.clip((TH_PLANTIO_MM - df['prec_plantio']) / TH_PLANTIO_MM, 0, 1)
    stress_anom = np.clip((-df['anom_prec_plantio']) / TH_PLANTIO_MM, 0, 1)
    df['stress_plantio']   = np.maximum(stress_def, stress_anom)
    df['stress_floracao']  = np.maximum(df['tmax_anom'], 0) * (-df['spi'].clip(upper=0))
    df['stress_maturacao'] = (
        np.maximum(df['tmax_anom'], 0) * (dur_rep / (dur_ciclo + eps))
        + np.maximum(df['anom_prec_colheita'], 0) / TH_PLANTIO_MM
    )

    # ----- Eventos extremos -----
    df['seca_extrema']  = (df['spi'] < SPI_SECA_EXT).astype(int)
    df['calor_extremo'] = (df['tmax_anom'] > CALOR_EXTREMO).astype(int)
    df['geada_risco']   = ((ini_doy <= 270) & (df['tmin_min'] < 2.0)).astype(int)

    # ----- Interações temporais -----
    df['chuva_pos_plantio']    = df['chuva_vs_roi']
    df['calor_antes_colheita'] = np.maximum(df['tmax_anom'], 0) * np.minimum(dur_rep.fillna(0), JANELA_PRE_COL)

    # ----- Índices agregados (sem BH) -----
    df['indice_estresse'] = df['estresse_termico_hidrico']

    # ----- Sazonalidade -----
    df['plantio_tardio']   = ((ini_doy > 60) & (ini_doy < 244)).astype(int)
    los_ref = dur_ciclo.fillna((fim_doy - ini_doy).clip(lower=0))
    df['colheita_precoce'] = (los_ref < PRECOCE_DIAS).astype(int)
    df['saldo_hidrico'] = df['prec_safra'] + df['spi']
    df['moisture_relief'] = df['spi'] + df['prec_anom_safra']

    return df

def criar_features_com_tecnologia(df):
    """Cria features com componente tecnológico explícito - TECNOLOGIA NÃO REGRIDE!"""
    df_fe = df.copy()
    
    # 1. Normalizar ano para componente tecnológico
    ano_min, ano_max = df_fe[ID_ANO].min(), df_fe[ID_ANO].max()
    df_fe['ano_normalizado'] = (df_fe[ID_ANO] - ano_min) / (ano_max - ano_min)
    
    # 2. Calcular tendência tecnológica do PAÍS (regressão com todos os municípios)
    def calcular_tendencia_tecnologica_pais(df_pais):
        """Calcula a tendência tecnológica para todo o país usando regressão linear"""
        # Agrupar por ano e calcular média nacional (apenas valores não-NaN)
        df_agg = df_pais[df_pais[TARGET].notna()].groupby(ID_ANO)[TARGET].mean().reset_index()
        
        if len(df_agg) < 2:
            return 0
            
        X = df_agg[ID_ANO].values.astype(float)
        y = df_agg[TARGET].values.astype(float)
        
        try:
            coef, intercept = np.polyfit(X.flatten(), y, 1)
            print(f"    🌎 Tendência tecnológica do país: {coef:.6f}")
            return coef
        except:
            return 0
    
    print("🌎 Calculando tendência tecnológica do país...")
    tendencia_pais = calcular_tendencia_tecnologica_pais(df_fe)
    
    # 3. Calcular tendência tecnológica por município (apenas dados não-NaN)
    def calcular_tendencia_tecnologica_municipio(subdf):
        """Calcula taxa de crescimento anual por município"""
        # Filtrar apenas valores não-NaN
        subdf = subdf[~subdf[ID_ANO].isin(ANOS_TESTE)]
        mask_nao_nan = subdf[TARGET].notna()
        if mask_nao_nan.sum() < 2:  # Precisa de pelo menos 2 pontos
            return tendencia_pais  # Se não tem dados suficientes, usa tendência do país
            
        X = subdf.loc[mask_nao_nan, [ID_ANO]].values.astype(float)
        y = subdf.loc[mask_nao_nan, TARGET].values.astype(float)
        
        try:
            coef, intercept = np.polyfit(X.flatten(), y, 1)
            # 🔥 SE tendência for negativa, usar tendência do PAÍS
            if coef < 0:
                return tendencia_pais
            return coef
        except:
            return tendencia_pais
    
    print("📈 Calculando tendências tecnológicas por município...")
    tendencias_municipio = df_fe.groupby(ID_MUNICIPIO).apply(calcular_tendencia_tecnologica_municipio)
    
    # Contar estatísticas antes de aplicar
    municipios_com_tendencia_negativa = (tendencias_municipio == tendencia_pais).sum()
    municipios_com_tendencia_positiva = (tendencias_municipio != tendencia_pais).sum()
    
    print(f"    Municípios com tendência positiva própria: {municipios_com_tendencia_positiva}")
    print(f"    Municípios usando tendência do país: {municipios_com_tendencia_negativa}")
    
    df_fe['taxa_crescimento_anual'] = df_fe[ID_MUNICIPIO].map(tendencias_municipio)
    
    # 4. Calcular baseline tecnológica para cada município
    def calcular_baseline_tecnologica(subdf):
        """Calcula a linha de tendência tecnológica para cada município"""
        municipio_id = subdf[ID_MUNICIPIO].iloc[0]
        taxa_crescimento = tendencias_municipio.get(municipio_id, tendencia_pais)
        
        # Filtrar apenas valores não-NaN para calcular a regressão
        mask_nao_nan = subdf[TARGET].notna()
        if mask_nao_nan.sum() < 2:
            return pd.Series(np.nan, index=subdf.index)
            
        X = subdf.loc[mask_nao_nan, [ID_ANO]].values.astype(float)
        y = subdf.loc[mask_nao_nan, TARGET].values.astype(float)
        
        try:
            # Calcular intercept baseado nos dados reais do município
            if mask_nao_nan.sum() >= 2:
                # Usar a taxa de crescimento definida (própria ou do país) mas ajustar o intercept aos dados do município
                anos = subdf.loc[mask_nao_nan, ID_ANO].values.astype(float)
                valores = subdf.loc[mask_nao_nan, TARGET].values.astype(float)
                
                # Calcular intercept que minimize o erro para a taxa de crescimento fixada
                # intercept = mean(y) - taxa_crescimento * mean(X)
                intercept = np.mean(valores) - taxa_crescimento * np.mean(anos)
                
                baseline = taxa_crescimento * subdf[ID_ANO] + intercept
            else:
                # Fallback: usar média
                media_y = np.mean(y)
                baseline = np.full(len(subdf), media_y)
            
            return pd.Series(baseline, index=subdf.index)
        except:
            return pd.Series(np.nan, index=subdf.index)
    
    print("🎯 Calculando baseline tecnológica...")
    df_fe['prod_baseline_tec'] = df_fe.groupby(ID_MUNICIPIO, group_keys=False).apply(calcular_baseline_tecnologica)
    
    # 5. Onde prod for NaN, usar prod_baseline_tec
    mask_nan = df_fe[TARGET].isna()
    df_fe.loc[mask_nan, TARGET] = df_fe.loc[mask_nan, 'prod_baseline_tec']
    
    # 6. Calcular desvio da tendência
    df_fe['prod_desvio_tec'] = df_fe[TARGET] - df_fe['prod_baseline_tec']
    
    # 7. Estatísticas finais
    print(f"\n🔥 CORREÇÕES APLICADAS - TECNOLOGIA NÃO REGRIDE:")
    print(f"    Tendência tecnológica do país: {tendencia_pais:.6f}")
    print(f"    Municípios com tendência positiva própria: {municipios_com_tendencia_positiva}")
    print(f"    Municípios usando tendência do país: {municipios_com_tendencia_negativa}")
    print(f"    Taxa de crescimento média final: {df_fe['taxa_crescimento_anual'].mean():.6f}")
    
    return df_fe

print("🔧 Aplicando engenharia de features com fator tecnológico...")
DF = criar_features_com_tecnologia(DF)
DF = criar_interacoes_estrategicas(DF)

# ==============================================================================
# CRIAÇÃO DE ZONAS
# ==============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def _slope_vs_ano(g, col):
    a = g['ano'].values.astype(float)
    y = g[col].values.astype(float)
    if len(np.unique(a)) < 2:
        return 0.0
    a_n = (a - a.mean()) / (a.std() + 1e-9)
    try:
        return float(np.polyfit(a_n, y, 1)[0])
    except Exception:
        return 0.0

def _agg_simples_por_muni(df):
    # lon/lat do centróide + resumos de POS e prod
    base = (
        df.dropna(subset=['lon','lat','POS','prod'])
          .groupby('cod')
          .apply(lambda g: pd.Series({
              'lon'        : g['lon'].median(),
              'lat'        : g['lat'].median(),
              'POS_med'    : g['POS'].median(),
              'POS_std'    : g['POS'].std(),
              'POS_slope'  : _slope_vs_ano(g, 'POS'),
              'prod_med'   : g['prod'].median(),
              'prod_std'   : g['prod'].std(),
              'prod_slope' : _slope_vs_ano(g, 'prod'),
          }))
          .reset_index()
          .fillna(0.0)
    )
    # pesos leves em lon/lat para coerência espacial
    base['lon_w'] = base['lon'] * 0.2
    base['lat_w'] = base['lat'] * 0.2
    return base

def criar_tendencia_por_zonas(df, n_zones):
    print(f"🔧 Criando tendência por zonas (lat/lon em km peso {0.60}; POS+prod considerados)")
    n_munis = df['cod'].nunique()
    anos_min, anos_max = int(df['ano'].min()), int(df['ano'].max())
    print(f"🗂️ Dados: {n_munis} munis | anos {anos_min}–{anos_max} | registros {len(df)}")

    # === base por município (precisa fornecer lon/lat, POS_* e prod_*) ===
    g = _agg_simples_por_muni(df).copy()
    
    # converter graus -> km para distância euclidiana aproximada
    lat_rad = np.deg2rad(g['lat'].astype(float))
    cos_lat = np.cos(lat_rad.clip(-np.pi/2, np.pi/2))
    g['lat_km'] = g['lat'].astype(float) * 111.0
    g['lon_km'] = g['lon'].astype(float) * (111.0 * cos_lat.mean())

    geo_weight = 0.60
    g['lat_w'] = g['lat_km'] * geo_weight
    g['lon_w'] = g['lon_km'] * geo_weight

    print(f"✅ Agregação por muni ok: {len(g)} munis válidos")

    feat_cols = ['lon_w','lat_w','POS_med','POS_std','POS_slope','prod_med','prod_std','prod_slope']
    scaler = StandardScaler()
    X = scaler.fit_transform(g[feat_cols].values)

    # --------- KMeans inicial ----------
    kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    g['zona'] = labels

    inertia = float(kmeans.inertia_)
    print(f"🧩 KMeans k={n_zones} | Inércia: {inertia:.2f}")

    # ----------------- aperto geográfico por raio --------------------
    def _geo_centroids(df_z):
        return df_z.groupby('zona')[['lon','lat']].mean()

    def _dist_km(lon1, lat1, lon2, lat2):
        lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
        lon1r, lon2r = np.deg2rad(lon1), np.deg2rad(lon2)
        x = (lon2r - lon1r) * np.cos(0.5*(lat1r + lat2r))
        y = (lat2r - lat1r)
        return 6371.0 * np.sqrt(x*x + y*y)

    R_MAX_KM = 100.0
    MAX_ITERS = 5
    changed = True
    it = 0
    while changed and it < MAX_ITERS:
        it += 1
        changed = False
        geo_cent = _geo_centroids(g)
        
        centers_X = {}
        for z in range(n_zones):
            idx = (g['zona'] == z).values
            if idx.sum() == 0:
                continue
            centers_X[z] = X[idx].mean(axis=0)

        reassign = []
        for i, row in g.iterrows():
            z = row['zona']
            if z not in geo_cent.index:
                continue
            d = _dist_km(row['lon'], row['lat'], geo_cent.loc[z, 'lon'], geo_cent.loc[z, 'lat'])
            if d > R_MAX_KM:
                best_z = z
                best_score = np.inf
                for z2 in range(n_zones):
                    if z2 not in geo_cent.index:
                        continue
                    d_geo = _dist_km(row['lon'], row['lat'], geo_cent.loc[z2, 'lon'], geo_cent.loc[z2, 'lat'])
                    if z2 in centers_X:
                        d_attr = np.linalg.norm(X[i] - centers_X[z2])
                    else:
                        d_attr = np.inf
                    score = d_geo / max(R_MAX_KM, 1.0) + 0.25 * d_attr
                    if score < best_score:
                        best_score = score
                        best_z = z2
                if best_z != z:
                    reassign.append((i, best_z))

        if reassign:
            changed = True
            for i, new_z in reassign:
                g.at[i, 'zona'] = new_z
            labels = g['zona'].to_numpy()
            centers = []
            for z in range(n_zones):
                idx = (labels == z)
                if idx.sum():
                    centers.append(X[idx].mean(axis=0))
                else:
                    centers.append(kmeans.cluster_centers_[z])
            kmeans.cluster_centers_ = np.vstack(centers)

        print(f"🔁 Ajuste geográfico iteração {it}: reatribuições = {len(reassign)}")

    counts = g['zona'].value_counts().sort_index()
    print("🧭 Tamanho das zonas:", ", ".join(f"Z{z}:{c}" for z,c in counts.items()))

    # mapear zonas ao df principal
    df['zona'] = df['cod'].map(g.set_index('cod')['zona'].to_dict())
    print("🗺️ Zonas mapeadas para o dataframe principal")

    # tendência por zona
    df_sorted = df.sort_values(['zona','ano'])
    all_anos = range(anos_min, anos_max + 1)

    n_interp_total = 0
    for zona in range(n_zones):
        zona_data = df_sorted[df_sorted['zona'] == zona]
        if zona_data.empty:
            print(f"⚠️ Zona {zona} sem dados")
            continue

        anos_cobertos = sorted(zona_data['ano'].unique())
        print(f"🏷️ Zona {zona}: {len(zona_data['cod'].unique())} munis | anos {anos_cobertos[0]}–{anos_cobertos[-1]}")

        tendencia_por_ano = zona_data.groupby('ano')['prod'].median().reindex(all_anos)
        n_missing = tendencia_por_ano.isna().sum()

        tendencia_interp = tendencia_por_ano.interpolate(method='linear', limit_direction='both')
        tendencia_suavizada = tendencia_interp.rolling(window=3, center=True, min_periods=1).mean()

        n_interp_total += int(n_missing)

    print(f"🧵 Anos sem mediana por zona (antes da interpolação): {n_interp_total}")
    print(f"✅ Tendência calculada por {n_zones} zonas com restrição geográfica")

    return df

DF = criar_tendencia_por_zonas(DF, n_zones=22)

# ==============================================================================
# IMPLEMENTAÇÕES DAS RECOMENDAÇÕES
# ==============================================================================

def criar_features_espaciais(df, n_vizinhos=5):
    """
    Adiciona média zonal dos vizinhos mais próximos para suavização
    """
    print("🗺️ Aplicando suavização por vizinhança...")
    
    # Calcular centróides das zonas
    centroides = df.groupby('zona')[['lon', 'lat']].median().reset_index()
    
    # Calcular matriz de distâncias entre zonas
    coords_rad = np.radians(centroides[['lat', 'lon']].values)
    distancias = haversine_distances(coords_rad) * 6371  # km
    
    # Encontrar zonas vizinhas para cada zona
    zonas_vizinhas = {}
    for i, zona in enumerate(centroides['zona']):
        dists = distancias[i]
        vizinhos_idx = np.argsort(dists)[1:n_vizinhos+1]
        zonas_vizinhas[zona] = centroides.iloc[vizinhos_idx]['zona'].tolist()
    
    # Calcular médias dos vizinhos para features críticas
    features_suavizar = ['prod', 'prod_desvio_tec', 'spi', 'tmax_anom', 'prec_safra']
    
    for feature in features_suavizar:
        if feature in df.columns:
            df[f'{feature}_zonal'] = 0.0
            for zona in df['zona'].unique():
                vizinhas = zonas_vizinhas.get(zona, [])
                if vizinhas:
                    mask = df['zona'].isin([zona] + vizinhas)
                    media_zonal = df.loc[mask, feature].mean()
                    df.loc[df['zona'] == zona, f'{feature}_zonal'] = media_zonal
    
    print(f"✅ Suavização aplicada para {len(features_suavizar)} features")
    return df

def criar_features_temporais_avancadas(df):
    """
    Cria features de tendência temporal para combater drift
    """
    df = df.copy()
    #df = df[~df[ID_ANO].isin(ANOS_TESTE)]
    
    # Tendência linear por município
    def calcular_tendencia_linear(grupo):
        if len(grupo) < 2:
            return pd.Series([0] * len(grupo), index=grupo.index)
        
        X = grupo['ano'].values.astype(float)
        y = grupo['prod'].values.astype(float)
        
        try:
            coef = np.polyfit(X, y, 1)[0]
            return pd.Series([coef] * len(grupo), index=grupo.index)
        except:
            return pd.Series([0] * len(grupo), index=grupo.index)
    
    df['tendencia_linear_municipio'] = df.groupby('cod', group_keys=False).apply(calcular_tendencia_linear)
    
    # Média móvel por zona
    df = df.sort_values(['zona', 'ano'])
    df['prod_media_movel_zona'] = df.groupby('zona')['prod'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # Ano normalizado para tendência global
    ano_min, ano_max = df['ano'].min(), df['ano'].max()
    df['ano_norm_tendencia'] = (df['ano'] - ano_min) / (ano_max - ano_min)
    
    # Feature de safra recente (peso maior para anos recentes)
    df['peso_safra_recente'] = (df['ano'] - 2010) / (2024 - 2010)
    
    print("📈 Features temporais avançadas criadas")
    return df

def aplicar_winsorizacao_avancada(df):
    """
    Winsorização mais agressiva para lidar com caudas pesadas
    """
    df = df.copy()
    
    # Winsorização por zona para o target principal
    q1_prod = df.groupby('zona')['prod_desvio_tec'].transform(lambda s: s.quantile(0.03))
    q99_prod = df.groupby('zona')['prod_desvio_tec'].transform(lambda s: s.quantile(0.97))
    
    out_antes = ((df['prod_desvio_tec'] < q1_prod) | (df['prod_desvio_tec'] > q99_prod)).sum()
    df['prod_desvio_tec_winsor'] = df['prod_desvio_tec'].clip(lower=q1_prod, upper=q99_prod)
    out_depois = ((df['prod_desvio_tec_winsor'] < q1_prod) | (df['prod_desvio_tec_winsor'] > q99_prod)).sum()
    
    print(f"📊 Winsorização aplicada:")
    print(f"    Outliers removidos: {out_antes} → {out_depois}")
    print(f"    Novos limites: [{df['prod_desvio_tec_winsor'].min():.2f}, {df['prod_desvio_tec_winsor'].max():.2f}]")
    
    return df

def criar_pesos_zonas_criticas(df, zonas_criticas, peso_normal=1.0, peso_critico=3.0):
    """
    Cria sample weights para dar mais importância às zonas críticas
    """
    weights = np.ones(len(df), dtype=np.float32)
    for zona in zonas_criticas:
        mask = df['zona'] == zona
        weights[mask] = peso_critico
    
    print(f"🎯 Pesos aplicados para {len(zonas_criticas)} zonas críticas")
    print(f"    Peso normal: {peso_normal}, Peso crítico: {peso_critico}")
    return weights

# ==============================================================================
# IMPLEMENTAÇÕES FALTANTES
# ==============================================================================

def criar_pesos_por_ano(df, ano_critico=2024, peso_normal=1.0, peso_critico=2.5):
    """
    Cria sample weights para dar mais importância a anos críticos
    """
    weights = np.ones(len(df), dtype=np.float32)
    mask_ano_critico = df['ano'] == ano_critico
    weights[mask_ano_critico] = peso_critico
    
    print(f"🎯 Pesos aplicados para ano crítico {ano_critico}:")
    print(f"    Peso normal: {peso_normal}, Peso crítico: {peso_critico}")
    print(f"    Amostras com peso crítico: {mask_ano_critico.sum()}")
    
    return weights

def aplicar_transformacao_monotonica_alvo(df):
    """
    Aplica transformação monotônica ao target para lidar com tendência erro×prod
    """
    df = df.copy()
    
    # Calcular relação entre erro e produtividade
    if 'erro' not in df.columns:
        # Se não há erro calculado, usar desvio como proxy
        df['erro'] = df['prod_desvio_tec']
    
    # Transformação para reduzir heterocedasticidade
    df['prod_desvio_tec_transformado'] = np.arcsinh(df['prod_desvio_tec_winsor'])
    
    print("📈 Transformação monotônica aplicada ao target:")
    print(f"    Antes: {df['prod_desvio_tec_winsor'].mean():.2f} ± {df['prod_desvio_tec_winsor'].std():.2f}")
    print(f"    Depois: {df['prod_desvio_tec_transformado'].mean():.2f} ± {df['prod_desvio_tec_transformado'].std():.2f}")
    
    return df



# Aplicar todas as melhorias
print("\n🚀 APLICANDO ESTRATÉGIAS DE MELHORIA...")
DF = criar_features_espaciais(DF)
DF = criar_features_temporais_avancadas(DF)
DF = aplicar_winsorizacao_avancada(DF)
DF = aplicar_transformacao_monotonica_alvo(DF)

# ==============================================================================
# FEATURES FINAIS - DEFINIR APÓS CRIAR TODAS AS FEATURES
# ==============================================================================

# Primeiro criar a coluna 'distancia_mediana' que estava faltando
print("📊 Criando feature 'distancia_mediana'...")
mediana_por_zona = DF.groupby('zona')['prod'].transform('median')
DF['distancia_mediana'] = (DF['prod'] - mediana_por_zona).abs()

FEATURES_TENDENCIAS = [
    # climático
    'FORCA_CALOR_MAX','INTERACAO_CALOR_SECA','tmed_std','calor_antes_colheita','stress_floracao','estresse_termico_hidrico','indice_estresse','calor_fase_sensivel_aprox','tmax_anom','estresse_termico_rel','amplitude_termica','gradiente_lat_spi','STRESS_CALOR_SECA','stress_plantio','seca_extrema','prec_anom_safra','moisture_relief','tmin_min','spi','produtividade_termica','anom_prec_plantio','prec_plantio','log_prec_plantio','saldo_hidrico','prec_safra','gradiente_lat_tmax','log_prec_safra',
    # fenologico
    'ROD','MOS_DT1','SIOS','AOS','vigor_x_ciclo','FENO_POS_MOS_SPI10','MOS','FENO_POS_MOS','POS','POS_relativo','distancia_mediana',
]

# Adicionar as novas features
FEATURES_TENDENCIAS += [
    'prod_zonal', 'prod_desvio_tec_zonal', 'spi_zonal', 'tmax_anom_zonal',
    'tendencia_linear_municipio', 'prod_media_movel_zona', 'ano_norm_tendencia',
    'peso_safra_recente'
]

print(f"🎯 Total de features: {len(FEATURES_TENDENCIAS)}")

# ==============================================================================
# PRÉ-PROCESSAMENTO FINAL
# ==============================================================================

def preprocessamento_agressivo(df):
    """Pré-processamento focado em reduzir viés de baixa sem distorcer o alvo"""
    df = df.copy()
    print("🔄 APLICANDO PRÉ-PROCESSAMENTO...")

    # Zonas com maior variância
    var_prod_zona = df.groupby('zona')['prod'].std()
    zonas_problematicas = var_prod_zona.nlargest(4).index.tolist()
    df['zona_problematica'] = df['zona'].isin(zonas_problematicas).astype(int)
    print(f"    Zonas problemáticas (alta variância): {zonas_problematicas}")

    print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO")
    return df

DF = preprocessamento_agressivo(DF)

# ==============================================================================
# DIVISÃO DOS DADOS COM PESOS
# ==============================================================================

# Verificar se todas as colunas necessárias existem antes do dropna
colunas_necessarias = FEATURES_TENDENCIAS + [TARGET, 'prod_desvio_tec', 'zona']
colunas_faltantes = [col for col in colunas_necessarias if col not in DF.columns]

if colunas_faltantes:
    print(f"⚠️ Colunas faltantes: {colunas_faltantes}")
    # Remover colunas faltantes da lista
    colunas_necessarias = [col for col in colunas_necessarias if col in DF.columns]
    print(f"✅ Usando {len(colunas_necessarias)} colunas disponíveis")

DF2025 = DF[DF['ano'] == 2025].copy()
DF.dropna(subset=colunas_necessarias, inplace=True)
DF = DF[DF['ano'] != 2025]

# ==============================================================================
# IDENTIFICAÇÃO DE ZONAS CRÍTICAS BASEADA NO KMEANS
# ==============================================================================

def identificar_zonas_criticas(df, n_zonas_criticas=12):
    """
    Identifica zonas críticas baseado no desempenho histórico (produtividade)
    usando os dados de treino para evitar data leakage
    """
    print("🎯 Identificando zonas críticas baseado no KMeans...")
    
    # Usar apenas dados de treino para evitar data leakage
    df_treino = df[~df['ano'] <= 2022].copy()  # Dados históricos
    
    # Calcular métricas de desempenho por zona
    desempenho_zonas = df_treino.groupby('zona').agg({
        'prod': ['median', 'std', 'count'],  # Mediana, variabilidade e quantidade de dados
        'prod_desvio_tec': ['mean', 'std'],  # Desvio da tendência
        'spi': 'mean',                       # Condições hídricas
        'tmax_anom': 'mean'                  # Condições térmicas
    }).round(4)
    
    # Flatten column names
    desempenho_zonas.columns = ['_'.join(col).strip() for col in desempenho_zonas.columns.values]
    
    # Calcular score de criticidade (quanto menor pior)
    desempenho_zonas['score_criticidade'] = (
        -desempenho_zonas['prod_median'] +          # Produtividade baixa
        desempenho_zonas['prod_std'] +              # Alta variabilidade
        -desempenho_zonas['prod_desvio_tec_mean'] + # Desvios negativos frequentes
        desempenho_zonas['spi_mean'] +              # Condições secas
        desempenho_zonas['tmax_anom_mean']          # Calor excessivo
    )
    
    # Ordenar por criticidade (mais críticas primeiro)
    zonas_ordenadas = desempenho_zonas.sort_values('score_criticidade', ascending=True)
    
    # Selecionar as n zonas mais críticas
    zonas_criticas = zonas_ordenadas.head(n_zonas_criticas).index.tolist()
    
    print(f"✅ Zonas críticas identificadas: {zonas_criticas}")
    print("📊 Estatísticas das zonas críticas:")
    for zona in zonas_criticas:
        stats = zonas_ordenadas.loc[zona]
        print(f"   Zona {zona}: Prod={stats['prod_median']:.1f} ± {stats['prod_std']:.1f}, "
              f"SPI={stats['spi_mean']:.2f}, TmaxAnom={stats['tmax_anom_mean']:.2f}")
    
    return zonas_criticas

# ==============================================================================
# ATUALIZAR A DIVISÃO DOS DADOS PARA USAR ZONAS CRÍTICAS DINÂMICAS
# ==============================================================================

def dividir_dados_anos_com_pesos(df):
    """Divide os dados com pesos por zona crítica E por ano crítico"""
    # Separar teste: anos específicos
    anos_teste = {2005, 2012, 2017, 2023, 2024}

    test = df[df['ano'].isin(anos_teste)].copy()
    treino = df[~df['ano'].isin(anos_teste)].copy()
    
    # Embaralhar o conjunto até 2023
    df_shuffled = treino.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    idx_train = int(0.8 * n_total)
    
    train = df_shuffled.iloc[:idx_train]
    val = df_shuffled.iloc[idx_train:]
    
    # 🔥 IDENTIFICAR ZONAS CRÍTICAS DINAMICAMENTE
    ZONAS_CRITICAS = identificar_zonas_criticas(train, n_zonas_criticas=12)
    
    # Criar pesos para zonas críticas
    sample_weights_zona_train = criar_pesos_zonas_criticas(train, ZONAS_CRITICAS)
    sample_weights_zona_val = criar_pesos_zonas_criticas(val, ZONAS_CRITICAS)
    
    # 🔥 ADICIONAR PESOS POR ANO CRÍTICO (2024 não está no treino/val, mas podemos dar peso para 2023)
    sample_weights_ano_train = criar_pesos_por_ano(train, ano_critico=2023, peso_critico=2.0)
    sample_weights_ano_val = criar_pesos_por_ano(val, ano_critico=2023, peso_critico=2.0)
    
    # 🔥 COMBINAR PESOS (multiplicação)
    sample_weights_train = sample_weights_zona_train * sample_weights_ano_train
    sample_weights_val = sample_weights_zona_val * sample_weights_ano_val
    
    print(f"📊 Distribuição balanceada COM PESOS COMBINADOS:")
    print(f"    Treino: {len(train)} amostras")
    print(f"    Validação: {len(val)} amostras") 
    print(f"    Teste: {len(test)} amostras")
    print(f"    Zonas críticas: {ZONAS_CRITICAS}")
    print(f"    Peso médio treino: {sample_weights_train.mean():.2f}")
    print(f"    Peso médio validação: {sample_weights_val.mean():.2f}")
    
    return train, val, test, sample_weights_train, sample_weights_val, ZONAS_CRITICAS

# ==============================================================================
# REMOVER A DEFINIÇÃO GLOBAL DE ZONAS_CRITICAS
# ==============================================================================

# REMOVER ESTA LINHA:
# ZONAS_CRITICAS = [4, 10, 0, 9, 15, 18, 8, 6, 16, 21, 1, 11]

# ==============================================================================
# ATUALIZAR A CHAMADA DA DIVISÃO DOS DADOS
# ==============================================================================

# Substituir esta linha:
# train, val, test, sample_weights_train, sample_weights_val = dividir_dados_anos_com_pesos(DF)

# Por esta:
train, val, test, sample_weights_train, sample_weights_val, ZONAS_CRITICAS = dividir_dados_anos_com_pesos(DF)
# ==============================================================================
# BASELINE INTELIGENTE COM TECNOLOGIA
# ==============================================================================

def baseline_com_tecnologia(train, val, test):
    """Baseline: prever tendência tecnológica (desvio zero)"""
    pred_val_abs = val['prod_baseline_tec'].to_numpy()
    pred_test_abs = test['prod_baseline_tec'].to_numpy()
    mae_val = mean_absolute_error(val[TARGET], pred_val_abs)
    mae_test = mean_absolute_error(test[TARGET], pred_test_abs)

    print(f"🎯 BASELINE TECNOLÓGICA (Tendência por Município):")
    print(f"    Validação MAE: {mae_val:.2f}")
    print(f"    Teste MAE: {mae_test:.2f}")
    print(f"    📈 Baseline considera crescimento tecnológico médio de {train['taxa_crescimento_anual'].mean():.2f} unidades/ano")
    return mae_val, mae_test

baseline_val, baseline_test = baseline_com_tecnologia(train, val, test)

# ==============================================================================
# PREPARAÇÃO DOS DADOS
# ==============================================================================

imputador   = SimpleImputer(strategy='median')
escalonador = RobustScaler()

X_train_num = escalonador.fit_transform(imputador.fit_transform(train[FEATURES_TENDENCIAS].to_numpy(dtype=float)))
X_val_num   = escalonador.transform(imputador.transform(val[FEATURES_TENDENCIAS].to_numpy(dtype=float)))
X_test_num  = escalonador.transform(imputador.transform(test[FEATURES_TENDENCIAS].to_numpy(dtype=float)))

# Targets: desvios winsorizados da tendência tecnológica
y_train_desvio = train['prod_desvio_tec_winsor'].to_numpy(dtype=float)
y_val_desvio   = val['prod_desvio_tec_winsor'].to_numpy(dtype=float)
y_test_desvio  = test['prod_desvio_tec_winsor'].to_numpy(dtype=float)

# Dados de zona
zona_train = train['zona'].astype(int).to_numpy()
zona_val = val['zona'].astype(int).to_numpy()
zona_test = test['zona'].astype(int).to_numpy()

# ==============================================================================
# MODELO COM EMBEDDING ZONAL
# ==============================================================================

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

from tensorflow.keras import regularizers

def criar_modelo_com_embedding_zonal(num_features, n_zonas=22):
    """Modelo com embeddings para calibrar bias por zona"""
    
    # Inputs
    inp_num = L.Input(shape=(num_features,), name="num_feat")
    inp_zona = L.Input(shape=(1,), name="zona_input")
    
    # Embedding para zonas (aprende calibração por zona)
    zona_embedding = L.Embedding(
        input_dim=n_zonas, 
        output_dim=4,
        name="zona_embedding"
    )(inp_zona)
    zona_embedding = L.Flatten()(zona_embedding)
    
    # ---------------------------------- bloco utilitário ----------------------------------
    def res_block(x, units, drop_rate, name):
        shortcut = x
        if shortcut.shape[-1] != units:
            shortcut = L.Dense(units, 
                              kernel_regularizer=regularizers.l2(1e-5),
                              name=f"{name}_shortcut")(shortcut)

        x = L.Dense(units, activation=None, 
                   kernel_regularizer=regularizers.l2(1e-5),
                   name=f"{name}_dense1")(x)
        x = L.BatchNormalization(name=f"{name}_bn1")(x)
        x = L.Activation('swish', name=f"{name}_swish1")(x)
        x = L.Dropout(drop_rate, name=f"{name}_drop1")(x)

        x = L.Dense(units, activation=None,
                   kernel_regularizer=regularizers.l2(1e-5),
                   name=f"{name}_dense2")(x)
        x = L.BatchNormalization(name=f"{name}_bn2")(x)
        x = L.Activation('swish', name=f"{name}_swish2")(x)
        x = L.Dropout(drop_rate, name=f"{name}_drop2")(x)

        # Squeeze-and-Excitation para vetor
        se = L.GlobalAveragePooling1D(keepdims=True)(L.Reshape((x.shape[-1], 1))(x))
        se = L.Flatten()(se)
        se = L.Dense(max(units // 8, 8), activation='relu', 
                    kernel_regularizer=regularizers.l2(1e-5),
                    name=f"{name}_se1")(se)
        se = L.Dense(units, activation='sigmoid', 
                    kernel_regularizer=regularizers.l2(1e-5),
                    name=f"{name}_se2")(se)
        x = L.Multiply(name=f"{name}_se_scale")([x, se])

        return L.Add(name=f"{name}_add")([shortcut, x])
    
    # ==========================================================================
    # BRANCH PRINCIPAL
    # ==========================================================================
    att_w = L.Dense(num_features, activation='softmax', name="feature_attention")(inp_num)
    attended = L.Multiply(name="apply_attention")([inp_num, att_w])

    x_main = L.Dense(512, activation=None, 
                    kernel_regularizer=regularizers.l2(1e-5),
                    name="main_dense1")(attended)
    x_main = L.BatchNormalization(name="main_bn1")(x_main)
    x_main = L.Activation('swish', name="main_swish1")(x_main)
    x_main = L.Dropout(0.4, name="main_drop1")(x_main)

    x_main = res_block(x_main, 512, 0.35, "main_res1")
    x_main = res_block(x_main, 256, 0.30, "main_res2")
    x_main = res_block(x_main, 128, 0.25, "main_res3")

    # ==========================================================================
    # BRANCH CLIMÁTICA
    # ==========================================================================
    CLIMATE_K = 26
    climate_feat = L.Lambda(lambda x: x[:, :min(CLIMATE_K, num_features)],
                            name="extract_climate_features")(inp_num)

    x_clim = L.Dense(256, activation=None,
                    kernel_regularizer=regularizers.l2(1e-5),
                    name="climate_dense1")(climate_feat)
    x_clim = L.BatchNormalization(name="climate_bn1")(x_clim)
    x_clim = L.Activation('gelu', name="climate_gelu1")(x_clim)
    x_clim = L.Dropout(0.4, name="climate_drop1")(x_clim)

    x_clim = res_block(x_clim, 256, 0.30, "climate_res1")
    x_clim = res_block(x_clim, 128, 0.25, "climate_res2")

    # ==========================================================================
    # BRANCH NDVI / FENOLOGIA
    # ==========================================================================
    NDVI_START = CLIMATE_K + 1
    ndvi_feat = L.Lambda(lambda x: x[:, NDVI_START:],
                         name="extract_ndvi_features")(inp_num)

    x_ndvi = L.Dense(192, activation=None,
                    kernel_regularizer=regularizers.l2(1e-5),
                    name="ndvi_dense1")(ndvi_feat)
    x_ndvi = L.BatchNormalization(name="ndvi_bn1")(x_ndvi)
    x_ndvi = L.Activation('gelu', name="ndvi_gelu1")(x_ndvi)
    x_ndvi = L.Dropout(0.35, name="ndvi_drop1")(x_ndvi)

    x_ndvi = res_block(x_ndvi, 192, 0.30, "ndvi_res1")
    x_ndvi = res_block(x_ndvi, 96,  0.25, "ndvi_res2")

    # ==========================================================================
    # FUSÃO COM EMBEDDING ZONAL
    # ==========================================================================
    fused_branches = L.Concatenate(name="fusion_branches")([x_main, x_clim, x_ndvi])
    fused = L.Concatenate(name="fusion_com_zona")([fused_branches, zona_embedding])
    
    x = L.Dense(512, activation=None,
               kernel_regularizer=regularizers.l2(1e-5),
               name="fusion_dense1")(fused)
    x = L.BatchNormalization(name="fusion_bn1")(x)
    x = L.Activation('swish', name="fusion_swish1")(x)
    x = L.Dropout(0.3, name="fusion_drop1")(x)

    x = res_block(x, 256, 0.25, "fusion_res1")
    x = res_block(x, 128, 0.20, "fusion_res2")
    x = res_block(x, 64,  0.15, "fusion_res3")

    # Output principal
    main_output = L.Dense(32, activation='swish', 
                         kernel_regularizer=regularizers.l2(1e-5),
                         name="main_head1")(x)
    main_output = L.Dropout(0.1, name="main_head_drop")(main_output)
    main_output = L.Dense(1, name="main_output")(main_output)
    
    # Output de calibração zonal (offset por zona)
    zona_offset = L.Dense(1, name="zona_offset")(zona_embedding)
    
    # Output final = predição + offset zonal
    final_output = L.Add(name="final_output")([main_output, zona_offset])

    model = keras.Model([inp_num, inp_zona], final_output)

    # Loss e otimizador - VERSÃO CORRIGIDA
    def quantile_loss_suave(tau=0.70, alpha=0.1):
        def ql_suave(y_true, y_pred):
            # Garantir que os tensores estão no dtype correto
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            e = y_true - y_pred
            quantile_loss = tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
            mse_component = alpha * tf.reduce_mean(tf.square(e))
            return quantile_loss + mse_component
        return ql_suave

    def mbe(y_true, y_pred):
        # Garantir tipos consistentes
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(y_pred - y_true)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=2e-4,  # Aumentado ligeiramente
            weight_decay=1e-4,   # Reduzido para permitir mais flexibilidade
            beta_1=0.9, 
            beta_2=0.999, 
            clipnorm=1.0         # Aumentado para permitir updates maiores
        ),
        loss=quantile_loss_suave(0.7, alpha=0.05),
        metrics=['mae', 'mse', mbe]
    )
    
    return model

def criar_callbacks_radicais():
    return [
        # MODEL CHECKPOINT MAIS PERMISSIVO
        keras.callbacks.ModelCheckpoint(
            'best_model_long_training.keras',
            monitor='val_mae', 
            mode='min', 
            save_best_only=True, 
            verbose=1
        ),
        
        # LEARNING RATE SCHEDULER MAIS CONSERVADOR
        keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.995 if epoch > 10 else lr,  # Redução mais lenta
            verbose=0  # Reduz verbosidade
        ),
        
        # REDUCE LR ON PLATEAU MAIS PACIENTE
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae', 
            mode='min',
            factor=0.7,           # Redução menos agressiva
            patience=16,          # Muito mais paciente
            min_delta=0.3,        # Margem menor para considerar melhoria
            cooldown=3,           # Cooldown reduzido
            min_lr=1e-7, 
            verbose=1
        ),
        
        # EARLY STOPPING MUITO MAIS PACIENTE
        keras.callbacks.EarlyStopping(
            monitor='val_mae', 
            mode='min',
            patience=64,          # Paciência drasticamente aumentada
            min_delta=0.2,        # Melhorias menores são consideradas
            restore_best_weights=True, 
            verbose=1
        ),
        
        # CALLBACK PERSONALIZADO PARA LOGS DETALHADOS
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: 
            print(f"Época {epoch}: LR = {keras.backend.get_value(modelo.optimizer.learning_rate):.2e}, "
                  f"Val_MAE = {logs.get('val_mae', 0):.3f}")
        ),
        
        keras.callbacks.TensorBoard(
            log_dir='./logs_long_training', 
            histogram_freq=4,  # A cada 5 épocas para economizar memória
            update_freq='epoch'
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

# ==============================================================================
# TREINAMENTO
# ==============================================================================

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

modelo = criar_modelo_com_embedding_zonal(len(FEATURES_TENDENCIAS))
callbacks = criar_callbacks_radicais()

print("🚀 INICIANDO TREINAMENTO DO MODELO COM CALIBRAÇÃO ZONAL...")
historico = modelo.fit(
    [X_train_num, zona_train],
    y_train_desvio,
    sample_weight=sample_weights_train,
    validation_data=([X_val_num, zona_val], y_val_desvio, sample_weights_val),
    epochs=1024, 
    batch_size=32, 
    verbose=1, 
    callbacks=callbacks
)

# ==============================================================================
# CALIBRAÇÃO PÓS-TREINO
# ==============================================================================

def calibrar_modelo_isotonic(model, X_val, zona_val, y_val):
    """
    Aplica calibração isotônica para correção monotônica
    """
    # Fazer previsões na validação
    y_pred_val = model.predict([X_val, zona_val]).flatten()
    
    # Treinar calibrador isotônico
    calibrador = IsotonicRegression(out_of_bounds='clip')
    calibrador.fit(y_pred_val, y_val)
    
    print("🎯 Calibração isotônica aplicada")
    return calibrador

def prever_com_calibracao(model, X, zonas, calibrador):
    pred_raw = model.predict([X, zonas]).flatten()
    return calibrador.predict(pred_raw)

# Aplicar calibração
print("🔧 Aplicando calibração pós-treino...")
calibrador = calibrar_modelo_isotonic(modelo, X_val_num, zona_val, y_val_desvio)

# ==============================================================================
# AVALIAÇÃO FINAL
# ==============================================================================

def avaliar_modelo_completo(model, calibrador, X_test, zona_test, y_test, baseline_mae):
    """
    Avaliação completa com métricas detalhadas
    """
    # Previsões calibradas
    y_pred_calibrado = prever_com_calibracao(model, X_test, zona_test, calibrador)
    
    # Métricas
    mae = mean_absolute_error(y_test, y_pred_calibrado)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_calibrado))
    r2 = r2_score(y_test, y_pred_calibrado)
    
    # Comparação com baseline
    melhoria_vs_baseline = ((baseline_mae - mae) / baseline_mae) * 100
    
    print("=" * 60)
    print("📊 AVALIAÇÃO FINAL DO MODELO")
    print("=" * 60)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Melhoria vs Baseline: {melhoria_vs_baseline:+.1f}%")
    print("=" * 60)
    
    return mae, rmse, r2

print("📈 Avaliando modelo final...")
mae_final, rmse_final, r2_final = avaliar_modelo_completo(
    modelo, calibrador, X_test_num, zona_test, y_test_desvio, baseline_test
)

print("✅ TREINAMENTO E AVALIAÇÃO CONCLUÍDOS!")
# ==============================================================================
# AVALIAÇÃO (SIMPLIFICADA)
# ==============================================================================
# Larguras para alinhar tudo bonitinho (apenas visual)
NAME_W, RMSE_W, MAE_W, R2_W, VSB_W = 8, 7, 6, 7, 11

def _row(name, rmse, mae, r2, vsb, flag):
    return (f"{name:<{NAME_W}} | {rmse:{RMSE_W}.2f} | {mae:{MAE_W}.2f} | "
            f"{r2:{R2_W}.3f} | {vsb:+{VSB_W}.2f} | {flag}")

def avaliar_modelo_hibrido(modelo, calibrador, dados, features_num, zonas, nome, baseline_mae):
    # Fazer previsões calibradas
    desvios_pred = prever_com_calibracao(modelo, features_num, zonas, calibrador)
    prod_pred = desvios_pred + dados['prod_baseline_tec'].to_numpy()
    prod_real = dados[TARGET].to_numpy()

    rmse = np.sqrt(np.mean((prod_pred - prod_real) ** 2))
    mae  = mean_absolute_error(prod_real, prod_pred)
    r2   = r2_score(prod_real, prod_pred)

    vsb = baseline_mae - mae
    flag = "✅" if vsb > 0 else "❌"
    print(_row(nome, rmse, mae, r2, vsb, flag))
    return mae, prod_pred

# Cabeçalho alinhado com as mesmas larguras
header = (f"{'Conjunto':<{NAME_W}} | {'RMSE':>{RMSE_W}} | {'MAE':>{MAE_W}} | "
          f"{'R²':>{R2_W}} | {'vs Baseline':>{VSB_W}} | Flag")
sep = "-" * len(header)

print("\n" + "=" * len(header))
print("AVALIAÇÃO FINAL DO MODELO")
print("=" * len(header))
print(header)
print(sep)

# Baseline (comparação direta prod_baseline_tec vs prod)
y_true = DF['prod'].to_numpy(dtype=float)
y_base = DF['prod_baseline_tec'].to_numpy(dtype=float)
rmse_b = np.sqrt(mean_squared_error(y_true, y_base))
mae_b  = mean_absolute_error(y_true, y_base)
r2_b   = r2_score(y_true, y_base)
print(_row("Baseline", rmse_b, mae_b, r2_b, 0.0, "—"))

# Partições - CORREÇÃO: usar zonas e calibrador
mae_train, pred_train = avaliar_modelo_hibrido(modelo, calibrador, train, X_train_num, zona_train, "Treino", baseline_val)
mae_val,   pred_val   = avaliar_modelo_hibrido(modelo, calibrador, val,   X_val_num,   zona_val,   "Valida", baseline_val)
mae_test,  pred_test  = avaliar_modelo_hibrido(modelo, calibrador, test,  X_test_num,  zona_test,  "Teste",  baseline_test)

print("=" * len(header))

# ==============================================================================
# CRIAÇÃO DO DATAFRAME DE RESULTADOS DO TESTE (CORRIGIDO)
# ==============================================================================

print("\n" + "="*80)
print("RESULTADOS FINAIS - DATAFRAME DE TESTE")
print("="*80)

# Criar DataFrame com resultados do teste
resultados_test = test.copy()

# CORREÇÃO: Usar as previsões (prod_pred) já calculadas na avaliação
resultados_test['prod_pred'] = pred_test
# Calcular o desvio predito com base na previsão absoluta
resultados_test['desvio_pred'] = resultados_test['prod_pred'] - resultados_test['prod_baseline_tec']
resultados_test['erro'] = resultados_test['prod'] - resultados_test['prod_pred']

# Calcular métricas finais para o teste (serão iguais ao mae_test já calculado)
mae_test_final = mean_absolute_error(resultados_test['prod'], resultados_test['prod_pred'])
rmse_test_final = np.sqrt(np.mean((resultados_test['prod'] - resultados_test['prod_pred']) ** 2))
r2_test_final = r2_score(resultados_test['prod'], resultados_test['prod_pred'])

print(f"📊 MÉTRICAS FINAIS - TESTE:")
print(f"    MAE:  {mae_test_final:.2f} (Confirmado)")
print(f"    RMSE: {rmse_test_final:.2f} (Confirmado)")
print(f"    R²:   {r2_test_final:.3f} (Confirmado)")
print(f"    Melhoria vs Baseline: {((baseline_test - mae_test_final) / baseline_test) * 100:+.1f}%")

print(f"\n📋 AMOSTRA DOS RESULTADOS (10 primeiras linhas do teste):")
colunas_mostrar = ['cod', 'ano', 'prod', 'prod_baseline_tec', 'desvio_pred', 'prod_pred', 'erro']
print(resultados_test[colunas_mostrar].head(10).round(2))

print(f"\n📈 ESTATÍSTICAS DOS ERROS NO TESTE:")
print(f"    Erro médio: {resultados_test['erro'].mean():.2f} ± {resultados_test['erro'].std():.2f}")
print(f"    Erro máximo: {resultados_test['erro'].max():.2f}")
print(f"    Erro mínimo: {resultados_test['erro'].min():.2f}")

# Quantis básicos (0%, 25%, 50%, 75%, 100%)
quantis_basicos = resultados_test['erro'].quantile([0, 0.25, 0.5, 0.75, 1])
print("📈 QUANTIS BÁSICOS DO ERRO:")
print(quantis_basicos.round(2))

print(f"    % erros < 60 unidades: {(np.abs(resultados_test['erro']) < 60).mean() * 100:.1f}%")
print(f"    % erros < 120 unidades: {(np.abs(resultados_test['erro']) < 120).mean() * 100:.1f}%")
print(f"    % erros < 180 unidades: {(np.abs(resultados_test['erro']) < 180).mean() * 100:.1f}%")
print(f"    % erros < 240 unidades: {(np.abs(resultados_test['erro']) < 240).mean() * 100:.1f}%")
print(f"    % erros < 300 unidades: {(np.abs(resultados_test['erro']) < 300).mean() * 100:.1f}%")

print(f"\n🏙️  RESUMO DO DATAFRAME DE TESTE:")
print(f"    Total de amostras: {len(resultados_test)}")
print(f"    Municípios únicos: {resultados_test['cod'].nunique()}")
print(f"    Período: {resultados_test['ano'].min()} a {resultados_test['ano'].max()}")
print(f"    Produtividade real: {resultados_test['prod'].mean():.1f} ± {resultados_test['prod'].std():.1f}")
print(f"    Produtividade predita: {resultados_test['prod_pred'].mean():.1f} ± {resultados_test['prod_pred'].std():.1f}")

# Mostrar apenas as colunas mais importantes do DataFrame de teste
print(f"\n🎯 DATAFRAME DE TESTE - COLUNAS PRINCIPAIS:")
print(resultados_test[['cod', 'ano', 'prod', 'prod_pred', 'erro']].head(15).round(2))

# ==============================================================================
# ANÁLISE DE MELHORIA
# ==============================================================================

print(f"\n📈 ANÁLISE DE MELHORIA:")
# CORREÇÃO: Usar os MAEs calculados e retornados pela função de avaliação
melhoria_val = ((baseline_val - mae_val) / baseline_val) * 100
melhoria_test = ((baseline_test - mae_test) / baseline_test) * 100

print(f"    Melhoria na validação: {melhoria_val:+.1f}%")
print(f"    Melhoria no teste: {melhoria_test:+.1f}%")

if melhoria_test > 0:
    print("    🎉 Modelo melhor que baseline tecnológica!")
else:
    print("    💡 Modelo precisa de ajustes")

# ==============================================================================
# ESTATÍSTICAS ADICIONAIS
# ==============================================================================

print(f"\n📊 ESTATÍSTICAS DOS DESVIOS TECNOLÓGICOS:")
print(f"    Desvio médio Treino: {y_train_desvio.mean():.2f} ± {y_train_desvio.std():.2f}")
print(f"    Desvio médio Validação: {y_val_desvio.mean():.2f} ± {y_val_desvio.std():.2f}")
print(f"    Desvio médio Teste: {y_test_desvio.mean():.2f} ± {y_test_desvio.std():.2f}")

print(f"\n🏙️  Informações do modelo:")
print(f"    Features utilizadas: {len(FEATURES_TENDENCIAS)}")
print(f"    Épocas treinadas: {len(historico.history['loss'])}")

# ==============================================================================
# ADICIONAR PREVISÕES AO DATAFRAME ORIGINAL (APENAS FEATURES)
# ==============================================================================

def adicionar_previsoes_com_zona(df_original, modelo, calibrador, escalonador, imputador):
    """Adiciona previsões do modelo como nova coluna no DataFrame original"""
    
    # Preparar dados para previsão
    X_num_total = escalonador.transform(imputador.transform(df_original[FEATURES_TENDENCIAS].to_numpy(dtype=float)))
    zonas_total = df_original['zona'].astype(int).to_numpy()
    
    # Fazer previsões calibradas para todos os dados
    desvios_pred_total = prever_com_calibracao(modelo, X_num_total, zonas_total, calibrador)
    
    # Calcular baseline tecnológica para todo o DataFrame (se não existir)
    if 'prod_baseline_tec' not in df_original.columns:
        print("    🔧 Recalculando baseline tecnológica para todo o dataset...")
        # Simplificado: assumindo que já foi calculado antes
        pass
    
    # Converter desvios para produtividade absoluta
    df_original['prod_pred'] = desvios_pred_total + df_original['prod_baseline_tec']
    
    # Adicionar também o desvio previsto
    df_original['desvio_pred'] = desvios_pred_total
    
    # Calcular componente tecnológico explícito
    df_original['componente_tecnologico'] = df_original['prod_baseline_tec'] - df_original['prod_baseline_tec'].mean()
    df_original['componente_climatico'] = df_original['desvio_pred']
    
    print(f"✅ Previsões adicionadas para {len(df_original)} registros")
    print(f"📊 Estatísticas das previsões:")
    print(f"    Produtividade prevista: {df_original['prod_pred'].mean():.1f} ± {df_original['prod_pred'].std():.1f}")
    print(f"    Desvio previsto: {df_original['desvio_pred'].mean():.1f} ± {df_original['desvio_pred'].std():.1f}")
    print(f"    Componente tecnológico: {df_original['componente_tecnologico'].mean():.1f} ± {df_original['componente_tecnologico'].std():.1f}")
    
    return df_original

# Adicionar previsões ao DataFrame original (DF) - CORREÇÃO: usar função com zona
DF_com_previsoes = adicionar_previsoes_com_zona(DF, modelo, calibrador, escalonador, imputador)

# ==============================================================================
# EXEMPLO DE VISUALIZAÇÃO DAS PREVISÕES
# ==============================================================================

print(f"\n📋 AMOSTRA DAS PREVISÕES (últimas 10 linhas):")
colunas_mostrar = [ID_MUNICIPIO, ID_ANO, TARGET, 'prod_baseline_tec', 'desvio_pred', 'prod_pred', 'taxa_crescimento_anual']
print(DF_com_previsoes[colunas_mostrar].tail(10).round(2))

# ==============================================================================
# ANÁLISE DO IMPACTO TECNOLÓGICO
# ==============================================================================

print(f"\n🔬 ANÁLISE DO FATOR TECNOLÓGICO:")
print(f"    Taxa média de crescimento: {DF_com_previsoes['taxa_crescimento_anual'].mean():.2f} unidades/ano")
print(f"    Municípios com crescimento positivo: {(DF_com_previsoes['taxa_crescimento_anual'] > 0).mean() * 100:.1f}%")
print(f"    Variação entre municípios: {DF_com_previsoes['taxa_crescimento_anual'].min():.2f} a {DF_com_previsoes['taxa_crescimento_anual'].max():.2f}")

# Análise temporal do componente tecnológico
crescimento_medio_por_ano = DF_com_previsoes.groupby(ID_ANO)['prod_baseline_tec'].mean()
if len(crescimento_medio_por_ano) > 1:
    crescimento_total = crescimento_medio_por_ano.iloc[-1] - crescimento_medio_por_ano.iloc[0]
    print(f"    Crescimento tecnológico médio no período: {crescimento_total:.1f} unidades")

print(f"\n🎯 ESTRUTURA FINAL DO DATAFRAME:")
print(f"    Colunas: {list(DF_com_previsoes.columns)}")
print(f"    Total de registros: {len(DF_com_previsoes)}")
print(f"    Período: {DF_com_previsoes[ID_ANO].min()} a {DF_com_previsoes[ID_ANO].max()}")
print(f"    Municípios: {DF_com_previsoes[ID_MUNICIPIO].nunique()}")

# ==============================================================================
# ANÁLISE DETALHADA DE CAUSAS DOS ERROS (VERSÃO SIMPLIFICADA)
# ==============================================================================

def analisar_causas_erro(resultados_test):
    """Versão simplificada que usa os erros já calculados sem precisar do modelo"""
    import numpy as np
    import pandas as pd

    print("\n🔍 ANÁLISE DETALHADA DE CAUSAS RAÍZES DOS ERROS")
    print("=" * 70)
    
    # Criar coluna de erro absoluto se não existir
    if 'erro_abs' not in resultados_test.columns:
        resultados_test['erro_abs'] = resultados_test['erro'].abs()
    
    # Estatísticas gerais do dataset
    n_amostras = len(resultados_test)
    n_municipios = resultados_test['cod'].nunique()
    n_anos = resultados_test['ano'].nunique()
    
    print(f"📈 ESTATÍSTICAS GERAIS DO DATASET:")
    print(f"    • Total de amostras: {n_amostras:,}")
    print(f"    • Municípios distintos: {n_municipios}")
    print(f"    • Anos analisados: {n_anos}")
    print(f"    • Período: {resultados_test['ano'].min()} - {resultados_test['ano'].max()}")
    
    # 1. Análise estatística completa dos erros
    print(f"\n📊 ESTATÍSTICAS DETALHADAS DOS ERROS:")
    print(f"    • Média do erro: {resultados_test['erro'].mean():.3f}")
    print(f"    • Mediana do erro: {resultados_test['erro'].median():.3f}")
    print(f"    • Desvio padrão: {resultados_test['erro'].std():.3f}")
    print(f"    • Erro absoluto médio (MAE): {resultados_test['erro_abs'].mean():.3f}")
    print(f"    • Raiz do erro quadrático médio (RMSE): {np.sqrt((resultados_test['erro']**2).mean()):.3f}")
    print(f"    • Erro máximo: {resultados_test['erro'].max():.3f}")
    print(f"    • Erro mínimo: {resultados_test['erro'].min():.3f}")
    print(f"    • Amplitude dos erros: {resultados_test['erro'].max() - resultados_test['erro'].min():.3f}")
    
    # ➕ Métricas extras (mantendo tudo que já existe)
    mae = resultados_test['erro_abs'].mean()
    rmse = np.sqrt((resultados_test['erro']**2).mean())
    rmse_mae_ratio = rmse / (mae + 1e-9)
    se_mae = resultados_test['erro_abs'].std(ddof=1) / np.sqrt(max(n_amostras, 1))
    ci95_low, ci95_high = mae - 1.96*se_mae, mae + 1.96*se_mae
    print(f"    • RMSE/MAE: {rmse_mae_ratio:.3f}")
    print(f"    • MAE (IC≈95%): [{ci95_low:.2f}, {ci95_high:.2f}]")
    
    # Distribuição dos erros
    erro_q25, erro_q75 = resultados_test['erro'].quantile([0.25, 0.75])
    print(f"    • 25º percentil: {erro_q25:.3f}")
    print(f"    • 75º percentil: {erro_q75:.3f}")
    print(f"    • IQR: {erro_q75 - erro_q25:.3f}")
    
    # Contagem de erros por magnitude
    limites_erro = pd.IntervalIndex.from_breaks([-np.inf, -600, -300, -180, -120, -60, 60, 120, 180, 300, 600, np.inf])
    labels_erro = ['< -600', '-600 a -300', '-300 a -180', '-180 a -120', '-120 a -60', '-60 a 60', '60 a 120', '120 a 180', '180 a 300', '300 a 600', '> 600']
    
    try:
        resultados_test['faixa_erro'] = pd.cut(resultados_test['erro'], bins=limites_erro)
        distribuicao_erros = resultados_test['faixa_erro'].value_counts().sort_index()
        
        print(f"\n📋 DISTRIBUIÇÃO DOS ERROS POR MAGNITUDE:")
        for faixa, count in distribuicao_erros.items():
            percentual = (count / n_amostras) * 100
            print(f"    • {faixa}: {count:>4} amostras ({percentual:5.1f}%)")
        # faixa central como indicador de estabilidade
        centro = distribuicao_erros.get(pd.Interval(-60.0, 60.0, closed='right'), 0)
        print(f"    • Share na faixa central (-60 a 60): {100*centro/max(n_amostras,1):.1f}%")
    except Exception as e:
        print(f"Erro ao criar faixas de erro: {e}")

    
    # 2. Erro por faixa de produtividade (mais detalhado)
    print(f"\n📊 ERRO POR FAIXA DE PRODUTIVIDADE (DETALHADO):")
    resultados_test['faixa_prod'] = pd.cut(resultados_test['prod'], bins=8)
    erro_por_faixa = resultados_test.groupby('faixa_prod', observed=True).agg(
        erro_count=('erro', 'count'),
        erro_mean=('erro', 'mean'),
        erro_std=('erro', 'std'),
        erro_min=('erro', 'min'),
        erro_max=('erro', 'max'),
        prod_mean=('prod', 'mean')
    ).round(3)
    
    for faixa, dados in erro_por_faixa.iterrows():
        count = dados['erro_count']
        mean_erro = dados['erro_mean']
        std_erro = dados['erro_std']
        mean_prod = dados['prod_mean']
        percentual = (count / n_amostras) * 100
        print(f"    • {faixa}:")
        print(f"          Amostras: {count} ({percentual:.1f}%) | Prod média: {mean_prod:.1f}")
        print(f"          Erro médio: {mean_erro:.3f} ± {std_erro:.3f}")
    
    # ➕ Erro relativo (quando possível)
    if (resultados_test['prod'] > 0).any():
        mask_pos = resultados_test['prod'] > 0
        mape = (resultados_test.loc[mask_pos, 'erro_abs'] / resultados_test.loc[mask_pos, 'prod']).mean()*100
        yhat = resultados_test['prod'].astype(float) + resultados_test['erro'].astype(float)
        smape = (200*np.abs(resultados_test['erro']) / (np.abs(resultados_test['prod']) + np.abs(yhat) + 1e-9)).mean()
        print(f"    • MAPE (prod>0): {mape:.2f}%")
        print(f"    • sMAPE: {smape:.2f}%")
    
    # 3. Análise geográfica detalhada
    print(f"\n🌎 ANÁLISE GEOGRÁFICA DETALHADA:")
    
    # Por latitude
    if 'lat' in resultados_test.columns:
        resultados_test['faixa_lat'] = pd.cut(resultados_test['lat'], bins=6, precision=2)
        erro_por_lat = resultados_test.groupby('faixa_lat', observed=True).agg(
            erro_mean=('erro', 'mean'),
            erro_std=('erro', 'std'),
            erro_count=('erro', 'count'),
            prod_mean=('prod', 'mean')
        ).round(3)
        
        print(f"    📍 POR LATITUDE:")
        for faixa, dados in erro_por_lat.iterrows():
            mean_erro = dados['erro_mean']
            count = dados['erro_count']
            mean_prod = dados['prod_mean']
            print(f"          {faixa}: Erro {mean_erro:.3f} | {count} amostras | Prod {mean_prod:.1f}")
    
    # Por longitude
    if 'lon' in resultados_test.columns:
        resultados_test['faixa_lon'] = pd.cut(resultados_test['lon'], bins=6, precision=2)
        erro_por_lon = resultados_test.groupby('faixa_lon', observed=True).agg(
            erro_mean=('erro', 'mean'),
            erro_std=('erro', 'std'),
            erro_count=('erro', 'count'),
            prod_mean=('prod', 'mean')
        ).round(3)
        
        print(f"    📍 POR LONGITUDE:")
        for faixa, dados in erro_por_lon.iterrows():
            mean_erro = dados['erro_mean']
            count = dados['erro_count']
            mean_prod = dados['prod_mean']
            print(f"          {faixa}: Erro {mean_erro:.3f} | {count} amostras | Prod {mean_prod:.1f}")
    
    # 3.1 ANÁLISE POR ZONAS EXISTENTES
    print(f"\n🗺️  ANÁLISE DETALHADA POR ZONAS:")
    
    erro_por_zona = resultados_test.groupby('zona').agg(
        erro_mean=('erro', 'mean'),
        erro_std=('erro', 'std'),
        erro_count=('erro', 'count'),
        erro_min=('erro', 'min'),
        erro_max=('erro', 'max'),
        erro_abs_mean=('erro_abs', 'mean'),
        erro_abs_std=('erro_abs', 'std'),
        prod_mean=('prod', 'mean'),
        prod_std=('prod', 'std')
    ).round(3)
    
    # Adicionar coordenadas médias se disponíveis
    if 'lat' in resultados_test.columns and 'lon' in resultados_test.columns:
        coords_por_zona = resultados_test.groupby('zona')[['lat', 'lon']].mean().round(3)
        erro_por_zona = erro_por_zona.join(coords_por_zona)
    
    print(f"    📊 DESEMPENHO POR ZONA GEOGRÁFICA:")
    zonas_ordenadas = erro_por_zona.sort_values('erro_abs_mean', ascending=False)
    
    for zona, dados in zonas_ordenadas.iterrows():
        n_amostras_zona = dados['erro_count']
        percentual_amostras = (n_amostras_zona / n_amostras) * 100
        erro_medio = dados['erro_mean']
        erro_abs_medio = dados['erro_abs_mean']
        std_erro = dados['erro_std']
        prod_media = dados['prod_mean']
        
        if erro_abs_medio > 600:
            severidade = "🟣 PÉSSIMO"
        elif erro_abs_medio > 480:
            severidade = "🔴 RUIM"
        elif erro_abs_medio > 360:
            severidade = "🟠 MODERADO"
        elif erro_abs_medio > 180:
            severidade = "🟡 RAZOÁVEL"
        elif erro_abs_medio > 60:
            severidade = "🟢 BOM"
        else:
            severidade = "⚪ EXCELENTE"
        
        print(f"          Zona: {zona} {severidade} | Erro médio: {erro_medio:8.1f} | Erro abs: {erro_abs_medio:6.1f} ± {std_erro:.1f}")
        print(f"          Amostras: {n_amostras_zona:4.0f} ({percentual_amostras:4.1f}%) | Prod média: {prod_media:6.1f}")
    
    zonas_criticas = zonas_ordenadas[zonas_ordenadas['erro_abs_mean'] > 300]
    if not zonas_criticas.empty:
        print(f"\n    🚨 ZONAS CRÍTICAS (ERRO > 300):")
        for zona in zonas_criticas.index:
            erro_abs = zonas_criticas.loc[zona, 'erro_abs_mean']
            n_amostras_zona = zonas_criticas.loc[zona, 'erro_count']
            print(f"          • {zona}: Erro absoluto médio = {erro_abs:.1f} ({n_amostras_zona:.0f} amostras)")
    
    # ➕ Balanced MAE entre zonas
    balanced_mae = erro_por_zona['erro_abs_mean'].mean()
    desvio_balanced = erro_por_zona['erro_abs_mean'].std()
    print(f"\n⚖️  BALANCED MAE ENTRE ZONAS: {balanced_mae:.2f} (desvio {desvio_balanced:.2f})")
    
    # 4. Análise temporal detalhada
    print(f"\n📅 ANÁLISE TEMPORAL DETALHADA:")
    
    erro_por_ano = resultados_test.groupby('ano').agg(
        erro_mean=('erro', 'mean'),
        erro_std=('erro', 'std'),
        erro_count=('erro', 'count'),
        prod_mean=('prod', 'mean')
    ).round(3)
    
    print(f"    📊 POR ANO:")
    for ano, dados in erro_por_ano.iterrows():
        mean_erro = dados['erro_mean']
        std_erro = dados['erro_std']
        count = dados['erro_count']
        mean_prod = dados['prod_mean']
        print(f"          {ano}: Erro {mean_erro:.3f} ± {std_erro:.3f} | Prod {mean_prod:.1f}")
    
    # 5. Análise climática detalhada
    print(f"\n🌧️  ANÁLISE CLIMÁTICA DETALHADA:")
    
    if 'spi' in resultados_test.columns:
        bins_spi = [
            -np.inf, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
             0.5,   1.0,  1.5,  2.0,  2.5,  3.0,  np.inf
        ]

        labels_spi = [
            'Extremamente Seco',
            'Muito Seco',
            'Severamente Seco',
            'Seco',
            'Moderadamente Seco',
            'Levemente Seco',
            'Normal',
            'Levemente Úmido',
            'Moderadamente Úmido',
            'Úmido',
            'Muito Úmido',
            'Severamente Úmido',
            'Extremamente Úmido'
        ]

        resultados_test['tipo_ano_detalhado'] = pd.cut(
            resultados_test['spi'].astype(float),
            bins=bins_spi,
            labels=labels_spi,
            right=True,
            include_lowest=True
        )
        
        erro_por_clima = resultados_test.groupby('tipo_ano_detalhado', observed=True).agg(
            erro_mean=('erro', 'mean'),
            erro_std=('erro', 'std'),
            erro_count=('erro', 'count'),
            prod_mean=('prod', 'mean')
        ).round(3)
        
        for tipo, dados in erro_por_clima.iterrows():
            mean_erro = dados['erro_mean']
            std_erro = dados['erro_std']
            count = dados['erro_count']
            mean_prod = dados['prod_mean']
            percentual = (count / n_amostras) * 100
            print(f"    • {tipo:20}: Erro {mean_erro:7.3f} ± {std_erro:.3f} | {count:>3.0f} amostras ({percentual:4.1f}%)")
    
    # 6. Municípios problemáticos - análise aprofundada
    print(f"\n🚨 ANÁLISE APROFUNDADA DOS MUNICÍPIOS PROBLEMÁTICOS:")
    
    erro_por_municipio_detalhado = resultados_test.groupby('cod').agg(
        erro_mean=('erro', 'mean'),
        erro_std=('erro', 'std'),
        erro_count=('erro', 'count'),
        erro_min=('erro', 'min'),
        erro_max=('erro', 'max'),
        erro_abs_mean=('erro_abs', 'mean'),
        erro_abs_std=('erro_abs', 'std'),
        prod_mean=('prod', 'mean'),
        prod_std=('prod', 'std'),
        zona_first=('zona', 'first'),
        ano_nunique=('ano', 'nunique')
    ).round(3)
    
    # Adicionar coordenadas se disponíveis
    if 'lat' in resultados_test.columns and 'lon' in resultados_test.columns:
        coords_por_municipio = resultados_test.groupby('cod')[['lat', 'lon']].first()
        erro_por_municipio_detalhado = erro_por_municipio_detalhado.join(coords_por_municipio)
    
    top_problematicos = erro_por_municipio_detalhado.nlargest(15, 'erro_abs_mean')
    
    print(f"    📍 TOP 15 MUNICÍPIOS COM MAIOR ERRO ABSOLUTO MÉDIO:")
    for idx, (municipio, dados) in enumerate(top_problematicos.iterrows(), 1):
        erro_medio = dados['erro_mean']
        erro_abs_medio = dados['erro_abs_mean']
        std_erro = dados['erro_std']
        n_amostras_mun = dados['erro_count']
        prod_media = dados['prod_mean']
        n_anos_mun = dados['ano_nunique']
        zona = dados['zona_first']
        
        print(f"       {idx:2d}. Município {municipio} (Zona: {int(zona)}):")
        print(f"             Erro médio: {erro_medio:7.3f} | Erro abs médio: {erro_abs_medio:7.3f} ± {std_erro:.3f}")
        print(f"             Amostras: {n_amostras_mun:3.0f} | Anos: {n_anos_mun:2.0f} | Prod média: {prod_media:6.1f}")
    
    # ➕ Concentração de erro (Pareto simples)
    total_abs = resultados_test['erro_abs'].sum()
    contrib_mun = resultados_test.groupby('cod')['erro_abs'].sum().sort_values(ascending=False)
    frac_80 = (contrib_mun.cumsum()/max(total_abs,1e-9) >= 0.80).idxmax() if not contrib_mun.empty else None
    k80 = (contrib_mun.cumsum()/max(total_abs,1e-9) >= 0.80).idxmax() if not contrib_mun.empty else None
    if not contrib_mun.empty:
        n80 = (contrib_mun.cumsum()/max(total_abs,1e-9) <= 0.80).sum() + 1
        print(f"\n📌 PARETO: Top {n80} municípios concentram ~80% do erro absoluto total.")

    # 7. Análise de correlações
    print(f"\n📈 ANÁLISE DE CORRELAÇÕES:")
    variaveis_correlacao = ['prod', 'lat', 'lon', 'prec_anom_safra', 'ano', 'tmax_anom', 'spi', 'POS', 'SOS', 'EOS']
    variaveis_disponiveis = [var for var in variaveis_correlacao if var in resultados_test.columns]
    
    if variaveis_disponiveis:
        correlacoes = resultados_test[['erro'] + variaveis_disponiveis].corr(numeric_only=True)['erro'].drop('erro', errors='ignore')
        
        for var, corr in correlacoes.items():
            significancia = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
            print(f"    • Correlação erro vs {var:15}: {corr:7.3f} {significancia}")
    else:
        correlacoes = pd.Series(dtype=float)
    
    # 8. Análise de viés sistemático
    print(f"\n🎯 ANÁLISE DE VIÉS SISTEMÁTICO:")
    n_erros_positivos = (resultados_test['erro'] > 0).sum()
    n_erros_negativos = (resultados_test['erro'] < 0).sum()
    n_erros_zero = (resultados_test['erro'] == 0).sum()
    
    print(f"    • Erros positivos (subestimação): {n_erros_positivos:>5} ({n_erros_positivos/n_amostras*100:5.1f}%)")
    print(f"    • Erros negativos (superestimação): {n_erros_negativos:>5} ({n_erros_negativos/n_amostras*100:5.1f}%)")
    print(f"    • Erros zero: {n_erros_zero:>5} ({n_erros_zero/n_amostras*100:5.1f}%)")
    
    skewness = resultados_test['erro'].skew()
    kurtosis = resultados_test['erro'].kurtosis()
    print(f"    • Assimetria (skewness): {skewness:.3f} {'(viés positivo)' if skewness > 0.5 else '(viés negativo)' if skewness < -0.5 else '(próximo de normal)'}")
    print(f"    • Curtose: {kurtosis:.3f} {'(caudas pesadas)' if kurtosis > 1 else '(caudas leves)' if kurtosis < -1 else '(normal)'}")
    
    # ➕ Checagens de calibração (reconstruindo predição: yhat = y + erro)
    y = resultados_test['prod'].astype(float)
    yhat = y + resultados_test['erro'].astype(float)
    tau = 0.70
    cobertura = float((y <= yhat).mean())  # para quantil τ, P(Y ≤ ŷ) ≈ τ
    desv_cobertura = cobertura - tau
    # Regressão simples y ~ yhat (slope ≈ 1 e intercept ≈ 0 indicam boa calibração)
    try:
        coef = np.polyfit(yhat, y, 1)
        slope_calib, intercept_calib = coef[0], coef[1]
    except Exception:
        slope_calib, intercept_calib = np.nan, np.nan
    print(f"\n📐 CHECAGEM DE CALIBRAÇÃO (τ=0.70): cobertura={cobertura:.3f} (desvio {desv_cobertura:+.3f}) | slope={slope_calib:.3f} | intercept={intercept_calib:.2f}")
    # Decis de predição
    try:
        resultados_test['decile_pred'] = pd.qcut(yhat, 10, duplicates='drop')
        calib_decis = resultados_test.groupby('decile_pred', observed=True).agg(
            yhat_mean=('decile_pred', lambda s: np.nan),  # placeholder
            y_mean=('prod', 'mean'),
            err_mean=('erro', 'mean'),
            mae_mean=('erro_abs', 'mean')
        )
        # Mostrar resumo compactado: apenas MAE por decil
        mae_decis = resultados_test.groupby('decile_pred', observed=True)['erro_abs'].mean().round(2)
        print("    • MAE por decil de predição:", list(mae_decis.values))
    except Exception:
        pass
    
    # 9. Resumo executivo
    print(f"\n💡 RESUMO EXECUTIVO E RECOMENDAÇÕES:")
    maior_erro_abs = resultados_test['erro_abs'].max()
    
    # Encontrar faixa mais problemática
    if not erro_por_faixa.empty:
        faixa_mais_problematica_idx = erro_por_faixa['erro_mean'].abs().idxmax()
        print(f"    📊 FAIXA MAIS PROBLEMÁTICA: {faixa_mais_problematica_idx}")
    
    # Encontrar clima mais problemático
    if 'tipo_ano_detalhado' in resultados_test.columns and 'erro_mean' in locals().get('erro_por_clima', pd.DataFrame()):
        if not erro_por_clima.empty:
            clima_mais_problematico_idx = erro_por_clima['erro_mean'].abs().idxmax()
            print(f"    🌧️  CLIMA MAIS PROBLEMÁTICO: {clima_mais_problematico_idx}")
    
    if not zonas_ordenadas.empty:
        zona_mais_problematica = zonas_ordenadas.iloc[0]
        print(f"    🗺️  ZONA MAIS PROBLEMÁTICA: {zonas_ordenadas.index[0]} (Erro abs: {zona_mais_problematica['erro_abs_mean']:.1f})")
    
    print(f"    🔴 MAIOR PROBLEMA: Erro máximo de {maior_erro_abs:.2f}")
    
    recomendacoes = []
    
    # Regras já existentes
    if abs(skewness) > 0.5:
        recomendacoes.append("Viés sistemático detectado — testar transformação do alvo (ex.: log1p) ou correção pós-treino por zona/ano")
    
    if 'prec_anom_safra' in correlacoes and abs(correlacoes['prec_anom_safra']) > 0.2:
        recomendacoes.append("Erro sensível à precipitação — enriquecer janelas climáticas (SPI multi-escala, CHUVA_30/60/90) e interações clima×fenologia")
    
    if 'erro_abs_mean' in top_problematicos:
        if top_problematicos['erro_abs_mean'].mean() > 300:
            recomendacoes.append("Muitos municípios críticos — revisar qualidade/consistência das séries e aplicar suavização por vizinhança (média zonal)")
    
    if not zonas_criticas.empty:
        zonas_criticas_str = [str(z) for z in zonas_criticas.index]
        recomendacoes.append(f"Focar nas zonas críticas ({', '.join(zonas_criticas_str)}) com calibração local (offset) e/ou pesos por zona")
    
    # ✅ Novas regras automáticas
    
    # 9.1 Caudas pesadas e outliers
    total_abs = resultados_test['erro_abs'].sum()
    top1_cut = resultados_test['erro_abs'].quantile(0.99)
    peso_top1 = resultados_test.loc[resultados_test['erro_abs'] >= top1_cut, 'erro_abs'].sum() / max(total_abs, 1e-9)
    share_extremos = (resultados_test.eval("erro<-200 or erro>200").mean())
    if kurtosis > 1.5 or peso_top1 > 0.15 or share_extremos > 0.25 or rmse_mae_ratio > 1.8:
        recomendacoes.append("Caudas pesadas — usar Huber/QuantileLoss, winsorizar alvos e aplicar cap de previsão em produção")
    
    # 9.2 Heteroscedasticidade por produtividade
    if 'prod' in resultados_test.columns:
        corr_abs_prod = resultados_test[['erro_abs','prod']].corr(numeric_only=True).loc['erro_abs','prod']
        if abs(corr_abs_prod) > 0.25:
            recomendacoes.append("Erro cresce com produtividade — ponderar por quantis de prod e modelar variância (loss robusta)")
        if len(erro_por_faixa) > 3 and erro_por_faixa['prod_mean'].notna().all():
            rho = erro_por_faixa[['prod_mean','erro_mean']].corr(method='spearman').loc['prod_mean','erro_mean']
            if abs(rho) > 0.5:
                recomendacoes.append("Tendência monotônica erro×prod — calibrar função monotônica (Isotonic) ou ajustar transformação do alvo")
    
    # 9.3 Drift temporal
    if 'ano' in resultados_test.columns:
        corr_abs_ano = resultados_test[['erro_abs','ano']].corr(numeric_only=True).loc['erro_abs','ano']
        if abs(corr_abs_ano) > 0.2:
            recomendacoes.append("Sinal de drift no tempo — incluir features de tendência/ano, validação temporal e re-treinos frequentes")
        ano_pior = erro_por_ano['erro_mean'].abs().idxmax()
        if erro_por_ano.loc[ano_pior,'erro_count'] >= max(50, 0.02*n_amostras):
            recomendacoes.append(f"Ano crítico {int(ano_pior)} — ajustar pesos por ano e avaliar recalibração por safra")
    
    # 9.4 Padrão espacial
    if 'lat' in resultados_test.columns and 'lon' in resultados_test.columns:
        corr_erro_lat = resultados_test[['erro','lat']].corr(numeric_only=True).loc['erro','lat'] 
        corr_erro_lon = resultados_test[['erro','lon']].corr(numeric_only=True).loc['erro','lon']
        if max(abs(corr_erro_lat), abs(corr_erro_lon)) > 0.2:
            recomendacoes.append("Gradiente espacial dos resíduos — incluir termos de coordenadas (splines) ou krigagem dos resíduos")
    
    # Bias por zona consistente
    sinais_zona = resultados_test.groupby('zona')['erro'].mean().pipe(lambda s: (s.abs()>s.abs().median()).mean())
    if sinais_zona > 0.4:
        recomendacoes.append("Bias consistente por zona — offsets fixos por zona ou embeddings regionais na rede")
    
    # 9.5 Extremos climáticos
    if 'tipo_ano_detalhado' in resultados_test.columns and 'erro_mean' in locals().get('erro_por_clima', pd.DataFrame()):
        if not erro_por_clima.empty and erro_por_clima['erro_mean'].abs().idxmax() in ['Muito Seco','Muito Chuvoso']:
            recomendacoes.append("Piora em extremos — reforçar amostragem de extremos e criar features de limiar/saturação")
    
    if 'spi' in correlacoes and abs(correlacoes['spi']) > 0.2:
        recomendacoes.append("Erro ligado ao SPI — avaliar janelas 1/3/6 meses e defasagens alinhadas à fenologia")
    
    if 'tmax_anom' in correlacoes and abs(correlacoes['tmax_anom']) > 0.2:
        recomendacoes.append("Sensibilidade térmica — inserir ondas de calor (dias > p90) e graus-dia extremos")
    
    # 9.6 Esparsidade e desbalanceamento
    share_poucos_anos = (erro_por_municipio_detalhado['ano_nunique'] < 3).mean()
    if share_poucos_anos > 0.4:
        recomendacoes.append("Muitos municípios com poucas safras — pooling por zona e shrinkage bayesiano")
    
    desbalance_zona = erro_por_zona['erro_count'].max() / max(1, erro_por_zona['erro_count'].min())
    if desbalance_zona > 5:
        recomendacoes.append("Desbalanceamento entre zonas — usar sample_weight por zona na etapa de treino")
    
    # 9.7 Fenologia (se disponível)
    for fen in ['POS','SOS','EOS']:
        if fen in resultados_test.columns:
            corr_fen = resultados_test[['erro', fen]].corr(numeric_only=True).loc['erro', fen]
            if abs(corr_fen) > 0.2:
                recomendacoes.append(f"Resíduo ligado a {fen} — revisar janelas fenológicas regionais e interações clima×{fen}")
    
    # 9.8 Calibração por sinal
    frac_pos = n_erros_positivos / max(n_amostras,1)
    frac_neg = n_erros_negativos / max(n_amostras,1)
    if max(frac_pos, frac_neg) > 0.6:
        recomendacoes.append("Desequilíbrio de sinal — aplicar correção de viés pós-treino (regressão dos resíduos) por zona/ano")
    
    # 9.9 Calibração quantílica
    if not np.isnan(slope_calib) and (abs(desv_cobertura) > 0.05 or slope_calib < 0.9 or slope_calib > 1.1):
        recomendacoes.append("Desvio de cobertura/escala — manter Isotonic, ajustar τ efetivo e revisar pesos da QuantileLoss")
    
    # 9.10 Balanced MAE alto vs MAE global
    if balanced_mae > 1.25*mae:
        recomendacoes.append("Erro médio por zona elevado frente ao global — aplicar reamostragem/ponderação para equilibrar regiões")
    
    # 9.11 Concentração de erro (Pareto)
    if not contrib_mun.empty and n_municipios > 0:
        n80 = (contrib_mun.cumsum()/max(total_abs,1e-9) <= 0.80).sum() + 1
        if n80 / max(n_municipios,1) < 0.3:
            recomendacoes.append(f"Erro concentrado — priorizar dados/ajustes nos ~{n80} municípios que somam 80% do erro")
    
    # 9.12 MAPE elevado
    if (resultados_test['prod'] > 0).any():
        if 'mape' in locals() and mape > 15:
            recomendacoes.append("Erro relativo alto — testar transformação do alvo e perdas sensíveis a escala relativa (sMAPE/Log-cosh)")
    
    if len(recomendacoes) > 0:
        print(f"    💡 RECOMENDAÇÕES:")
        for i, rec in enumerate(recomendacoes, 1):
            print(f"          {i}. {rec}")
    
    # Limpar colunas de análise
    cols_to_drop = ['faixa_erro', 'faixa_prod', 'faixa_lat', 'faixa_lon', 'tipo_ano_detalhado', 'decile_pred']
    resultados_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 4. CHAMAR A VERSÃO SIMPLIFICADA (SEM PRECISAR DO model.predict)
analisar_causas_erro(resultados_test)

# ==============================================================================
# PREVISÃO PARA 2025
# ==============================================================================

print("\nPREVISÃO 2025 — Amostra (10 primeiras linhas):")

# Preparar dados de 2025 - CORRIGIDO
X_2025_num = escalonador.transform(
    imputador.transform(DF2025[FEATURES_TENDENCIAS].to_numpy(dtype=float))
)
zonas_2025 = DF2025['zona'].astype(int).to_numpy()

# Predição do DESVIO (componente climático) - CORRIGIDO: usar calibrador
desvios_pred_2025 = prever_com_calibracao(modelo, X_2025_num, zonas_2025, calibrador)

# Converte para produtividade absoluta (tendência + desvio)
prod_pred_2025 = DF2025['prod_baseline_tec'].to_numpy(dtype=float) + desvios_pred_2025

# Monta resultado
prev_2025 = DF2025.copy()
prev_2025['desvio_pred'] = desvios_pred_2025
prev_2025['prod_pred'] = prod_pred_2025
prev_2025['componente_climatico'] = prev_2025['desvio_pred']
prev_2025['componente_tecnologico'] = prev_2025['prod_baseline_tec'] - prev_2025['prod_baseline_tec'].mean()

# Ordena visão rápida
cols_show = ['cod','ano','prod_baseline_tec','desvio_pred','prod_pred','lon','lat']
print(prev_2025[cols_show].head(10).round(2))

# Resumos úteis
print("\nResumo 2025:")
print(f"    Municípios: {prev_2025['cod'].nunique()}")
print(f"    Prod. prevista (média ± dp): {prev_2025['prod_pred'].mean():.1f} ± {prev_2025['prod_pred'].std():.1f}")
print(f"    Desvio previsto (média): {prev_2025['desvio_pred'].mean():.1f}")

DF_final = pd.concat([DF_com_previsoes,prev_2025])
DF_final.sort_values(by=['ano','cod'],inplace=True)
DF_final.to_csv('result-train.final.csv', index=False)
print("\n✅ Arquivo final 'result-train.final.csv' salvo com sucesso.")