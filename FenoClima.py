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
# GLOBAL SETTINGS
# ==============================================================================
TARGET = 'prod'
ID_MUNICIPIO = 'cod'
ID_ANO = 'ano'
ANOS_TESTE = {2005, 2012, 2017, 2023, 2024}

# ==============================================================================
# DATA LOADING AND PREPARATION
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
        print(f"Error reading {f}: {e}")

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

# convert date columns to day-of-year
datas_cols = ['data_ini', 'data_max', 'data_fim', 'ROI_DT', 'ROD_DT', 'MOS_DT1', 'MOS_DT2']
DF[datas_cols] = DF[datas_cols].apply(lambda s: pd.to_datetime(s, errors='coerce').dt.dayofyear.astype('Int16'))

DF.drop('safra_temporada', axis=1, inplace=True, errors='ignore')
original_features = list(DF.columns)

# ==============================================================================
# SEQUENTIAL FILTERS
# ==============================================================================

print("INITIAL DATAFRAME STATE:")
print(f"    Total records: {len(DF)}")
print(f"    Unique municipalities: {DF['cod'].nunique()}")
print(f"    Period: {DF['ano'].min()} to {DF['ano'].max()}")

# Filter 1: remove duplicates keeping the smallest data_max
print(f"\nAPPLYING FILTER 1: Remove duplicate records (smallest data_max)")
registros_antes_dup = len(DF)
duplicatas_antes = DF.duplicated(subset=['cod', 'ano']).sum()

# sort by cod, ano and data_max (ascending so the smallest data_max comes first)
DF = DF.sort_values(['cod', 'ano', 'data_max'])

# drop duplicates keeping the first occurrence (smallest data_max)
DF = DF.drop_duplicates(subset=['cod', 'ano'], keep='first')
DF.reset_index(drop=True, inplace=True)

duplicatas_removidas = registros_antes_dup - len(DF)
print(f"    Duplicate records found: {duplicatas_antes}")
print(f"    Records removed: {duplicatas_removidas}")
print(f"    After filter 1 - Records: {len(DF)}, Municipalities: {DF['cod'].nunique()}")

# Filter 2: remove municipalities with fewer than 5 records (up to 2022)
print(f"\nAPPLYING FILTER 2: Remove municipalities with < 5 records (up to 2022)")

contagem_por_municipio = DF[DF['ano'] <= 2022].groupby('cod').size()
municipios_validos = contagem_por_municipio[contagem_por_municipio >= 5].index
municipios_insuficientes = contagem_por_municipio[contagem_por_municipio < 5]

print(f"    Municipalities with insufficient data (<5 records): {len(municipios_insuficientes)}")
print(f"    Valid municipalities (>=5 records): {len(municipios_validos)}")

if len(municipios_insuficientes) > 0:
    print(f"    Removing municipalities: {municipios_insuficientes.index.tolist()}")

    registros_antes = len(DF)
    DF = DF[DF['cod'].isin(municipios_validos)].copy()
    DF.reset_index(drop=True, inplace=True)

    print(f"    Records removed: {registros_antes - len(DF)}")
    print(f"    Records retained: {len(DF)}")
else:
    print(f"    No municipalities with insufficient data found")

print(f"    After filter 2 - Records: {len(DF)}, Municipalities: {DF['cod'].nunique()}")

# Final structure check
print(f"\nFINAL STRUCTURE AFTER FILTERS:")
print(f"    Total records: {len(DF)}")
print(f"    Unique municipalities: {DF['cod'].nunique()}")
print(f"    Period: {DF['ano'].min()} to {DF['ano'].max()}")

contagem_final = DF.groupby('cod').size()
print(f"    Mean records per municipality: {contagem_final.mean():.1f}")
print(f"    Min records per municipality: {contagem_final.min()}")
print(f"    Max records per municipality: {contagem_final.max()}")

# ==============================================================================
# FEATURE ENGINEERING WITH TECHNOLOGY FACTOR
# ==============================================================================

def criar_interacoes_estrategicas(df):
    def relu(x):
        return np.maximum(0, x)

    # z-score per season (or global if preferred)
    def z(s):
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    df = df.copy()
    eps = 1e-6

    # easily tunable thresholds
    TH_PLANTIO_MM   = 40.0     # minimum rainfall for crop establishment
    CALOR_EXTREMO   = 2.0      # Tmax anomaly (°C) for extreme heat
    SPI_SECA_EXT    = -1.5     # extreme drought SPI threshold
    PRECOCE_DIAS    = 95       # LOS < PRECOCE_DIAS => early harvest
    JANELA_PRE_COL  = 15       # last days before harvest (most heat-sensitive)

    # DOY already numeric
    ini_doy = pd.to_numeric(df['data_ini'], errors='coerce')
    pos_doy = pd.to_numeric(df['POS'], errors='coerce')
    fim_doy = pd.to_numeric(df['data_fim'], errors='coerce')

    dur_veg   = pd.to_numeric(df['MOS_DT1'], errors='coerce')  # SOS→POS
    dur_rep   = pd.to_numeric(df['MOS_DT2'], errors='coerce')  # POS→EOS
    dur_ciclo = pd.to_numeric(df['LOS'],     errors='coerce')  # SOS→EOS

    # -------------------------------------------------------------------------
    # Municipality-relative POS: POS_year(municipality) / POS_max(municipality)
    chave_muni = 'cod' if 'cod' in df.columns else ('municipio' if 'municipio' in df.columns else None)
    if chave_muni is not None:
        pos_max_muni = pos_doy.groupby(df[chave_muni]).transform('max')
        df['POS_relativo'] = (pos_doy / (pos_max_muni + eps)).clip(0, 1)
    else:
        # no municipality key — use global maximum
        df['POS_relativo'] = (pos_doy / (pos_doy.max() + eps)).clip(0, 1)
    # -------------------------------------------------------------------------

    df['FENO_POS_MOS'] = df['POS'] + df['MOS']
    df['FORCA_CALOR_MAX'] = (df['tmax_anom'] + df['tmed_std']) * df['tmax_max']
    df['STRESS_CALOR_SECA'] = df['tmax_anom'] + (-df['spi'])
    df['FENO_POS_MOS_SPI10'] = df['POS'] + df['MOS'] + 0.1 * df['spi']

    C  = relu(z(df["tmax_anom"].astype(float)))
    S1 = relu(z(-df["prec_anom_safra"].astype(float)))
    S2 = relu(z(-df["spi"].astype(float)))
    V  = relu(z(df["tmed_std"].astype(float)))
    X  = C * (S1 + S2)
    df["INTERACAO_CALOR_SECA"] = C + (S1 + S2) + X + V

    # canopy amplitude
    amp = df['AOS'] if 'AOS' in df.columns else df['MOS']
    amp = amp.fillna(df['MOS'])

    # ----- Water -----
    df['sazonalidade_chuva_plantio']  = df['prec_plantio']  / (df['prec_safra'] + eps)
    df['sazonalidade_chuva_colheita'] = df['prec_colheita'] / (df['prec_safra'] + eps)
    df['chuva_vs_roi'] = df['prec_plantio'] / (np.minimum(dur_veg, 30).clip(lower=1) + eps)
    df['chuva_vs_pos'] = df['anom_prec_colheita'] / (dur_rep.clip(lower=1) + eps)

    # ----- Heat / Cold -----
    df['amplitude_termica'] = df['tmax_max'] - df['tmin_min']
    df['estresse_termico_rel'] = np.maximum(df['tmax_anom'], 0) / (
        (df['tmax_mean'] - df['tmin_min']).clip(lower=0.5) + eps
    )
    df['calor_fase_sensivel_aprox'] = np.maximum(df['tmax_anom'], 0) * dur_rep.fillna(0)

    # ----- Phenology -----
    df['assimetria_fases'] = (dur_veg + eps) / (dur_rep + eps)
    df['rod_roi_ratio']    = np.abs(df['ROD']) / (np.abs(df['ROI']) + eps)
    df['vigor_x_ciclo']    = amp * (dur_ciclo + 1)

    # ----- Geo-climate -----
    if 'lat' in df.columns:
        df['gradiente_lat_tmax'] = df['lat'] * df['tmax_anom']
        df['gradiente_lat_spi']  = df['lat'] * df['spi']

    # ----- Thermal and hydro-thermal indicators (no water-balance model) -----
    df['eficiencia_termica']       = df['gdd'] / (dur_ciclo + eps)
    df['estresse_termico_hidrico'] = np.maximum(df['tmax_anom'], 0) * (-df['spi'].clip(upper=0))
    df['produtividade_termica']    = (df['gdd'] / (dur_ciclo + eps)) * amp
    if 'tmin_anom' in df.columns and 'lat' in df.columns:
        df['lat_x_tmin'] = df['lat'] * df['tmin_anom']
    df['consistencia_fenologica']  = amp / (dur_ciclo + eps)

    # ----- Log transformations -----
    for col in ['prec_safra', 'prec_plantio', 'prec_colheita']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    # ----- Phase-level stress indices -----
    stress_def  = np.clip((TH_PLANTIO_MM - df['prec_plantio']) / TH_PLANTIO_MM, 0, 1)
    stress_anom = np.clip((-df['anom_prec_plantio']) / TH_PLANTIO_MM, 0, 1)
    df['stress_plantio']   = np.maximum(stress_def, stress_anom)
    df['stress_floracao']  = np.maximum(df['tmax_anom'], 0) * (-df['spi'].clip(upper=0))
    df['stress_maturacao'] = (
        np.maximum(df['tmax_anom'], 0) * (dur_rep / (dur_ciclo + eps))
        + np.maximum(df['anom_prec_colheita'], 0) / TH_PLANTIO_MM
    )

    # ----- Extreme event flags -----
    df['seca_extrema']  = (df['spi'] < SPI_SECA_EXT).astype(int)
    df['calor_extremo'] = (df['tmax_anom'] > CALOR_EXTREMO).astype(int)
    df['geada_risco']   = ((ini_doy <= 270) & (df['tmin_min'] < 2.0)).astype(int)

    # ----- Temporal interactions -----
    df['chuva_pos_plantio']    = df['chuva_vs_roi']
    df['calor_antes_colheita'] = np.maximum(df['tmax_anom'], 0) * np.minimum(dur_rep.fillna(0), JANELA_PRE_COL)

    # ----- Aggregate stress index (no water-balance) -----
    df['indice_estresse'] = df['estresse_termico_hidrico']

    # ----- Seasonality flags -----
    df['plantio_tardio']   = ((ini_doy > 60) & (ini_doy < 244)).astype(int)
    los_ref = dur_ciclo.fillna((fim_doy - ini_doy).clip(lower=0))
    df['colheita_precoce'] = (los_ref < PRECOCE_DIAS).astype(int)
    df['saldo_hidrico']  = df['prec_safra'] + df['spi']
    df['moisture_relief'] = df['spi'] + df['prec_anom_safra']

    return df

def criar_features_com_tecnologia(df):
    """Creates features with an explicit technology component — TECHNOLOGY NEVER REGRESSES!"""
    df_fe = df.copy()

    # 1. Normalise year for the technology component
    ano_min, ano_max = df_fe[ID_ANO].min(), df_fe[ID_ANO].max()
    df_fe['ano_normalizado'] = (df_fe[ID_ANO] - ano_min) / (ano_max - ano_min)

    # 2. Compute national technology trend (regression across all municipalities)
    def calcular_tendencia_tecnologica_pais(df_pais):
        """Computes the technology trend for the entire country via linear regression."""
        df_agg = df_pais[df_pais[TARGET].notna()].groupby(ID_ANO)[TARGET].mean().reset_index()

        if len(df_agg) < 2:
            return 0

        X = df_agg[ID_ANO].values.astype(float)
        y = df_agg[TARGET].values.astype(float)

        try:
            coef, intercept = np.polyfit(X.flatten(), y, 1)
            print(f"    National technology trend: {coef:.6f}")
            return coef
        except:
            return 0

    print("Computing national technology trend...")
    tendencia_pais = calcular_tendencia_tecnologica_pais(df_fe)

    # 3. Compute per-municipality technology trend (non-NaN data only)
    def calcular_tendencia_tecnologica_municipio(subdf):
        """Computes the annual growth rate for a single municipality."""
        subdf = subdf[~subdf[ID_ANO].isin(ANOS_TESTE)]
        mask_nao_nan = subdf[TARGET].notna()
        if mask_nao_nan.sum() < 2:
            return tendencia_pais  # fall back to national trend if insufficient data

        X = subdf.loc[mask_nao_nan, [ID_ANO]].values.astype(float)
        y = subdf.loc[mask_nao_nan, TARGET].values.astype(float)

        try:
            coef, intercept = np.polyfit(X.flatten(), y, 1)
            # negative municipal trend → use the national trend instead
            if coef < 0:
                return tendencia_pais
            return coef
        except:
            return tendencia_pais

    print("Computing per-municipality technology trends...")
    tendencias_municipio = df_fe.groupby(ID_MUNICIPIO).apply(calcular_tendencia_tecnologica_municipio)

    municipios_com_tendencia_negativa = (tendencias_municipio == tendencia_pais).sum()
    municipios_com_tendencia_positiva = (tendencias_municipio != tendencia_pais).sum()

    print(f"    Municipalities with positive own trend: {municipios_com_tendencia_positiva}")
    print(f"    Municipalities using national trend: {municipios_com_tendencia_negativa}")

    df_fe['taxa_crescimento_anual'] = df_fe[ID_MUNICIPIO].map(tendencias_municipio)

    # 4. Compute technology baseline for each municipality
    def calcular_baseline_tecnologica(subdf):
        """Computes the technology trend line for each municipality."""
        municipio_id = subdf[ID_MUNICIPIO].iloc[0]
        taxa_crescimento = tendencias_municipio.get(municipio_id, tendencia_pais)

        mask_nao_nan = subdf[TARGET].notna()
        if mask_nao_nan.sum() < 2:
            return pd.Series(np.nan, index=subdf.index)

        X = subdf.loc[mask_nao_nan, [ID_ANO]].values.astype(float)
        y = subdf.loc[mask_nao_nan, TARGET].values.astype(float)

        try:
            if mask_nao_nan.sum() >= 2:
                anos = subdf.loc[mask_nao_nan, ID_ANO].values.astype(float)
                valores = subdf.loc[mask_nao_nan, TARGET].values.astype(float)

                # intercept minimises error for the fixed growth rate:
                # intercept = mean(y) - growth_rate * mean(X)
                intercept = np.mean(valores) - taxa_crescimento * np.mean(anos)

                baseline = taxa_crescimento * subdf[ID_ANO] + intercept
            else:
                media_y = np.mean(y)
                baseline = np.full(len(subdf), media_y)

            return pd.Series(baseline, index=subdf.index)
        except:
            return pd.Series(np.nan, index=subdf.index)

    print("Computing technology baseline...")
    df_fe['prod_baseline_tec'] = df_fe.groupby(ID_MUNICIPIO, group_keys=False).apply(calcular_baseline_tecnologica)

    # 5. Where prod is NaN, fill with prod_baseline_tec
    mask_nan = df_fe[TARGET].isna()
    df_fe.loc[mask_nan, TARGET] = df_fe.loc[mask_nan, 'prod_baseline_tec']

    # 6. Compute deviation from the trend
    df_fe['prod_desvio_tec'] = df_fe[TARGET] - df_fe['prod_baseline_tec']

    # 7. Summary statistics
    print(f"\nCORRECTIONS APPLIED — TECHNOLOGY NEVER REGRESSES:")
    print(f"    National technology trend: {tendencia_pais:.6f}")
    print(f"    Municipalities with positive own trend: {municipios_com_tendencia_positiva}")
    print(f"    Municipalities using national trend: {municipios_com_tendencia_negativa}")
    print(f"    Final mean growth rate: {df_fe['taxa_crescimento_anual'].mean():.6f}")

    return df_fe

print("Applying feature engineering with technology factor...")
DF = criar_features_com_tecnologia(DF)
DF = criar_interacoes_estrategicas(DF)

# ==============================================================================
# ZONE CREATION
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
    # centroid lon/lat + POS and productivity summaries per municipality
    base = (
        df.dropna(subset=['lon', 'lat', 'POS', 'prod'])
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
    # light lon/lat weights for spatial coherence
    base['lon_w'] = base['lon'] * 0.2
    base['lat_w'] = base['lat'] * 0.2
    return base

def criar_tendencia_por_zonas(df, n_zones):
    print(f"Creating zone trends (lat/lon in km weight {0.60}; POS+prod included)")
    n_munis = df['cod'].nunique()
    anos_min, anos_max = int(df['ano'].min()), int(df['ano'].max())
    print(f"Data: {n_munis} municipalities | years {anos_min}–{anos_max} | records {len(df)}")

    g = _agg_simples_por_muni(df).copy()

    # convert degrees to km for approximate Euclidean distances
    lat_rad = np.deg2rad(g['lat'].astype(float))
    cos_lat = np.cos(lat_rad.clip(-np.pi/2, np.pi/2))
    g['lat_km'] = g['lat'].astype(float) * 111.0
    g['lon_km'] = g['lon'].astype(float) * (111.0 * cos_lat.mean())

    geo_weight = 0.60
    g['lat_w'] = g['lat_km'] * geo_weight
    g['lon_w'] = g['lon_km'] * geo_weight

    print(f"Per-municipality aggregation done: {len(g)} valid municipalities")

    feat_cols = ['lon_w', 'lat_w', 'POS_med', 'POS_std', 'POS_slope', 'prod_med', 'prod_std', 'prod_slope']
    scaler = StandardScaler()
    X = scaler.fit_transform(g[feat_cols].values)

    # --------- initial KMeans ----------
    kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    g['zona'] = labels

    inertia = float(kmeans.inertia_)
    print(f"KMeans k={n_zones} | Inertia: {inertia:.2f}")

    # ----------------- geographic radius refinement --------------------
    def _geo_centroids(df_z):
        return df_z.groupby('zona')[['lon', 'lat']].mean()

    def _dist_km(lon1, lat1, lon2, lat2):
        lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
        lon1r, lon2r = np.deg2rad(lon1), np.deg2rad(lon2)
        x = (lon2r - lon1r) * np.cos(0.5 * (lat1r + lat2r))
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

        print(f"Geographic adjustment iteration {it}: reassignments = {len(reassign)}")

    counts = g['zona'].value_counts().sort_index()
    print("Zone sizes:", ", ".join(f"Z{z}:{c}" for z, c in counts.items()))

    # map zones back to the main dataframe
    df['zona'] = df['cod'].map(g.set_index('cod')['zona'].to_dict())
    print("Zones mapped to main dataframe")

    # zone trend
    df_sorted = df.sort_values(['zona', 'ano'])
    all_anos = range(anos_min, anos_max + 1)

    n_interp_total = 0
    for zona in range(n_zones):
        zona_data = df_sorted[df_sorted['zona'] == zona]
        if zona_data.empty:
            print(f"Zone {zona} has no data")
            continue

        anos_cobertos = sorted(zona_data['ano'].unique())
        print(f"Zone {zona}: {len(zona_data['cod'].unique())} municipalities | years {anos_cobertos[0]}–{anos_cobertos[-1]}")

        tendencia_por_ano = zona_data.groupby('ano')['prod'].median().reindex(all_anos)
        n_missing = tendencia_por_ano.isna().sum()

        tendencia_interp = tendencia_por_ano.interpolate(method='linear', limit_direction='both')
        tendencia_suavizada = tendencia_interp.rolling(window=3, center=True, min_periods=1).mean()

        n_interp_total += int(n_missing)

    print(f"Years without zone median before interpolation: {n_interp_total}")
    print(f"Trend computed for {n_zones} zones with geographic constraint")

    return df

DF = criar_tendencia_por_zonas(DF, n_zones=22)

# ==============================================================================
# SPATIAL, TEMPORAL, AND PREPROCESSING FUNCTIONS
# ==============================================================================

def criar_features_espaciais(df, n_vizinhos=5):
    """Adds zonal neighbour averages for spatial smoothing."""
    print("Applying neighbourhood smoothing...")

    # compute zone centroids
    centroides = df.groupby('zona')[['lon', 'lat']].median().reset_index()

    # pairwise distance matrix between zones
    coords_rad = np.radians(centroides[['lat', 'lon']].values)
    distancias = haversine_distances(coords_rad) * 6371  # km

    # find the nearest zones for each zone
    zonas_vizinhas = {}
    for i, zona in enumerate(centroides['zona']):
        dists = distancias[i]
        vizinhos_idx = np.argsort(dists)[1:n_vizinhos + 1]
        zonas_vizinhas[zona] = centroides.iloc[vizinhos_idx]['zona'].tolist()

    # compute neighbour means for critical features
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

    print(f"Smoothing applied to {len(features_suavizar)} features")
    return df

def criar_features_temporais_avancadas(df):
    """Creates temporal trend features to counteract drift."""
    df = df.copy()

    # linear trend per municipality
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

    # 3-year rolling mean per zone
    df = df.sort_values(['zona', 'ano'])
    df['prod_media_movel_zona'] = df.groupby('zona')['prod'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # normalised year for global trend (0–1)
    ano_min, ano_max = df['ano'].min(), df['ano'].max()
    df['ano_norm_tendencia'] = (df['ano'] - ano_min) / (ano_max - ano_min)

    # recent-season weight (higher weight for more recent years)
    df['peso_safra_recente'] = (df['ano'] - 2010) / (2024 - 2010)

    print("Advanced temporal features created")
    return df

def aplicar_winsorizacao_avancada(df):
    """Aggressive winsorisation to handle heavy tails in the target."""
    df = df.copy()

    # per-zone winsorisation of the primary target
    q1_prod  = df.groupby('zona')['prod_desvio_tec'].transform(lambda s: s.quantile(0.03))
    q99_prod = df.groupby('zona')['prod_desvio_tec'].transform(lambda s: s.quantile(0.97))

    out_antes = ((df['prod_desvio_tec'] < q1_prod) | (df['prod_desvio_tec'] > q99_prod)).sum()
    df['prod_desvio_tec_winsor'] = df['prod_desvio_tec'].clip(lower=q1_prod, upper=q99_prod)
    out_depois = ((df['prod_desvio_tec_winsor'] < q1_prod) | (df['prod_desvio_tec_winsor'] > q99_prod)).sum()

    print(f"Winsorisation applied:")
    print(f"    Outliers removed: {out_antes} → {out_depois}")
    print(f"    New limits: [{df['prod_desvio_tec_winsor'].min():.2f}, {df['prod_desvio_tec_winsor'].max():.2f}]")

    return df

def criar_pesos_zonas_criticas(df, zonas_criticas, peso_normal=1.0, peso_critico=3.0):
    """Creates sample weights giving higher importance to critical zones."""
    weights = np.ones(len(df), dtype=np.float32)
    for zona in zonas_criticas:
        mask = df['zona'] == zona
        weights[mask] = peso_critico

    print(f"Weights applied to {len(zonas_criticas)} critical zones")
    print(f"    Normal weight: {peso_normal}, Critical weight: {peso_critico}")
    return weights

# ==============================================================================
# ADDITIONAL WEIGHTING FUNCTIONS
# ==============================================================================

def criar_pesos_por_ano(df, ano_critico=2024, peso_normal=1.0, peso_critico=2.5):
    """Creates sample weights giving higher importance to critical years."""
    weights = np.ones(len(df), dtype=np.float32)
    mask_ano_critico = df['ano'] == ano_critico
    weights[mask_ano_critico] = peso_critico

    print(f"Weights applied to critical year {ano_critico}:")
    print(f"    Normal weight: {peso_normal}, Critical weight: {peso_critico}")
    print(f"    Samples with critical weight: {mask_ano_critico.sum()}")

    return weights

def aplicar_transformacao_monotonica_alvo(df):
    """Applies a monotonic transformation to the target to reduce error×productivity heteroscedasticity."""
    df = df.copy()

    if 'erro' not in df.columns:
        df['erro'] = df['prod_desvio_tec']

    df['prod_desvio_tec_transformado'] = np.arcsinh(df['prod_desvio_tec_winsor'])

    print("Monotonic transformation applied to target:")
    print(f"    Before: {df['prod_desvio_tec_winsor'].mean():.2f} ± {df['prod_desvio_tec_winsor'].std():.2f}")
    print(f"    After:  {df['prod_desvio_tec_transformado'].mean():.2f} ± {df['prod_desvio_tec_transformado'].std():.2f}")

    return df


# Apply all improvement strategies
print("\nAPPLYING IMPROVEMENT STRATEGIES...")
DF = criar_features_espaciais(DF)
DF = criar_features_temporais_avancadas(DF)
DF = aplicar_winsorizacao_avancada(DF)
DF = aplicar_transformacao_monotonica_alvo(DF)

# ==============================================================================
# FINAL FEATURE SET — DEFINED AFTER ALL FEATURES ARE CREATED
# ==============================================================================

# create 'distancia_mediana' feature
print("Creating 'distancia_mediana' feature...")
mediana_por_zona = DF.groupby('zona')['prod'].transform('median')
DF['distancia_mediana'] = (DF['prod'] - mediana_por_zona).abs()

FEATURES_TENDENCIAS = [
    # climate
    'FORCA_CALOR_MAX','INTERACAO_CALOR_SECA','tmed_std','calor_antes_colheita','stress_floracao',
    'estresse_termico_hidrico','indice_estresse','calor_fase_sensivel_aprox','tmax_anom',
    'estresse_termico_rel','amplitude_termica','gradiente_lat_spi','STRESS_CALOR_SECA',
    'stress_plantio','seca_extrema','prec_anom_safra','moisture_relief','tmin_min','spi',
    'produtividade_termica','anom_prec_plantio','prec_plantio','log_prec_plantio','saldo_hidrico',
    'prec_safra','gradiente_lat_tmax','log_prec_safra',
    # phenology
    'ROD','MOS_DT1','SIOS','AOS','vigor_x_ciclo','FENO_POS_MOS_SPI10','MOS','FENO_POS_MOS',
    'POS','POS_relativo','distancia_mediana',
]

# add spatial and temporal features
FEATURES_TENDENCIAS += [
    'prod_zonal', 'prod_desvio_tec_zonal', 'spi_zonal', 'tmax_anom_zonal',
    'tendencia_linear_municipio', 'prod_media_movel_zona', 'ano_norm_tendencia',
    'peso_safra_recente'
]

print(f"Total features: {len(FEATURES_TENDENCIAS)}")

# ==============================================================================
# FINAL PRE-PROCESSING
# ==============================================================================

def preprocessamento_agressivo(df):
    """Pre-processing focused on reducing low-value bias without distorting the target."""
    df = df.copy()
    print("APPLYING PRE-PROCESSING...")

    # zones with highest variance
    var_prod_zona = df.groupby('zona')['prod'].std()
    zonas_problematicas = var_prod_zona.nlargest(4).index.tolist()
    df['zona_problematica'] = df['zona'].isin(zonas_problematicas).astype(int)
    print(f"    High-variance zones: {zonas_problematicas}")

    print("PRE-PROCESSING COMPLETE")
    return df

DF = preprocessamento_agressivo(DF)

# ==============================================================================
# TRAIN / VALIDATION / TEST SPLIT WITH SAMPLE WEIGHTS
# ==============================================================================

# verify all required columns exist before dropna
colunas_necessarias = FEATURES_TENDENCIAS + [TARGET, 'prod_desvio_tec', 'zona']
colunas_faltantes = [col for col in colunas_necessarias if col not in DF.columns]

if colunas_faltantes:
    print(f"Missing columns: {colunas_faltantes}")
    colunas_necessarias = [col for col in colunas_necessarias if col in DF.columns]
    print(f"Using {len(colunas_necessarias)} available columns")

DF2025 = DF[DF['ano'] == 2025].copy()
DF.dropna(subset=colunas_necessarias, inplace=True)
DF = DF[DF['ano'] != 2025]

# ==============================================================================
# CRITICAL ZONE IDENTIFICATION VIA KMEANS
# ==============================================================================

def identificar_zonas_criticas(df, n_zonas_criticas=12):
    """
    Identifies critical zones based on historical performance (productivity)
    using only training data to avoid data leakage.
    """
    print("Identifying critical zones via KMeans...")

    # use only historical data to avoid leakage
    df_treino = df[~df['ano'] <= 2022].copy()

    # compute zone performance metrics
    desempenho_zonas = df_treino.groupby('zona').agg({
        'prod': ['median', 'std', 'count'],
        'prod_desvio_tec': ['mean', 'std'],
        'spi': 'mean',
        'tmax_anom': 'mean'
    }).round(4)

    desempenho_zonas.columns = ['_'.join(col).strip() for col in desempenho_zonas.columns.values]

    # criticality score (lower = worse)
    desempenho_zonas['score_criticidade'] = (
        -desempenho_zonas['prod_median'] +
        desempenho_zonas['prod_std'] +
        -desempenho_zonas['prod_desvio_tec_mean'] +
        desempenho_zonas['spi_mean'] +
        desempenho_zonas['tmax_anom_mean']
    )

    # sort by criticality (most critical first)
    zonas_ordenadas = desempenho_zonas.sort_values('score_criticidade', ascending=True)

    # select the n most critical zones
    zonas_criticas = zonas_ordenadas.head(n_zonas_criticas).index.tolist()

    print(f"Critical zones identified: {zonas_criticas}")
    print("Critical zone statistics:")
    for zona in zonas_criticas:
        stats = zonas_ordenadas.loc[zona]
        print(f"   Zone {zona}: Prod={stats['prod_median']:.1f} ± {stats['prod_std']:.1f}, "
              f"SPI={stats['spi_mean']:.2f}, TmaxAnom={stats['tmax_anom_mean']:.2f}")

    return zonas_criticas

# ==============================================================================
# DATA SPLIT WITH DYNAMIC CRITICAL-ZONE WEIGHTS
# ==============================================================================

def dividir_dados_anos_com_pesos(df):
    """Splits the data with combined zone and year sample weights."""
    anos_teste = {2005, 2012, 2017, 2023, 2024}

    test = df[df['ano'].isin(anos_teste)].copy()
    treino = df[~df['ano'].isin(anos_teste)].copy()

    # shuffle the non-test set
    df_shuffled = treino.sample(frac=1, random_state=42).reset_index(drop=True)

    n_total = len(df_shuffled)
    idx_train = int(0.8 * n_total)

    train = df_shuffled.iloc[:idx_train]
    val   = df_shuffled.iloc[idx_train:]

    # identify critical zones dynamically
    ZONAS_CRITICAS = identificar_zonas_criticas(train, n_zonas_criticas=12)

    # zone-based weights
    sample_weights_zona_train = criar_pesos_zonas_criticas(train, ZONAS_CRITICAS)
    sample_weights_zona_val   = criar_pesos_zonas_criticas(val,   ZONAS_CRITICAS)

    # year-based weights (2023 = severe La Niña)
    sample_weights_ano_train = criar_pesos_por_ano(train, ano_critico=2023, peso_critico=2.0)
    sample_weights_ano_val   = criar_pesos_por_ano(val,   ano_critico=2023, peso_critico=2.0)

    # combine weights multiplicatively
    sample_weights_train = sample_weights_zona_train * sample_weights_ano_train
    sample_weights_val   = sample_weights_zona_val   * sample_weights_ano_val

    print(f"Split with COMBINED WEIGHTS:")
    print(f"    Train: {len(train)} samples")
    print(f"    Validation: {len(val)} samples")
    print(f"    Test: {len(test)} samples")
    print(f"    Critical zones: {ZONAS_CRITICAS}")
    print(f"    Mean train weight: {sample_weights_train.mean():.2f}")
    print(f"    Mean validation weight: {sample_weights_val.mean():.2f}")

    return train, val, test, sample_weights_train, sample_weights_val, ZONAS_CRITICAS

# ==============================================================================
# REMOVE HARD-CODED CRITICAL ZONES (NOW COMPUTED DYNAMICALLY)
# ==============================================================================

# REMOVED: ZONAS_CRITICAS = [4, 10, 0, 9, 15, 18, 8, 6, 16, 21, 1, 11]

train, val, test, sample_weights_train, sample_weights_val, ZONAS_CRITICAS = dividir_dados_anos_com_pesos(DF)

# ==============================================================================
# TECHNOLOGY BASELINE
# ==============================================================================

def baseline_com_tecnologia(train, val, test):
    """Baseline: predict technology trend (zero climate deviation)."""
    pred_val_abs  = val['prod_baseline_tec'].to_numpy()
    pred_test_abs = test['prod_baseline_tec'].to_numpy()
    mae_val  = mean_absolute_error(val[TARGET],  pred_val_abs)
    mae_test = mean_absolute_error(test[TARGET], pred_test_abs)

    print(f"TECHNOLOGY BASELINE (Per-Municipality Trend):")
    print(f"    Validation MAE: {mae_val:.2f}")
    print(f"    Test MAE: {mae_test:.2f}")
    print(f"    Baseline uses mean growth rate of {train['taxa_crescimento_anual'].mean():.2f} units/year")
    return mae_val, mae_test

baseline_val, baseline_test = baseline_com_tecnologia(train, val, test)

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

imputador   = SimpleImputer(strategy='median')
escalonador = RobustScaler()

X_train_num = escalonador.fit_transform(imputador.fit_transform(train[FEATURES_TENDENCIAS].to_numpy(dtype=float)))
X_val_num   = escalonador.transform(imputador.transform(val[FEATURES_TENDENCIAS].to_numpy(dtype=float)))
X_test_num  = escalonador.transform(imputador.transform(test[FEATURES_TENDENCIAS].to_numpy(dtype=float)))

# targets: winsorised deviations from the technology trend
y_train_desvio = train['prod_desvio_tec_winsor'].to_numpy(dtype=float)
y_val_desvio   = val['prod_desvio_tec_winsor'].to_numpy(dtype=float)
y_test_desvio  = test['prod_desvio_tec_winsor'].to_numpy(dtype=float)

# zone inputs
zona_train = train['zona'].astype(int).to_numpy()
zona_val   = val['zona'].astype(int).to_numpy()
zona_test  = test['zona'].astype(int).to_numpy()

# ==============================================================================
# HYBRID MODEL WITH ZONE EMBEDDING
# ==============================================================================

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

from tensorflow.keras import regularizers

def criar_modelo_com_embedding_zonal(num_features, n_zonas=22):
    """Model with zone embeddings for per-zone bias calibration."""

    # Inputs
    inp_num  = L.Input(shape=(num_features,), name="num_feat")
    inp_zona = L.Input(shape=(1,), name="zona_input")

    # zone embedding (learns per-zone calibration)
    zona_embedding = L.Embedding(
        input_dim=n_zonas,
        output_dim=4,
        name="zona_embedding"
    )(inp_zona)
    zona_embedding = L.Flatten()(zona_embedding)

    # ---------------------------------- utility block ----------------------------------
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

        # Squeeze-and-Excitation for feature vectors
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
    # MAIN BRANCH
    # ==========================================================================
    att_w    = L.Dense(num_features, activation='softmax', name="feature_attention")(inp_num)
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
    # CLIMATE BRANCH
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
    # NDVI / PHENOLOGY BRANCH
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
    # FUSION WITH ZONE EMBEDDING
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

    # main prediction head
    main_output = L.Dense(32, activation='swish',
                          kernel_regularizer=regularizers.l2(1e-5),
                          name="main_head1")(x)
    main_output = L.Dropout(0.1, name="main_head_drop")(main_output)
    main_output = L.Dense(1, name="main_output")(main_output)

    # zone calibration offset
    zona_offset = L.Dense(1, name="zona_offset")(zona_embedding)

    # final output = main prediction + zone offset
    final_output = L.Add(name="final_output")([main_output, zona_offset])

    model = keras.Model([inp_num, inp_zona], final_output)

    def quantile_loss_suave(tau=0.70, alpha=0.1):
        def ql_suave(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            e = y_true - y_pred
            quantile_loss = tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
            mse_component = alpha * tf.reduce_mean(tf.square(e))
            return quantile_loss + mse_component
        return ql_suave

    def mbe(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(y_pred - y_true)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=2e-4,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=1.0
        ),
        loss=quantile_loss_suave(0.7, alpha=0.05),
        metrics=['mae', 'mse', mbe]
    )

    return model

def criar_callbacks_radicais():
    return [
        keras.callbacks.ModelCheckpoint(
            'best_model_long_training.keras',
            monitor='val_mae',
            mode='min',
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.995 if epoch > 10 else lr,
            verbose=0
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            mode='min',
            factor=0.7,
            patience=16,
            min_delta=0.3,
            cooldown=3,
            min_lr=1e-7,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            mode='min',
            patience=64,
            min_delta=0.2,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs:
            print(f"Epoch {epoch}: LR = {keras.backend.get_value(modelo.optimizer.learning_rate):.2e}, "
                  f"Val_MAE = {logs.get('val_mae', 0):.3f}")
        ),

        keras.callbacks.TensorBoard(
            log_dir='./logs_long_training',
            histogram_freq=4,
            update_freq='epoch'
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

# ==============================================================================
# TRAINING
# ==============================================================================

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

modelo = criar_modelo_com_embedding_zonal(len(FEATURES_TENDENCIAS))
callbacks = criar_callbacks_radicais()

print("STARTING MODEL TRAINING WITH ZONE CALIBRATION...")
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
# POST-TRAINING CALIBRATION
# ==============================================================================

def calibrar_modelo_isotonic(model, X_val, zona_val, y_val):
    """Applies isotonic regression for monotonic post-training calibration."""
    y_pred_val = model.predict([X_val, zona_val]).flatten()

    calibrador = IsotonicRegression(out_of_bounds='clip')
    calibrador.fit(y_pred_val, y_val)

    print("Isotonic calibration applied")
    return calibrador

def prever_com_calibracao(model, X, zonas, calibrador):
    pred_raw = model.predict([X, zonas]).flatten()
    return calibrador.predict(pred_raw)

print("Applying post-training calibration...")
calibrador = calibrar_modelo_isotonic(modelo, X_val_num, zona_val, y_val_desvio)

# ==============================================================================
# FINAL EVALUATION
# ==============================================================================

def avaliar_modelo_completo(model, calibrador, X_test, zona_test, y_test, baseline_mae):
    """Full evaluation with detailed metrics."""
    y_pred_calibrado = prever_com_calibracao(model, X_test, zona_test, calibrador)

    mae  = mean_absolute_error(y_test, y_pred_calibrado)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_calibrado))
    r2   = r2_score(y_test, y_pred_calibrado)

    melhoria_vs_baseline = ((baseline_mae - mae) / baseline_mae) * 100

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Improvement vs Baseline: {melhoria_vs_baseline:+.1f}%")
    print("=" * 60)

    return mae, rmse, r2

print("Evaluating final model...")
mae_final, rmse_final, r2_final = avaliar_modelo_completo(
    modelo, calibrador, X_test_num, zona_test, y_test_desvio, baseline_test
)

print("TRAINING AND EVALUATION COMPLETE!")

# ==============================================================================
# EVALUATION (TABULAR SUMMARY)
# ==============================================================================
NAME_W, RMSE_W, MAE_W, R2_W, VSB_W = 8, 7, 6, 7, 11

def _row(name, rmse, mae, r2, vsb, flag):
    return (f"{name:<{NAME_W}} | {rmse:{RMSE_W}.2f} | {mae:{MAE_W}.2f} | "
            f"{r2:{R2_W}.3f} | {vsb:+{VSB_W}.2f} | {flag}")

def avaliar_modelo_hibrido(modelo, calibrador, dados, features_num, zonas, nome, baseline_mae):
    desvios_pred = prever_com_calibracao(modelo, features_num, zonas, calibrador)
    prod_pred    = desvios_pred + dados['prod_baseline_tec'].to_numpy()
    prod_real    = dados[TARGET].to_numpy()

    rmse = np.sqrt(np.mean((prod_pred - prod_real) ** 2))
    mae  = mean_absolute_error(prod_real, prod_pred)
    r2   = r2_score(prod_real, prod_pred)

    vsb  = baseline_mae - mae
    flag = "OK" if vsb > 0 else "NEEDS WORK"
    print(_row(nome, rmse, mae, r2, vsb, flag))
    return mae, prod_pred

header = (f"{'Split':<{NAME_W}} | {'RMSE':>{RMSE_W}} | {'MAE':>{MAE_W}} | "
          f"{'R²':>{R2_W}} | {'vs Baseline':>{VSB_W}} | Flag")
sep = "-" * len(header)

print("\n" + "=" * len(header))
print("FINAL MODEL EVALUATION")
print("=" * len(header))
print(header)
print(sep)

# Baseline
y_true = DF['prod'].to_numpy(dtype=float)
y_base = DF['prod_baseline_tec'].to_numpy(dtype=float)
rmse_b = np.sqrt(mean_squared_error(y_true, y_base))
mae_b  = mean_absolute_error(y_true, y_base)
r2_b   = r2_score(y_true, y_base)
print(_row("Baseline", rmse_b, mae_b, r2_b, 0.0, "—"))

mae_train, pred_train = avaliar_modelo_hibrido(modelo, calibrador, train, X_train_num, zona_train, "Train",  baseline_val)
mae_val,   pred_val   = avaliar_modelo_hibrido(modelo, calibrador, val,   X_val_num,   zona_val,   "Val",    baseline_val)
mae_test,  pred_test  = avaliar_modelo_hibrido(modelo, calibrador, test,  X_test_num,  zona_test,  "Test",   baseline_test)

print("=" * len(header))

# ==============================================================================
# TEST RESULTS DATAFRAME
# ==============================================================================

print("\n" + "="*80)
print("FINAL RESULTS — TEST DATAFRAME")
print("="*80)

resultados_test = test.copy()
resultados_test['prod_pred'] = pred_test
resultados_test['desvio_pred'] = resultados_test['prod_pred'] - resultados_test['prod_baseline_tec']
resultados_test['erro'] = resultados_test['prod'] - resultados_test['prod_pred']

mae_test_final  = mean_absolute_error(resultados_test['prod'], resultados_test['prod_pred'])
rmse_test_final = np.sqrt(np.mean((resultados_test['prod'] - resultados_test['prod_pred']) ** 2))
r2_test_final   = r2_score(resultados_test['prod'], resultados_test['prod_pred'])

print(f"FINAL TEST METRICS:")
print(f"    MAE:  {mae_test_final:.2f} (confirmed)")
print(f"    RMSE: {rmse_test_final:.2f} (confirmed)")
print(f"    R²:   {r2_test_final:.3f} (confirmed)")
print(f"    Improvement vs Baseline: {((baseline_test - mae_test_final) / baseline_test) * 100:+.1f}%")

print(f"\nRESULT SAMPLE (first 10 rows of test set):")
colunas_mostrar = ['cod', 'ano', 'prod', 'prod_baseline_tec', 'desvio_pred', 'prod_pred', 'erro']
print(resultados_test[colunas_mostrar].head(10).round(2))

print(f"\nTEST ERROR STATISTICS:")
print(f"    Mean error: {resultados_test['erro'].mean():.2f} ± {resultados_test['erro'].std():.2f}")
print(f"    Max error: {resultados_test['erro'].max():.2f}")
print(f"    Min error: {resultados_test['erro'].min():.2f}")

quantis_basicos = resultados_test['erro'].quantile([0, 0.25, 0.5, 0.75, 1])
print("BASIC ERROR QUANTILES:")
print(quantis_basicos.round(2))

print(f"    % errors < 60 units: {(np.abs(resultados_test['erro']) < 60).mean() * 100:.1f}%")
print(f"    % errors < 120 units: {(np.abs(resultados_test['erro']) < 120).mean() * 100:.1f}%")
print(f"    % errors < 180 units: {(np.abs(resultados_test['erro']) < 180).mean() * 100:.1f}%")
print(f"    % errors < 240 units: {(np.abs(resultados_test['erro']) < 240).mean() * 100:.1f}%")
print(f"    % errors < 300 units: {(np.abs(resultados_test['erro']) < 300).mean() * 100:.1f}%")

print(f"\nTEST DATAFRAME SUMMARY:")
print(f"    Total samples: {len(resultados_test)}")
print(f"    Unique municipalities: {resultados_test['cod'].nunique()}")
print(f"    Period: {resultados_test['ano'].min()} to {resultados_test['ano'].max()}")
print(f"    Actual productivity: {resultados_test['prod'].mean():.1f} ± {resultados_test['prod'].std():.1f}")
print(f"    Predicted productivity: {resultados_test['prod_pred'].mean():.1f} ± {resultados_test['prod_pred'].std():.1f}")

print(f"\nTEST DATAFRAME — MAIN COLUMNS:")
print(resultados_test[['cod', 'ano', 'prod', 'prod_pred', 'erro']].head(15).round(2))

# ==============================================================================
# IMPROVEMENT ANALYSIS
# ==============================================================================

print(f"\nIMPROVEMENT ANALYSIS:")
melhoria_val  = ((baseline_val  - mae_val)  / baseline_val)  * 100
melhoria_test = ((baseline_test - mae_test) / baseline_test) * 100

print(f"    Validation improvement: {melhoria_val:+.1f}%")
print(f"    Test improvement: {melhoria_test:+.1f}%")

if melhoria_test > 0:
    print("    Model outperforms the technology baseline!")
else:
    print("    Model needs further adjustment")

# ==============================================================================
# ADDITIONAL STATISTICS
# ==============================================================================

print(f"\nTECHNOLOGY DEVIATION STATISTICS:")
print(f"    Mean deviation Train: {y_train_desvio.mean():.2f} ± {y_train_desvio.std():.2f}")
print(f"    Mean deviation Val:   {y_val_desvio.mean():.2f} ± {y_val_desvio.std():.2f}")
print(f"    Mean deviation Test:  {y_test_desvio.mean():.2f} ± {y_test_desvio.std():.2f}")

print(f"\nModel information:")
print(f"    Features used: {len(FEATURES_TENDENCIAS)}")
print(f"    Epochs trained: {len(historico.history['loss'])}")

# ==============================================================================
# ADD PREDICTIONS TO THE FULL DATAFRAME
# ==============================================================================

def adicionar_previsoes_com_zona(df_original, modelo, calibrador, escalonador, imputador):
    """Adds model predictions as new columns to the original DataFrame."""

    X_num_total    = escalonador.transform(imputador.transform(df_original[FEATURES_TENDENCIAS].to_numpy(dtype=float)))
    zonas_total    = df_original['zona'].astype(int).to_numpy()

    # calibrated predictions for the entire dataset
    desvios_pred_total = prever_com_calibracao(modelo, X_num_total, zonas_total, calibrador)

    if 'prod_baseline_tec' not in df_original.columns:
        print("    Recomputing technology baseline for the full dataset...")
        pass

    # convert deviations to absolute productivity
    df_original['prod_pred']  = desvios_pred_total + df_original['prod_baseline_tec']
    df_original['desvio_pred'] = desvios_pred_total

    # explicit technology and climate components
    df_original['componente_tecnologico'] = df_original['prod_baseline_tec'] - df_original['prod_baseline_tec'].mean()
    df_original['componente_climatico']   = df_original['desvio_pred']

    print(f"Predictions added to {len(df_original)} records")
    print(f"Prediction statistics:")
    print(f"    Predicted productivity: {df_original['prod_pred'].mean():.1f} ± {df_original['prod_pred'].std():.1f}")
    print(f"    Predicted deviation: {df_original['desvio_pred'].mean():.1f} ± {df_original['desvio_pred'].std():.1f}")
    print(f"    Technology component: {df_original['componente_tecnologico'].mean():.1f} ± {df_original['componente_tecnologico'].std():.1f}")

    return df_original

DF_com_previsoes = adicionar_previsoes_com_zona(DF, modelo, calibrador, escalonador, imputador)

# ==============================================================================
# PREDICTION SAMPLE
# ==============================================================================

print(f"\nPREDICTION SAMPLE (last 10 rows):")
colunas_mostrar = [ID_MUNICIPIO, ID_ANO, TARGET, 'prod_baseline_tec', 'desvio_pred', 'prod_pred', 'taxa_crescimento_anual']
print(DF_com_previsoes[colunas_mostrar].tail(10).round(2))

# ==============================================================================
# TECHNOLOGY FACTOR ANALYSIS
# ==============================================================================

print(f"\nTECHNOLOGY FACTOR ANALYSIS:")
print(f"    Mean growth rate: {DF_com_previsoes['taxa_crescimento_anual'].mean():.2f} units/year")
print(f"    Municipalities with positive growth: {(DF_com_previsoes['taxa_crescimento_anual'] > 0).mean() * 100:.1f}%")
print(f"    Growth range across municipalities: {DF_com_previsoes['taxa_crescimento_anual'].min():.2f} to {DF_com_previsoes['taxa_crescimento_anual'].max():.2f}")

crescimento_medio_por_ano = DF_com_previsoes.groupby(ID_ANO)['prod_baseline_tec'].mean()
if len(crescimento_medio_por_ano) > 1:
    crescimento_total = crescimento_medio_por_ano.iloc[-1] - crescimento_medio_por_ano.iloc[0]
    print(f"    Total technology gain over the period: {crescimento_total:.1f} units")

print(f"\nFINAL DATAFRAME STRUCTURE:")
print(f"    Columns: {list(DF_com_previsoes.columns)}")
print(f"    Total records: {len(DF_com_previsoes)}")
print(f"    Period: {DF_com_previsoes[ID_ANO].min()} to {DF_com_previsoes[ID_ANO].max()}")
print(f"    Municipalities: {DF_com_previsoes[ID_MUNICIPIO].nunique()}")

# ==============================================================================
# DETAILED ROOT-CAUSE ERROR ANALYSIS
# ==============================================================================

def analisar_causas_erro(resultados_test):
    """Simplified version that uses pre-computed errors without requiring model.predict."""
    import numpy as np
    import pandas as pd

    print("\nDETAILED ROOT-CAUSE ERROR ANALYSIS")
    print("=" * 70)

    if 'erro_abs' not in resultados_test.columns:
        resultados_test['erro_abs'] = resultados_test['erro'].abs()

    n_amostras    = len(resultados_test)
    n_municipios  = resultados_test['cod'].nunique()
    n_anos        = resultados_test['ano'].nunique()

    print(f"DATASET OVERVIEW:")
    print(f"    Total samples: {n_amostras:,}")
    print(f"    Distinct municipalities: {n_municipios}")
    print(f"    Years analysed: {n_anos}")
    print(f"    Period: {resultados_test['ano'].min()} - {resultados_test['ano'].max()}")

    # 1. Detailed error statistics
    print(f"\nDETAILED ERROR STATISTICS:")
    print(f"    Mean error: {resultados_test['erro'].mean():.3f}")
    print(f"    Median error: {resultados_test['erro'].median():.3f}")
    print(f"    Std deviation: {resultados_test['erro'].std():.3f}")
    print(f"    MAE: {resultados_test['erro_abs'].mean():.3f}")
    print(f"    RMSE: {np.sqrt((resultados_test['erro']**2).mean()):.3f}")
    print(f"    Max error: {resultados_test['erro'].max():.3f}")
    print(f"    Min error: {resultados_test['erro'].min():.3f}")
    print(f"    Error range: {resultados_test['erro'].max() - resultados_test['erro'].min():.3f}")

    mae = resultados_test['erro_abs'].mean()
    rmse = np.sqrt((resultados_test['erro']**2).mean())
    rmse_mae_ratio = rmse / (mae + 1e-9)
    se_mae = resultados_test['erro_abs'].std(ddof=1) / np.sqrt(max(n_amostras, 1))
    ci95_low, ci95_high = mae - 1.96*se_mae, mae + 1.96*se_mae
    print(f"    RMSE/MAE: {rmse_mae_ratio:.3f}")
    print(f"    MAE (approx 95% CI): [{ci95_low:.2f}, {ci95_high:.2f}]")

    erro_q25, erro_q75 = resultados_test['erro'].quantile([0.25, 0.75])
    print(f"    25th percentile: {erro_q25:.3f}")
    print(f"    75th percentile: {erro_q75:.3f}")
    print(f"    IQR: {erro_q75 - erro_q25:.3f}")

    limites_erro = pd.IntervalIndex.from_breaks([-np.inf, -600, -300, -180, -120, -60, 60, 120, 180, 300, 600, np.inf])

    try:
        resultados_test['faixa_erro'] = pd.cut(resultados_test['erro'], bins=limites_erro)
        distribuicao_erros = resultados_test['faixa_erro'].value_counts().sort_index()

        print(f"\nERROR DISTRIBUTION BY MAGNITUDE:")
        for faixa, count in distribuicao_erros.items():
            percentual = (count / n_amostras) * 100
            print(f"    {faixa}: {count:>4} samples ({percentual:5.1f}%)")
        centro = distribuicao_erros.get(pd.Interval(-60.0, 60.0, closed='right'), 0)
        print(f"    Share in central band (-60 to 60): {100*centro/max(n_amostras,1):.1f}%")
    except Exception as e:
        print(f"Error creating error bands: {e}")

    # 2. Error by productivity band
    print(f"\nERROR BY PRODUCTIVITY BAND (DETAILED):")
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
        count      = dados['erro_count']
        mean_erro  = dados['erro_mean']
        std_erro   = dados['erro_std']
        mean_prod  = dados['prod_mean']
        percentual = (count / n_amostras) * 100
        print(f"    {faixa}:")
        print(f"          Samples: {count} ({percentual:.1f}%) | Mean prod: {mean_prod:.1f}")
        print(f"          Mean error: {mean_erro:.3f} ± {std_erro:.3f}")

    if (resultados_test['prod'] > 0).any():
        mask_pos = resultados_test['prod'] > 0
        mape = (resultados_test.loc[mask_pos, 'erro_abs'] / resultados_test.loc[mask_pos, 'prod']).mean() * 100
        yhat = resultados_test['prod'].astype(float) + resultados_test['erro'].astype(float)
        smape = (200*np.abs(resultados_test['erro']) / (np.abs(resultados_test['prod']) + np.abs(yhat) + 1e-9)).mean()
        print(f"    MAPE (prod>0): {mape:.2f}%")
        print(f"    sMAPE: {smape:.2f}%")

    # 3. Geographic analysis
    print(f"\nGEOGRAPHIC ANALYSIS:")

    if 'lat' in resultados_test.columns:
        resultados_test['faixa_lat'] = pd.cut(resultados_test['lat'], bins=6, precision=2)
        erro_por_lat = resultados_test.groupby('faixa_lat', observed=True).agg(
            erro_mean=('erro', 'mean'),
            erro_std=('erro', 'std'),
            erro_count=('erro', 'count'),
            prod_mean=('prod', 'mean')
        ).round(3)

        print(f"    BY LATITUDE:")
        for faixa, dados in erro_por_lat.iterrows():
            print(f"          {faixa}: Error {dados['erro_mean']:.3f} | {dados['erro_count']} samples | Prod {dados['prod_mean']:.1f}")

    if 'lon' in resultados_test.columns:
        resultados_test['faixa_lon'] = pd.cut(resultados_test['lon'], bins=6, precision=2)
        erro_por_lon = resultados_test.groupby('faixa_lon', observed=True).agg(
            erro_mean=('erro', 'mean'),
            erro_std=('erro', 'std'),
            erro_count=('erro', 'count'),
            prod_mean=('prod', 'mean')
        ).round(3)

        print(f"    BY LONGITUDE:")
        for faixa, dados in erro_por_lon.iterrows():
            print(f"          {faixa}: Error {dados['erro_mean']:.3f} | {dados['erro_count']} samples | Prod {dados['prod_mean']:.1f}")

    # 3.1 Zone-level analysis
    print(f"\nDETAILED ZONE ANALYSIS:")

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

    if 'lat' in resultados_test.columns and 'lon' in resultados_test.columns:
        coords_por_zona = resultados_test.groupby('zona')[['lat', 'lon']].mean().round(3)
        erro_por_zona = erro_por_zona.join(coords_por_zona)

    print(f"    PERFORMANCE BY GEOGRAPHIC ZONE:")
    zonas_ordenadas = erro_por_zona.sort_values('erro_abs_mean', ascending=False)

    for zona, dados in zonas_ordenadas.iterrows():
        n_amostras_zona  = dados['erro_count']
        percentual_amostras = (n_amostras_zona / n_amostras) * 100
        erro_medio       = dados['erro_mean']
        erro_abs_medio   = dados['erro_abs_mean']
        std_erro         = dados['erro_std']
        prod_media       = dados['prod_mean']

        if erro_abs_medio > 600:
            severidade = "VERY POOR"
        elif erro_abs_medio > 480:
            severidade = "POOR"
        elif erro_abs_medio > 360:
            severidade = "MODERATE"
        elif erro_abs_medio > 180:
            severidade = "ACCEPTABLE"
        elif erro_abs_medio > 60:
            severidade = "GOOD"
        else:
            severidade = "EXCELLENT"

        print(f"          Zone: {zona} [{severidade}] | Mean error: {erro_medio:8.1f} | Abs error: {erro_abs_medio:6.1f} ± {std_erro:.1f}")
        print(f"          Samples: {n_amostras_zona:4.0f} ({percentual_amostras:4.1f}%) | Mean prod: {prod_media:6.1f}")

    zonas_criticas = zonas_ordenadas[zonas_ordenadas['erro_abs_mean'] > 300]
    if not zonas_criticas.empty:
        print(f"\n    CRITICAL ZONES (ERROR > 300):")
        for zona in zonas_criticas.index:
            erro_abs        = zonas_criticas.loc[zona, 'erro_abs_mean']
            n_amostras_zona = zonas_criticas.loc[zona, 'erro_count']
            print(f"          {zona}: Mean absolute error = {erro_abs:.1f} ({n_amostras_zona:.0f} samples)")

    balanced_mae   = erro_por_zona['erro_abs_mean'].mean()
    desvio_balanced = erro_por_zona['erro_abs_mean'].std()
    print(f"\nBALANCED MAE ACROSS ZONES: {balanced_mae:.2f} (std {desvio_balanced:.2f})")

    # 4. Temporal analysis
    print(f"\nTEMPORAL ANALYSIS:")

    erro_por_ano = resultados_test.groupby('ano').agg(
        erro_mean=('erro', 'mean'),
        erro_std=('erro', 'std'),
        erro_count=('erro', 'count'),
        prod_mean=('prod', 'mean')
    ).round(3)

    print(f"    BY YEAR:")
    for ano, dados in erro_por_ano.iterrows():
        print(f"          {ano}: Error {dados['erro_mean']:.3f} ± {dados['erro_std']:.3f} | Prod {dados['prod_mean']:.1f}")

    # 5. Climate analysis
    print(f"\nCLIMATE ANALYSIS:")

    if 'spi' in resultados_test.columns:
        bins_spi = [
            -np.inf, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
             0.5,    1.0,  1.5,  2.0,  2.5,  3.0,  np.inf
        ]

        labels_spi = [
            'Extremely Dry',
            'Very Dry',
            'Severely Dry',
            'Dry',
            'Moderately Dry',
            'Slightly Dry',
            'Normal',
            'Slightly Wet',
            'Moderately Wet',
            'Wet',
            'Very Wet',
            'Severely Wet',
            'Extremely Wet'
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
            percentual = (dados['erro_count'] / n_amostras) * 100
            print(f"    {tipo:25}: Error {dados['erro_mean']:7.3f} ± {dados['erro_std']:.3f} | {dados['erro_count']:>3.0f} samples ({percentual:4.1f}%)")

    # 6. Problematic municipality deep-dive
    print(f"\nPROBLEMATIC MUNICIPALITY ANALYSIS:")

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

    if 'lat' in resultados_test.columns and 'lon' in resultados_test.columns:
        coords_por_municipio = resultados_test.groupby('cod')[['lat', 'lon']].first()
        erro_por_municipio_detalhado = erro_por_municipio_detalhado.join(coords_por_municipio)

    top_problematicos = erro_por_municipio_detalhado.nlargest(15, 'erro_abs_mean')

    print(f"    TOP 15 MUNICIPALITIES WITH HIGHEST MEAN ABSOLUTE ERROR:")
    for idx, (municipio, dados) in enumerate(top_problematicos.iterrows(), 1):
        print(f"       {idx:2d}. Municipality {municipio} (Zone: {int(dados['zona_first'])}):")
        print(f"             Mean error: {dados['erro_mean']:7.3f} | Mean abs error: {dados['erro_abs_mean']:7.3f} ± {dados['erro_std']:.3f}")
        print(f"             Samples: {dados['erro_count']:3.0f} | Years: {dados['ano_nunique']:2.0f} | Mean prod: {dados['prod_mean']:6.1f}")

    # Pareto concentration
    total_abs  = resultados_test['erro_abs'].sum()
    contrib_mun = resultados_test.groupby('cod')['erro_abs'].sum().sort_values(ascending=False)
    if not contrib_mun.empty:
        n80 = (contrib_mun.cumsum()/max(total_abs, 1e-9) <= 0.80).sum() + 1
        print(f"\nPARETO: Top {n80} municipalities account for ~80% of total absolute error.")

    # 7. Correlation analysis
    print(f"\nCORRELATION ANALYSIS:")
    variaveis_correlacao = ['prod', 'lat', 'lon', 'prec_anom_safra', 'ano', 'tmax_anom', 'spi', 'POS', 'SOS', 'EOS']
    variaveis_disponiveis = [var for var in variaveis_correlacao if var in resultados_test.columns]

    if variaveis_disponiveis:
        correlacoes = resultados_test[['erro'] + variaveis_disponiveis].corr(numeric_only=True)['erro'].drop('erro', errors='ignore')

        for var, corr in correlacoes.items():
            significancia = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
            print(f"    Correlation error vs {var:15}: {corr:7.3f} {significancia}")
    else:
        correlacoes = pd.Series(dtype=float)

    # 8. Systematic bias analysis
    print(f"\nSYSTEMATIC BIAS ANALYSIS:")
    n_erros_positivos = (resultados_test['erro'] > 0).sum()
    n_erros_negativos = (resultados_test['erro'] < 0).sum()
    n_erros_zero      = (resultados_test['erro'] == 0).sum()

    print(f"    Positive errors (under-prediction): {n_erros_positivos:>5} ({n_erros_positivos/n_amostras*100:5.1f}%)")
    print(f"    Negative errors (over-prediction):  {n_erros_negativos:>5} ({n_erros_negativos/n_amostras*100:5.1f}%)")
    print(f"    Zero errors: {n_erros_zero:>5} ({n_erros_zero/n_amostras*100:5.1f}%)")

    skewness = resultados_test['erro'].skew()
    kurtosis = resultados_test['erro'].kurtosis()
    print(f"    Skewness: {skewness:.3f} {'(positive bias)' if skewness > 0.5 else '(negative bias)' if skewness < -0.5 else '(near-normal)'}")
    print(f"    Kurtosis: {kurtosis:.3f} {'(heavy tails)' if kurtosis > 1 else '(light tails)' if kurtosis < -1 else '(normal)'}")

    # calibration check (yhat = y + error)
    y    = resultados_test['prod'].astype(float)
    yhat = y + resultados_test['erro'].astype(float)
    tau  = 0.70
    cobertura      = float((y <= yhat).mean())
    desv_cobertura = cobertura - tau
    try:
        coef = np.polyfit(yhat, y, 1)
        slope_calib, intercept_calib = coef[0], coef[1]
    except Exception:
        slope_calib, intercept_calib = np.nan, np.nan
    print(f"\nCALIBRATION CHECK (tau=0.70): coverage={cobertura:.3f} (deviation {desv_cobertura:+.3f}) | slope={slope_calib:.3f} | intercept={intercept_calib:.2f}")
    try:
        resultados_test['decile_pred'] = pd.qcut(yhat, 10, duplicates='drop')
        mae_decis = resultados_test.groupby('decile_pred', observed=True)['erro_abs'].mean().round(2)
        print("    MAE by prediction decile:", list(mae_decis.values))
    except Exception:
        pass

    # 9. Executive summary and recommendations
    print(f"\nEXECUTIVE SUMMARY AND RECOMMENDATIONS:")
    maior_erro_abs = resultados_test['erro_abs'].max()

    if not erro_por_faixa.empty:
        faixa_mais_problematica_idx = erro_por_faixa['erro_mean'].abs().idxmax()
        print(f"    MOST PROBLEMATIC PRODUCTIVITY BAND: {faixa_mais_problematica_idx}")

    if not zonas_ordenadas.empty:
        zona_mais_problematica = zonas_ordenadas.iloc[0]
        print(f"    MOST PROBLEMATIC ZONE: {zonas_ordenadas.index[0]} (Abs error: {zona_mais_problematica['erro_abs_mean']:.1f})")

    print(f"    BIGGEST ISSUE: Max error of {maior_erro_abs:.2f}")

    recomendacoes = []

    if abs(skewness) > 0.5:
        recomendacoes.append("Systematic bias detected — test target transformation (e.g., log1p) or post-training correction per zone/year")

    if 'prec_anom_safra' in correlacoes and abs(correlacoes['prec_anom_safra']) > 0.2:
        recomendacoes.append("Error sensitive to precipitation — enrich climate windows (multi-scale SPI, RAIN_30/60/90) and climate×phenology interactions")

    if 'erro_abs_mean' in top_problematicos:
        if top_problematicos['erro_abs_mean'].mean() > 300:
            recomendacoes.append("Many critical municipalities — review series quality/consistency and apply neighbourhood smoothing (zonal average)")

    if not zonas_criticas.empty:
        zonas_criticas_str = [str(z) for z in zonas_criticas.index]
        recomendacoes.append(f"Focus on critical zones ({', '.join(zonas_criticas_str)}) with local calibration (offset) and/or zone-based weights")

    if kurtosis > 1.5 or rmse_mae_ratio > 1.8:
        recomendacoes.append("Heavy tails — use Huber/QuantileLoss, winsorise targets and apply prediction cap in production")

    if 'prod' in resultados_test.columns:
        corr_abs_prod = resultados_test[['erro_abs', 'prod']].corr(numeric_only=True).loc['erro_abs', 'prod']
        if abs(corr_abs_prod) > 0.25:
            recomendacoes.append("Error grows with productivity — weight by prod quantiles and model variance (robust loss)")

    if 'ano' in resultados_test.columns:
        corr_abs_ano = resultados_test[['erro_abs', 'ano']].corr(numeric_only=True).loc['erro_abs', 'ano']
        if abs(corr_abs_ano) > 0.2:
            recomendacoes.append("Temporal drift signal — add trend/year features, temporal validation, and frequent retraining")

    if 'lat' in resultados_test.columns and 'lon' in resultados_test.columns:
        corr_erro_lat = resultados_test[['erro', 'lat']].corr(numeric_only=True).loc['erro', 'lat']
        corr_erro_lon = resultados_test[['erro', 'lon']].corr(numeric_only=True).loc['erro', 'lon']
        if max(abs(corr_erro_lat), abs(corr_erro_lon)) > 0.2:
            recomendacoes.append("Spatial gradient in residuals — add coordinate spline terms or residual kriging")

    sinais_zona = resultados_test.groupby('zona')['erro'].mean().pipe(lambda s: (s.abs() > s.abs().median()).mean())
    if sinais_zona > 0.4:
        recomendacoes.append("Consistent zone bias — fixed per-zone offsets or regional embeddings in the network")

    if 'spi' in correlacoes and abs(correlacoes['spi']) > 0.2:
        recomendacoes.append("Error linked to SPI — evaluate 1/3/6-month windows and phenology-aligned lags")

    if 'tmax_anom' in correlacoes and abs(correlacoes['tmax_anom']) > 0.2:
        recomendacoes.append("Thermal sensitivity — add heat-wave indicators (days > p90) and extreme degree-days")

    share_poucos_anos = (erro_por_municipio_detalhado['ano_nunique'] < 3).mean()
    if share_poucos_anos > 0.4:
        recomendacoes.append("Many municipalities with few seasons — zone pooling and Bayesian shrinkage")

    desbalance_zona = erro_por_zona['erro_count'].max() / max(1, erro_por_zona['erro_count'].min())
    if desbalance_zona > 5:
        recomendacoes.append("Zone imbalance — apply sample_weight per zone during training")

    for fen in ['POS', 'SOS', 'EOS']:
        if fen in resultados_test.columns:
            corr_fen = resultados_test[['erro', fen]].corr(numeric_only=True).loc['erro', fen]
            if abs(corr_fen) > 0.2:
                recomendacoes.append(f"Residual linked to {fen} — review regional phenological windows and climate×{fen} interactions")

    frac_pos = n_erros_positivos / max(n_amostras, 1)
    frac_neg = n_erros_negativos / max(n_amostras, 1)
    if max(frac_pos, frac_neg) > 0.6:
        recomendacoes.append("Sign imbalance — apply post-training bias correction (residual regression) per zone/year")

    if not np.isnan(slope_calib) and (abs(desv_cobertura) > 0.05 or slope_calib < 0.9 or slope_calib > 1.1):
        recomendacoes.append("Coverage/scale deviation — keep Isotonic, adjust effective tau, and review QuantileLoss weights")

    if balanced_mae > 1.25 * mae:
        recomendacoes.append("High zone-level MAE vs global — resample/reweight to balance regions")

    if not contrib_mun.empty and n_municipios > 0:
        n80 = (contrib_mun.cumsum()/max(total_abs, 1e-9) <= 0.80).sum() + 1
        if n80 / max(n_municipios, 1) < 0.3:
            recomendacoes.append(f"Concentrated error — prioritise data/adjustments for the ~{n80} municipalities that account for 80% of error")

    if len(recomendacoes) > 0:
        print(f"    RECOMMENDATIONS:")
        for i, rec in enumerate(recomendacoes, 1):
            print(f"          {i}. {rec}")

    # clean up temporary analysis columns
    cols_to_drop = ['faixa_erro', 'faixa_prod', 'faixa_lat', 'faixa_lon', 'tipo_ano_detalhado', 'decile_pred']
    resultados_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

analisar_causas_erro(resultados_test)

# ==============================================================================
# 2025 FORECAST
# ==============================================================================

print("\n2025 FORECAST — Sample (first 10 rows):")

X_2025_num = escalonador.transform(
    imputador.transform(DF2025[FEATURES_TENDENCIAS].to_numpy(dtype=float))
)
zonas_2025 = DF2025['zona'].astype(int).to_numpy()

# predict the climate deviation component
desvios_pred_2025 = prever_com_calibracao(modelo, X_2025_num, zonas_2025, calibrador)

# convert to absolute productivity (trend + deviation)
prod_pred_2025 = DF2025['prod_baseline_tec'].to_numpy(dtype=float) + desvios_pred_2025

prev_2025 = DF2025.copy()
prev_2025['desvio_pred']          = desvios_pred_2025
prev_2025['prod_pred']            = prod_pred_2025
prev_2025['componente_climatico'] = prev_2025['desvio_pred']
prev_2025['componente_tecnologico'] = prev_2025['prod_baseline_tec'] - prev_2025['prod_baseline_tec'].mean()

cols_show = ['cod', 'ano', 'prod_baseline_tec', 'desvio_pred', 'prod_pred', 'lon', 'lat']
print(prev_2025[cols_show].head(10).round(2))

print("\n2025 Summary:")
print(f"    Municipalities: {prev_2025['cod'].nunique()}")
print(f"    Predicted productivity (mean ± std): {prev_2025['prod_pred'].mean():.1f} ± {prev_2025['prod_pred'].std():.1f}")
print(f"    Predicted deviation (mean): {prev_2025['desvio_pred'].mean():.1f}")

DF_final = pd.concat([DF_com_previsoes, prev_2025])
DF_final.sort_values(by=['ano', 'cod'], inplace=True)
DF_final.to_csv('result-train.final.csv', index=False)
