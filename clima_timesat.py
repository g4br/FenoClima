import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import datetime as dt
import geopandas as gpd
import pandas as pd
import numpy as np

import shapely.vectorized
def roi(matrix,geom,xx,yy):
    #mask = shapely.vectorized.contains(geom.buffer(0.05).geometry.item(), xx, yy)
    mask = shapely.vectorized.contains(geom.buffer(0.05), xx, yy)
    vmasked = np.where(mask, matrix, np.nan)
    return vmasked

def calcular_graus_dia(TM, Tm, TB=38, Tb=9):
    # Zera se TM excede TB (limite superior) – condição de excesso
    if TM > TB or Tm < Tb:
        GD = 0
    # TB > TM > Tm > Tb: faixa de temperatura máxima e mínima fica totalmente dentro dos limites das bases
    elif TB > TM > Tm > Tb:
        GD = (TM - Tm) / 2 + Tm - Tb
    # TB > TM > Tb > Tm: limite superior de base está entre Tm e TM
    elif TB > TM > Tb > Tm:
        GD = ((TM - Tb)**2) / (2 * (TM - Tm))
    # TB > Tb > TM > Tm: ambas as temperaturas do dia ficam abaixo do limite superior
    elif TB > Tb > TM > Tm:
        GD = 0
    # TM > TB > Tm > Tb: limite inferior de base está entre Tm e TM
    elif TM > TB > Tm > Tb:
        GD = (2*(TM-Tm)*(Tm-Tb) + (TM-Tm)**2 - (TM-TB)**2) / (2*(TM-Tm))
    # TM > TB > Tb > Tm: ambos os limites de base estão entre Tm e TM
    elif TM > TB > Tb > Tm:
        GD = 0.5 * ((TM-Tb)**2 - (TM-TB)**2) / (TM-Tm)
    else:
        # Qualquer outra situação não prevista explicitamente
        GD = 0
    return GD


# função que remove a data 366 dos anos bissextos
def day_of_year_no29(ds):
    day_of_year = ds.time.dt.dayofyear
    if ( day_of_year == 366).any():
        return xr.where(day_of_year > 61, day_of_year-1, day_of_year)
    else:
        return day_of_year


#----------------------------------------
# Calculo das anomalias e médias
#----------------------------------------
# então a partir daqui começam todas as manipulações em relação às datas e a composição do dataframe com as infos explicativas...
# temperatura me interessa do inicio até o momento antes da colheita... 
# na precipitação da pra avaliar em 3 momentos... 
# seca no inicio da safra, data_ini → mos_dt1
# chuva na colheita mos_dt2 → data_fim
# toda safra data_ini → data_fim
# SPI de toda a safra
def process_climate_safra(row,geometry, tmed_ds, clim_tmed_ds, tmax_ds, 
                         clim_tmax_ds, tmin_ds, clim_tmin_ds, prec_ds, 
                         clim_prec_ds, prec_std):
    minx, miny, maxx, maxy = geometry.bounds
    safra_temporada = row['safra_temporada']
    print(f'init: {safra_temporada}')
    data_temp1 = row['data_ini']
    data_temp2 = row['MOS_DT2']
    data_prec1 = data_temp1
    data_prec2 = row['MOS_DT1']
    data_prec3 = data_temp2
    data_prec4 = row['data_fim']
    #
    # tmed
    tmed_clip = tmed_ds.sel(time=slice(data_temp1, data_temp2),
                            lon=slice(minx,maxx),
                            lat=slice(miny,maxy))
    tmed_clip['dayofyear'] = day_of_year_no29(tmed_clip)
    tmed_clip = tmed_clip.set_index({'time': 'dayofyear'})
    tmed_clip['anom'] = tmed_clip['tmed'] - clim_tmed_ds.sel(time=tmed_clip['time'])['tmed']
    #
    # tmin
    tmin_clip = tmin_ds.sel(time=slice(data_temp1, data_temp2),
                            lon=slice(minx,maxx),
                            lat=slice(miny,maxy))
    tmin_clip['dayofyear'] = day_of_year_no29(tmin_clip)
    tmin_clip = tmin_clip.set_index({'time': 'dayofyear'})
    tmin_clip['anom'] = tmin_clip['tmin'] - clim_tmin_ds.sel(time=tmin_clip['time'])['tmin']
    #
    # tmax
    tmax_clip = tmax_ds.sel(time=slice(data_temp1, data_temp2),
                            lon=slice(minx,maxx),
                            lat=slice(miny,maxy))
    tmax_clip['dayofyear'] = day_of_year_no29(tmax_clip)
    tmax_clip = tmax_clip.set_index({'time': 'dayofyear'})
    tmax_clip['anom'] = tmax_clip['tmax'] - clim_tmax_ds.sel(time=tmax_clip['time'])['tmax']
    #
    # pegando as lats e lons para a roi
    lon = tmed_clip.lon.values
    lat = tmed_clip.lat.values
    xx, yy = np.meshgrid(lon, lat)
    #
    # Calculando GDD para  área
    graus_dia_xr = xr.apply_ufunc(
        calcular_graus_dia,
        tmax_clip.tmax, tmin_clip.tmin,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    graus_dia = graus_dia_xr.sum(axis=0).values
    roi_gdd = roi(graus_dia,geometry,xx,yy)
    masknan = (roi_gdd/roi_gdd)
    masknan = xr.DataArray(~np.isnan(masknan), dims=("lat", "lon"))
    #
    gdd = np.nanmean(roi_gdd)
    tmin_min = tmin_clip.tmin.where(masknan).min(dim='time').min().compute()
    tmin_mean = tmin_clip.tmin.where(masknan).mean(dim='time').mean().compute()
    tmin_anom = tmin_clip.anom.where(masknan).mean(dim='time').mean().compute()
    tmed_mean = tmed_clip.tmed.where(masknan).mean(dim='time').mean().compute()
    tmed_std = tmed_clip.tmed.where(masknan).std(dim='time').mean().compute()
    tmed_anom = tmed_clip.anom.where(masknan).mean(dim='time').mean().compute()
    tmax_mean = tmax_clip.tmax.where(masknan).mean(dim='time').mean().compute()
    tmax_max = tmax_clip.tmax.where(masknan).max(dim='time').max().compute()
    tmax_anom = tmax_clip.anom.where(masknan).mean(dim='time').mean().compute()
    #
    # aqui manipulando a precipitação
    prec_clip = prec_ds.sel(time=slice(data_prec1, data_prec4),
                            longitude=slice(minx,maxx),
                            latitude=slice(miny,maxy))
    # aqui apenas crio uma máscara de nans com a grade de prec
    lon = prec_clip.longitude.values
    lat = prec_clip.latitude.values
    xx, yy = np.meshgrid(lon, lat)
    masknan = roi(prec_clip['PREC_surface'][0].values,geometry,xx,yy)
    masknan = (masknan + 1) / (masknan + 1)
    masknan = xr.DataArray(~np.isnan(masknan), dims=("latitude", "longitude"))
    #
    prec_clip = prec_clip.where(masknan)
    dias_do_ano = day_of_year_no29(prec_clip)
    clim_clip = clim_prec_ds['PREC_surface'].sel(time=dias_do_ano,
                                                 longitude=slice(minx,maxx),
                                                 latitude=slice(miny,maxy))
    clim_clip['time'] = prec_clip.time
    std_periodo = prec_std['PREC_surface'].sel(time=dias_do_ano,
                                               longitude=slice(minx,maxx),
                                               latitude=slice(miny,maxy))
    #
    std_periodo['time'] = prec_clip.time
    prec_clip['anom'] = (prec_clip - clim_clip)['PREC_surface']
    #
    # calculando o SPI do período
    prec_acumulada = prec_clip['PREC_surface'].sum(dim='time').mean().compute()
    clim_acumulada_esperada = clim_clip.sum(dim='time').mean().compute()
    anom_acumulada = prec_acumulada - clim_acumulada_esperada
    std_acumulada = np.sqrt((std_periodo ** 2).sum(dim='time').mean().compute())
    spi = ((prec_acumulada - clim_acumulada_esperada) / std_acumulada).values
    #
    prec_plantio = prec_clip['PREC_surface'].sel(time=slice(data_prec1, data_prec2)).sum(dim='time').mean().compute()
    anom_prec_plantio = prec_clip['anom'].sel(time=slice(data_prec1, data_prec2)).sum(dim='time').mean().compute()
    #
    prec_colheita = prec_clip['PREC_surface'].sel(time=slice(data_prec3, data_prec4)).sum(dim='time').mean().compute()
    anom_prec_colheita = prec_clip['anom'].sel(time=slice(data_prec3, data_prec4)).sum(dim='time').mean().compute()
    #
    return {
        'safra_temporada':safra_temporada,
        'gdd':round(float(gdd),2),
        'tmin_min':round(float(tmin_min),2),
        'tmin_mean':round(float(tmin_mean),2),
        'tmin_anom':round(float(tmin_anom),2),
        'tmed_mean':round(float(tmed_mean),2),
        'tmed_std':round(float(tmed_std),4),
        'tmed_anom':round(float(tmed_anom),2),
        'tmax_mean':round(float(tmax_mean),2),
        'tmax_max':round(float(tmax_max),2),
        'tmax_anom':round(float(tmax_anom),2),
        'prec_safra':round(float(prec_acumulada),2),
        'prec_anom_safra':round(float(anom_acumulada),2), 
        'spi':round(float(spi),4),
        'prec_plantio':round(float(prec_plantio),2),
        'anom_prec_plantio':round(float(anom_prec_plantio),2),
        'prec_colheita':round(float(prec_colheita),2),
        'anom_prec_colheita':round(float(anom_prec_colheita),2)
    }


#----------------------------------------
# leitura dos arquivos
#----------------------------------------

path_media = '/media/ED520C85-64E3-48F9-88F6-2F3550735AE9/MESTRADO'

tmed_ds = xr.open_zarr('tmed.zarr')
tmed_ds = tmed_ds.sortby('time').astype('float32')
clim_tmed_ds = xr.open_dataset('tmed_climatology.nc')
clim_tmed_ds['dayofyear'] = clim_tmed_ds.time.dt.dayofyear
clim_tmed_ds = clim_tmed_ds.set_index({'time': 'dayofyear'})

tmax_ds = xr.open_zarr('tmax.zarr')
tmax_ds = tmax_ds.sortby('time').astype('float32')
clim_tmax_ds = xr.open_dataset('tmax_climatology.nc')
clim_tmax_ds['dayofyear'] = clim_tmax_ds.time.dt.dayofyear
clim_tmax_ds = clim_tmax_ds.set_index({'time': 'dayofyear'})

tmin_ds = xr.open_zarr('tmin.zarr')
tmin_ds = tmin_ds.sortby('time').astype('float32')
clim_tmin_ds = xr.open_dataset('tmin_climatology.nc')
clim_tmin_ds['dayofyear'] = clim_tmin_ds.time.dt.dayofyear
clim_tmin_ds = clim_tmin_ds.set_index({'time': 'dayofyear'})

# mesma metodologia para PREC
prec_ds = xr.open_zarr('prec.zarr')
prec_ds = prec_ds.sortby('time').astype('float32')
clim_prec_ds = xr.open_dataset('no29-prec-ydaymean.nc')
clim_prec_ds['dayofyear'] = clim_prec_ds.time.dt.dayofyear
clim_prec_ds = clim_prec_ds.set_index({'time': 'dayofyear'})

prec_std = xr.open_dataset('prec_ydaystd.nc')
prec_std['dayofyear'] = prec_std.time.dt.dayofyear
prec_std = prec_std.set_index({'time': 'dayofyear'})

#gdf = gpd.read_file(f'{path_media}/shapeSoja/regressao_tecnologica2.zip/regressao_tecnologica2.shp')
gdf = pd.read_pickle('safras_municipios_2000-2024.pkl')
#----------------------------------------
# INIT do loop
#----------------------------------------
import glob as glob 
from pathlib import Path
import os
import sys
mun = sys.argv[1]

files = glob.glob(f'{path_media}/results/csv/timesat/{mun}.csv')
CSV_OUT = f'{path_media}/results/csv/complete/{mun}.csv'
if os.path.exists(CSV_OUT):
    print(f"The file '{CSV_OUT}' exists.")
else:
    print(f"The file '{CSV_OUT}' does not exist.")
    #basenames_stem = [Path(file).stem for file in files]
    #basenames_stem[:5]

    #RESULTS = []
    #for mun in basenames_stem:
    df = pd.read_csv(f'{path_media}/results/csv/timesat/{mun}.csv',index_col=0)
    df = df[(df['ERROR'] == 0)]
    if len(df) <= 3:
        print('safra pequena')
        quit()
        #continue
    #
    prod = gdf.loc[mun].dropna()
    if len(prod) < len(df):
        print('menos safras do IBGE do que do timesat')
        quit()
        #continue
    #
    print(f'gerando para {mun}')
    print(f'safras Timesat : {len(df)}')
    print(f'safras IBGE : {len(prod)}')
    #
    geometry = prod.geometry
    #
    df['data_ini'] = pd.to_datetime(df['data_ini'])
    df['ROI_DT'] = pd.to_datetime(df['ROI_DT'])
    df['data_max'] = pd.to_datetime(df['data_max'])
    df['ROD_DT'] = pd.to_datetime(df['ROD_DT'])
    df['MOS_DT1'] = pd.to_datetime(df['MOS_DT1'])
    df['MOS_DT2'] = pd.to_datetime(df['MOS_DT2'])
    df['data_fim'] = pd.to_datetime(df['data_fim'])
    df['ano'] = df['data_fim'].dt.year
    df['prod'] = df.apply(lambda r: prod[str(r.ano)] if str(r.ano) in prod else np.nan, axis=1)
    calculations = []
    for _,row in df.iterrows():
        calculations.append(process_climate_safra(
                row, geometry, tmed_ds, clim_tmed_ds, tmax_ds, clim_tmax_ds,
                tmin_ds, clim_tmin_ds, prec_ds, clim_prec_ds, prec_std
            ))
    #
    result = pd.merge(pd.DataFrame(calculations), df, on='safra_temporada')
    result.to_csv(f'{path_media}/results/csv/complete/{mun}.csv')
    print(f'{path_media}/results/csv/complete/{mun}.csv')
'''   
all_results = pd.concat(RESULTS)
result['variacao'] = result['prod'] - result['prod'].mean() 
result[['gdd', 'tmin_min', 'tmin_mean', 'tmin_anom', 'tmed_mean', 'tmed_std', 'tmed_anom', 'tmax_mean', 'tmax_max', 'tmax_anom', 'prec_safra', 'prec_anom_safra', 'spi', 'prec_plantio', 'anom_prec_plantio', 'prec_colheita', 'anom_prec_colheita', 'quant_dias','SOS', 'ROI', 'ROI_START_VALUE', 'POS', 'ROD', 'ROD_END_VALUE', 'EOS', 'SIOS', 'AOS', 'MOS', 'LOS', 'BSE','prod']].corr()['prod'].sort_values()
result[['ROD_END_VALUE','tmed_std','tmax_max','prec_colheita','variacao']]
result[['gdd', 'tmin_min', 'tmin_mean', 'tmin_anom', 'tmed_mean', 'tmed_std', 'tmed_anom', 'tmax_mean', 'tmax_max', 'tmax_anom', 'prec_safra', 'prec_anom_safra', 'spi', 'prec_plantio', 'anom_prec_plantio', 'prec_colheita', 'anom_prec_colheita','prod','variacao']].sort_values(by='variacao')
result[['anom_prec_colheita','variacao']].sort_values(by='variacao')
'''