import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
matplotlib.use('Agg')

import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import datetime as dt
import matplotlib.pyplot as plt

import concurrent.futures

#--------------------------------------------
# Works on interpolated data
# https://github.com/lewistrotter/PhenoloPy
#--------------------------------------------
def double_logistic(t, a, b, c1, d, e, c2, f0):
    return (a / (1 + np.exp(-b * (t - c1)))) + (d / (1 + np.exp(-e * (t - c2)))) + f0

def to_x(dt):
    if dt < df.index[0] or dt > df.index[-1]:
        return None
    return df.index.get_indexer([dt], method='nearest')[0]

def criar_resposta_erro(error_code, error_str, data_max):
    print(data_max.strftime('%F'))
    return {
        'safra_temporada': np.nan,
        'data_ini': np.nan,
        'data_max': data_max.strftime('%F'),
        'data_fim': np.nan,
        'quant_dias': np.nan,
        'SOS': np.nan,
        'ROI': np.nan,
        'ROI_DT': np.nan,
        'ROI_START_VALUE': np.nan,
        'POS': np.nan,
        'ROD': np.nan,
        'ROD_DT': np.nan,
        'ROD_END_VALUE': np.nan,
        'EOS': np.nan,
        'SIOS': np.nan,
        'AOS': np.nan,
        'MOS': np.nan,
        'MOS_DT1': np.nan,
        'MOS_DT2': np.nan,
        'LOS': np.nan,
        'BSE': np.nan,
        'ERROR': error_code,
        'ER_DISCR': error_str
    }

def parametros_safra_curva_df(data):
    # data must be a DataFrame indexed by date with columns: ndvi_fit, ndvi_p50 (and optionally ndvi_poly2)
    # 1. Peak date (maximum of the smoothed curve)
    curva_suave = data['ndvi_fit']
    data_max = data['ndvi_p50'].idxmax()
    pos = data['ndvi_p50'][data_max]
    if data_max.month in [6, 7, 8, 9]:
        return criar_resposta_erro(1, 'peak out of season', data_max)
    # 2. Season start: first value > 0 in the original NDVI BEFORE the peak
    antes_pico = data.loc[:data_max]
    data_ini = antes_pico[antes_pico['ndvi_p50'] > 0]['ndvi_p50'].idxmin()
    sos = antes_pico.loc[data_ini, 'ndvi_p50']
    depois_pico = data.loc[data_max:]
    data_fim = depois_pico[depois_pico['ndvi_p50'] > 0]['ndvi_p50'].idxmin()
    eos = depois_pico.loc[data_fim, 'ndvi_p50']
    quant_dias = (data_fim - data_ini).days
    # >>> Validation rules
    if data_ini.month in [4, 5, 6, 7]:
        return criar_resposta_erro(2, 'planting out of season', peak)
    elif data_fim.month in [7, 8, 9, 10, 11]:
        return criar_resposta_erro(3, 'harvest out of season', data_max)
    elif quant_dias > 240:
        return criar_resposta_erro(4, 'cycle too long', data_max)
    elif quant_dias < 90:
        return criar_resposta_erro(5, 'cycle too short', data_max)
    else:
        safra_window = curva_suave.loc[data_ini:data_fim]
        mos = safra_window.min() + ((safra_window.max() - safra_window.min()) * .8)
        mos_values = safra_window[safra_window >= mos]
        mos_d1 = mos_values.index[0]
        mos_d2 = mos_values.index[-1]
        bse = (sos + eos) / 2
        los = safra_window.count()
        # ROI calculation
        roi_values = curva_suave.loc[data_ini:mos_d1]
        init20 = roi_values.min() + ((roi_values.max() - roi_values.min()) * .1)
        roi_values = roi_values[roi_values >= init20]
        roi_start = roi_values.index[0]
        roi_start_value = roi_values.iloc[0]
        if len(roi_values) < 2 or np.any(np.isnan(roi_values.values)) or np.any(np.isinf(roi_values.values)) or np.ptp(roi_values.values) == 0:
            return criar_resposta_erro(6, 'ROI calculation failed', data_max)
        else:
            y = roi_values.values
            x = np.arange(y.size)
            fit = np.polyfit(x, y, 1)
            roi_slope = fit[0]
        # ROD calculation
        rod_values = curva_suave.loc[mos_d2:data_fim]
        fim20 = rod_values.min() + ((rod_values.max() - rod_values.min()) * .1)
        rod_values = rod_values[rod_values >= fim20]
        rod_end = rod_values.index[-1]
        rod_end_value = rod_values.iloc[-1]
        if len(rod_values) < 2 or np.any(np.isnan(rod_values.values)) or np.any(np.isinf(rod_values.values)) or np.ptp(rod_values.values) == 0:
            return criar_resposta_erro(7, 'ROD calculation failed', data_max)
        else:
            y = rod_values.values
            x = np.arange(y.size)
            fit = np.polyfit(x, y, 1)
            rod_slope = fit[0]
        aos = pos - bse
        sios = np.trapz(safra_window.values)
        if np.isnan([sos, roi_slope, pos, rod_slope, eos, sios, aos, mos, los, bse]).any():
            return criar_resposta_erro(8, 'parameter is NaN', data_max)
        else:
            nomesafra = f'{data_ini.year}/{data_fim.year}'
            err = 0
            timesdat_dict = {
                'safra_temporada': nomesafra,
                'data_ini': data_ini,
                'data_max': data_max,
                'data_fim': data_fim,
                'quant_dias': quant_dias,
                'SOS': sos,
                'ROI': roi_slope,
                'ROI_DT': roi_start,
                'ROI_START_VALUE': roi_start_value,
                'POS': pos,
                'ROD': rod_slope,
                'ROD_DT': rod_end,
                'ROD_END_VALUE': rod_end_value,
                'EOS': eos,
                'SIOS': sios,
                'AOS': aos,
                'MOS': mos,
                'MOS_DT1': mos_d1,
                'MOS_DT2': mos_d2,
                'LOS': los,
                'BSE': bse,
                'ERROR': err,
                'ER_DISCR': None
                }
        return timesdat_dict


def ajustar_curva_safra(peak, df, ndvi_quad, range_window=105, blend_points=4):
    """
    Fits a double logistic or polynomial curve to a time window centred on an
    NDVI peak to model the vegetation growth cycle.

    Args:
        peak: index of the peak to process
        df: DataFrame with the original NDVI data
        ndvi_quad: time series with NDVI values used for fitting
        range_window: time window in days for analysis (default: 105)
        blend_points: number of blending points at the edges (default: 4)

    Returns:
        dict with fitted curve parameters, or None on failure
    """
    # get the date corresponding to the current peak in the DataFrame
    peak_date = df.iloc[peak].name

    # define the time window for analysis: range_window days before and after the peak
    start = peak_date - dt.timedelta(days=range_window)
    # refine the start: find the minimum NDVI before the peak and subtract 8 days
    start = ndvi_quad[start:peak_date].idxmin() - dt.timedelta(days=8)

    # define the end of the window: range_window days after the peak
    end = peak_date + dt.timedelta(days=range_window)
    # refine the end: find the minimum NDVI after the peak and add 8 days
    end = ndvi_quad[peak_date:end].idxmin() + dt.timedelta(days=8)

    # extract the NDVI sub-series within the defined window
    sub_ndvi = ndvi_quad.loc[start:end].copy()
    # drop any NaN values
    sub_ndvi = sub_ndvi.dropna()

    # ensure there are enough data points (minimum 10)
    if len(sub_ndvi) < 10:
        print(f"Insufficient data for peak at {peak}: only {len(sub_ndvi)} points")
        return None

    # convert NDVI values to a numpy array
    y = sub_ndvi.values
    # create time array in relative days (days from the start of the sub-series)
    t = (sub_ndvi.index - sub_ndvi.index[0]).days

    # main curve-fitting block
    try:
        # initialise equal weights for all points (used by curve_fit)
        weights = np.ones_like(y)

        # locate the position of the NDVI maximum within the sub-series
        idx_peak = np.argmax(y)
        # find the left-side minimum before the peak
        idx_min_esq = np.argmin(y[:idx_peak])
        # find the right-side minimum after the peak (offset by peak index)
        idx_min_dir = np.argmin(y[idx_peak:]) + idx_peak

        # compute NDVI amplitude (max − min)
        a = y.max() - y.min()
        # minimum gap to avoid degenerate bounds
        gap = 1e-3

        # lower bounds for double_logistic parameters
        lower = [
            0,          # minimum amplitude
            0.001,      # minimum growth rate
            max(0, idx_min_esq - 10),  # minimum left-valley position (with safety margin)
            -1,         # minimum decline rate
            0.001,      # minimum weight of the second logistic
            max(0, idx_min_dir - 10),  # minimum right-valley position (with safety margin)
            0           # minimum base value
        ]

        # upper bounds for the parameters
        upper = [
            2 * a,      # maximum amplitude (twice the computed amplitude)
            1,          # maximum growth rate
            min(len(y) - 1, idx_min_esq + 10),  # maximum left-valley position
            0,          # maximum decline rate
            1,          # maximum weight of the second logistic
            min(len(y) - 1, idx_min_dir + 10),  # maximum right-valley position
            y.max()     # maximum base value
        ]

        # ensure all upper bounds exceed their lower bounds
        for i in range(len(lower)):
            if upper[i] <= lower[i]:
                upper[i] = lower[i] + gap

        # smart initial parameter guesses
        p0 = [
            a * 0.8,       # initial amplitude (80% of computed amplitude)
            0.1,           # initial growth rate
            idx_min_esq,   # initial left-valley position
            -0.1,          # initial decline rate
            0.1,           # initial weight of the second logistic
            idx_min_dir,   # initial right-valley position
            y.min()        # initial base value
        ]

        # first curve-fitting attempt
        try:
            params, _ = curve_fit(double_logistic, t, y, p0, sigma=1/weights,
                                  bounds=(lower, upper), maxfev=5000)
        except RuntimeError:
            # if the first attempt fails, retry with wider bounds
            print(f"First attempt failed for peak {peak}, trying wider bounds")
            lower = [0, 0.0001, 0, -2, 0.0001, 0, 0]
            upper = [3 * a, 2, len(y) - 1, 0, 2, len(y) - 1, y.max() * 1.5]
            params, _ = curve_fit(double_logistic, t, y, p0, sigma=1/weights,
                                  bounds=(lower, upper), maxfev=5000)

        # evaluate the fitted curve with the optimised parameters
        ndvi_fit = double_logistic(t, *params)

        # get the actual NDVI value at the peak from the original DataFrame
        peak_ndvi = df.iloc[peak]['ndvi_p50']
        fit_max = ndvi_fit.max()

        # scale the fitted curve to match the actual peak value
        if fit_max > 0 and peak_ndvi > 0:
            scaling = peak_ndvi / fit_max
            ndvi_fit_scaled = ndvi_fit * scaling
        else:
            ndvi_fit_scaled = ndvi_fit

        # clip all fitted values to [0, 1]
        ndvi_fit_scaled = np.clip(ndvi_fit_scaled, 0, 1)

        return {
            'peak': peak,        # original peak index
            'params': params,    # optimised curve parameters
            'data': pd.DataFrame({
                'date': sub_ndvi.index,       # sub-series dates
                'ndvi_poly2': y,              # original NDVI values
                'ndvi_fit': ndvi_fit_scaled   # fitted curve values
            })
        }

    except Exception as e:
        print(f'Curve fit failed for peak at {peak}: {e}')
        # fallback: polynomial fit
        try:
            fit_coeffs = np.polyfit(t, y, 2)
            ndvi_fit = np.polyval(fit_coeffs, t)
            peak_ndvi = df.iloc[peak]['ndvi_p50']
            fit_max = ndvi_fit.max()
            if fit_max > 0 and peak_ndvi > 0:
                scaling = peak_ndvi / fit_max
                ndvi_fit_scaled = ndvi_fit * scaling
            else:
                ndvi_fit_scaled = ndvi_fit
            ndvi_fit_scaled = np.clip(ndvi_fit_scaled, 0, 1)
            print(f"Using polynomial fallback for peak {peak}")
            return {
                    'peak': peak,
                    'params': fit_coeffs,     # polynomial coefficients
                    'data': pd.DataFrame({
                        'date': sub_ndvi.index,
                        'ndvi_poly2': y,
                        'ndvi_fit': ndvi_fit_scaled
                        }),
                    'fallback': True  # flag indicating the alternative method was used
                    }

        except Exception as e2:
            print(f"Fallback also failed for peak {peak}: {e2}")
            return None

def aplicar_savitzky_golay(ndvi_series, window_length=48, polyorder=1):
    """
    Applies a Savitzky-Golay filter to smooth the time series.
    """
    try:
        if len(ndvi_series) < window_length:
            print(f"Insufficient data for Savitzky-Golay: {len(ndvi_series)} points, need {window_length}")
            return ndvi_series

        original_index = ndvi_series.index
        valid_mask = ~ndvi_series.isna()
        valid_values = ndvi_series[valid_mask].values

        if len(valid_values) < window_length:
            print(f"Insufficient valid values for Savitzky-Golay: {len(valid_values)}")
            return ndvi_series

        smoothed_values = savgol_filter(valid_values, window_length, polyorder)

        smoothed_series = pd.Series(index=ndvi_series.index, dtype=float)
        smoothed_series[valid_mask] = smoothed_values

        if smoothed_series.isna().any():
            smoothed_series = smoothed_series.interpolate(method='linear')
            smoothed_series = smoothed_series.bfill().ffill()

        print(f"Savitzky-Golay filter applied: window={window_length}, order={polyorder}")
        return smoothed_series

    except Exception as e:
        print(f"Error applying Savitzky-Golay: {e}")
        return ndvi_series

def calcular_dados(mun, path_csv):
    print(f'exec {mun}')
    files = glob.glob(f'{path_csv_gee}/*{mun}.csv')
    DF = []
    for f in files:
        DF.append(pd.read_csv(f))
    DF = pd.concat(DF)
    DF.index = pd.to_datetime(DF.date)
    DF.drop('date', axis=1, inplace=True)
    DF = DF.sort_index()
    ano_inicial = DF.index.year[0]
    DF = DF[f'{ano_inicial}-06':]
    df = DF.copy()
    df.loc[df.pct_cloud_soy > 70, 'ndvi_p50'] = np.nan
    print(f'find peaks {mun}')
    peaks, _ = find_peaks(DF['ndvi_p50'].values, distance=16, height=.65)
    date_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    ndvi_quad = df['ndvi_p50'].reindex(date_range).interpolate(method='polynomial', order=3)
    ndvi_quad = aplicar_savitzky_golay(ndvi_quad, window_length=33, polyorder=1)
    print(f'Curve fit {mun}')
    if peaks.size > 0:
        print(f'Identifying parameters for peaks: {peaks}')
        curvas_ajustadas = [
            r for peak in peaks
            if (r := ajustar_curva_safra(peak, df, ndvi_quad)) is not None
        ]
    else:
        curvas_ajustadas = []
    parametros_lista = []
    dados_curvas = []
    for _, curva in enumerate(curvas_ajustadas):
        try:
            curva_data = curva['data'].set_index('date')
            data = pd.concat((curva_data, df['ndvi_p50'][curva_data.index[0]:curva_data.index[-1]]), axis=1)
            parametros = parametros_safra_curva_df(data)
            print(parametros)
            parametros_lista.append(parametros)
            dados_curvas.append(data)
        except Exception as e:
            print(f'Error in season {_}: {e}')
            pass
    pd.DataFrame(parametros_lista).to_csv(path_csv)
    print(f'{path_csv}')
    return df, ndvi_quad, peaks, dados_curvas, parametros_lista


def plotar_safra(mun, df, ndvi_quad, peaks, dados_curvas, parametros_lista, path_png):
    print(df)
    fig, ax1 = plt.subplots(figsize=(18, 6))
    x = np.arange(len(df))
    ax1.bar(x, df['pct_cloud_soy'], color='grey', width=1, label='Cloud (%)', alpha=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Cloud Cover (%)')
    datas_peaks = df.iloc[peaks].index.strftime('%Y/%m')
    ax1.set_xticks(peaks)
    ax1.set_xticklabels(datas_peaks, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel("Date")
    ax2 = ax1.twinx()

    ax2.plot(x, df['ndvi_p50'], color='k', linewidth=2, label='NDVI P50')
    ax2.scatter(x, df['ndvi_p50'], color='k', s=40, marker='o', label='NDVI P50')
    ax2.scatter(peaks, df['ndvi_p50'].iloc[peaks], color='orange', s=60, marker='*', zorder=99, label='Peaks')
    ax2.set_ylim(.1, .95)
    ax2.set_ylabel('NDVI P50')

    # Axis 3: Savitzky-Golay (used for ROI/ROD detection)
    ax3 = ax1.twinx()
    # map Savitzky-Golay dates to x positions
    date_to_x = {date: i for i, date in enumerate(df.index)}

    ndvi_smooth_x = []
    ndvi_smooth_y = []

    for date, value in ndvi_quad.items():
        if date in date_to_x:
            ndvi_smooth_x.append(date_to_x[date])
            ndvi_smooth_y.append(value)
        else:
            # interpolate position for dates between original points
            dates_df = np.array([d.timestamp() for d in df.index])
            current_ts = date.timestamp()
            idx = np.searchsorted(dates_df, current_ts)

            if idx == 0:
                ndvi_smooth_x.append(0)
                ndvi_smooth_y.append(value)
            elif idx == len(dates_df):
                ndvi_smooth_x.append(len(df.index) - 1)
                ndvi_smooth_y.append(value)
            else:
                prev_date = df.index[idx - 1]
                next_date = df.index[idx]
                prev_ts = prev_date.timestamp()
                next_ts = next_date.timestamp()

                fraction = (current_ts - prev_ts) / (next_ts - prev_ts)
                x_pos = (idx - 1) + fraction

                ndvi_smooth_x.append(x_pos)
                ndvi_smooth_y.append(value)

    # plot the Savitzky-Golay series
    ax3.plot(ndvi_smooth_x, ndvi_smooth_y, color='blue', linewidth=1,
             label='NDVI Savitzky-Golay', alpha=0.7)
    ax3.set_ylim(.1, .95)
    ax3.set_ylabel('NDVI Savitzky-Golay', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    already_legend = set()
    for i, (data, params) in enumerate(zip(dados_curvas, parametros_lista)):
        if params['ERROR'] == 0:
            idx = df.index.get_indexer(data.index)
            valid = idx >= 0
            label = 'Double Logistic Fit' if 'Double Logistic Fit' not in already_legend else ""
            ax2.plot(idx[valid], data['ndvi_fit'][valid], color='red', linewidth=2, alpha=0.7, label=label)
            already_legend.add('Double Logistic Fit')
            x_sos = to_x(params['data_ini'])
            x_pos = to_x(params['data_max'])
            x_eos = to_x(params['data_fim'])
            x_mos1 = to_x(params['MOS_DT1'])
            x_mos2 = to_x(params['MOS_DT2'])
            x_roi_d = to_x(params['ROI_DT'])
            x_rod_d = to_x(params['ROD_DT'])
            if x_sos:
                ax2.scatter(x_sos, params['SOS'], facecolors='none', edgecolors='green', s=110, linewidths=2, label='SOS' if 'SOS' not in already_legend else "")
                ax2.annotate('SOS', (x_sos, params['SOS']), xytext=(5, 10), textcoords='offset points', fontsize=10, color='green')
                already_legend.add('SOS')
            if x_pos:
                ax2.scatter(x_pos, params['POS'], facecolors='none', edgecolors='blue', s=110, linewidths=2, label='POS' if 'POS' not in already_legend else "")
                ax2.annotate('POS', (x_pos, params['POS']), xytext=(5, -15), textcoords='offset points', fontsize=10, color='blue')
                already_legend.add('POS')
            if x_eos:
                ax2.scatter(x_eos, params['EOS'], facecolors='none', edgecolors='orange', s=110, linewidths=2, label='EOS' if 'EOS' not in already_legend else "")
                ax2.annotate('EOS', (x_eos, params['EOS']), xytext=(-40, 10), textcoords='offset points', fontsize=10, color='orange')
                already_legend.add('EOS')
            if x_sos and x_eos:
                ax2.hlines(params['BSE'], x_sos, x_eos, linestyles=':', color='grey', label='BSE' if 'BSE' not in already_legend else "")
                already_legend.add('BSE')
            if x_pos:
                ax2.vlines(x_pos, params['BSE'], params['POS'], linestyles=':', color='grey', label='POS Line' if 'POS Line' not in already_legend else "")
                already_legend.add('POS Line')
            if 'MOS' in params and not np.isnan(params['MOS']):
                ax2.hlines(params['MOS'], x_mos1, x_mos2, linestyles=':', color='purple', label='MOS' if 'MOS' not in already_legend else "")
                ax2.plot([x_roi_d, x_mos1], [params['ROI_START_VALUE'], params['MOS']], color='green', linestyle='--', lw=2, label='ROI' if 'ROI' not in already_legend else "")
                ax2.plot([x_mos2, x_rod_d], [params['MOS'], params['ROD_END_VALUE']], color='orange', linestyle='--', lw=2, label='ROD' if 'ROD' not in already_legend else "")
                already_legend.add('MOS')
                already_legend.update(['ROI', 'ROD'])

    # combine legends from all axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()

    all_handles = handles1 + handles2 + handles3
    all_labels = labels1 + labels2 + labels3

    by_label = dict(zip(all_labels, all_handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=len(by_label), fontsize=8)

    plt.tight_layout()
    plt.title(mun)
    plt.savefig(f'{path_png}', dpi=300)
    print(f'{path_png}')
    plt.close()


'''
    if os.path.exists(path_png) and os.path.exists(path_csv):
        print(f'Already exists: {mun}')
        return
'''
import os
global path_media_png, path_media_csv, path_csv_gee
path_media = '/media/ED520C85-64E3-48F9-88F6-2F3550735AE9/MESTRADO'
path_csv_gee = '/home/metop/tmp/gee'
path_media_png = f'{path_media}/results/figs/all_seasons_plot'
path_media_csv = f'{path_media}/results/csv/timesat'
list_mun = f'{path_media}/list_mun.gee'
with open(list_mun) as f:
    list_mun = [line.rstrip('\n') for line in f]

ranges = [
    (1000000, 2000000),
    (2000000, 3000000),
    (3000000, 4000000),
    (4100000, 4200000),
    (4200000, 4300000),
    (4300000, 4400000),
    (5100000, 5200000),
    (5200000, 5300000),
    (5300000, 5400000),
 ]

import sys
grupo = int(sys.argv[1])
region = list(filter(lambda x: x.startswith(str(grupo)), list_mun))
for mun in region:
    mun = str(mun)
    path_png = f'{path_media_png}/{mun}.png'
    path_csv = f'{path_media_csv}/{mun}.csv'
    if os.path.exists(path_png) and os.path.exists(path_csv):
        print(f'Already exists: {mun}')
    else:
        df, ndvi_quad, peaks, dados_curvas, parametros_lista = calcular_dados(mun, path_csv)
        plotar_safra(mun, df, ndvi_quad, peaks, dados_curvas, parametros_lista, path_png)
        del df, peaks, dados_curvas, parametros_lista
#-----------------------------------------------
# Plot all seasons
#-----------------------------------------------
'''
fig, ax1 = plt.subplots(figsize=(14, 5))
x = np.arange(len(df))
ax1.bar(x, df['pct_cloud_soy'], color='grey', width=1, label='Cloud (%)')
ax1.set_ylim(0, 100)
ax1.set_ylabel('Cloud Cover (%)')
ax1.set_xticks(list(range(len(df))))
ax1.set_xticklabels(df.index.strftime('%F'), rotation=45, ha='right', fontsize=8)
ax1.set_xlabel("Date")

ax2 = ax1.twinx()
ax2.plot(x, df['ndvi_p50'], color='k', linewidth=2, label='NDVI P50')
ax2.scatter(x, df['ndvi_p50'], color='k', s=100, marker='o', label='NDVI P50')
ax2.scatter(peaks, df['ndvi_p50'].iloc[peaks], color='orange', s=50, marker='*', zorder=99)
ax2.set_ylim(0, 1)
ax2.set_ylabel('NDVI P50')

# plot each fitted curve
for i, curva in enumerate(curvas_ajustadas):
    data = curva['data']
    # only plot where dates exist in the main index (df)
    idx = df.index.get_indexer(data['date'])
    valid = idx >= 0
    ax2.plot(idx[valid], data['ndvi_fit'][valid], color='red', linewidth=2, alpha=0.7,
             label='Double Logistic Fit' if i == 0 else "")

# deduplicate legend
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.show()

#-----------------------------------------------
# Plot a single season with full detail
#-----------------------------------------------
df['ndvi_p50'][start:end]
curva['data'].set_index('date', inplace=True)
data = pd.concat((curva['data'], df['ndvi_p50'][start:end]), axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['ndvi_interp'], '-o', label='NDVI polynomial interpolation', color='tab:blue', markersize=4)
ax.plot(data.index, data['ndvi_fit'], label='Fitted Curve (Double Logistic)', color='red', linewidth=2)
ax.plot(data.index, data['ndvi_p50'], 'd', label='Original NDVI (df["ndvi_p50"])', color='black', markersize=12, zorder=99)
ax.set_ylabel("NDVI")
ax.set_xlabel("Date")
ax.set_title("Fitted curve vs observed and original NDVI (season)")
ax.legend()
plt.tight_layout()
plt.show()
'''
