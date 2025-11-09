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
# Trabalha nos dados interpolados
# https://github.com/lewistrotter/PhenoloPy
#--------------------------------------------
def double_logistic(t, a, b, c1, d, e, c2, f0):
    return (a / (1 + np.exp(-b * (t - c1)))) + (d / (1 + np.exp(-e * (t - c2)))) + f0

def to_x(dt):
    if dt < df.index[0] or dt > df.index[-1]:
        return None
    return df.index.get_indexer([dt], method='nearest')[0]

def criar_resposta_erro(error_code,error_str,data_max):
    print(data_max.strftime('%F'))
    return {
        'safra_temporada': np.nan, 
        'data_ini': np.nan, 
        'data_max': data_max.strftime('%F'), 
        'data_fim': np.nan,
        'quant_dias': np.nan, 
        'SOS': np.nan, 
        'ROI': np.nan, 
        'ROI_DT':np.nan,  
        'ROI_START_VALUE':np.nan,
        'POS': np.nan, 
        'ROD': np.nan, 
        'ROD_DT':np.nan,
        'ROD_END_VALUE':np.nan,
        'EOS': np.nan,
        'SIOS': np.nan, 
        'AOS': np.nan, 
        'MOS': np.nan, 
        'MOS_DT1':np.nan, 
        'MOS_DT2':np.nan, 
        'LOS': np.nan, 
        'BSE': np.nan, 
        'ERROR': error_code,
        'ER_DISCR': error_str
    }

def parametros_safra_curva_df(data):
    # data deve ser um DataFrame já indexado por data, com colunas: ndvi_fit, ndvi_p50 (e opcionalmente ndvi_poly2)
    # 1. Data do pico (máximo da curva suavizada)
    curva_suave = data['ndvi_fit']
    data_max = data['ndvi_p50'].idxmax()
    pos = data['ndvi_p50'][data_max]
    if data_max.month in [6,7,8,9]:
        return criar_resposta_erro(1,'peak fora de época',data_max)
    # 2. Início da safra: primeiro valor > 0 no NDVI original ANTES do pico
    antes_pico = data.loc[:data_max]
    data_ini = antes_pico[antes_pico['ndvi_p50'] > 0]['ndvi_p50'].idxmin()
    sos = antes_pico.loc[data_ini, 'ndvi_p50']
    depois_pico = data.loc[data_max:]
    data_fim = depois_pico[depois_pico['ndvi_p50'] > 0]['ndvi_p50'].idxmin()
    eos = depois_pico.loc[data_fim, 'ndvi_p50']
    quant_dias = (data_fim - data_ini).days
    # >>> Validações com as novas regras
    if data_ini.month in [4,5,6,7]:
        return criar_resposta_erro(2,'plantio fora de época',peak)
    elif data_fim.month in [7,8,9,10,11]:
        return criar_resposta_erro(3,'colheita fora de época',data_max)
    elif quant_dias > 240:
        return criar_resposta_erro(4,'ciclo muito longo',data_max)
    elif quant_dias < 90:
        return criar_resposta_erro(5,'ciclo muito curto',data_max)
    else:
        safra_window = curva_suave.loc[data_ini:data_fim]
        mos = safra_window.min() + ((safra_window.max() - safra_window.min()) * .8)
        mos_values = safra_window[safra_window >= mos]
        mos_d1 = mos_values.index[0]
        mos_d2 = mos_values.index[-1]
        bse = (sos + eos) / 2
        los = safra_window.count()
        # Cálculo ROI
        roi_values = curva_suave.loc[data_ini:mos_d1]
        init20 = roi_values.min() + ((roi_values.max() - roi_values.min()) * .1)
        roi_values = roi_values[roi_values >= init20]
        roi_start = roi_values.index[0]
        roi_start_value = roi_values.iloc[0]
        if len(roi_values) < 2 or np.any(np.isnan(roi_values.values)) or np.any(np.isinf(roi_values.values)) or np.ptp(roi_values.values) == 0:
            return criar_resposta_erro(6,'calculo do ROI',data_max)  # Use código de erro único para debug
        else:
            y = roi_values.values
            x = np.arange(y.size)
            fit = np.polyfit(x, y, 1)
            roi_slope = fit[0]
        # Cálculo ROD
        rod_values = curva_suave.loc[mos_d2:data_fim]
        fim20 = rod_values.min() + ((rod_values.max() - rod_values.min()) * .1)
        rod_values = rod_values[rod_values >= fim20]
        rod_end = rod_values.index[-1]
        rod_end_value = rod_values.iloc[-1]
        if len(rod_values) < 2 or np.any(np.isnan(rod_values.values)) or np.any(np.isinf(rod_values.values)) or np.ptp(rod_values.values) == 0:
            return criar_resposta_erro(7,'calculo ROD',data_max)
        else:
            y = rod_values.values
            x = np.arange(y.size)
            fit = np.polyfit(x, y, 1)
            rod_slope = fit[0]
        aos = pos - bse
        sios = np.trapz(safra_window.values)
        if np.isnan([sos, roi_slope, pos, rod_slope, eos, sios, aos, mos, los, bse]).any():
            return criar_resposta_erro(8,'algum parâmetro como NAN',data_max)
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
                'ROI_DT':roi_start,  
                'ROI_START_VALUE':roi_start_value,
                'POS': pos, 
                'ROD': rod_slope, 
                'ROD_DT':rod_end,
                'ROD_END_VALUE':rod_end_value,
                'EOS': eos,
                'SIOS': sios, 
                'AOS': aos, 
                'MOS': mos, 
                'MOS_DT1':mos_d1, 
                'MOS_DT2':mos_d2, 
                'LOS': los, 
                'BSE': bse, 
                'ERROR': err,
                'ER_DISCR':None
                }
        return timesdat_dict


#for peak in peaks_validos:
def ajustar_curva_safra(peak, df, ndvi_quad, range_window=105, blend_points=4):
    """
    Ajusta uma curva double logistic ou polinomial a uma janela temporal
    centrada em um pico de NDVI para modelar o comportamento vegetativo.
    
    Args:
        peak: Índice do pico a ser processado
        df: DataFrame com os dados de NDVI original
        ndvi_quad: Série temporal com dados de NDVI para ajuste
        range_window: Janela temporal em dias para análise (padrão: 105)
        blend_points: Número de pontos para blending nas extremidades (padrão: 4)
    
    Returns:
        Dicionário com parâmetros da curva ajustada ou None em caso de falha
    """
    # Obtém a data correspondente ao pico atual no DataFrame
    peak_date = df.iloc[peak].name
    
    # Define a janela temporal para análise: range_window dias antes e depois do pico
    start = peak_date - dt.timedelta(days=range_window)
    # Refina o início da janela: encontra o ponto mínimo de NDVI no período anterior ao pico
    # e subtrai 8 dias para garantir capturar o início completo da curva
    start = ndvi_quad[start:peak_date].idxmin() - dt.timedelta(days=8)
    
    # Define o fim da janela temporal: range_window dias após o pico
    end = peak_date + dt.timedelta(days=range_window)
    # Refina o fim da janela: encontra o ponto mínimo de NDVI no período posterior ao pico
    # e adiciona 8 dias para garantir capturar o final completo da curva
    end = ndvi_quad[peak_date:end].idxmin() + dt.timedelta(days=8)
    
    # Extrai a sub-série temporal de NDVI dentro da janela definida
    sub_ndvi = ndvi_quad.loc[start:end].copy()
    # Remove quaisquer valores NaN que possam estar presentes
    sub_ndvi = sub_ndvi.dropna()
    
    # Verifica se há dados suficientes para processamento (mínimo de 10 pontos)
    if len(sub_ndvi) < 10:
        print(f"Dados insuficientes para o pico em {peak}: apenas {len(sub_ndvi)} pontos")
        return None  # Pula para o próximo pico se não houver dados suficientes
    
    # Converte os valores de NDVI para um array numpy
    y = sub_ndvi.values
    # Cria array de tempo em dias relativos (dias desde o início da sub-série)
    t = (sub_ndvi.index - sub_ndvi.index[0]).days
    
    # Bloco principal de tentativa de ajuste da curva
    try:
        # Inicializa pesos iguais para todos os pontos (para uso no curve_fit)
        weights = np.ones_like(y)
        
        # Identifica a posição do valor máximo de NDVI na sub-série
        idx_peak = np.argmax(y)
        # Encontra o ponto mínimo à esquerda do pico
        idx_min_esq = np.argmin(y[:idx_peak])
        # Encontra o ponto mínimo à direita do pico (ajustando pelo índice do pico)
        idx_min_dir = np.argmin(y[idx_peak:]) + idx_peak
        
        # Calcula a amplitude dos valores de NDVI (diferença entre máximo e mínimo)
        a = y.max() - y.min()
        # Define um valor mínimo de gap para diferenças entre limites
        gap = 1e-3
        
        # Define limites inferiores para os parâmetros da função double_logistic
        lower = [
            0,          # Amplitude mínima
            0.001,      # Taxa de crescimento mínima
            max(0, idx_min_esq-10),  # Posição mínima do vale esquerdo (com margem de segurança)
            -1,         # Taxa de declínio mínima
            0.001,      # Peso mínimo da segunda logística
            max(0, idx_min_dir-10),  # Posição mínima do vale direito (com margem de segurança)
            0           # Valor base mínimo
        ]
        
        # Define limites superiores para os parâmetros
        upper = [
            2*a,        # Amplitude máxima (o dobro da amplitude calculada)
            1,          # Taxa de crescimento máxima
            min(len(y)-1, idx_min_esq+10),  # Posição máxima do vale esquerdo
            0,          # Taxa de declínio máxima
            1,          # Peso máximo da segunda logística
            min(len(y)-1, idx_min_dir+10),  # Posição máxima do vale direito
            y.max()     # Valor base máximo
        ]
        
        # Garante que todos os limites superiores sejam maiores que os inferiores
        for i in range(len(lower)):
            if upper[i] <= lower[i]:
                upper[i] = lower[i] + gap
        
        # Define valores iniciais inteligentes para os parâmetros
        p0 = [
            a * 0.8,           # Amplitude inicial (80% da amplitude calculada)
            0.1,               # Taxa de crescimento inicial
            idx_min_esq,       # Posição inicial do vale esquerdo
            -0.1,              # Taxa de declínio inicial
            0.1,               # Peso inicial da segunda logística
            idx_min_dir,       # Posição inicial do vale direito
            y.min()            # Valor base inicial
        ]
        
        # Primeira tentativa de ajuste da curva double_logistic
        try:
            # Ajusta a curva double_logistic aos dados usando os parâmetros iniciais e limites
            params, _ = curve_fit(double_logistic, t, y, p0, sigma=1/weights, 
                                 bounds=(lower, upper), maxfev=5000)
        except RuntimeError:
            # Se a primeira tentativa falhar, tenta com limites mais amplos
            print(f"Primeira tentativa falhou para pico {peak}, tentando com limites mais amplos")
            # Limites mais amplos para permitir maior flexibilidade no ajuste
            lower = [0, 0.0001, 0, -2, 0.0001, 0, 0]
            upper = [3*a, 2, len(y)-1, 0, 2, len(y)-1, y.max()*1.5]
            # Segunda tentativa com limites ampliados
            params, _ = curve_fit(double_logistic, t, y, p0, sigma=1/weights, 
                                 bounds=(lower, upper), maxfev=5000)
        
        # Calcula os valores da curva ajustada usando os parâmetros otimizados
        ndvi_fit = double_logistic(t, *params)
        
        # Obtém o valor real de NDVI no pico a partir do DataFrame original
        peak_ndvi = df.iloc[peak]['ndvi_p50']
        # Obtém o valor máximo da curva ajustada
        fit_max = ndvi_fit.max()
        
        # Ajusta a escala da curva ajustada para corresponder ao valor real do pico
        if fit_max > 0 and peak_ndvi > 0:
            scaling = peak_ndvi / fit_max
            ndvi_fit_scaled = ndvi_fit * scaling
        else:
            ndvi_fit_scaled = ndvi_fit
        
        # Garante que todos os valores da curva ajustada estejam no intervalo [0, 1]
        ndvi_fit_scaled = np.clip(ndvi_fit_scaled, 0, 1)
        
        # Armazena os resultados do ajuste bem-sucedido
        return {
            'peak': peak,  # Índice do pico original
            'params': params,  # Parâmetros otimizados da curva
            'data': pd.DataFrame({
                'date': sub_ndvi.index,  # Datas da sub-série
                'ndvi_poly2': y,  # Valores originais de NDVI
                'ndvi_fit': ndvi_fit_scaled  # Valores da curva ajustada
            })
        }
        
    # Se ocorrer qualquer exceção durante o processo de ajuste principal
    except Exception as e:
        print(f'Ajuste falhou na safra com peak em {peak}: {e}')
        # Tenta uma abordagem alternativa usando ajuste polinomial
        try:
            # Ajusta um polinômio de segundo grau aos dados
            fit_coeffs = np.polyfit(t, y, 2)
            # Calcula os valores do polinômio ajustado
            ndvi_fit = np.polyval(fit_coeffs, t)
            # Obtém o valor real de NDVI no pico
            peak_ndvi = df.iloc[peak]['ndvi_p50']
            fit_max = ndvi_fit.max()
            # Ajusta a escala do polinômio para corresponder ao valor real do pico
            if fit_max > 0 and peak_ndvi > 0:
                scaling = peak_ndvi / fit_max
                ndvi_fit_scaled = ndvi_fit * scaling
            else:
                ndvi_fit_scaled = ndvi_fit
            # Garante que os valores estejam no intervalo [0, 1]
            ndvi_fit_scaled = np.clip(ndvi_fit_scaled, 0, 1)
            print(f"Usando fallback polinomial para pico {peak}")
            # Armazena os resultados do ajuste polinomial (fallback)
            return {
                    'peak': peak,
                    'params': fit_coeffs,  # Coeficientes do polinômio
                    'data': pd.DataFrame({
                        'date': sub_ndvi.index, 
                        'ndvi_poly2': y, 
                        'ndvi_fit': ndvi_fit_scaled
                        }),
                    'fallback': True  # Flag indicando que foi usado método alternativo
                    }
            
        # Se o método alternativo também falhar
        except Exception as e2:
            print(f"Fallback também falhou para pico {peak}: {e2}")
            return None

def aplicar_savitzky_golay(ndvi_series, window_length=48, polyorder=1):
    """
    Aplica filtro Savitzky-Golay para suavizar a série temporal
    """
    try:
        # Verifica se há dados suficientes para o filtro
        if len(ndvi_series) < window_length:
            print(f"Dados insuficientes para Savitzky-Golay: {len(ndvi_series)} pontos, necessários {window_length}")
            return ndvi_series
        
        # Remove NaNs temporariamente para aplicar o filtro
        original_index = ndvi_series.index
        valid_mask = ~ndvi_series.isna()
        valid_values = ndvi_series[valid_mask].values
        
        if len(valid_values) < window_length:
            print(f"Valores válidos insuficientes para Savitzky-Golay: {len(valid_values)}")
            return ndvi_series
        
        # Aplica o filtro Savitzky-Golay
        smoothed_values = savgol_filter(valid_values, window_length, polyorder)
        
        # Recria a série com os valores suavizados
        smoothed_series = pd.Series(index=ndvi_series.index, dtype=float)
        smoothed_series[valid_mask] = smoothed_values
        
        # Interpola quaisquer valores faltantes
        if smoothed_series.isna().any():
            smoothed_series = smoothed_series.interpolate(method='linear')
            # Preenche valores restantes nas extremidades
            smoothed_series = smoothed_series.bfill().ffill()
        
        print(f"Filtro Savitzky-Golay aplicado: janela={window_length}, ordem={polyorder}")
        return smoothed_series
        
    except Exception as e:
        print(f"Erro ao aplicar Savitzky-Golay: {e}")
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
    if peaks.size > 0:  # para numpy array
        print(f'Identificando parâmetros para peaks: {peaks}')
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
            print(f'Error em {_}: {e}')
            pass
    pd.DataFrame(parametros_lista).to_csv(path_csv)
    print(f'{path_csv}')
    return df, ndvi_quad, peaks, dados_curvas, parametros_lista


def plotar_safra(mun, df, ndvi_quad, peaks, dados_curvas, parametros_lista, path_png):
    print(df)
    fig, ax1 = plt.subplots(figsize=(18, 6))
    x = np.arange(len(df))
    ax1.bar(x, df['pct_cloud_soy'], color='grey', width=1, label='Nuvem (%)', alpha=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Cobertura de Nuvem (%)')
    datas_peaks = df.iloc[peaks].index.strftime('%Y/%m')
    ax1.set_xticks(peaks)
    ax1.set_xticklabels(datas_peaks, rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel("Data")
    ax2 = ax1.twinx()
    
    ax2.plot(x, df['ndvi_p50'], color='k', linewidth=2, label='NDVI P50')
    ax2.scatter(x, df['ndvi_p50'], color='k', s=40, marker='o', label='NDVI P50')
    ax2.scatter(peaks, df['ndvi_p50'].iloc[peaks], color='orange', s=60, marker='*', zorder=99, label='Peaks')
    ax2.set_ylim(.1, .95)
    ax2.set_ylabel('NDVI P50')

    # Eixo 3: Savitzky-Golay (para ROI/ROD e detecção)
    ax3 = ax1.twinx()
    # Mapear datas do savgol para posições x
    date_to_x = {date: i for i, date in enumerate(df.index)}
    
    ndvi_smooth_x = []
    ndvi_smooth_y = []
    
    for date, value in ndvi_quad.items():
        if date in date_to_x:
            ndvi_smooth_x.append(date_to_x[date])
            ndvi_smooth_y.append(value)
        else:
            # Interpolar posição para datas entre os pontos originais
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
    
    # Plot da série Savitzky-Golay
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
                ax2.annotate('SOS', (x_sos, params['SOS']), xytext=(5,10), textcoords='offset points', fontsize=10, color='green')
                already_legend.add('SOS')
            if x_pos: 
                ax2.scatter(x_pos, params['POS'], facecolors='none', edgecolors='blue', s=110, linewidths=2, label='POS' if 'POS' not in already_legend else "")
                ax2.annotate('POS', (x_pos, params['POS']), xytext=(5,-15), textcoords='offset points', fontsize=10, color='blue')
                already_legend.add('POS')
            if x_eos: 
                ax2.scatter(x_eos, params['EOS'], facecolors='none', edgecolors='orange', s=110, linewidths=2, label='EOS' if 'EOS' not in already_legend else "")
                ax2.annotate('EOS', (x_eos, params['EOS']), xytext=(-40,10), textcoords='offset points', fontsize=10, color='orange')
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
                already_legend.update(['ROI','ROD'])
    
    # Combinar as legendas de todos os eixos
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
        print(f'Já existe: {mun}')
        return
'''
import os
global path_media_png, path_media_csv, path_csv_gee
path_media = '/media/ED520C85-64E3-48F9-88F6-2F3550735AE9/MESTRADO' 
#path_csv_gee = f'{path_media}/gee'
path_csv_gee = '/home/metop/tmp/gee'
path_media_png = f'{path_media}/results/figs/all_seasons_plot'
path_media_csv = f'{path_media}/results/csv/timesat'
list_mun = f'{path_media}/list_mun.gee'
with open(list_mun) as f:
    list_mun = [line.rstrip('\n') for line in f]

#list_mun = np.array(list_mun).astype(int)

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
#start, end = ranges[grupo]
#region = list_mun[(list_mun > start) & (list_mun < end)]
region = list(filter(lambda x: x.startswith(str(grupo)), list_mun))
for mun in region:
    mun = str(mun)
    path_png = f'{path_media_png}/{mun}.png'
    path_csv = f'{path_media_csv}/{mun}.csv'
    if os.path.exists(path_png) and os.path.exists(path_csv):
        print(f'Já existe: {mun}')
    else:
        df, ndvi_quad, peaks, dados_curvas, parametros_lista = calcular_dados(mun, path_csv)
        plotar_safra(mun, df, ndvi_quad, peaks, dados_curvas, parametros_lista, path_png)
        del df, peaks, dados_curvas, parametros_lista
#-----------------------------------------------
# Plot de todas as safras
#-----------------------------------------------
'''
fig, ax1 = plt.subplots(figsize=(14, 5))
x = np.arange(len(df))
ax1.bar(x, df['pct_cloud_soy'], color='grey', width=1, label='Nuvem (%)')
ax1.set_ylim(0, 100)
ax1.set_ylabel('Cobertura de Nuvem (%)')
ax1.set_xticks(list(range(len(df))))
ax1.set_xticklabels(df.index.strftime('%F'), rotation=45, ha='right', fontsize=8)
ax1.set_xlabel("Data")

ax2 = ax1.twinx()
ax2.plot(x, df['ndvi_p50'], color='k', linewidth=2, label='NDVI P50')
ax2.scatter(x, df['ndvi_p50'], color='k', s=100, marker='o', label='NDVI P50')
ax2.scatter(peaks, df['ndvi_p50'].iloc[peaks], color='orange', s=50, marker='*', zorder=99)
ax2.set_ylim(0, 1)
ax2.set_ylabel('NDVI P50')

# Plotando cada curva ajustada
for i, curva in enumerate(curvas_ajustadas):
    data = curva['data']
    # Garantir que só plote onde as datas estão no índice principal (df)
    idx = df.index.get_indexer(data['date'])
    valid = idx >= 0
    ax2.plot(idx[valid], data['ndvi_fit'][valid], color='red', linewidth=2, alpha=0.7,label='Double Logistic Fit' if i == 0 else "")

# Só para não repetir legenda
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.show()

#-----------------------------------------------
# Plot de uma safra com os dados completos
#-----------------------------------------------
df['ndvi_p50'][start:end]
curva['data'].set_index('date',inplace=True)
data = pd.concat((curva['data'],df['ndvi_p50'][start:end]),axis=1)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data.index, data['ndvi_interp'], '-o', label='NDVI interpolação polynomial', color='tab:blue', markersize=4)
ax.plot(data.index, data['ndvi_fit'], label='Curva Ajustada (Double Logistic)', color='red', linewidth=2)
ax.plot(data.index, data['ndvi_p50'], 'd', label='NDVI Original (df["ndvi_p50"])', color='black', markersize=12, zorder=99)
ax.set_ylabel("NDVI")
ax.set_xlabel("Data")
ax.set_title("Curva ajustada vs NDVI observado e original (safra)")
ax.legend()
plt.tight_layout()
plt.show()
'''