import ee
ee.Initialize(project='ee-gabrielluanrodrigues')

# ================== CONFIG ==================
ASSET_TPL = {
    'mask': 'projects/ee-gabrielluanrodrigues/assets/{year}/mask',
    'soil': 'projects/ee-gabrielluanrodrigues/assets/{year}/soil',
    'shp' : 'projects/ee-gabrielluanrodrigues/assets/{year}/shp'
}
PROP_MUNI_ID     = 'D1C'   # ajuste se mudar
PROP_FILTRA_SOJA = 'V'     # > 0
YEAR_START, YEAR_END = 2025,2025  # Período completo
CHUNK_SIZE = 1  # Processar 1 município por vez
SELECTORS = [
    PROP_MUNI_ID,                # id do município que vem do seu shapefile
    'date',
    'ndvi_mean','ndvi_stdDev','ndvi_p5','ndvi_p50','ndvi_p95',
    'area_total_soja_sum','area_nuvem_soja_sum','pct_cloud_soy'
]

IC = ee.ImageCollection('MODIS/061/MOD09Q1')
pixel_area = ee.Image.pixelArea()

# ================== FUNÇÕES ==================
def add_cloud_info_mod09q1(img):
    qa = img.select('State')
    cloud_state  = qa.rightShift(10).bitwiseAnd(3).neq(0)  # bits 10-11
    cloud_shadow = qa.bitwiseAnd(1 << 2).neq(0)            # bit 2
    cirrus       = qa.rightShift(8).bitwiseAnd(3).gt(0)    # bits 8-9
    cloud = cloud_state.Or(cloud_shadow).Or(cirrus).rename('cloud')  # 1 nuvem
    clear = cloud.Not().rename('clear')
    return img.addBands([cloud, clear])

def add_ndvi(img):
    ndvi = img.normalizedDifference(['sur_refl_b02','sur_refl_b01']).rename('ndvi')
    return img.addBands(ndvi)

def chunk_fc(fc, size):
    n = fc.size()
    lst = fc.toList(n)
    chunks = []
    i = 0
    total = n.getInfo()
    while i < total:
        chunks.append(ee.FeatureCollection(lst.slice(i, i + size)))
        i += size
    return chunks

def path_for(year, kind):
    return ASSET_TPL[kind].format(year=year)

# ================== LOOP ANUAL ==================
for year in range(YEAR_START, YEAR_END + 1):
    y0 = ee.Date.fromYMD(year, 1, 1)
    y1 = y0.advance(1, 'year')

    ic_year = (IC.filterDate(y0, y1)
                 .map(add_cloud_info_mod09q1)
                 .map(add_ndvi))

    if ic_year.size().getInfo() == 0:
        print(f'{year}: sem imagens')
        continue

    # Imagem de referência para projeção/escala (250 m)
    img_ref = ee.Image(ic_year.first())
    proj = img_ref.projection()

    # ===== assets daquele ano =====
    mask_path = path_for(year, 'mask')
    soil_path = path_for(year, 'soil')
    shp_path  = path_for(year, 'shp')

    # Carrega e alinha
    mask_soja = ee.Image(mask_path) \
        .reproject(crs=proj, scale=proj.nominalScale()).gt(0)

    solo = ee.Image(soil_path).select('b1') \
        .reproject(crs=proj, scale=proj.nominalScale())

    # Municípios com soja > 0 (do ano)
    munis_all = ee.FeatureCollection(shp_path).filter(ee.Filter.gt(PROP_FILTRA_SOJA, 0))
    if munis_all.size().getInfo() == 0:
        print(f'{year}: shp vazio (sem municípios com V>0?)')
        continue

    muni_chunks = chunk_fc(munis_all, CHUNK_SIZE)

    for idx, muni_fc in enumerate(muni_chunks, start=1):
        # Obter ID do município atual (chunk size 1)
        muni_feature = ee.Feature(muni_fc.first())
        muni_id = muni_feature.get(PROP_MUNI_ID).getInfo()
        print(f"\nProcessando município {muni_id} ({idx}/{len(muni_chunks)}) - Ano {year}")

        def per_image(img):
            date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')

            ndvi_soja  = img.select('ndvi').updateMask(mask_soja).rename('ndvi')
            area_total = pixel_area.updateMask(mask_soja).rename('area_total_soja')
            area_cloud = pixel_area.updateMask(img.select('cloud').And(mask_soja)).rename('area_nuvem_soja')
            stack = ndvi_soja.addBands([area_total, area_cloud])

            # ---- Redutores estatísticos ----
            red_mean = ee.Reducer.mean().setOutputs(['ndvi_mean'])
            red_std  = ee.Reducer.stdDev().setOutputs(['ndvi_stdDev'])
            red_pct  = ee.Reducer.percentile([5,50,95]).setOutputs(['ndvi_p5','ndvi_p50','ndvi_p95'])
            reducer_stats = red_mean.combine(red_std, sharedInputs=True).combine(red_pct, sharedInputs=True)

            reducer_sum_total = ee.Reducer.sum().setOutputs(['area_total_soja_sum'])
            reducer_sum_cloud = ee.Reducer.sum().setOutputs(['area_nuvem_soja_sum'])

            reducer_all = (reducer_stats
                           .combine(reducer_sum_total, sharedInputs=False)
                           .combine(reducer_sum_cloud, sharedInputs=False))

            # Aplicar redução apenas no município atual
            reduced = stack.reduceRegions(
                collection=muni_fc,
                reducer=reducer_all,
                scale=proj.nominalScale(),
                tileScale=4  # Aumentado para melhor performance
            )

            # Calcular porcentagem de nuvem
            def add_cloud_pct(f):
                area_total = ee.Number(f.get('area_total_soja_sum')).max(1)
                cloud_pct = ee.Number(f.get('area_nuvem_soja_sum')).divide(area_total).multiply(100)
                return f.set({
                    'date': date_str,
                    'pct_cloud_soy': cloud_pct
                })
                
            return reduced.map(add_cloud_pct)

        # Processar todas as imagens do ano para este município
        fc_year_muni = ee.FeatureCollection(ic_year.map(per_image)).flatten()

        # Configurar exportação
        desc = f'modis09q1_ndvi_cloud_y{year}_muni{muni_id}'
        print(f'Exportando: {desc}')

        task = ee.batch.Export.table.toDrive(
            collection=fc_year_muni,
            description=desc,
            folder='gee_exports',
            fileNamePrefix=desc,
            fileFormat='CSV',
            selectors=SELECTORS
        )
        task.start()
        print(f"Task iniciada para município {muni_id} ({task.id})")

# Monitorar tasks (opcional)
print("\nMonitoramento de tasks:")
for t in ee.batch.Task.list():
    st = t.status()
    print(f"{st['description']} ({st['id']}): {st['state']}")
    if st['state'] == 'FAILED':
        print(f"    ERRO: {st.get('error_message', 'Sem detalhes')}")
'''
import ee
ee.Initialize(project='ee-gabrielluanrodrigues')

# ================== CONFIG ==================
ASSET_TPL = {
    'mask': 'projects/ee-gabrielluanrodrigues/assets/{year}/mask',
    'soil': 'projects/ee-gabrielluanrodrigues/assets/{year}/soil',
    'shp' : 'projects/ee-gabrielluanrodrigues/assets/{year}/shp'
}
PROP_MUNI_ID     = 'D1C'   # ajuste se mudar
PROP_FILTRA_SOJA = 'V'     # > 0
YEAR_START, YEAR_END = 2001, 2001
CHUNK_SIZE = 250
SELECTORS = [
    PROP_MUNI_ID,                # id do município que vem do seu shapefile
    'date',
    'ndvi_mean','ndvi_stdDev','ndvi_p5','ndvi_p50','ndvi_p95',
    'area_total_soja_sum','area_nuvem_soja_sum','pct_cloud_soy'
]

IC = ee.ImageCollection('MODIS/061/MOD09Q1')
pixel_area = ee.Image.pixelArea()

# ================== FUNÇÕES ==================
def add_cloud_info_mod09q1(img):
    qa = img.select('State')
    cloud_state  = qa.rightShift(10).bitwiseAnd(3).neq(0)  # bits 10-11
    cloud_shadow = qa.bitwiseAnd(1 << 2).neq(0)            # bit 2
    cirrus       = qa.rightShift(8).bitwiseAnd(3).gt(0)    # bits 8-9
    cloud = cloud_state.Or(cloud_shadow).Or(cirrus).rename('cloud')  # 1 nuvem
    clear = cloud.Not().rename('clear')
    return img.addBands([cloud, clear])

def add_ndvi(img):
    ndvi = img.normalizedDifference(['sur_refl_b02','sur_refl_b01']).rename('ndvi')
    return img.addBands(ndvi)

def chunk_fc(fc, size):
    n = fc.size()
    lst = fc.toList(n)
    chunks = []
    i = 0
    total = n.getInfo()
    while i < total:
        chunks.append(ee.FeatureCollection(lst.slice(i, i + size)))
        i += size
    return chunks

def path_for(year, kind):
    return ASSET_TPL[kind].format(year=year)

# ================== LOOP ANUAL ==================
for year in range(YEAR_START, YEAR_END + 1):
    y0 = ee.Date.fromYMD(year, 1, 1)
    y1 = y0.advance(1, 'year')

    ic_year = (IC.filterDate(y0, y1)
                 .map(add_cloud_info_mod09q1)
                 .map(add_ndvi))

    if ic_year.size().getInfo() == 0:
        print(f'{year}: sem imagens')
        continue

    # Imagem de referência para projeção/escala (250 m)
    img_ref = ee.Image(ic_year.first())
    proj = img_ref.projection()

    # ===== assets daquele ano =====
    mask_path = path_for(year, 'mask')
    soil_path = path_for(year, 'soil')
    shp_path  = path_for(year, 'shp')

    # Carrega e alinha
    mask_soja = ee.Image(mask_path) \
        .reproject(crs=proj, scale=proj.nominalScale()).gt(0)

    solo = ee.Image(soil_path).select('b1') \
        .reproject(crs=proj, scale=proj.nominalScale())

    # Municípios com soja > 0 (do ano)
    munis_all = ee.FeatureCollection(shp_path).filter(ee.Filter.gt(PROP_FILTRA_SOJA, 0))
    if munis_all.size().getInfo() == 0:
        print(f'{year}: shp vazio (sem municípios com V>0?)')
        continue

    muni_chunks = chunk_fc(munis_all, CHUNK_SIZE)

    # --- DEBUG: primeiro chunk + primeiro município ---
    first_chunk = muni_chunks[0]                         # 1º chunk
    first_muni  = ee.Feature(first_chunk.first())        # 1º município do chunk
    geom        = first_muni.geometry()
    muni_id     = first_muni.get(PROP_MUNI_ID).getInfo()

    # primeira imagem do ano (já com ndvi/cloud)
    img_first   = ee.Image(ic_year.first())
    date_first  = ee.Date(img_first.get('system:time_start')).format('YYYY-MM-dd').getInfo()

    # NDVI mascarado pela soja (mesmo que no per_image)
    ndvi_soja_dbg = img_first.select('ndvi').updateMask(mask_soja).rename('ndvi')

    # 1) Contagem "à moda do seu snippet" (vai contar tudo > -2, inclusive 0 do unmask)
    cnt_gtm2 = ndvi_soja_dbg.unmask(0).gt(-2).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geom,
        scale=proj.nominalScale(),
        tileScale=2
    ).getInfo()

    # 2) Contagem de pixels VÁLIDOS de NDVI (recomendado)
    cnt_valid = ndvi_soja_dbg.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geom,
        scale=proj.nominalScale(),
        tileScale=2
    ).getInfo()

    # 3) Área total de soja e área com nuvem (mesma lógica do export)
    px = ee.Image.pixelArea()
    area_total = px.updateMask(mask_soja).reduceRegion(
        ee.Reducer.sum(), geom, proj.nominalScale(), tileScale=2
    ).getInfo()
    area_cloud = (pixel_area
              .updateMask(img.select('cloud').And(mask_soja))
              .unmask(0)  # vira 0 quando não houver pixels de nuvem
              .rename('area_nuvem_soja'))

    print(f"Ano {year} | Data da imagem: {date_first} | Munícipe: {muni_id}")
    print("count_gt(-2) (com unmask):", cnt_gtm2)   # dicionário {'ndvi': N}
    print("count_válido NDVI:",        cnt_valid)   # dicionário {'ndvi': N}
    print("area_total_soja(m²):",      area_total)  # {'pixelarea': m2}
    print("area_nuvem_soja(m²):",      area_cloud)  # {'pixelarea': m2}


    for idx, muni_fc in enumerate(muni_chunks, start=1):
        def per_image(img):
            date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')

            ndvi_soja  = img.select('ndvi').updateMask(mask_soja).rename('ndvi')
            area_total = pixel_area.updateMask(mask_soja).rename('area_total_soja')
            area_cloud = pixel_area.updateMask(img.select('cloud').And(mask_soja)).rename('area_nuvem_soja')
            stack = ndvi_soja.addBands([area_total, area_cloud])

            # ---- Nomes explícitos p/ bater com SELECTORS ----
            red_mean = ee.Reducer.mean().setOutputs(['ndvi_mean'])
            red_std  = ee.Reducer.stdDev().setOutputs(['ndvi_stdDev'])
            red_pct  = ee.Reducer.percentile([5,50,95]).setOutputs(['ndvi_p5','ndvi_p50','ndvi_p95'])
            reducer_stats = red_mean.combine(red_std, sharedInputs=True).combine(red_pct, sharedInputs=True)

            reducer_sum_total = ee.Reducer.sum().setOutputs(['area_total_soja_sum'])
            reducer_sum_cloud = ee.Reducer.sum().setOutputs(['area_nuvem_soja_sum'])

            reducer_all = (reducer_stats
                           .combine(reducer_sum_total, sharedInputs=False)
                           .combine(reducer_sum_cloud, sharedInputs=False))

            fc = stack.reduceRegions(
                collection=muni_fc,
                reducer=reducer_all,
                scale=proj.nominalScale(),
                tileScale=2
            )

            fc = fc.map(lambda f: f.set({
                'date': date_str,
                'pct_cloud_soy': ee.Number(f.get('area_nuvem_soja_sum'))
                                   .divide(ee.Number(f.get('area_total_soja_sum')).max(1))
                                   .multiply(100)
            }))
            return fc

        fc_year_chunk = ee.FeatureCollection(ic_year.map(per_image)).flatten()

        desc = f'modis09q1_ndvi_cloud_y{year}_chunk{idx:02d}'
        print('Exportando:', desc)

        task = ee.batch.Export.table.toDrive(
            collection=fc_year_chunk,
            description=desc,
            folder='gee_exports',             # pasta no seu Google Drive (será criada)
            fileNamePrefix=desc,
            fileFormat='CSV',
            selectors=SELECTORS               # mantém só essas colunas
        )
        task.start()

for t in ee.batch.Task.list():
    st = t.status()
    if st['state'] == 'FAILED':
        print(st.get('description'), '=>', st.get('error_message'))
'''
