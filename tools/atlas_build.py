"""Build the offline world atlas of BrineLab (js/data/atlas_*.js) from open global data sets.

The app normally pulls site data live in the browser. When a live service cannot be reached, the missing
fields are answered from the modules written by this script. Everything is fetched once, here, at build time.

Usage (from the project root; needs Python 3 with numpy, scipy, Pillow, shapely and requests):

    python3 tools/atlas_build.py fetch  [--cache DIR] [relief ocean sst climate nations marine waves]
    python3 tools/atlas_build.py build  [--cache DIR] [--out js/data]
    python3 tools/atlas_build.py all    [--cache DIR]
    node tools/atlas_check.mjs          verify the result against live reference values
    node tools/build.mjs                afterwards: refresh the service-worker pre-cache list

`fetch` downloads into the cache directory (default: $ATLAS_CACHE or ./.atlas-cache; about 1.4 GB). Every
response is kept on disk, so the script can be stopped and re-run: finished downloads are never requested
again. `build` (tools/atlas_pack.py) reads only the cache and writes the data modules, printing the size of
each one. To refresh a data set, delete its folder in the cache and run `fetch` and `build` again.

Sources (all public, no key):
  relief   NOAA NCEI global DEM mosaic (ArcGIS ImageServer exportImage), resampled by the server to 1.5 arc-minutes
           (72 tiles of 30 x 30 degrees)
  ocean    SeaDataCloud global temperature-salinity climatology V2.1 (EMODnet Physics ERDDAP), 0.25 deg, monthly,
           5 m level — used for salinity
  sst      NOAA Coral Reef Watch CoralTemp monthly mean sea-surface temperature, 5 km, 2017-2024, sampled every
           0.25 deg (NOAA CoastWatch ERDDAP, NOAA_DHW_monthly)
  climate  NASA POWER climatology API (regional endpoint, one parameter per call), 2001-2020: all-sky irradiation,
           wind speed at 10 m, air temperature at 2 m
  nations  Natural Earth 1:50m admin-0 countries; World Bank Open Data; Our World in Data electricity mix;
           open.er-api.com exchange rates; ISO 4217 currency codes (datasets/country-codes)
  marine   Open-Meteo Marine API: hourly sea level and currents of the last 60 days at one sea point per 1 degree
           cell that holds a shoreline (about 3000 points)
  waves    Open-Meteo Marine API: hourly significant wave height, period and direction for one full year at one
           sea point per 2 degree shoreline cell (about 1200 points)

The marine and wave jobs are rate limited by the provider (an hourly allowance that about 700 location-years
of hourly data exhaust): the script paces itself, backs off on HTTP 429, stops after repeated refusals and can
simply be started again in the next hour to continue where it stopped.
"""
import datetime, json, os, sys, time
import numpy as np
import requests

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
UA = {'User-Agent': 'BrineLab-atlas-build/1.0 (offline atlas builder; one-off batch)'}
WAVE_YEAR = 2025
MARINE_DAYS = 60
SEA_STEP, WAVE_STEP = 1.0, 2.0  # degrees: grid of the tide/current table and of the wave table


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cached(cache, name, url, *, post=None, tries=6, timeout=180, pause=0.0, check=None, binary=True):
    """Download url to cache/name unless it is already there. Returns the path, or None after all retries failed."""
    path = os.path.join(cache, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wait = 4.0
    for k in range(tries):
        try:
            r = requests.post(url, data=post, headers=UA, timeout=timeout) if post else requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 429:
                raise RuntimeError('HTTP 429 (rate limit)')
            if r.status_code != 200:
                raise RuntimeError('HTTP %d %s' % (r.status_code, r.text[:160].replace('\n', ' ')))
            body = r.content
            if check and not check(body):
                raise RuntimeError('unexpected content')
            with open(path + '.part', 'wb') as f:
                f.write(body)
            os.replace(path + '.part', path)
            if pause:
                time.sleep(pause)
            return path
        except Exception as e:  # network errors, time-outs, rate limits: wait and try again
            log('  retry %d/%d %s: %s' % (k + 1, tries, name, str(e)[:160]))
            time.sleep(wait * (4 if '429' in str(e) else 1))
            wait = min(wait * 2, 120)
    return None


# ------------------------------------------------------------------------------------------------ fetch
RELIEF_PX = 1200  # pixels per 30 degree tile  ->  1.5 arc-minutes


def fetch_relief(cache):
    base = 'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage'
    for y in range(-90, 90, 30):
        for x in range(-180, 180, 30):
            q = 'bbox=%d,%d,%d,%d&bboxSR=4326&imageSR=4326&size=%d,%d&format=tiff&pixelType=S16&interpolation=RSP_BilinearInterpolation&f=image' % (x, y, x + 30, y + 30, RELIEF_PX, RELIEF_PX)
            p = cached(cache, 'relief/r_%d_%d.tif' % (x, y), base + '?' + q, check=lambda b: b[:2] in (b'II', b'MM'), pause=0.5)
            log('relief', x, y, 'ok' if p else 'FAILED')


def fetch_ocean(cache):
    for var in ('Salinity', 'Temperature'):
        for m in range(12):
            q = '%s%%5B%d%%5D%%5B0%%5D%%5B(-80):1:(80)%%5D%%5B(-180):1:(179.75)%%5D' % (var, m)
            p = cached(cache, 'ocean/%s_%02d.nc' % (var, m), 'https://erddap.emodnet-physics.eu/erddap/griddap/SDC_GLO_CLIM_TS_V2_1.nc?' + q, timeout=400, check=lambda b: b[:3] == b'CDF', pause=1.0)
            log('ocean', var, m, 'ok' if p else 'FAILED')


SST_YEARS = range(2017, 2025)  # months averaged into the sea-surface temperature climatology


def fetch_sst(cache):
    """NOAA Coral Reef Watch CoralTemp monthly mean SST (5 km), sampled every 0.25 degrees; two downloads at a time."""
    from concurrent.futures import ThreadPoolExecutor
    def one(ym):
        y, m = ym
        q = 'sea_surface_temperature%%5B(%d-%02d-16T00:00:00Z)%%5D%%5B(89.875):5:(-89.875)%%5D%%5B(-179.875):5:(179.875)%%5D' % (y, m)
        p = cached(cache, 'sst/crw_%d_%02d.nc' % (y, m), 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW_monthly.nc?' + q, timeout=300, check=lambda b: b[:3] == b'CDF', pause=1.0)
        if not p or m == 12:
            log('sst', y, m, 'ok' if p else 'FAILED')
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, [(y, m) for y in SST_YEARS for m in range(1, 13)]))


POWER_PARAMS = ('ALLSKY_SFC_SW_DWN', 'WS10M', 'T2M')


def fetch_climate(cache):
    for par in POWER_PARAMS:
        for y in range(-60, 80, 10):
            for x in range(-180, 180, 10):
                u = 'https://power.larc.nasa.gov/api/temporal/climatology/regional?parameters=%s&community=RE&latitude-min=%d&latitude-max=%d&longitude-min=%d&longitude-max=%d&format=JSON' % (par, y, y + 10, x, x + 10)
                p = cached(cache, 'power/%s_%d_%d.json' % (par, x, y), u, timeout=120, check=lambda b: b'features' in b, pause=0.4)
                if not p:
                    log('climate', par, x, y, 'FAILED')
        log('climate', par, 'done')


WB = {'inflation': 'FP.CPI.TOTL.ZG', 'lendingRate': 'FR.INR.LEND', 'gdpPerCapita': 'NY.GDP.PCAP.CD', 'waterStress': 'ER.H2O.FWST.ZS', 'renewableElectricity': 'EG.ELC.RNEW.ZS',
      'tariff': 'IC.ELC.PRI.KH.DB1619', 'freshwaterPerCapita': 'ER.H2O.INTR.PC', 'safeWaterAccess': 'SH.H2O.SMDW.ZS'}


def fetch_nations(cache):
    cached(cache, 'nations/countries50.geojson', 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson')
    for k, ind in WB.items():
        cached(cache, 'nations/wb_%s.json' % k, 'https://api.worldbank.org/v2/country/all/indicator/%s?format=json&mrnev=1&per_page=400' % ind, pause=0.5)
    cached(cache, 'nations/owid_carbon.csv', 'https://ourworldindata.org/grapher/electricity-mix.csv?frequency=annual&metric=carbon_intensity&source=total&csvType=filtered&time=latest')
    cached(cache, 'nations/owid_renew.csv', 'https://ourworldindata.org/grapher/electricity-mix.csv?frequency=annual&metric=share_of_generation&source=renewables&csvType=filtered&time=latest')
    cached(cache, 'nations/fx.json', 'https://open.er-api.com/v6/latest/USD')
    cached(cache, 'nations/codes.csv', 'https://raw.githubusercontent.com/datasets/country-codes/main/data/country-codes.csv')
    log('nations done')


def load_relief(cache):
    """Whole-world relief at 1.5 arc-minutes as int16 [row 0 = south], from the cached tiles."""
    from PIL import Image
    n = RELIEF_PX
    out = np.zeros((6 * n, 12 * n), dtype=np.int16)
    for iy, y in enumerate(range(-90, 90, 30)):
        for ix, x in enumerate(range(-180, 180, 30)):
            a = np.array(Image.open(os.path.join(cache, 'relief/r_%d_%d.tif' % (x, y)))).astype(np.int32)
            a[np.abs(a) > 12000] = 0
            out[iy * n:(iy + 1) * n, ix * n:(ix + 1) * n] = a[::-1].astype(np.int16)
    return out


def block_mean(a, f):
    h, w = a.shape
    return a.reshape(h // f, f, w // f, f).astype(np.float64).mean(axis=(1, 3))


def marine_points(cache, step):
    """Sea cells of a `step` degree grid that contain a shoreline (at least 20 % sea and some land), 56 S to 72 N,
    each with one representative sea point: the sample deeper than 5 m that lies nearest the cell centre."""
    path = os.path.join(cache, 'marine/points_%g.json' % step)
    if os.path.exists(path):
        return json.load(open(path))
    fine = load_relief(cache)
    f = int(round(step * 40))  # 1.5' samples per cell
    ny, nx = fine.shape[0] // f, fine.shape[1] // f
    sea = (fine < 0).reshape(ny, f, nx, f).mean(axis=(1, 3))
    pts = []
    yy, xx = np.mgrid[0:f, 0:f]
    for i in range(ny):
        la = -90 + (i + 0.5) * step
        if la < -56 or la > 72:
            continue
        for j in range(nx):
            if sea[i, j] < 0.2 or sea[i, j] > 0.98:
                continue
            blk = fine[i * f:(i + 1) * f, j * f:(j + 1) * f]
            score = np.where(blk < -5, np.hypot(yy - f / 2, xx - f / 2), 1e9)
            k = int(np.argmin(score))
            if score.flat[k] >= 1e9:
                continue
            pts.append([round(-90 + (i * f + k // f + 0.5) / 40, 3), round(-180 + (j * f + k % f + 0.5) / 40, 3)])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(pts, open(path, 'w'))
    return pts


def _marine_batches(cache, kind, step, query, batch, pause):
    pts = marine_points(cache, step)
    log(kind, len(pts), 'coastal cells')
    failed = 0
    for b in range(0, len(pts), batch):
        chunk = pts[b:b + batch]
        u = 'https://marine-api.open-meteo.com/v1/marine?latitude=%s&longitude=%s&%s&timezone=GMT&cell_selection=sea' % (','.join(str(p[0]) for p in chunk), ','.join(str(p[1]) for p in chunk), query)
        name = 'marine/%s_%05d.json' % (kind, b)
        fresh = not os.path.exists(os.path.join(cache, name))
        p = cached(cache, name, u, timeout=240, tries=7, check=lambda x: x[:1] in (b'[', b'{') and b'"hourly"' in x, pause=pause if fresh else 0)
        if not p:
            failed += 1
            log(kind, 'batch', b, 'FAILED — run the fetch again later')
            if failed >= 3:
                log(kind, 'stopping after repeated failures (rate limit?)')
                return False
        elif fresh and (b // batch) % 10 == 0:
            log(kind, 'batch', b, 'of', len(pts))
    return failed == 0


def fetch_marine(cache):
    end = datetime.date.today() - datetime.timedelta(days=2)
    stamp = os.path.join(cache, 'marine/period.json')
    if os.path.exists(stamp):
        per = json.load(open(stamp))
    else:
        per = {'start': str(end - datetime.timedelta(days=MARINE_DAYS - 1)), 'end': str(end)}
        os.makedirs(os.path.dirname(stamp), exist_ok=True)
        json.dump(per, open(stamp, 'w'))
    return _marine_batches(cache, 'sea', SEA_STEP, 'hourly=sea_level_height_msl,ocean_current_velocity,ocean_current_direction&start_date=%s&end_date=%s' % (per['start'], per['end']), 50, 8.0)


def fetch_waves(cache):
    return _marine_batches(cache, 'wave', WAVE_STEP, 'hourly=wave_height,wave_period,wave_direction&start_date=%d-01-01&end_date=%d-12-31' % (WAVE_YEAR, WAVE_YEAR), 20, 15.0)


FETCH = {'relief': fetch_relief, 'ocean': fetch_ocean, 'sst': fetch_sst, 'climate': fetch_climate, 'nations': fetch_nations, 'marine': fetch_marine, 'waves': fetch_waves}


def main():
    args = sys.argv[1:]
    cache = os.environ.get('ATLAS_CACHE', os.path.join(os.getcwd(), '.atlas-cache'))
    out = os.path.join(ROOT, 'js', 'data')
    if '--cache' in args:
        i = args.index('--cache'); cache = args[i + 1]; del args[i:i + 2]
    if '--out' in args:
        i = args.index('--out'); out = args[i + 1]; del args[i:i + 2]
    if not args or args[0] not in ('fetch', 'build', 'all'):
        print(__doc__); sys.exit(2)
    os.makedirs(cache, exist_ok=True)
    if args[0] in ('fetch', 'all'):
        for k in (args[1:] or list(FETCH)):
            FETCH[k](cache)
    if args[0] in ('build', 'all'):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import atlas_pack
        atlas_pack.build(cache, out)


if __name__ == '__main__':
    main()
