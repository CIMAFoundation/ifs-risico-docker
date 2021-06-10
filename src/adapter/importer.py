#!/usr/bin/env python
import xarray as xr
from progressbar import progressbar
import numpy as np

import sys
from os import path, makedirs, listdir
from datetime import datetime
from utils.zbin import read_gzip_binary, write_gzip_binary
from utils.grid import Grid

from scipy.interpolate import griddata

def adapt_grid_file(grid):
    full_grid = Grid.from_file('risico/GRID/GRID_TEMPLATE.txt')
    print(full_grid.lats.shape, full_grid.lats.max(), full_grid.lats.min())
    print(full_grid.lons.shape, full_grid.lons.max(), full_grid.lons.min())
    
    dlon = (full_grid.lons.max() - full_grid.lons.min())/full_grid.lons.shape[1]
    dlat = (full_grid.lats.max() - full_grid.lats.min())/full_grid.lats.shape[0]

    print(dlon, dlat)

    MINLON, MAXLON = grid.lons.min(), grid.lons.max()
    MINLAT, MAXLAT = grid.lats.min(), grid.lats.max()

    NROWS = int(round((MAXLAT-MINLAT)/dlat) + 1)
    NCOLS = int(round((MAXLON-MINLON)/dlon) + 1)
    
    grid_string = \
    f'''GRIDREG=true
MINLON={MINLON}
MAXLON={MAXLON}
MINLAT={MINLAT}
MAXLAT={MAXLAT}
GRIDNCOLS={NCOLS}
GRIDNROWS={NROWS}
'''

    with open('risico/GRID/GRID.txt', 'w') as f:
        f.write(grid_string)
    

def configure_risico(grid):
    cells = np.loadtxt('risico/STATIC/world.txt')
    lat, lon = cells[:,1], cells[:,0]
    min_lon, max_lon = grid.lons.min(), grid.lons.max()
    min_lat, max_lat = grid.lats.min(), grid.lats.max()
    selected_cells = cells[(lon>=min_lon)&(lon<=max_lon)&(lat>=min_lat)&(lat<=max_lat), :]
    np.savetxt('risico/STATIC/region.txt', selected_cells, fmt=['%.8f','%.8f','%.8f','%.8f','%d'])




def roll_grid(interp_lons, interp_lats, interp_indexes):
     # find the first element where the longitude is over 180

    roll_point = np.where(interp_lons[0,:]>=180)[0][0]

    interp_lons_ = np.concatenate((
        interp_lons[:, roll_point:]-360, 
        interp_lons[:, 0:roll_point]), 
        axis=1
    )

    interp_lats_ = np.flipud(np.concatenate((
        interp_lats[:, roll_point:], 
        interp_lats[:, 0:roll_point]), 
        axis=1
    ))

    interp_indexes_ = np.flipud(np.concatenate((
        interp_indexes[:, roll_point:], 
        interp_indexes[:, 0:roll_point]), 
        axis=1
    ))
    return interp_lons_, interp_lats_, interp_indexes_

def generate_interp(ds, lat_bounds=(-56.9, 83.6)):
    lons = ds.longitude.values[:]
    lats = ds.latitude.values[:]

    lat_bounds_min, lat_bounds_max = lat_bounds
    index_filter = (lats<=lat_bounds_max) & (lats>=lat_bounds_min)
    lats = lats[index_filter]
    lons = lons[index_filter]

    diff_lons = np.diff(lons)
    diff_lats = np.diff(lats)

    d_lon, d_lat = np.min(np.abs(diff_lons[diff_lons!=0])), np.min(np.abs(diff_lats[diff_lats!=0]))
    d_lon = min(d_lon, d_lat)
    d_lat = d_lon
    lon_min, lat_min, lon_max, lat_max = lons.min(), lats.min(), lons.max(), lats.max()
    interp_lons, interp_lats = np.meshgrid(np.arange(lon_min, lon_max, d_lon), np.arange(lat_max, lat_min, -d_lat))
    
    points = np.array((lons, lats)).T
    values_index = np.arange(points.shape[0], dtype='int')
    interp_indexes = griddata(points, values_index, (interp_lons, interp_lats), method='nearest')

    interp_lons, interp_lats, interp_indexes = roll_grid(interp_lons, interp_lats, interp_indexes)
    grid = Grid(interp_lats, interp_lons, regular=True)
    
    return grid, interp_indexes, index_filter


if __name__ == '__main__':
    # ifs files
    input_dir = sys.argv[1]
    # risico input files
    output_dir = sys.argv[2]
    # list of input files
    file_list_file = sys.argv[3]

    makedirs(output_dir, exist_ok=True)
    input_file_dir = path.dirname(file_list_file)
    if input_file_dir != '':
        makedirs(input_file_dir, exist_ok=True)


    vars_risico = {
     't2m': 'T',
     'u10': 'W',
     'tp': 'P', 
     'd2m': 'H',
    }

    nc_files = list(map(
        lambda f: path.join(input_dir, f),
        sorted(list(filter(
            lambda s: (s.endswith('.grb') or s.endswith('.grib')) and not s.startswith('._'), listdir(input_dir)
    )))))



    grid, interp_indexes, index_filter = None, None, None



    with open(file_list_file, 'w') as file_list:
        for f in nc_files:
            try:
                ds = xr.open_dataset(f, engine='cfgrib')
            except Exception as exp:
                print(exp)
                print('skipping file ', f)
                continue
            for v in ds.variables:
                if grid is None or interp_indexes is None:
                    print('interpolating grid')
                    grid, interp_indexes, index_filter = generate_interp(ds)
                if v in vars_risico:
                    print(f'Reading {v} from {f}')
                    var_risico = vars_risico[v]
                    for idx, step in progressbar(enumerate(ds.step)):
                        date_np = step.valid_time.values
                        date = datetime.utcfromtimestamp(date_np.tolist()/1e9)
                        out_date_str = date.strftime('%Y%m%d%H%M')
                        out_file = f'{output_dir}/{out_date_str}_IFS_{var_risico}.zbin'

                        values = ds[v].values[idx, :]
                        interp_values = values[index_filter][interp_indexes]
                        write_gzip_binary(out_file, interp_values, grid)
                        file_list.write(path.abspath(out_file) + '\n')

                        if var_risico == 'W':
                            wd_values = np.zeros_like(interp_values)
                            out_file = f'{output_dir}/{out_date_str}_IFS_D.zbin'
                            write_gzip_binary(out_file, wd_values, grid)
                            file_list.write(path.abspath(out_file) + '\n')

    configure_risico(grid)
    adapt_grid_file(grid)






