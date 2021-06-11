#!/usr/bin/env python
import os
import sys
from datetime import datetime
from os import listdir, makedirs, path
from shutil import copyfile

import numpy as np
import xarray as xr
from progressbar import progressbar
from scipy.interpolate import griddata

from utils.grid import Grid
from utils.zbin import read_gzip_binary, write_gzip_binary


def adapt_grid_file(selected_cells):
    lons, lats = selected_cells[:,0], selected_cells[:,1]
    MINLON, MAXLON = lons.min(), lons.max()
    MINLAT, MAXLAT = lats.min(), lats.max()

    diff_lons = abs(np.diff(lons))
    diff_lats = abs(np.diff(lats))
    dlon = diff_lons[diff_lons>0].min()
    dlat = diff_lats[diff_lats>0].min()

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
    

def configure_risico(grid, cells_file):
    cells = np.loadtxt(cells_file)
    lat, lon = cells[:,1], cells[:,0]
    min_lon, max_lon = grid.lons.min(), grid.lons.max()
    min_lat, max_lat = grid.lats.min(), grid.lats.max()
    selected_cells = cells[(lon>=min_lon)&(lon<=max_lon)&(lat>=min_lat)&(lat<=max_lat), :]
    np.savetxt('risico/STATIC/region.txt', selected_cells, fmt=['%.8f','%.8f','%.8f','%.8f','%d'])
    return selected_cells




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
    print(sys.argv)
    # ifs files
    input_dir = sys.argv[1]
    # risico input files
    output_dir = sys.argv[2]
    # list of input files
    file_list_file = sys.argv[3]

    cells_file = os.environ.get('CELLS_FILE', 'risico/STATIC/world.txt')
    veg_file = os.environ.get('PVEG_FILE', 'risico/STATIC/pveg_world.csv')
    copyfile(veg_file, 'risico/STATIC/pveg.csv')
    
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

    selected_cells = configure_risico(grid, cells_file)
    adapt_grid_file(selected_cells)






