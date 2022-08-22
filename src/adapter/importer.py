#!/usr/bin/env python
import os
import argparse
from datetime import datetime
from os import listdir, makedirs, path
from shutil import copyfile

import numpy as np
import xarray as xr

from progressbar import progressbar
from scipy.interpolate import griddata

from utils.grid import Grid
from utils.zbin import read_gzip_binary, write_gzip_binary

RISICO_VARS = {
    'temperature': dict(name='T'),
    'relative_humidity': dict(name='H'),
    '10m_u_wind_component': dict(name='U', transform=lambda kmh: kmh/3.6),
    '10m_v_wind_component': dict(name='V', transform=lambda kmh: kmh/3.6),
    'total_precipitation': dict(name='P', transform=lambda m: m*1000)
}
SUFFIX = 'IFS'


def adapt_grid_file(selected_cells: np.ndarray):
    """
    write grid file adapted to selected cells
    :param selected_cells: selected cells
    """
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
    

def configure_risico(grid: Grid, cells_file: str):
    """
    Configure risico cell file removing cells outside the grid
    :param grid: grid
    :param cells_file: cells file
    :return: selected cells
    """
    cells = np.loadtxt(cells_file)
    lat, lon = cells[:,1], cells[:,0]
    min_lon, max_lon = grid.lons.min(), grid.lons.max()
    min_lat, max_lat = grid.lats.min(), grid.lats.max()
    selected_cells = cells[(lon>=min_lon)&(lon<=max_lon)&(lat>=min_lat)&(lat<=max_lat), :]

    np.savetxt('risico/STATIC/region.txt', selected_cells, fmt=['%.8f','%.8f','%.8f','%.8f','%d'])
    return selected_cells


def roll_grid(interp_lons: np.ndarray, interp_lats: np.ndarray, interp_indexes: np.ndarray):
    """
    Roll grid to convert it to -180 to 180 longitude range
    :param interp_lons: interpolated longitudes
    :param interp_lats: interpolated latitudes
    :param interp_indexes: interpolated indexes
    :return: rolled grid
    """
     # find the first element where the longitude is over 180


    roll_point = np.where(interp_lons[0, :] >= 180)
    if len(roll_point[0]) == 0:
        return interp_lons, interp_lats, interp_indexes

    roll_point = roll_point[0][0]

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

def generate_grid(ds: xr.Dataset):
    """
    Generate grid from dataset
    :param ds: dataset
    :return: grid
    """
    lons = ds.lon.values[:]
    lats = np.flip(ds.lat.values[:])
    lons, lats = np.meshgrid(lons, lats)
    grid = Grid(lats, lons, regular=True)
    
    return grid


def get_nc_files(input_dir: str):
    """
    Get all netcdf files in input directory
    """

    nc_files = list(map(
        lambda f: path.join(input_dir, f),
        sorted(list(filter(
            lambda s: (s.endswith('.nc')) and not s.startswith('._'), listdir(input_dir)
    )))))

    return nc_files

def process_dataset(ds: xr.Dataset, output_dir: str):
    """
    Process dataset and write output files
    :param ds: Dataset to process
    :param output_dir: Output directory
    :return: list of output files and grid
    """
    grid: Grid = None
    
    file_list = []

    for v in ds.variables:
        if v not in RISICO_VARS:
            continue

        if grid is None:
            grid = generate_grid(ds)

        
        print(f'Reading {v}')
        
        var = RISICO_VARS[v]
        var_name = var['name']
        transform = None
        
        if 'transform' in var:
            transform = RISICO_VARS[v]['transform']

        for idx, time in progressbar(enumerate(ds.time.values)):
            date = datetime.utcfromtimestamp(time.tolist()/1e9)
            out_date_str = date.strftime('%Y%m%d%H%M')
            
            out_file = f'{output_dir}/{out_date_str}_{SUFFIX}_{var_name}.zbin'
            values = ds[v].sel(time=time).values[0,:,:]
            values = np.flipud(values)

            if transform is not None:
                values = transform(values)
            
            write_gzip_binary(out_file, values, grid)
            file_list.append(path.abspath(out_file))

            if var_name == 'W':
                wd_values = np.zeros_like(values)
                out_file = f'{output_dir}/{out_date_str}_{SUFFIX}_D.zbin'
                write_gzip_binary(out_file, wd_values, grid)
                file_list.append(path.abspath(out_file))

    return file_list, grid

def main(input_dir: str, output_dir: str, file_list_file: str, cells_file: str, veg_file: str):
    copyfile(veg_file, 'risico/STATIC/pveg.csv')
    
    makedirs(output_dir, exist_ok=True)
    input_file_dir = path.dirname(file_list_file)
    
    if input_file_dir != '':
        makedirs(input_file_dir, exist_ok=True)

    nc_files = get_nc_files(input_dir)  
    print('found nc files', nc_files)

    with open(file_list_file, 'w') as file_list:
        for f in nc_files:
            print('reading file ', f)

            try:
                ds = xr.open_dataset(f)
            except Exception as exc:
                print('skipping file ', f)
                continue

            files, grid = process_dataset(ds, output_dir)
            file_list.write(path.abspath('\n'.join(files)) + '\n')

    selected_cells = configure_risico(grid, cells_file)
    adapt_grid_file(selected_cells)



def parse_args():
    """
    Parse command line arguments
    :return: Namespace with arguments
    """
    parser = argparse.ArgumentParser(description='Generate IFS files')
    
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('file_list_file', help='file list file')

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    # ifs files
    input_dir = args.input_dir
    # risico input files
    output_dir = args.output_dir
    # list of input files
    file_list_file = args.file_list_file

    cells_file = os.environ.get('CELLS_FILE', 'risico/STATIC/world.txt')
    veg_file = os.environ.get('PVEG_FILE', 'risico/STATIC/pveg_world.csv')

    main(input_dir, output_dir, file_list_file, cells_file, veg_file)


