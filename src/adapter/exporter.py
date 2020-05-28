from utils.zbin import read_gzip_binary
import sys
import xarray as xr
import os
from datetime import datetime
import numpy as np
from progressbar import progressbar as bar
import pandas as pd
from datetime import timedelta



from functools import partial
def perc_mean(dataset, axis=0, perc=50, inverse=False):
    threshold = np.percentile(dataset, [perc], axis).squeeze()
    if inverse:
        perc_data = dataset.where(dataset < threshold).mean(axis=0)
    else:
        perc_data = dataset.where(dataset >= threshold).mean(axis=0)
    return perc_data

data_min = partial(xr.DataArray.min, dim='time')
data_mean = partial(xr.DataArray.mean, dim='time')
data_max = partial(xr.DataArray.max, dim='time')
perc50_mean = partial(perc_mean, perc=50)
perc75_mean = partial(perc_mean, perc=75)
perc90_mean = partial(perc_mean, perc=90)
perc10_inv_mean = partial(perc_mean, perc=10)
perc25_inv_mean = partial(perc_mean, perc=25)
perc50_inv_mean = partial(perc_mean, perc=50)

ext_names = {
    'W': 'W',
    'V': 'ROS',
    'I': 'I',
    'UMB': 'FFM'
}

variables = [
    dict(var='W', fun=data_max, out_name='W_MAX'),
    dict(var='W', fun=data_mean, out_name='W_MEAN'),
    dict(var='W', fun=perc50_mean, out_name='W_P50'),
#    dict(var='W', fun=perc75_mean, out_name='W_P75'),
#    dict(var='W', fun=perc90_mean, out_name='W_P90'),    

    dict(var='ROS', fun=data_max, out_name='ROS_MAX'),
    dict(var='ROS', fun=data_mean, out_name='ROS_MEAN'),
    dict(var='ROS', fun=perc50_mean, out_name='ROS_P50'),
#    dict(var='ROS', fun=perc75_mean, out_name='ROS_P75'),
#    dict(var='ROS', fun=perc90_mean, out_name='ROS_P90'),

    dict(var='I', fun=data_max, out_name='I_MAX'),
    dict(var='I', fun=data_mean, out_name='I_MEAN'),
    dict(var='I', fun=perc50_mean, out_name='I_P50'),
#    dict(var='I', fun=perc75_mean, out_name='I_P75'),
#    dict(var='I', fun=perc90_mean, out_name='I_P90'),

    dict(var='FFM', fun=data_min, out_name='FFM_MIN'),
    dict(var='FFM', fun=data_mean, out_name='FFM_MEAN'),
#   dict(var='FFM', fun=perc10_inv_mean, out_name='FFM_P10'),
#   dict(var='FFM', fun=perc25_inv_mean, out_name='FFM_P25'),
    dict(var='FFM', fun=perc50_inv_mean, out_name='FFM_P50')
]


if __name__ == '__main__':
    # risico output folder
    out_folder = sys.argv[1]
    # netcdf output file
    filename = sys.argv[2]
    aggr_filename = sys.argv[3]    

    files = os.listdir(out_folder)
    grid = None
    outputs = {}


    for f in bar(files):
        if f.endswith('.zbin'):
            model, model_date, date_ref, variable = f.split('_')
            variable = variable.replace('.zbin','')

            if variable not in ['UMB', 'V', 'I', 'W']: continue

            if variable not in outputs:
                outputs[variable] = []

            if not grid:
                values, grid = read_gzip_binary(filename=out_folder+f, read_grid=True)
            else:
                values, _ = read_gzip_binary(filename=out_folder+f, read_grid=False)


            outputs[variable].append(dict(date=date_ref, values=values))



    ds = xr.Dataset()
    for var in bar(outputs.keys()):
        output = sorted(outputs[var], key=lambda d:d['date'])
        data = np.stack([d['values']for d in output], axis=0)
        ds[ext_names[var]] = xr.DataArray(data=data,
                     dims=('time', 'latitude','longitude'), 
                     coords={
                        'time': [datetime.strptime(d['date'], '%Y%m%d%H%M') for d in output],
                        'longitude': grid.lons[0,:],
                        'latitude': grid.lats[:,0]
                     }
        )

    print('creating dataset %s' % filename)
    ds.to_netcdf(filename)


    _ds = ds.copy()
    _ds['time'] = _ds['time'] - pd.to_timedelta(timedelta(hours=1))
    datasets = []
    for d in bar(variables):
        var = d['var']
        fun = d['fun']
        out_name = d['out_name']
        da =  _ds[var].resample(time='1D').apply(fun)
        da = da.rename(out_name)
        datasets.append(da)

    aggr_ds = xr.merge(datasets)        
    aggr_ds.to_netcdf(aggr_filename)