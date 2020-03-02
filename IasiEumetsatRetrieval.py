import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from datetime import timedelta

from typhon.plots import worldmap, profile_p, styles
from typhon.physics import vmr2relative_humidity, mixing_ratio2vmr, relative_humidity2vmr
from typhon.collocations import Collocator, expand

plt.style.use(styles.get('typhon'))


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def plot_wv_map(data_list, plevel):
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    for data in data_list:
        p_ind = find_nearest(data['pressure_levels_humidity'], plevel * 100.)
        wv = get_domain_array(data, 'atmospheric_water_vapor')
        wv_vmr = mixing_ratio2vmr(wv)
        T = get_domain_array(data, 'atmospheric_temperature')
        P = data['pressure_levels_humidity'].values
        RH = vmr2relative_humidity(wv_vmr, P, T) * 100.
        scat = worldmap(
            data["lat"], data["lon"], RH[:, :, p_ind],
            draw_coastlines=False, cmap='density', ax=ax, draw_grid=True,
        )
        ax.coastlines(resolution='50m')
        s = f"{np.round(data['pressure_levels_humidity'][p_ind].values / 100., 0)}"
        plt.title(f"p = {s} hPa", y=1.1)
    plt.colorbar(scat, fraction=0.015, label='Relative Humidity [%]')
    ax.set_extent([-75, -40, 0, 25], crs=ccrs.PlateCarree())


def plot_temp_map(data_list, plevel):
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    for data in data_list:
        p_ind = find_nearest(data['pressure_levels_humidity'], plevel * 100.)
        T = get_domain_array(data, 'atmospheric_temperature')
        scat = worldmap(
            data["lat"], data["lon"], T[:, :, p_ind],
            draw_coastlines=False, cmap='temperature', ax=ax, draw_grid=True,
        )
        ax.coastlines(resolution='50m')
        s = f"{np.round(data['pressure_levels_humidity'][p_ind].values / 100., 0)}"
        plt.title(f"p = {s} hPa", y=1.1)
    plt.colorbar(scat, fraction=0.015, label='Temperature [K]')
    ax.set_extent([-75, -40, 0, 25], crs=ccrs.PlateCarree())


def get_domain_array(data, var_name):
    return data[var_name].where(
        (data.lat > 6.) & (data.lat < 24.) &
        (data.lon > -90.) & (data.lon < -10.)).values


def collocate_iasi_with_dropsondes(iasi_data, dropsonde_data, collocation_radius, max_interval,
                                   profiles_available=False):
    iasi_collocation_ds = prepare_iasi_ds_for_collocation(iasi_data)
    dropsonde_collocation_ds = prepare_dropsonde_ds_for_collocation(dropsonde_data)
    collocater = Collocator(threads=4, name='eurec4a_loc_iasi')
    print("Collocating...")
    collocations = collocater.collocate(
        primary=('dropsondes', dropsonde_collocation_ds),
        secondary=('iasi', iasi_collocation_ds),
        max_distance=collocation_radius,
        max_interval=max_interval
    )
    collocations = expand(collocations)
    if profiles_available:
        p_ind = find_nearest(collocations['iasi/pressure_levels_humidity'].values, 500.)
        collocations = collocations.isel({
            'collocation': ~np.isnan(collocations['iasi/atmospheric_water_vapor'][p_ind, :]).values,
        }
        )
    return collocations


def get_iasi_start_end_dates(iasi_data):
    sy = iasi_data.start_sensing_data_time[:4]
    sm = iasi_data.start_sensing_data_time[4:6]
    sd = iasi_data.start_sensing_data_time[6:8]
    sH = iasi_data.start_sensing_data_time[8:10]
    sM = iasi_data.start_sensing_data_time[10:12]
    sS = iasi_data.start_sensing_data_time[12:14]
    start_date = f"{sy}-{sm}-{sd} {sH}:{sM}:{sS}"
    ey = iasi_data.end_sensing_data_time[:4]
    em = iasi_data.end_sensing_data_time[4:6]
    ed = iasi_data.end_sensing_data_time[6:8]
    eH = iasi_data.end_sensing_data_time[8:10]
    eM = iasi_data.end_sensing_data_time[10:12]
    eS = iasi_data.end_sensing_data_time[12:14]
    end_date = f"{ey}-{em}-{ed} {eH}:{eM}:{eS}"
    return start_date, end_date


def prepare_dropsonde_ds_for_collocation(dropsonde_ds):
    dropsonde_ds = dropsonde_ds.rename({'time': 'continuous_time'})
    dropsonde_ds = dropsonde_ds.assign_coords(
        {'time': ('sonde_number', dropsonde_ds.launch_time.values)})
    dropsonde_ds = xr.Dataset(
        {
            'lat': (('sonde_number'), dropsonde_ds.lat[:, 0]),
            'lon': (('sonde_number'), dropsonde_ds.lon[:, 0]),
            'time': (('sonde_number'), dropsonde_ds.time),
            'sonde_number': (('sonde_number'), dropsonde_ds.sonde_number),
            'height': (('height'), dropsonde_ds.height),
            'p': (('sonde_number', 'height'), dropsonde_ds.p * 100.),
            'rh': (('sonde_number', 'height'), dropsonde_ds.rh),
            'ta': (('sonde_number', 'height'), dropsonde_ds.ta + 273.15),
        })
    return dropsonde_ds


def prepare_iasi_ds_for_collocation(iasi_data):
    start_date, end_date = get_iasi_start_end_dates(iasi_data)
    iasi_collocation_ds = xr.Dataset(
        {
            'lat': iasi_data.lat,
            'lon': iasi_data.lon,
            'time': (('along_track'),
                     np.arange(np.datetime64(start_date),
                               np.datetime64(end_date),
                               np.timedelta64(8, 's'),
                               dtype='datetime64')),
            'atmospheric_water_vapor': iasi_data.atmospheric_water_vapor,
            'atmospheric_temperature': iasi_data.atmospheric_temperature,

        })
    print("Constraining iasi_data to general EUREC4A region before collocating...")
    # TODO: CURRENTLY RAISES ISSUE
    # iasi_collocation_ds = iasi_collocation_ds.where(
    #     (iasi_collocation_ds.lat > 6.) & (iasi_collocation_ds.lat < 18.) &
    #     (iasi_collocation_ds.lon > -90.) & (iasi_collocation_ds.lon < -45.)
    # )
    return iasi_collocation_ds


def collocate_iasi_with_location(iasi_data, collocation_latitudes,
                                 collocation_longitudes, collocation_radius,
                                 profiles_available=False):
    iasi_collocation_ds = prepare_iasi_ds_for_collocation(iasi_data)
    start_date, end_date = get_iasi_start_end_dates(iasi_data)
    eurec4a_loc_ds = xr.Dataset(
        {
            'lat': collocation_latitudes,
            'lon': collocation_longitudes,
            'time': np.array([f'{start_date}'], dtype="datetime64[D]")
        }
    )
    collocater = Collocator(threads=4, name='eurec4a_loc_iasi')
    print("Collocating...")
    collocations = collocater.collocate(
        primary=('eurec4a_locations', eurec4a_loc_ds),
        secondary=('iasi', iasi_collocation_ds),
        max_distance=collocation_radius,
        max_interval=timedelta(days=1)
    )
    collocations = expand(collocations)
    if profiles_available:
        p_ind = find_nearest(collocations['iasi/pressure_levels_humidity'].values, 500.)
        collocations = collocations.isel({
            'collocation': ~np.isnan(collocations['iasi/atmospheric_water_vapor'][p_ind, :]).values,
        }
        )
    return collocations


def plot_collocated_iasi_profiles(collocations, collocation_indices=None, fig=None,
                                  axs=None, alpha=1.0, color='blue'):
    if collocation_indices is None:
        collocation_indices = np.arange(len(collocations['collocation']))
    if not fig and not axs:
        fig, axs = plt.subplots(ncols=3, sharey=True)
    for i in collocation_indices:
        P = collocations['iasi/pressure_levels_humidity'].values
        T = collocations['iasi/atmospheric_temperature'][:, i].values
        wv_vmr = mixing_ratio2vmr(collocations['iasi/atmospheric_water_vapor'][:, i].values)
        RH = vmr2relative_humidity(wv_vmr, P, T) * 100.
        profile_p(P, T, ax=axs[0], alpha=alpha, color=color)
        profile_p(P, RH, ax=axs[1], alpha=alpha, color=color)
        profile_p(P, wv_vmr, ax=axs[2], alpha=alpha, color=color,
                  label=f"IASI: {np.array(collocations['iasi/time'][i].values, dtype='datetime64[s]')}")
        axs[0].set_xlabel('Temperature [K]')
        axs[1].set_xlabel('Relative Humidity [%]')
        axs[2].set_xlabel('H${_2}$O VMR [-]')
        axs[1].set_xlim([0, 100])
    axs[2].legend(bbox_to_anchor=(1.1, 1))
    return fig, axs


def plot_radiosonde_profiles(radiosonde_data, fig=None, axs=None, show_legend=True):
    if not fig and not axs:
        fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(20, 10))
    P = radiosonde_data['pressure'][0].values * 100.
    T = radiosonde_data['temperature'][0].values + 273.15
    wv_vmr = mixing_ratio2vmr(radiosonde_data['mixingRatio'][0].values / 1000.)
    RH = radiosonde_data['humidity'][0].values
    profile_p(P, T, ax=axs[0], color='black')
    profile_p(P, RH, ax=axs[1], color='black')
    profile_p(P, wv_vmr, ax=axs[2], color='black',
              label=f"RS: {np.array(radiosonde_data['flight_time'][0, 0].values, dtype='datetime64[s]')}")
    axs[0].set_xlabel('Temperature [K]')
    axs[1].set_xlabel('Relative Humidity [%]')
    axs[2].set_xlabel('H${_2}$O VMR [-]')
    axs[1].set_xlim([0, 100])
    if show_legend:
        axs[2].legend(bbox_to_anchor=(1.1, 1))
    return fig, axs


def plot_collocated_dropsonde_profiles(collocations, collocation_indices=None,
                                       fig=None, axs=None, alpha=1.0, color='black',
                                       show_legend=True):
    if collocation_indices is None:
        collocation_indices = np.arange(len(collocations['collocation']))
    if not fig and not axs:
        fig, axs = plt.subplots(ncols=3, sharey=True)
    for i in collocation_indices:
        P = collocations['dropsondes/p'][i, :].values
        T = collocations['dropsondes/ta'][i, :].values
        RH = collocations['dropsondes/rh'][i, :].values
        wv_vmr = relative_humidity2vmr(RH / 100., P, T)
        profile_p(P, T, ax=axs[0], alpha=alpha, color=color)
        profile_p(P, RH, ax=axs[1], alpha=alpha, color=color)
        profile_p(P, wv_vmr, ax=axs[2], alpha=alpha, color=color,
                  label=f"DS: {np.array(collocations['dropsondes/time'][i].values, dtype='datetime64[s]')}")
        axs[0].set_xlabel('Temperature [K]')
        axs[1].set_xlabel('Relative Humidity [%]')
        axs[2].set_xlabel('H${_2}$O VMR [-]')
        axs[1].set_xlim([0, 100])
    if show_legend:
        axs[2].legend(bbox_to_anchor=(1.1, 1))
    return fig, axs
