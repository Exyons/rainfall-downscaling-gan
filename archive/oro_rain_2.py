import numpy as np
import rasterio
from pathlib import Path
import orographicRainLinearTheory as lt
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rasterio.warp import reproject, Resampling

# Configuration - Define all options here
DATA_DIR = '../../ERA5_daily_Uttarakhand'  # Root directory containing your .tif files
OUTPUT_DIR = './spectral_output'
DEM_FILE = 'DEM.tif'

# Model parameters
TAUC = 900  # conversion time scale (s)
TAUF = 900  # fallout time scale (s)

# Plotting
PLOT_RESULTS = True  # Set to False to disable plotting
PLOT_EVERY_N = 10  # Plot every Nth file
PLOT_DIR = './plots'  # Directory to save plots

# Number of parallel jobs (-1 uses all available cores)
N_JOBS = -1

MAX_REASONABLE = 300  # mm/day - typical maximum for orographic enhancement

def load_tif_file(filepath):
    """Load a single .tif file and return data with geospatial info"""
    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
        # Get lat/lon coordinates from pixel centers
        height, width = data.shape
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        lons, lats = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        lons = np.array(lons).reshape(height, width)
        lats = np.array(lats).reshape(height, width)
        
    return data, lons, lats, transform, crs, bounds

def load_era5_daily_data(filepath):
    """Load ERA5 .tif file with multiple bands
    
    Band structure:
    - Band 1: Air temperature
    - Band 5: Precipitation
    - Band 6: Pressure
    - Band 8: U-wind component
    - Band 9: V-wind component
    """
    with rasterio.open(filepath) as src:
        n_bands = src.count
        shape = src.shape
        height, width = shape
        
        # Read specific bands
        temperature = src.read(1)  # Band 1
        precipitation = src.read(5)*1000  # Band 5
        pressure = src.read(6)  # Band 6
        u_wind = src.read(8)  # Band 8
        v_wind = src.read(9)  # Band 9
        
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        dtype = src.dtypes[0]
        nodata = src.nodata

        # Get lat/lon coordinates from pixel centers
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        lons, lats = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        lons = np.array(lons).reshape(height, width)
        lats = np.array(lats).reshape(height, width)
        
    return {
        'temperature': temperature,
        'precipitation': precipitation,
        'pressure': pressure,
        'u_wind': u_wind,
        'v_wind': v_wind,
        'lons': lons,
        'lats': lats,
        'transform': transform,
        'crs': crs,
        'bounds': bounds,
        'dtype': dtype,
        'nodata': nodata,
        'shape': shape
    }

def calculate_grid_spacing(lats, lons):
    """Calculate dx and dy from lat/lon grid"""
    ny, nx = lats.shape
    center_lat = lats[ny//2, nx//2]
    center_lon = lons[ny//2, nx//2]
    
    # Calculate dx (spacing in x-direction)
    if nx > 1:
        dx = lt.haversine(center_lat, center_lon, center_lat, lons[ny//2, min(nx//2 + 1, nx-1)])
    else:
        dx = 25000  # Default 25km
    
    # Calculate dy (spacing in y-direction)
    if ny > 1:
        dy = lt.haversine(center_lat, center_lon, lats[min(ny//2 + 1, ny-1), nx//2], center_lon)
    else:
        dy = 25000  # Default 25km
    
    return dx, dy

def calculate_spatial_weights(lats, lons):
    """Calculate inverse distance weights for spatial averaging"""
    ny, nx = lats.shape
    center_lat = lats[ny//2, nx//2]
    center_lon = lons[ny//2, nx//2]
    
    weights = np.empty((ny, nx))
    for j in range(ny):
        for i in range(nx):
            dist = lt.haversine(center_lat, center_lon, lats[j, i], lons[j, i])
            weights[j, i] = 1 / (dist**2 + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Normalize weights
    weights_norm = weights / np.sum(weights)
    return weights_norm

def process_single_timestep(h_scaled, dx_scaled, dy_scaled, u_avg, v_avg, 
                           ts_mean, ps_mean, gamma_mean, tauc, tauf):
    """Calculate orographic rainfall for a single timestep"""
    # Check for NaN or invalid inputs
    if np.isnan(u_avg) or np.isnan(v_avg) or np.isnan(ts_mean) or np.isnan(ps_mean) or np.isnan(gamma_mean):
        tqdm.write(f"  WARNING: NaN in input parameters - u={u_avg:.2f}, v={v_avg:.2f}, T={ts_mean:.2f}, P={ps_mean:.2f}, gamma={gamma_mean:.6f}")
        return None
    
    # Check for zero or very small gamma
    if gamma_mean <= 0 or gamma_mean > 0.1:  # gamma should be positive and reasonable (~0.001-0.01 K/m)
        tqdm.write(f"  WARNING: Invalid gamma value: {gamma_mean:.6f} K/m")
        return None
    
    try:
        p = lt.compute_orographic_rain(
            h_scaled, u_avg, v_avg,
            ts_mean, ps_mean, gamma_mean,
            tauc, tauf, dx_scaled, dy_scaled
        )
        
        # The function already converts to mm/hr in the last line (* 3600)
        # Convert from mm/hr to mm/day
        p = p * 24
        
        # Sanity check - orographic rainfall should typically be < 100 mm/day
        # If values are too high, there might be an issue with the calculation
        max_val = np.nanmax(np.abs(p))
        if max_val > 500:
            tqdm.write(f"  WARNING: Very high orographic rainfall detected: {max_val:.2f} mm/day")
            tqdm.write(f"  Input params - U={u_avg:.2f}, V={v_avg:.2f}, T={ts_mean:.2f}, P={ps_mean:.2f}, gamma={gamma_mean:.6f}")
        
        # Check for NaN in output
        if np.all(np.isnan(p)):
            tqdm.write(f"  WARNING: All NaN in orographic rainfall output")
            return None
            
        return np.squeeze(p)
    except Exception as e:
        tqdm.write(f"  ERROR in compute_orographic_rain: {str(e)}")
        import traceback
        tqdm.write(traceback.format_exc())
        return None

def plot_results(dem_data, era5_data, orographic_rain, output_file, file_idx):
    """Create diagnostic plots"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot DEM
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dem_data, cmap='terrain')
    ax1.set_title('DEM (m)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot Temperature
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(era5_data['temperature'], cmap='RdYlBu_r')
    ax2.set_title(f'Temperature (K)\nMin={np.nanmin(era5_data["temperature"]):.1f}, Max={np.nanmax(era5_data["temperature"]):.1f}')
    plt.colorbar(im2, ax=ax2)
    
    # Plot Pressure
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(era5_data['pressure'], cmap='viridis')
    ax3.set_title(f'Pressure (Pa)\nMin={np.nanmin(era5_data["pressure"]):.0f}, Max={np.nanmax(era5_data["pressure"]):.0f}')
    plt.colorbar(im3, ax=ax3)
    
    # Plot U-wind
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(era5_data['u_wind'], cmap='RdBu_r')
    ax4.set_title(f'U-wind (m/s)\nMin={np.nanmin(era5_data["u_wind"]):.2f}, Max={np.nanmax(era5_data["u_wind"]):.2f}')
    plt.colorbar(im4, ax=ax4)
    
    # Plot V-wind
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(era5_data['v_wind'], cmap='RdBu_r')
    ax5.set_title(f'V-wind (m/s)\nMin={np.nanmin(era5_data["v_wind"]):.2f}, Max={np.nanmax(era5_data["v_wind"]):.2f}')
    plt.colorbar(im5, ax=ax5)
    
    # Plot ERA5 Precipitation
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(era5_data['precipitation'], cmap='turbo')
    ax6.set_title(f'ERA5 Precip (mm/day)\nMin={np.nanmin(era5_data["precipitation"]):.2f}, Max={np.nanmax(era5_data["precipitation"]):.2f}')
    plt.colorbar(im6, ax=ax6)
    
    # Plot Orographic Rainfall
    ax7 = fig.add_subplot(gs[2, :2])
    if orographic_rain is not None and not np.all(np.isnan(orographic_rain)):
        im7 = ax7.imshow(orographic_rain, cmap='turbo')
        ax7.set_title(f'Orographic Rainfall (mm/day)\nMean={np.nanmean(orographic_rain):.2f}, Max={np.nanmax(orographic_rain):.2f}')
        plt.colorbar(im7, ax=ax7)
    else:
        ax7.text(0.5, 0.5, 'No valid orographic rainfall\ncomputed', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=14)
        ax7.set_title('Orographic Rainfall - ERROR')
    
    # Plot histogram
    ax8 = fig.add_subplot(gs[2, 2])
    # if orographic_rain is not None and not np.all(np.isnan(orographic_rain)):
    #     valid_data = orographic_rain[~np.isnan(orographic_rain)]
    #     if len(valid_data) > 0:
    #         ax8.hist(valid_data.ravel(), bins=50, edgecolor='black', alpha=0.7)
    #         ax8.set_xlabel('Orographic Rainfall (mm/day)')
    #         ax8.set_ylabel('Frequency')
    #         ax8.set_title('Distribution')
    #         ax8.grid(True, alpha=0.3)
    combined = era5_data['precipitation'] + orographic_rain
    combined[combined < 0] = 9
    im8 = ax8.imshow(combined, cmap='turbo')
    ax8.set_title(f'ERA5 Precip (mm/day)\nMin={np.nanmin(combined):.2f}, Max={np.nanmax(combined):.2f}')
    plt.colorbar(im8, ax=ax8)
    
    plt.suptitle(f'File: {output_file.stem}', fontsize=14, fontweight='bold')
    
    # Save plot
    plot_path = Path(PLOT_DIR)
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_file = plot_path / f'diagnostic_{file_idx:04d}_{output_file.stem}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    tqdm.write(f"  Plot saved: {plot_file.name}")

def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DEM
    tqdm.write("Loading DEM...")
    dem_path = Path(DEM_FILE)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    
    dem_data, dem_lons, dem_lats, dem_transform, dem_crs, dem_bounds = load_tif_file(dem_path)
    
    # Calculate DEM grid spacing
    dem_dx, dem_dy = calculate_grid_spacing(dem_lats, dem_lons)
    tqdm.write(f"DEM shape: {dem_data.shape}")
    tqdm.write(f"DEM grid spacing: dx={dem_dx:.2f}m, dy={dem_dy:.2f}m")
    tqdm.write(f"DEM elevation range: {np.nanmin(dem_data):.2f}m to {np.nanmax(dem_data):.2f}m")
    
    # Find all ERA5 .tif files in data directory (excluding DEM)
    data_path = Path(DATA_DIR)
    era5_files = sorted([f for f in data_path.glob('*.tif') if f.name != DEM_FILE])
    
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 .tif files found in {DATA_DIR}")
    
    tqdm.write(f"\nFound {len(era5_files)} ERA5 files to process\n")
    
    # Process each ERA5 file with progress bar
    file_count = 0
    for era5_file in tqdm(era5_files, desc="Processing files", unit="file"):
        try:
            file_count += 1
            
            # Load ERA5 data
            era5_data = load_era5_daily_data(era5_file)
            
            # Extract variables
            temperature = era5_data['temperature']
            pressure = era5_data['pressure']
            u_wind = era5_data['u_wind']
            v_wind = era5_data['v_wind']
            rain_lats = era5_data['lats']
            rain_lons = era5_data['lons']
            era5_shape = temperature.shape
            
            tqdm.write(f"\n{era5_file.name}:")
            tqdm.write(f"  ERA5 shape: {era5_shape}")
            tqdm.write(f"  Temp: min={np.nanmin(temperature):.2f}, max={np.nanmax(temperature):.2f}K, NaNs={np.sum(np.isnan(temperature))}")
            tqdm.write(f"  Pressure: min={np.nanmin(pressure):.2f}, max={np.nanmax(pressure):.2f}Pa, NaNs={np.sum(np.isnan(pressure))}")
            tqdm.write(f"  U-wind: min={np.nanmin(u_wind):.2f}, max={np.nanmax(u_wind):.2f}m/s, NaNs={np.sum(np.isnan(u_wind))}")
            tqdm.write(f"  V-wind: min={np.nanmin(v_wind):.2f}, max={np.nanmax(v_wind):.2f}m/s, NaNs={np.sum(np.isnan(v_wind))}")
            
            # Resize DEM to match ERA5 resolution (25km)
            tqdm.write(f"  Resizing DEM from {dem_data.shape} to {era5_shape}...")

            # from scipy.ndimage import zoom
            # zoom_factors = (era5_shape[0] / dem_data.shape[0], era5_shape[1] / dem_data.shape[1])
            # h_era5 = zoom(dem_data, zoom_factors, order=1)  # Bilinear interpolation

            h_era5 = np.empty(era5_data['shape'], dtype=era5_data['dtype'])
            reproject(
                source=dem_data,
                destination=h_era5,
                src_transform=dem_transform,
                src_crs=dem_crs,
                dst_transform=era5_data['transform'],
                dst_crs=era5_data['crs'],
                dst_nodata=era5_data['nodata'],
                resampling=Resampling.lanczos
            )
            
            # Calculate grid spacing for ERA5 resolution
            era5_dx, era5_dy = calculate_grid_spacing(rain_lats, rain_lons)
            tqdm.write(f"  ERA5 grid spacing: dx={era5_dx:.2f}m, dy={era5_dy:.2f}m")
            
            # Ensure all arrays have the same shape
            assert temperature.shape == pressure.shape == u_wind.shape == v_wind.shape, \
                f"Data shape mismatch in {era5_file.name}"
            
            # Calculate spatial weights for averaging
            weights = calculate_spatial_weights(rain_lats, rain_lons)
            
            # Calculate spatially averaged wind components (ignoring NaN)
            u_avg = np.ma.average(np.ma.masked_invalid(u_wind), weights=weights)
            v_avg = np.ma.average(np.ma.masked_invalid(v_wind), weights=weights)
            
            tqdm.write(f"  Averaged U-wind: {u_avg:.2f} m/s, V-wind: {v_avg:.2f} m/s")
            
            # Calculate moist parameters
            gamma_m = lt.find_saturated_moist_adiabatic_lapse_rate(temperature, pressure)
            gamma = 0.99 * gamma_m  # Environmental lapse rate slightly less than moist adiabatic
            
            # Calculate spatial means (ignoring NaN)
            ts_mean = np.nanmean(temperature)
            ps_mean = np.nanmean(pressure)
            gamma_mean = np.nanmean(gamma)
            
            tqdm.write(f"  Mean T={ts_mean:.2f}K, P={ps_mean:.0f}Pa, gamma={gamma_mean:.6f}K/m")
            
            # Calculate orographic rainfall at ERA5 resolution
            orographic_rain = process_single_timestep(
                h_era5, era5_dx, era5_dy, u_avg, v_avg,
                ts_mean, ps_mean, gamma_mean, TAUC, TAUF
            )
            
            if orographic_rain is None:
                tqdm.write(f"✗ {era5_file.name}: Failed to compute orographic rainfall")
                continue
            
            # Ensure non-negative rainfall
            orographic_rain[orographic_rain < 0] = 0
            
            # Cap extreme values (likely numerical artifacts)
            if np.nanmax(orographic_rain) > MAX_REASONABLE:
                tqdm.write(f"  WARNING: Capping values above {MAX_REASONABLE} mm/day")
                orographic_rain = np.clip(orographic_rain, 0, MAX_REASONABLE)
            
            # Save results
            output_filename = f"oro_rain_{era5_file.stem}.tif"
            output_file = output_dir / output_filename
            
            # Save as single-band GeoTIFF at ERA5 resolution
            with rasterio.open(
                output_file, 'w',
                driver='GTiff',
                height=orographic_rain.shape[0],
                width=orographic_rain.shape[1],
                count=1,
                dtype=orographic_rain.dtype,
                crs=era5_data['crs'],
                transform=era5_data['transform'],
                compress='lzw'
            ) as dst:
                dst.write(orographic_rain, 1)
                dst.set_band_description(1, f'Orographic Rainfall - {era5_file.stem}')
            
            # Log statistics for this file
            valid_rain = orographic_rain[~np.isnan(orographic_rain)]
            mean_rain = np.mean(valid_rain) if len(valid_rain) > 0 else np.nan
            max_rain = np.max(valid_rain) if len(valid_rain) > 0 else np.nan
            
            tqdm.write(f"✓ {era5_file.name}: Mean={mean_rain:.2f}, Max={max_rain:.2f} mm/day")
            
            # Plot results if enabled
            if PLOT_RESULTS and (file_count % PLOT_EVERY_N == 0 or file_count <= 3):
                plot_results(h_era5, era5_data, orographic_rain, output_file, file_count)
        except Exception as e:
            tqdm.write(f"✗ Error processing {era5_file.name}: {str(e)}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue

if __name__ == "__main__":
    print("="*60)
    print("Orographic Rainfall Calculator")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"DEM file: {DEM_FILE}")
    print(f"Model parameters: tauc={TAUC}s, tauf={TAUF}s, scale={25}")
    print("="*60)
    print()
    
    main()
    
    print()
    print("="*60)
    print("All processing complete!")
    print("="*60)