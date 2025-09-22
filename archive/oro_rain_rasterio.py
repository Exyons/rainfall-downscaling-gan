import numpy as np
import rasterio
from pathlib import Path
from joblib import Parallel, delayed
import orographicRainLinearTheory as lt
from skimage.transform import rescale
from tqdm import tqdm

# Configuration - Define all options here
DATA_DIR = '../../ERA5_daily_Uttarakhand'  # Root directory containing your .tif files
OUTPUT_DIR = './spectral_output'
DEM_FILE = 'DEM.tif'  # DEM 30km resolution file name

# Model parameters
TAUC = 900  # conversion time scale (s)
TAUF = 900  # fallout time scale (s)
SCALE_FACTOR = 1.2  # DEM upscaling factor for spectral calculation

# Number of parallel jobs (-1 uses all available cores)
N_JOBS = -1

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

def load_era5_yearly_data(filepath):
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
        height, width = src.shape
        
        # Read specific bands
        temperature = src.read(1)  # Band 1
        precipitation = src.read(5)*1000  # Band 5
        pressure = src.read(6)  # Band 6
        u_wind = src.read(8)  # Band 8
        v_wind = src.read(9)  # Band 9
        
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
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
        'bounds': bounds
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
    p = lt.compute_orographic_rain(
        h_scaled, u_avg, v_avg, 
        ts_mean, ps_mean, gamma_mean,
        tauc, tauf, dx_scaled, dy_scaled
    )
    # Convert from mm/s to mm/day
    p = p * 3600 * 24
    return np.squeeze(p)

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
    tqdm.write(f"DEM grid spacing: dx={dem_dx:.2f}m, dy={dem_dy:.2f}m")
    
    # Scale DEM for spectral calculation
    h_scaled = rescale(dem_data, 1/SCALE_FACTOR, anti_aliasing=False)
    dx_scaled = dem_dx * SCALE_FACTOR
    dy_scaled = dem_dy * SCALE_FACTOR
    tqdm.write(f"Scaled DEM shape: {h_scaled.shape}, dx={dx_scaled:.2f}m, dy={dy_scaled:.2f}m\n")
    
    # Find all ERA5 .tif files in data directory (excluding DEM)
    data_path = Path(DATA_DIR)
    era5_files = sorted([f for f in data_path.glob('*.tif') if f.name != DEM_FILE])
    
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 .tif files found in {DATA_DIR}")
    
    tqdm.write(f"Found {len(era5_files)} ERA5 files to process\n")
    
    # Process each ERA5 file with progress bar
    for era5_file in tqdm(era5_files, desc="Processing files", unit="file"):
        try:
            # Load ERA5 data
            era5_data = load_era5_yearly_data(era5_file)
            
            # Extract variables
            temperature = era5_data['temperature']
            pressure = era5_data['pressure']
            u_wind = era5_data['u_wind']
            v_wind = era5_data['v_wind']
            rain_lats = era5_data['lats']
            rain_lons = era5_data['lons']
            
            # Ensure all arrays have the same shape
            assert temperature.shape == pressure.shape == u_wind.shape == v_wind.shape, \
                f"Data shape mismatch in {era5_file.name}"
            
            # Calculate spatial weights for averaging
            weights = calculate_spatial_weights(rain_lats, rain_lons)
            
            # Ensure weights match data shape
            assert weights.shape == u_wind.shape, \
                f"Weights shape {weights.shape} doesn't match data shape {u_wind.shape}"
            
            # Calculate spatially averaged wind components
            u_avg = np.average(u_wind, weights=weights)
            v_avg = np.average(v_wind, weights=weights)
            
            # Calculate moist parameters
            gamma_m = lt.find_saturated_moist_adiabatic_lapse_rate(temperature, pressure)
            gamma = 0.99 * gamma_m  # Environmental lapse rate slightly less than moist adiabatic
            
            # Calculate spatial means
            ts_mean = np.nanmean(temperature)
            ps_mean = np.nanmean(pressure)
            gamma_mean = np.nanmean(gamma)
            
            # Calculate orographic rainfall
            orographic_rain = process_single_timestep(
                h_scaled, dx_scaled, dy_scaled, u_avg, v_avg,
                ts_mean, ps_mean, gamma_mean, TAUC, TAUF
            )
            
            # Ensure non-negative rainfall
            orographic_rain[orographic_rain < 0] = 0
            
            # Save results
            output_filename = f"oro_rain_{era5_file.stem}.tif"
            output_file = output_dir / output_filename
            
            # Save as single-band GeoTIFF
            with rasterio.open(
                output_file, 'w',
                driver='GTiff',
                height=orographic_rain.shape[0],
                width=orographic_rain.shape[1],
                count=1,
                dtype=orographic_rain.dtype,
                crs=dem_crs,
                transform=rasterio.transform.from_bounds(
                    dem_bounds.left, dem_bounds.bottom,
                    dem_bounds.right, dem_bounds.top,
                    orographic_rain.shape[1], orographic_rain.shape[0]
                ),
                compress='lzw'
            ) as dst:
                dst.write(orographic_rain, 1)
                dst.set_band_description(1, f'Orographic Rainfall - {era5_file.stem}')
            
            # Log statistics for this file
            tqdm.write(f"✓ {era5_file.name}: Mean={np.mean(orographic_rain):.2f}, Max={np.max(orographic_rain):.2f} mm/day")
            
        except Exception as e:
            tqdm.write(f"✗ Error processing {era5_file.name}: {str(e)}")
            continue

if __name__ == "__main__":
    print("="*60)
    print("Orographic Rainfall Calculator")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"DEM file: {DEM_FILE}")
    print(f"Model parameters: tauc={TAUC}s, tauf={TAUF}s, scale={SCALE_FACTOR}")
    print("="*60)
    print()
    
    main()
    
    print()
    print("="*60)
    print("All processing complete!")
    print("="*60)