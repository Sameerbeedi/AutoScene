import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol

def get_terrain_elevation_at_coordinate(coord, elevation_data, transform, bounds):
    """Get terrain elevation at a specific coordinate using bilinear interpolation"""
    lon, lat = coord
    
    # Check if coordinate is within bounds
    if not (bounds.left <= lon <= bounds.right and 
            bounds.bottom <= lat <= bounds.top):
        return None
    
    # Convert geographic coordinate to pixel coordinates
    row, col = rowcol(transform, lon, lat)
    
    # Get array dimensions
    height, width = elevation_data.shape
    
    # Handle edge cases
    if row < 0 or row >= height or col < 0 or col >= width:
        return None
    
    # Bilinear interpolation
    row_floor = int(np.floor(row))
    col_floor = int(np.floor(col))
    row_ceil = min(height-1, row_floor + 1)
    col_ceil = min(width-1, col_floor + 1)
    
    # Get the four surrounding pixels
    tl = elevation_data[row_floor, col_floor]  # top-left
    tr = elevation_data[row_floor, col_ceil]   # top-right
    bl = elevation_data[row_ceil, col_floor]   # bottom-left
    br = elevation_data[row_ceil, col_ceil]    # bottom-right
    
    # Check for NaN values
    if np.isnan([tl, tr, bl, br]).any():
        return None
    
    # Interpolation weights
    row_weight = row - row_floor
    col_weight = col - col_floor
    
    # Bilinear interpolation
    top = tl * (1 - col_weight) + tr * col_weight
    bottom = bl * (1 - col_weight) + br * col_weight
    result = top * (1 - row_weight) + bottom * row_weight
    
    return float(result)

def calculate_polygon_terrain_elevation(polygon, elevation_data, dem_transform, dem_bounds):
    """Calculate mean terrain elevation for a polygon"""
    # Extract all coordinates from polygon
    coords = list(polygon.exterior.coords)
    
    # Get terrain elevation for each coordinate
    elevations = []
    for coord in coords:
        elevation = get_terrain_elevation_at_coordinate(coord, elevation_data, dem_transform, dem_bounds)
        if elevation is not None:
            elevations.append(elevation)
    
    # Return mean elevation or None
    if elevations:
        return np.mean(elevations)
    else:
        return None

def update_terrain_elevation_in_geojson(geojson_path, dem_path):
    """Update existing terrain_elevation column in GeoJSON file"""
    print("=== UPDATING TERRAIN ELEVATION IN GEOJSON ===")
    
    # Load existing GeoJSON
    print(f"Loading GeoJSON: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded {len(gdf)} features")
    
    # Load DEM data
    print(f"Loading DEM: {dem_path}")
    with rasterio.open(dem_path) as src:
        elevation_data = src.read(1)
        dem_transform = src.transform
        dem_bounds = src.bounds
    
    print(f"DEM elevation range: {np.nanmin(elevation_data):.2f}m to {np.nanmax(elevation_data):.2f}m")
    
    # Process each feature to update terrain elevation
    print(f"Updating terrain elevation for {len(gdf)} features...")
    
    updated_count = 0
    for idx, row in gdf.iterrows():
        polygon = row.geometry
        
        # Calculate terrain elevation for this polygon
        terrain_elevation = calculate_polygon_terrain_elevation(
            polygon, elevation_data, dem_transform, dem_bounds
        )
        
        # Update the terrain_elevation column
        gdf.at[idx, 'terrain_elevation'] = terrain_elevation
        
        if terrain_elevation is not None:
            updated_count += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(gdf)} features")
    
    print(f"✅ Updated {updated_count}/{len(gdf)} features with terrain elevation")
    
    # Show statistics
    valid_terrain = gdf['terrain_elevation'].dropna()
    if len(valid_terrain) > 0:
        print(f"Terrain elevation range: {valid_terrain.min():.2f}m to {valid_terrain.max():.2f}m")
        print(f"Mean terrain elevation: {valid_terrain.mean():.2f}m")
    
    # Save updated GeoJSON (overwrite existing)
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"✅ Updated GeoJSON saved to: {geojson_path}")
    
    return gdf

# Main execution
if __name__ == "__main__":
    # File paths
    geojson_file = "output/vector_map.geojson"  # Your existing GeoJSON (will be updated)
    dem_file = "input/cdne43u.tif"              # Your DEM file
    
    try:
        # Update terrain elevation in existing GeoJSON
        gdf_updated = update_terrain_elevation_in_geojson(
            geojson_path=geojson_file,
            dem_path=dem_file
        )
        
        if gdf_updated is not None:
            print(f"\n🎉 Success! Terrain elevation updated in {geojson_file}")
            
        else:
            print("❌ Failed to update terrain elevation data")
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure both the GeoJSON and DEM files exist")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()