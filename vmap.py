import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio import features
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
import os

def load_class_mask_from_npy(npy_path):
    """Load class mask from numpy array file."""
    mask = np.load(npy_path)
    print(f"Loaded mask with shape: {mask.shape}")
    return mask

def create_vector_map(npy_path, object_height_npy_path, output_path,class_names=None, geographic_bounds=None, crs='EPSG:4326'):
    """
    Creates a georeferenced vector map, calculating object height
    ONLY for building polygons and default heights for other classes.
    """
    print("=== CREATING VECTOR MAP WITH BUILDING HEIGHTS AND DEFAULT HEIGHTS ===")

    # 1. Load input data
    mask = load_class_mask_from_npy(npy_path)
    object_height_data = np.load(object_height_npy_path).astype(np.float32)

    # 2. Create geographic transform from the mask's bounds
    west, south, east, north = geographic_bounds
    mask_transform = from_bounds(west, south, east, north, mask.shape[1], mask.shape[0])

    all_polygons_data = []
    unique_classes = np.unique(mask)

    print("Step 1: Generating polygons for all classes...")
    for class_id in unique_classes:
        if class_id >= len(class_names):
            continue
        
        class_name = class_names[class_id]
        class_mask = (mask == class_id).astype(np.uint8)
        
        if np.sum(class_mask) > 0:
            shapes = features.shapes(
                class_mask, mask=(class_mask > 0),
                transform=mask_transform, connectivity=8
            )
            for geom, value in shapes:
                if value == 1:
                    all_polygons_data.append({
                        'geometry': shape(geom),
                        'class': class_name,
                        'class_id': int(class_id)
                    })
    
    if not all_polygons_data:
        print("❌ No polygons were generated!")
        return None

    # 3. Create a single, complete GeoDataFrame
    gdf = gpd.GeoDataFrame(all_polygons_data, crs=crs)
    print(f"Step 2: Created a GeoDataFrame with {len(gdf)} total polygons.")

    # 4. Initialize object_height column with default values
    print("Step 3: Assigning object heights...")
    
    # Define default heights for different classes (only buildings get calculated heights)
    default_heights = {
        'Water': 1.0,
        'Land (unpaved area)': 0.5,
        'Road': 1.0,
        'Building': 5.0,  # Will be overridden with calculated heights
        'Vegetation': 1.0,
        'Unlabeled': 0.0
    }
    
    # Initialize all heights based on class
    gdf['object_height'] = gdf['class'].map(default_heights).fillna(0.0)
    
    # # Filter to get only the building polygons for special processing
    # building_gdf = gdf[gdf['class'] == 'Building'].copy()

    # if not building_gdf.empty:
    #     print(f"  Calculating heights for {len(building_gdf)} building polygons...")
        
    #     # Run zonal stats only on the building polygons
    #     object_stats = zonal_stats(building_gdf, object_height_data, affine=mask_transform, stats="mean", nodata=-999)
        
    #     # Get the calculated mean heights for buildings
    #     building_heights = [s['mean'] if s and s['mean'] is not None else 5.0 for s in object_stats]  # Default 5.0 for buildings without data
        
    #     # Assign the calculated heights back to the main GeoDataFrame at the correct locations
    #     gdf.loc[building_gdf.index, 'object_height'] = building_heights
    #     print(f"  Assigned calculated heights to {len(building_gdf)} building polygons.")
    # else:
    #     print("  No 'Building' polygons found to process.")

    # Show height assignment summary
    print(f"\nStep 4: Height assignment summary:")
    for class_name in gdf['class'].unique():
        class_data = gdf[gdf['class'] == class_name]
        heights = class_data['object_height'].dropna()
        count = len(class_data)
        
        if class_name == 'Building':
            if len(heights) > 0:
                print(f"  {class_name}: {count} polygons (calculated heights: {heights.min():.2f} to {heights.max():.2f})")
            else:
                print(f"  {class_name}: {count} polygons (no height data)")
        else:
            default_height = default_heights.get(class_name, 0.0)
            print(f"  {class_name}: {count} polygons (default height: {default_height})")

    # 5. Save the final GeoDataFrame
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"✅ Vector map with building heights saved to: {output_path}")
    
    return gdf

def create_vector_map2(npy_path, object_height_npy_path, output_path, class_names=None):
    """
    Creates a vector map from a semantic mask in pixel coordinates without georeferencing.
    Calculates object height for buildings and assigns default heights to other classes.
    """
    print("=== CREATING VECTOR MAP (PIXEL COORDINATES) WITH DEFAULT HEIGHTS ===")

    # 1. Load input data
    mask = load_class_mask_from_npy(npy_path)
    object_height_data = np.load(object_height_npy_path).astype(np.float32)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Object height data shape: {object_height_data.shape}")
    
    # 2. Create identity transform for pixel coordinates (no georeferencing)
    height, width = mask.shape
    pixel_transform = from_bounds(0, 0, width, height, width, height)
    
    all_polygons_data = []
    unique_classes = np.unique(mask)
    
    print("Step 1: Generating polygons for all classes in pixel coordinates...")
    for class_id in unique_classes:
        if class_names and class_id >= len(class_names):
            continue
        
        class_name = class_names[class_id] if class_names else f"Class_{class_id}"
        class_mask = (mask == class_id).astype(np.uint8)
        
        if np.sum(class_mask) > 0:
            shapes = features.shapes(
                class_mask, mask=(class_mask > 0),
                transform=pixel_transform, connectivity=8
            )
            
            polygon_count = 0
            for geom, value in shapes:
                if value == 1:
                    all_polygons_data.append({
                        'geometry': shape(geom),
                        'class': class_name,
                        'class_id': int(class_id)
                    })
                    polygon_count += 1
            
            print(f"  {class_name}: {polygon_count} polygons")
    
    if not all_polygons_data:
        print("❌ No polygons were generated!")
        return None

    # 3. Create GeoDataFrame with no CRS (pixel coordinates)
    gdf = gpd.GeoDataFrame(all_polygons_data, crs=None)
    print(f"Step 2: Created GeoDataFrame with {len(gdf)} total polygons in pixel coordinates.")

    # 4. Initialize object_height column with default values
    print("Step 3: Assigning object heights...")
    
    # Define default heights for different classes
    default_heights = {
        'Water': 2.0,
        'Land (unpaved area)': 0.0,
        'Road': 2.0,
        'Vegetation': 2.0,
        'Unlabeled': 0.0  # Keep unlabeled at 0
    }
    
    # Initialize all heights based on class
    gdf['object_height'] = gdf['class'].map(default_heights).fillna(np.nan)
    
    # Filter to get only the building polygons for special processing
    building_gdf = gdf[gdf['class'] == 'Building'].copy()

    if not building_gdf.empty:
        print(f"  Found {len(building_gdf)} building polygons to process.")
        
        # Run zonal stats only on the building polygons using pixel coordinates
        object_stats = zonal_stats(
            building_gdf, 
            object_height_data, 
            affine=pixel_transform, 
            stats="mean", 
            nodata=-999
        )
        
        # Get the calculated mean heights
        building_heights = []
        valid_height_count = 0
        
        for s in object_stats:
            if s and s['mean'] is not None and not np.isnan(s['mean']):
                building_heights.append(s['mean'])
                valid_height_count += 1
            else:
                building_heights.append(5.0)  # Default height for buildings without data
        
        # Assign the heights back to the main GeoDataFrame
        gdf.loc[building_gdf.index, 'object_height'] = building_heights
        print(f"  Assigned heights to {len(building_gdf)} building polygons.")
        print(f"  {valid_height_count} buildings have valid height data.")
        
        # Show height statistics for buildings
        valid_heights = [h for h in building_heights if h > 0]
        if valid_heights:
            print(f"  Building height range: {min(valid_heights):.2f} to {max(valid_heights):.2f} units")
            print(f"  Mean building height: {np.mean(valid_heights):.2f} units")
    else:
        print("  No 'Building' polygons found to process.")

    # 5. Show height assignment summary
    print(f"\nStep 4: Height assignment summary:")
    for class_name in gdf['class'].unique():
        class_data = gdf[gdf['class'] == class_name]
        count = len(class_data)
        
        if class_name == 'Building':
            valid_heights = class_data['object_height'].dropna()
            if len(valid_heights) > 0:
                print(f"  {class_name}: {count} polygons (calculated heights)")
            else:
                print(f"  {class_name}: {count} polygons (no height data)")
        else:
            default_height = default_heights.get(class_name, 0.0)
            print(f"  {class_name}: {count} polygons (default height: {default_height})")

    # 6. Add pixel coordinate bounds info
    bounds = gdf.total_bounds
    print(f"Step 5: Pixel coordinate bounds: [{bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f}]")
    
    # 7. Save the final GeoDataFrame (Note: will be saved with no CRS)
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"✅ Vector map with heights saved to: {output_path}")
    print(f"   Coordinate system: Pixel coordinates (no georeferencing)")
    print(f"   Image dimensions: {width} x {height} pixels")
    
    return gdf

def visualize_vector_map(gdf, output_image=None):
    """Visualize the vector map."""
    if gdf is None or gdf.empty:
        print("No data to visualize")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    class_colors = {
        'Water': "#4156e2", 'Land (unpaved area)': "#E0E24E", 'Road': "#000000",
        'Building': '#380798', 'Vegetation': "#24660e", 'Unlabeled': "#9B9B9B"
    }
    
    for class_name in gdf['class'].unique():
        class_data = gdf[gdf['class'] == class_name]
        color = class_colors.get(class_name, '#CCCCCC')
        class_data.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5)
        
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=name) for name, color in class_colors.items() if name in gdf['class'].unique()]
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_title('Vector Map')
    ax.axis('off')
    plt.tight_layout()
    
    if output_image:
        plt.savefig(output_image, dpi=300)
        print(f"Visualization saved to: {output_image}")
    
    plt.show()

# ===============================================================
# Main execution block for testing this script directly
# ===============================================================
if __name__ == "__main__":
    print("--- Running vectormap.py (Simplified Version) ---")

    if not os.path.exists("output"):
        os.makedirs("output")

    # Input file paths (DEM is no longer needed)
    input_semantic_mask = "input/semantic_mask.npy"
    input_object_height = "input/depth_map.npy"
    output_geojson = "output/vector_map.geojson"
    
    # Scene parameters
    # geographic_bounds = (74.000070, 16.3396997, 74.00034778, 16.339970)
    geographic_bounds = (74.1, 16.2, 74.6, 16.7)
    class_names = [
        'Water', 'Land (unpaved area)', 'Road',
        'Building', 'Vegetation', 'Unlabeled'
    ]

    try:
        # Check if all input files exist
        for f in [input_semantic_mask, input_object_height]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required input file not found: {f}")

        generated_gdf = create_vector_map(
            npy_path=input_semantic_mask,
            object_height_npy_path=input_object_height,
            output_path=output_geojson,
            class_names=class_names,
            geographic_bounds=geographic_bounds
        )

        # generated_gdf = create_vector_map2(
        #     npy_path=input_semantic_mask,
        #     object_height_npy_path=input_object_height,
        #     output_path=output_geojson,
        #     class_names=class_names
        # )

        if generated_gdf is not None:
            print("\n--- Generating Visualization ---")
            visualize_vector_map(generated_gdf, "output/vector_map.png")

    except Exception as e:
        print(f"\n❌ An error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()