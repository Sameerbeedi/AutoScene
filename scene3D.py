import numpy as np
import geopandas as gpd
import open3d as o3d
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pandas as pd

def load_vector_map(geojson_path):
    """
    Load GeoJSON vector map
    """
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded {len(gdf)} polygons")
    print(f"Columns: {list(gdf.columns)}")
    
    # Check for height columns
    height_columns = [col for col in gdf.columns if 'height' in col.lower() or 'elevation' in col.lower()]
    print(f"Available height columns: {height_columns}")
    
    return gdf

def get_coordinate_bounds(gdf):
    """
    Get coordinate bounds from GeoDataFrame
    """
    bounds = gdf.total_bounds
    print(f"Data bounds: [{bounds[0]:.6f}, {bounds[1]:.6f}, {bounds[2]:.6f}, {bounds[3]:.6f}]")
    return bounds

def normalize_coordinates_to_meters(gdf):
    """
    Convert coordinates to a normalized meter-based system
    """
    bounds = gdf.total_bounds
    min_x, min_y, max_x, max_y = bounds
    
    # If coordinates are geographic (lat/lon), convert to approximate meters
    if gdf.crs and 'EPSG:4326' in str(gdf.crs):
        # Geographic coordinates - convert to approximate meters
        center_lat = (min_y + max_y) / 2
        lat_rad = np.radians(center_lat)
        
        meters_per_deg_lat = 111319.9
        meters_per_deg_lon = 111319.9 * np.cos(lat_rad)
        
        width_meters = (max_x - min_x) * meters_per_deg_lon
        height_meters = (max_y - min_y) * meters_per_deg_lat
        
        print(f"Geographic coordinates detected")
        print(f"Scene dimensions: {width_meters:.2f}m × {height_meters:.2f}m")
        
        # Create conversion function
        def convert_coords(x, y):
            x_meters = (x - min_x) * meters_per_deg_lon
            y_meters = (y - min_y) * meters_per_deg_lat
            return x_meters, y_meters
            
    else:
        # Pixel or projected coordinates - use as-is
        width_meters = max_x - min_x
        height_meters = max_y - min_y
        
        print(f"Pixel/projected coordinates detected")
        print(f"Scene dimensions: {width_meters:.2f} × {height_meters:.2f} units")
        
        # Create conversion function
        def convert_coords(x, y):
            x_meters = x - min_x
            y_meters = y - min_y
            return x_meters, y_meters
    
    return convert_coords, width_meters, height_meters

def polygon_to_3d_mesh(polygon, height, base_height=0.0, convert_coords=None):
    """
    Convert 2D polygon to 3D mesh using center-based triangulation
    """
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return None, None
    
    # Get exterior coordinates
    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate last point
    
    if len(coords) < 3:
        return None, None
    
    coords_array = np.array(coords)
    n_points = len(coords_array)
    
    # Ensure minimum height for visibility
    if height <= 0.01:
        height = 0.1
    
    # Convert coordinates if function provided
    if convert_coords:
        converted_coords = []
        for coord in coords_array:
            x_new, y_new = convert_coords(coord[0], coord[1])
            converted_coords.append([x_new, y_new])
        coords_array = np.array(converted_coords)
    
    vertices = []
    faces = []
    
    # Add center point for both top and bottom
    center_x = np.mean(coords_array[:, 0])
    center_y = np.mean(coords_array[:, 1])
    
    # Bottom vertices
    vertices.append([center_x, center_y, base_height])  # Bottom center (index 0)
    for coord in coords_array:
        vertices.append([coord[0], coord[1], base_height])  # Bottom perimeter (indices 1 to n)
    
    # Top vertices  
    vertices.append([center_x, center_y, base_height + height])  # Top center (index n+1)
    for coord in coords_array:
        vertices.append([coord[0], coord[1], base_height + height])  # Top perimeter (indices n+2 to 2n+1)
    
    # Bottom face - triangles from center to each edge
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([0, next_i + 1, i + 1])  # Center to edge (corrected order)
    
    # Top face - triangles from center to each edge
    top_center_idx = n_points + 1
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([top_center_idx, top_center_idx + i + 1, top_center_idx + next_i + 1])  # Center to edge
    
    # Side walls
    for i in range(n_points):
        next_i = (i + 1) % n_points
        
        bottom_curr = i + 1
        bottom_next = next_i + 1
        top_curr = top_center_idx + i + 1
        top_next = top_center_idx + next_i + 1
        
        # Two triangles per wall segment
        faces.append([bottom_curr, bottom_next, top_curr])
        faces.append([bottom_next, top_next, top_curr])
    
    return np.array(vertices), np.array(faces)

def create_base_plane(width, height, base_height=-0.1):
    """
    Create a base plane for the scene
    """
    # Create base plane vertices
    vertices = [
        [0, 0, base_height],           # Southwest corner
        [width, 0, base_height],       # Southeast corner
        [width, height, base_height],  # Northeast corner
        [0, height, base_height]       # Northwest corner
    ]
    
    # Create two triangular faces for the base plane
    faces = [
        [0, 1, 2],  # First triangle
        [0, 2, 3]   # Second triangle
    ]
    
    # Create Open3D mesh
    base_mesh = o3d.geometry.TriangleMesh()
    base_mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    base_mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    base_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Gray base
    base_mesh.compute_vertex_normals()
    
    return base_mesh

def create_3d_scene(gdf, height_column='object_height'):
    """
    Create 3D scene from vector map with base at minimum road elevation
    """
    print(f"Creating 3D scene from {len(gdf)} polygons...")
    
    # Get coordinate conversion function
    convert_coords, width_meters, height_meters = normalize_coordinates_to_meters(gdf)
    
    # Find minimum elevation from road polygons to set as base level
    road_data = gdf[gdf['class'] == 'Road']
    if not road_data.empty and height_column in road_data.columns:
        road_elevations = road_data[height_column].dropna()
        if len(road_elevations) > 0:
            base_elevation = road_elevations.min()
            print(f"Base elevation set to minimum road elevation: {base_elevation:.2f}")
        else:
            print("No road elevation data found, using 0 as base")
            base_elevation = 0.0
    else:
        print("No road polygons found, using 0 as base")
        base_elevation = 0.0
    
    # Create base plane at the base elevation level
    base_mesh = create_base_plane(width_meters, height_meters, base_height=base_elevation)
    
    # Define colors for different classes
    class_colors = {
        'Water': [0.1, 0.5, 0.8],        # Blue
        'Land (unpaved area)': [0.6, 0.4, 0.2],  # Brown
        'Road': [0.3, 0.3, 0.3],         # Dark gray
        'Building': [1.0, 0.0, 0.0],     # Red
        'Vegetation': [0.2, 0.7, 0.2],   # Green
        'Unlabeled': [0.7, 0.7, 0.7]     # Light gray
    }
    
    # Create meshes for all polygons
    meshes = []
    
    print(f"Processing polygons by class:")
    for class_name in gdf['class'].unique():
        class_data = gdf[gdf['class'] == class_name]
        color = class_colors.get(class_name, [0.5, 0.5, 0.5])
        
        print(f"  {class_name}: {len(class_data)} polygons")
        
        for idx, row in class_data.iterrows():
            polygon = row.geometry
            
            # Get the object height value directly from your vector map
            object_height = row.get(height_column, 0.0)
            
            # Handle missing data with appropriate defaults
            if pd.isna(object_height) or object_height is None:
                if class_name == 'Building':
                    object_height = 5.0  # Default building height
                elif class_name in ['Land (unpaved area)', 'Unlabeled']:
                    object_height = 0.0  # Flat
                else:
                    object_height = 2.0  # Default for water, roads, vegetation
            
            # For buildings, treat object_height as absolute height and calculate relative
            if class_name == 'Building':
                # Buildings: object_height is absolute elevation, calculate height above base
                height_above_base = object_height - base_elevation
                if height_above_base <= 0.5:
                    height_above_base = 3.0  # Minimum building height
            else:
                # Non-buildings: object_height is already the desired height above base
                height_above_base = object_height
            
            # Ensure minimum visibility
            if height_above_base <= 0.01 and class_name not in ['Land (unpaved area)', 'Unlabeled']:
                height_above_base = 0.5
            
            # Debug print for first polygon of each class
            if idx == class_data.index[0]:
                print(f"    Sample {class_name}: object_height={object_height:.2f}, height_above_base={height_above_base:.2f}")
            
            # Create 3D mesh with base at base_elevation
            vertices, faces = polygon_to_3d_mesh(
                polygon, height_above_base, base_height=base_elevation, convert_coords=convert_coords
            )
            
            if vertices is not None and faces is not None:
                # Create Open3D mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
                
                meshes.append(mesh)
    
    print(f"Created {len(meshes)} 3D meshes")
    print(f"Scene elevation range: {base_elevation:.2f} (base) to {base_elevation + 20:.2f} (estimated top)")
    
    return base_mesh, meshes, width_meters, height_meters, base_elevation

def export_3d_scene(base_mesh, meshes, output_path):
    """
    Export the complete 3D scene to file
    """
    # Combine all meshes
    combined_mesh = base_mesh
    
    for mesh in meshes:
        combined_mesh += mesh
    
    # Export as PLY file
    success = o3d.io.write_triangle_mesh(f"{output_path}.ply", combined_mesh)
    
    if success:
        print(f"✅ 3D scene exported to: {output_path}.ply")
    else:
        print("❌ Failed to export 3D scene")

def visualize_3d_scene(base_mesh, meshes, scene_info):
    """
    Visualize the 3D scene using Open3D with proper camera positioning
    """
    print("Launching 3D visualization...")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Vector Map Scene", width=1400, height=900)
    
    # Add base plane
    vis.add_geometry(base_mesh)
    
    # Add all meshes
    for mesh in meshes:
        vis.add_geometry(mesh)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.show_coordinate_frame = False
    opt.background_color = np.asarray([0.9, 0.9, 0.9])  # Light gray background
    opt.light_on = True
    
    # Set up camera view
    ctr = vis.get_view_control()
    if len(scene_info) >= 3:
        width_meters, height_meters, base_elevation = scene_info
    else:
        width_meters, height_meters = scene_info
        base_elevation = 0
    
    # Position camera for good overview - account for base elevation
    camera_height = max(width_meters, height_meters) * 0.5 + base_elevation
    ctr.set_front([0.3, 0.3, -0.9])  # Look down at angle
    ctr.set_lookat([width_meters/2, height_meters/2, base_elevation + 5])  # Look at scene center above base
    ctr.set_up([0, 1, 0])  # Y-axis up
    ctr.set_zoom(0.8)
    
    print("\n=== 3D Viewer Controls ===")
    print("Mouse: Rotate view")
    print("Scroll: Zoom in/out") 
    print("Ctrl+Mouse: Pan")
    print("R: Reset view")
    print("Q: Quit")
    print("==============================")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

# Update the main execution to handle the base elevation
if __name__ == "__main__":
    # Configuration
    geojson_file = "output/vector_map.geojson"  # Your vector map file
    output_3d_file = "output/3d_scene"          # Output 3D scene
    height_column = "object_height"             # Column containing height data
    
    try:
        print("=== LOADING VECTOR MAP ===")
        gdf = load_vector_map(geojson_file)
        
        if gdf is None or len(gdf) == 0:
            print("❌ No data loaded!")
            exit(1)
        
        # Analyze the data
        # analyze_scene_data(gdf, height_column)
        
        print(f"\n=== CREATING 3D SCENE ===")
        base_mesh, meshes, width_meters, height_meters, base_elevation = create_3d_scene(gdf, height_column)
        
        if not meshes:
            print("❌ No 3D meshes created!")
            exit(1)
        
        scene_info = (width_meters, height_meters, base_elevation)
        
        print(f"\n=== SCENE STATISTICS ===")
        print(f"Scene dimensions: {width_meters:.2f} × {height_meters:.2f}")
        print(f"Base elevation: {base_elevation:.2f}")
        print(f"Total 3D meshes: {len(meshes)}")
        
        # Export 3D scene
        print(f"\n=== EXPORTING 3D SCENE ===")
        export_3d_scene(base_mesh, meshes, output_3d_file)
        
        # Visualize
        print(f"\n=== LAUNCHING 3D VISUALIZATION ===")
        visualize_3d_scene(base_mesh, meshes, scene_info)
        
        print("\n✅ 3D scene generation complete!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {geojson_file}")
        print("Make sure your vector map file exists")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()