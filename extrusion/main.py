import os

# Import our modules
import extrusion.vmap as vmap
import extrusion.dem as dem
import extrusion.scene3D as scene3D

def run_pipeline():
    """
    Complete pipeline: semantic mask → vector map with heights → DEM update → 3D visualization
    """
    print('=' * 60)
    print("            3D VECTOR PIPELINE")
    print('=' * 60)
    
    # Define file paths 
    input_npy = "input/semantic_mask.npy"
    object_height_npy = "input/depth_map.npy"  
    dem_file = "input/cdne43u.tif"            
    
    # Output files
    vector_output = "output/vector_map.geojson"
    final_3d_output = "output/3d_scene"
    
    # Geographic bounds for your data (if using georeferenced version)
    west_lon = 74.000070
    south_lat = 16.3396997
    east_lon = 74.00034778
    north_lat = 16.339970
    # west_lon = 74.1
    # south_lat = 16.2
    # east_lon = 74.6
    # north_lat = 16.7
    geographic_bounds = (west_lon, south_lat, east_lon, north_lat)
    
    # Class names for segmentation
    class_names = [
        'Water',
        'Land (unpaved area)',
        'Road',
        'Building',
        'Vegetation',
        'Unlabeled'
    ]
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Check input files exist
    required_inputs = [input_npy, object_height_npy, dem_file]
    for file_path in required_inputs:
        if not os.path.exists(file_path):
            print(f"❌ Error: Input file '{file_path}' not found!")
            return False
    
    try:
        # STEP 1: Create vector map with object heights
        print("\nSTEP 1: Creating vector map with building heights...")
        print(f"📁 Semantic mask: {input_npy}")
        print(f"📁 Depth map: {object_height_npy}")
        print(f"📁 Output: {vector_output}")
        
        # Choose between georeferenced or pixel coordinates
        use_geographic = True  
        
        if use_geographic:
            print("🌍 Using geographic coordinates...")
            gdf_with_heights = vmap.create_vector_map(
                npy_path=input_npy,
                object_height_npy_path=object_height_npy,
                output_path=vector_output,
                class_names=class_names,
                geographic_bounds=geographic_bounds,
                crs='EPSG:4326'
            )
        else:
            print("🔲 Using pixel coordinates...")
            gdf_with_heights = vmap.create_vector_map2(
                npy_path=input_npy,
                object_height_npy_path=object_height_npy,
                output_path=vector_output,
                class_names=class_names
            )
        
        if gdf_with_heights is None or len(gdf_with_heights) == 0:
            print("❌ Step 1 failed: No vector polygons generated!")
            return False
        
        print(f"✅ Step 1 complete: Created {len(gdf_with_heights)} vector polygons with heights")
        
        # Create vector map visualization
        print(f"\n📊 Creating vector map visualization...")
        vmap.visualize_vector_map(gdf_with_heights, output_image="output/vector_map.png")
        
        # Show height statistics
        print("\n📈 Height Statistics:")
        for class_name in gdf_with_heights['class'].unique():
            class_data = gdf_with_heights[gdf_with_heights['class'] == class_name]
            if 'object_height' in class_data.columns:
                valid_heights = class_data['object_height'].dropna()
                if len(valid_heights) > 0:
                    print(f"   {class_name}: {valid_heights.min():.2f} to {valid_heights.max():.2f} "
                          f"(mean: {valid_heights.mean():.2f})")
                else:
                    print(f"   {class_name}: No height data")
        
        # STEP 2: Update terrain elevation from DEM 
        print(f"\nSTEP 2: Updating terrain elevation from DEM...")
        print(f"📁 Vector map: {vector_output}")
        print(f"📁 DEM file: {dem_file}")
        
        try:
            # Update terrain elevation in the GeoJSON
            gdf_updated = dem.update_terrain_elevation_in_geojson(
                geojson_path=vector_output,
                dem_path=dem_file
            )
            
            if gdf_updated is not None:
                print(f"✅ Step 2 complete: Terrain elevation updated successfully")
            else:
                print(f"⚠️ Step 2 warning: DEM update failed, continuing without terrain elevation")
                
        except Exception as e:
            print(f"⚠️ Step 2 warning: DEM update failed ({e}), continuing to 3D visualization")

        # STEP 3: Create 3D visualization
        print(f"\nSTEP 3: Creating 3D scene...")
        print(f"📁 Input: {vector_output}")
        print(f"📁 Output: {final_3d_output}")
        
        # Load vector map using scene3D module
        gdf_final = scene3D.load_vector_map(vector_output)
        
        if gdf_final is None or len(gdf_final) == 0:
            print("❌ Step 3 failed: Could not load vector map!")
            return False
        
        # Create 3D scene
        base_mesh, meshes, width_meters, height_meters, base_elevation = scene3D.create_3d_scene(
            gdf=gdf_final,
            height_column='object_height'
        )
        
        if not meshes:
            print("❌ Step 3 failed: No 3D meshes created!")
            return False
        
        scene_info = (width_meters, height_meters, base_elevation)
        
        print(f"✅ Step 3 complete: Created 3D scene with {len(meshes)} meshes")
        print(f"📐 Scene dimensions: {width_meters:.2f} × {height_meters:.2f}")
        print(f"🏔️ Base elevation: {base_elevation:.2f}")
        
        # Export 3D scene
        print(f"\n💾 Exporting 3D scene...")
        scene3D.export_3d_scene(base_mesh, meshes, final_3d_output)
        
        # STEP 4: Launch 3D visualization (scene3D.py)
        print(f"\nSTEP 4: Launching 3D visualization...")
        print("🖱️ Use mouse to navigate, 'Q' to quit")
        
        scene3D.visualize_3d_scene(base_mesh, meshes, scene_info)
        
        # PIPELINE COMPLETE
        print("\n" + "=" * 60)
        print("            PIPELINE COMPLETED SUCCESSFULLY! ✅")
        print("=" * 60)
        print(f"📁 Generated files:")
        print(f"   • Vector map with heights:  {vector_output}")
        print(f"   • Vector visualization:     output/vector_map.png")
        print(f"   • 3D scene:                 {final_3d_output}.ply")
        print(f"\n🎯 Summary:")
        print(f"   • {len(gdf_with_heights)} vector polygons created")
        print(f"   • Building heights calculated from depth map")
        print(f"   • Terrain elevation updated from DEM")
        print(f"   • {len(meshes)} 3D meshes generated")
        print(f"   • Scene size: {width_meters:.1f} × {height_meters:.1f}")
        print(f"   • Base elevation: {base_elevation:.2f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all module files (vmap.py, dem.py, scene3D.py) are in the same directory")
        return False
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        return False
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """
    Check if all required dependencies are installed
    """
    required_modules = [
        'numpy', 'geopandas', 'open3d', 'rasterio', 
        'shapely', 'matplotlib', 'pandas', 'PIL',
        'rasterstats', 'scipy'
    ]
    
    missing_modules = []
    
    print("🔍 Checking dependencies...")
    for module in required_modules:
        try:
            if module == 'PIL':
                __import__('PIL')
            else:
                __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"   ❌ {module}")
    
    if missing_modules:
        print(f"\n❌ Missing required modules:")
        for module in missing_modules:
            print(f"  • {module}")
        print(f"\n💡 Install missing modules with:")
        install_names = []
        for module in missing_modules:
            if module == 'PIL':
                install_names.append('Pillow')
            else:
                install_names.append(module)
        print(f"pip install {' '.join(install_names)}")
        return False
    
    print("✅ All dependencies available!")
    return True

def check_input_files():
    """
    Check if required input files exist
    """
    required_files = [
        "input/semantic_mask.npy",
        "input/depth_map.npy",
        "input/cdne43u.tif"
    ]
    
    print("🔍 Checking input files...")
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   ✅ {file_path} ({file_size:.2f} MB)")
        else:
            missing_files.append(file_path)
            print(f"   ❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing required input files:")
        for file_path in missing_files:
            print(f"  • {file_path}")
        print(f"\n💡 Please ensure these files exist before running the pipeline.")
        return False
    
    print("✅ All input files available!")
    return True

def main():
    """
    Main entry point
    """
    print("🚀 Starting 3D Vector Pipeline:")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies before running the pipeline.")
        return
    
    # Check input files
    if not check_input_files():
        print("Please provide required input files before running the pipeline.")
        return
    
    # Check module files
    required_modules = ['vmap.py', 'dem.py', 'scene3D.py']
    missing_modules = []
    
    for module_file in required_modules:
        if not os.path.exists(module_file):
            missing_modules.append(module_file)
    
    if missing_modules:
        print(f"❌ Missing module files:")
        for module_file in missing_modules:
            print(f"  • {module_file}")
        return
    
    # Run the pipeline
    print("\n" + "=" * 60)
    success = run_pipeline()
    
    if success:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"You can now view your 3D scene in the generated files.")
        print(f"\n💡 Next steps:")
        print(f"   • Open output/3d_scene.ply in a 3D viewer")
        print(f"   • Check output/vector_map.png for 2D visualization")
        print(f"   • Examine output/vector_map.geojson for vector data")
    else:
        print(f"\n💥 Pipeline failed!")
        print(f"Please check the error messages above and fix any issues.")

if __name__ == "__main__":
    main()