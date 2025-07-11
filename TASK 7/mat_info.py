import numpy as np
import cv2
import open3d as o3d
import os
from pathlib import Path
import matplotlib.pyplot as plt
import copy

class MultiViewPointCloudGenerator:
    def __init__(self, dataset_path, camera_intrinsics=None):
        """
        Initialize the multi-view point cloud generator
        
        Args:
            dataset_path: Path to the dataset folder
            camera_intrinsics: Camera intrinsic parameters (fx, fy, cx, cy)
        """
        self.dataset_path = dataset_path
        
        # Camera intrinsics - adjust these based on your camera
        if camera_intrinsics is None:
            self.fx = 525.0
            self.fy = 525.0
            self.cx = 319.5
            self.cy = 239.5
        else:
            self.fx, self.fy, self.cx, self.cy = camera_intrinsics
            
        self.depth_scale = 1000.0  # Depth in millimeters
        
    def load_rgbd_pair(self, color_path, depth_path):
        """Load RGB and depth image pair"""
        color_image = cv2.imread(color_path)
        if color_image is None:
            raise ValueError(f"Could not load color image: {color_path}")
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise ValueError(f"Could not load depth image: {depth_path}")
            
        return color_image, depth_image
    
    def rgbd_to_pointcloud(self, color_image, depth_image):
        """Convert single RGB-D pair to point cloud"""
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to meters
        depth_m = depth_image.astype(np.float32) / self.depth_scale
        
        # Filter valid depth values
        valid_mask = (depth_m > 0.1) & (depth_m < 5.0)  # Reasonable depth range
        
        # Calculate 3D coordinates
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        
        # Apply valid mask
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        colors = color_image[valid_mask] / 255.0
        
        points = np.stack([x_valid, y_valid, z_valid], axis=1)
        
        return points, colors
    
    def create_point_cloud(self, points, colors):
        """Create Open3D point cloud object"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    
    def preprocess_point_cloud(self, pcd, voxel_size=0.005):
        """Preprocess point cloud for better registration"""
        # Remove outliers
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample
        pcd_down = pcd_filtered.voxel_down_sample(voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        return pcd_down
    
    def generate_individual_pointclouds(self, start_idx=1, end_idx=10):
        """Generate individual point clouds for each image"""
        pointclouds = []
        
        print(f"Generating {end_idx - start_idx + 1} individual point clouds...")
        
        for i in range(start_idx, end_idx + 1):
            color_path = os.path.join(self.dataset_path, f"desk_2_{i}.png")
            depth_path = os.path.join(self.dataset_path, f"desk_2_{i}_depth.png")
            
            if not os.path.exists(color_path) or not os.path.exists(depth_path):
                print(f"Skipping image {i}: files not found")
                continue
            
            try:
                # Load images
                color_image, depth_image = self.load_rgbd_pair(color_path, depth_path)
                
                # Convert to point cloud
                points, colors = self.rgbd_to_pointcloud(color_image, depth_image)
                pcd = self.create_point_cloud(points, colors)
                
                # Preprocess for better registration
                pcd_processed = self.preprocess_point_cloud(pcd)
                
                pointclouds.append({
                    'pcd': pcd_processed,
                    'original': pcd,
                    'image_idx': i
                })
                
                print(f"Generated point cloud {i}: {len(pcd_processed.points)} points")
                
                # Save individual point cloud
                o3d.io.write_point_cloud(f"individual_desk_{i}.ply", pcd_processed)
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        print(f"Successfully generated {len(pointclouds)} individual point clouds")
        return pointclouds
    
    def compute_fpfh_features(self, pcd, voxel_size=0.005):
        """Compute FPFH features for point cloud"""
        radius_normal = voxel_size * 2
        radius_feature = voxel_size * 5
        
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        return fpfh
    
    def pairwise_registration(self, source, target, voxel_size=0.005):
        """Perform pairwise registration between two point clouds"""
        
        # Compute FPFH features
        source_fpfh = self.compute_fpfh_features(source, voxel_size)
        target_fpfh = self.compute_fpfh_features(target, voxel_size)
        
        # Global registration using RANSAC
        distance_threshold = voxel_size * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        # Refine with ICP
        distance_threshold = voxel_size * 0.4
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        return result_icp.transformation, result_icp.fitness
    
    def multiway_registration(self, pointclouds):
        """Perform multiway registration to align all point clouds"""
        if len(pointclouds) < 2:
            return pointclouds
        
        print("Performing multiway registration...")
        
        # Use first point cloud as reference
        reference_pcd = pointclouds[0]['pcd']
        transformations = [np.eye(4)]  # Identity for reference
        
        # Register each point cloud to the reference
        for i in range(1, len(pointclouds)):
            print(f"Registering point cloud {pointclouds[i]['image_idx']} to reference...")
            
            source = pointclouds[i]['pcd']
            transformation, fitness = self.pairwise_registration(source, reference_pcd)
            
            transformations.append(transformation)
            print(f"Registration fitness: {fitness:.4f}")
        
        # Apply transformations
        aligned_pointclouds = []
        for i, pc_data in enumerate(pointclouds):
            aligned_pcd = copy.deepcopy(pc_data['original'])
            aligned_pcd.transform(transformations[i])
            aligned_pointclouds.append(aligned_pcd)
            
            # Save aligned individual point cloud
            o3d.io.write_point_cloud(f"aligned_desk_{pc_data['image_idx']}.ply", aligned_pcd)
        
        return aligned_pointclouds
    
    def combine_aligned_pointclouds(self, aligned_pointclouds):
        """Combine all aligned point clouds into one"""
        if not aligned_pointclouds:
            return None
        
        print("Combining aligned point clouds...")
        
        # Start with first point cloud
        combined_pcd = aligned_pointclouds[0]
        
        # Add all other point clouds
        for i in range(1, len(aligned_pointclouds)):
            combined_pcd += aligned_pointclouds[i]
        
        print(f"Combined point cloud has {len(combined_pcd.points)} points")
        
        # Final processing
        combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.002)
        
        print(f"Final processed point cloud has {len(combined_pcd.points)} points")
        
        return combined_pcd
    
    def visualize_registration_process(self, pointclouds, aligned_pointclouds):
        """Visualize the registration process"""
        print("Visualizing registration process...")
        
        # Show original point clouds
        print("Showing original point clouds (before alignment)...")
        original_pcds = [pc_data['pcd'] for pc_data in pointclouds[:3]]  # Show first 3
        o3d.visualization.draw_geometries(original_pcds,
                                        window_name="Original Point Clouds (Before Alignment)",
                                        width=1000, height=800)
        
        # Show aligned point clouds
        print("Showing aligned point clouds (after alignment)...")
        o3d.visualization.draw_geometries(aligned_pointclouds[:3],
                                        window_name="Aligned Point Clouds (After Alignment)",
                                        width=1000, height=800)

def create_mesh_from_pointcloud(pcd):
    """Create mesh from point cloud using Poisson reconstruction"""
    try:
        print("Creating mesh from point cloud...")
        
        # Estimate normals if not already done
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
        
        # Poisson reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
        
        # Remove low-density vertices
        vertices_to_remove = []
        mesh.remove_vertices_by_index(vertices_to_remove)
        
        return mesh
        
    except Exception as e:
        print(f"Mesh creation failed: {e}")
        return None

def main_multiview():
    """Main function for multi-view point cloud generation"""
    
    # Configuration
    dataset_path = "desk_dataset/rgbd-scenes/desk/desk_2/"
    camera_intrinsics = (525.0, 525.0, 319.5, 239.5)  # Adjust based on your camera
    
    print("="*70)
    print("MULTI-VIEW POINT CLOUD GENERATION WITH PROPER ALIGNMENT")
    print("="*70)
    
    # Initialize generator
    generator = MultiViewPointCloudGenerator(dataset_path, camera_intrinsics)
    
    try:
        # Step 1: Generate individual point clouds
        print("\n1. GENERATING INDIVIDUAL POINT CLOUDS")
        print("-" * 40)
        pointclouds = generator.generate_individual_pointclouds(1, 10)
        
        if not pointclouds:
            print("No point clouds generated. Check your dataset path and files.")
            return None
        
        # Step 2: Perform multiway registration
        print("\n2. PERFORMING MULTIWAY REGISTRATION")
        print("-" * 40)
        aligned_pointclouds = generator.multiway_registration(pointclouds)
        
        # Step 3: Visualize registration process (optional)
        print("\n3. VISUALIZING REGISTRATION PROCESS")
        print("-" * 40)
        generator.visualize_registration_process(pointclouds, aligned_pointclouds)
        
        # Step 4: Combine aligned point clouds
        print("\n4. COMBINING ALIGNED POINT CLOUDS")
        print("-" * 40)
        final_pcd = generator.combine_aligned_pointclouds(aligned_pointclouds)
        
        if final_pcd is None:
            print("Failed to combine point clouds")
            return None
        
        # Step 5: Save final model
        print("\n5. SAVING FINAL MODEL")
        print("-" * 40)
        final_path = "final_aligned_desk_model.ply"
        o3d.io.write_point_cloud(final_path, final_pcd)
        print(f"Final aligned model saved as: {final_path}")
        
        # Step 6: Create mesh (optional)
        print("\n6. CREATING MESH")
        print("-" * 40)
        mesh = create_mesh_from_pointcloud(final_pcd)
        if mesh is not None:
            mesh_path = "final_desk_mesh.ply"
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Mesh saved as: {mesh_path}")
        
        # Step 7: Final visualization
        print("\n7. FINAL VISUALIZATION")
        print("-" * 40)
        o3d.visualization.draw_geometries([final_pcd],
                                        window_name="Final Aligned Desk Model",
                                        width=1200,
                                        height=900,
                                        left=50,
                                        top=50)
        
        if mesh is not None:
            o3d.visualization.draw_geometries([mesh],
                                            window_name="Final Desk Mesh",
                                            width=1200,
                                            height=900)
        
        print("\n" + "="*70)
        print("✅ MULTI-VIEW POINT CLOUD GENERATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 Individual point clouds saved as: individual_desk_*.ply")
        print(f"📁 Aligned point clouds saved as: aligned_desk_*.ply")
        print(f"📁 Final combined model saved as: {final_path}")
        if mesh is not None:
            print(f"📁 Final mesh saved as: {mesh_path}")
        
        return final_pcd
        
    except Exception as e:
        print(f"Error in multi-view generation: {e}")
        import traceback
        traceback.print_exc()
        return None

# Additional utility functions
def compare_models():
    """Compare the old and new models side by side"""
    try:
        # Load both models if they exist
        old_model_path = "enhanced_desk_model_with_objects.ply"
        new_model_path = "final_aligned_desk_model.ply"
        
        models_to_show = []
        
        if os.path.exists(old_model_path):
            old_pcd = o3d.io.read_point_cloud(old_model_path)
            old_pcd.paint_uniform_color([1, 0, 0])  # Red for old model
            models_to_show.append(old_pcd)
            print("Loaded old model (red)")
        
        if os.path.exists(new_model_path):
            new_pcd = o3d.io.read_point_cloud(new_model_path)
            new_pcd.paint_uniform_color([0, 1, 0])  # Green for new model
            models_to_show.append(new_pcd)
            print("Loaded new model (green)")
        
        if models_to_show:
            o3d.visualization.draw_geometries(models_to_show,
                                            window_name="Model Comparison: Old (Red) vs New (Green)",
                                            width=1200,
                                            height=900)
        else:
            print("No models found for comparison")
            
    except Exception as e:
        print(f"Error comparing models: {e}")

def batch_process_with_different_parameters():
    """Process with different parameter sets to find optimal results"""
    dataset_path = "desk_dataset/rgbd-scenes/desk/desk_2/"
    
    # Different parameter sets
    parameter_sets = [
        {"voxel_size": 0.003, "depth_range": (0.1, 5.0), "name": "fine"},
        {"voxel_size": 0.005, "depth_range": (0.2, 4.0), "name": "medium"},
        {"voxel_size": 0.008, "depth_range": (0.3, 3.0), "name": "coarse"}
    ]
    
    for params in parameter_sets:
        print(f"\nProcessing with {params['name']} parameters...")
        generator = MultiViewPointCloudGenerator(dataset_path)
        
        # Modify parameters
        generator.voxel_size = params["voxel_size"]
        generator.depth_range = params["depth_range"]
        
        try:
            pointclouds = generator.generate_individual_pointclouds(1, 10)
            aligned_pointclouds = generator.multiway_registration(pointclouds)
            final_pcd = generator.combine_aligned_pointclouds(aligned_pointclouds)
            
            if final_pcd:
                output_path = f"desk_model_{params['name']}.ply"
                o3d.io.write_point_cloud(output_path, final_pcd)
                print(f"Saved {params['name']} model as: {output_path}")
                
        except Exception as e:
            print(f"Error with {params['name']} parameters: {e}")

if __name__ == "__main__":
    # Required packages: pip install opencv-python open3d numpy matplotlib
    
    # Run main multi-view generation
    result = main_multiview()
    
    # Optional: Compare with previous model
    # compare_models()
    
    # Optional: Try different parameter sets
    # batch_process_with_different_parameters()
