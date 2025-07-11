import numpy as np
import cv2
import open3d as o3d
import os
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
from tqdm import tqdm
import warnings
import time

class RobustMultiViewGenerator:
    def __init__(self, dataset_path, bbox_data=None, total_views=20, camera_intrinsics=None):
        """
        Initialize the robust multi-view point cloud generator with progress tracking
        """
        self.dataset_path = dataset_path
        self.bbox_data = bbox_data
        self.total_views = total_views
        
        # Camera intrinsics
        if camera_intrinsics is None:
            self.fx = 525.0
            self.fy = 525.0
            self.cx = 319.5
            self.cy = 239.5
        else:
            self.fx, self.fy, self.cx, self.cy = camera_intrinsics
            
        self.depth_scale = 1000.0
        
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
        """Convert RGB-D pair to point cloud"""
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to meters
        depth_m = depth_image.astype(np.float32) / self.depth_scale
        
        # Filter valid depth values
        valid_mask = (depth_m > 0.1) & (depth_m < 5.0)
        
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
    
    def generate_selected_pointclouds(self, view_indices=None):
        """Generate point clouds for selected views with progress tracking"""
        if view_indices is None:
            view_indices = np.linspace(1, 95, 20, dtype=int)
        
        pointclouds = []
        
        print(f"🔄 Generating {len(view_indices)} selected point clouds...")
        print(f"📋 Selected views: {view_indices}")
        
        # Progress bar for point cloud generation
        pbar = tqdm(view_indices, desc="Generating point clouds", unit="view")
        
        for i in pbar:
            color_path = os.path.join(self.dataset_path, f"desk_2_{i}.png")
            depth_path = os.path.join(self.dataset_path, f"desk_2_{i}_depth.png")
            
            pbar.set_postfix({"Current": f"View {i}"})
            
            if not os.path.exists(color_path) or not os.path.exists(depth_path):
                print(f"⚠️  Skipping image {i}: files not found")
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
                
                # Save individual point cloud
                o3d.io.write_point_cloud(f"individual_desk_{i}.ply", pcd_processed)
                
            except Exception as e:
                print(f"❌ Error processing image {i}: {e}")
                continue
        
        pbar.close()
        print(f"✅ Successfully generated {len(pointclouds)} point clouds")
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
    
    def optimized_multiway_registration(self, pointclouds):
        """Perform optimized multiway registration with progress tracking"""
        if len(pointclouds) < 2:
            return pointclouds
        
        print("🔄 Performing optimized multiway registration...")
        
        # Use middle point cloud as reference
        reference_idx = len(pointclouds) // 2
        reference_pcd = pointclouds[reference_idx]['pcd']
        transformations = []
        
        print(f"📍 Using point cloud {pointclouds[reference_idx]['image_idx']} as reference")
        
        # Progress bar for registration
        pbar = tqdm(enumerate(pointclouds), total=len(pointclouds), 
                   desc="Registering point clouds", unit="view")
        
        for i, pc_data in pbar:
            pbar.set_postfix({"Current": f"View {pc_data['image_idx']}"})
            
            if i == reference_idx:
                transformations.append(np.eye(4))  # Identity for reference
                continue
            
            source = pc_data['pcd']
            transformation, fitness = self.pairwise_registration(source, reference_pcd)
            
            transformations.append(transformation)
            pbar.set_postfix({"Current": f"View {pc_data['image_idx']}", 
                             "Fitness": f"{fitness:.3f}"})
        
        pbar.close()
        
        # Apply transformations to original point clouds
        print("🔄 Applying transformations...")
        aligned_pointclouds = []
        
        for i, pc_data in enumerate(tqdm(pointclouds, desc="Applying transforms", unit="view")):
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
        
        print("🔄 Combining aligned point clouds...")
        
        # Start with first point cloud
        combined_pcd = aligned_pointclouds[0]
        
        # Add all other point clouds with progress
        for i in tqdm(range(1, len(aligned_pointclouds)), desc="Combining clouds", unit="view"):
            combined_pcd += aligned_pointclouds[i]
        
        print(f"📊 Combined point cloud has {len(combined_pcd.points)} points")
        
        # Final processing
        print("🔄 Final processing (outlier removal and downsampling)...")
        combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.002)
        
        print(f"✅ Final processed point cloud has {len(combined_pcd.points)} points")
        
        return combined_pcd
    
    def detect_and_track_objects_robust(self, pcd, cluster_eps=0.02, min_points=50):
        """
        Robust object detection with proper error handling for Qhull issues
        """
        print("🔄 Detecting objects using DBSCAN clustering...")
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
        
        # Perform DBSCAN clustering
        start_time = time.time()
        labels = np.array(pcd.cluster_dbscan(eps=cluster_eps, min_points=min_points))
        clustering_time = time.time() - start_time
        
        max_label = labels.max()
        print(f"📊 Point cloud has {max_label + 1} clusters (took {clustering_time:.1f}s)")
        
        # Create bounding boxes for each cluster with progress tracking
        bounding_boxes = []
        object_info = []
        failed_clusters = 0
        
        pbar = tqdm(range(max_label + 1), desc="Creating bounding boxes", unit="object")
        
        for i in pbar:
            cluster_indices = np.where(labels == i)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Get cluster points
            cluster_points = np.asarray(pcd.points)[cluster_indices]
            cluster_colors = np.asarray(pcd.colors)[cluster_indices]
            
            # Check if cluster has sufficient 3D extent
            if len(cluster_points) < 4:  # Need at least 4 points for 3D bounding box
                failed_clusters += 1
                continue
            
            # Check 3D extent to avoid flat clusters
            extent_check = np.ptp(cluster_points, axis=0)  # Range in each dimension
            if np.any(extent_check < 1e-6):  # Very flat cluster
                failed_clusters += 1
                continue
            
            try:
                # Create axis-aligned bounding box (more robust)
                aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(cluster_points))
                
                # Try to create oriented bounding box with error handling
                obb = None
                try:
                    # Add small random noise to break coplanarity
                    noise_points = cluster_points + np.random.normal(0, 1e-6, cluster_points.shape)
                    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                        o3d.utility.Vector3dVector(noise_points))
                except (RuntimeError, ValueError) as e:
                    # If OBB fails, create a simple OBB from AABB
                    obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb)
                
                # Calculate object properties
                center = aabb.get_center()
                extent = aabb.get_extent()
                volume = extent[0] * extent[1] * extent[2]
                
                # Color the bounding box
                aabb.color = (1, 0, 0)  # Red
                obb.color = (0, 1, 0)   # Green
                
                bounding_boxes.append({
                    'aabb': aabb,
                    'obb': obb,
                    'cluster_id': i,
                    'num_points': len(cluster_indices),
                    'center': center,
                    'extent': extent,
                    'volume': volume,
                    'avg_color': np.mean(cluster_colors, axis=0)
                })
                
                object_info.append({
                    'id': i,
                    'points': len(cluster_indices),
                    'center': center.tolist(),
                    'size': extent.tolist(),
                    'volume': volume,
                    'color': np.mean(cluster_colors, axis=0).tolist()
                })
                
                pbar.set_postfix({"Success": len(bounding_boxes), "Failed": failed_clusters})
                
            except Exception as e:
                failed_clusters += 1
                pbar.set_postfix({"Success": len(bounding_boxes), "Failed": failed_clusters})
                continue
        
        pbar.close()
        
        print(f"✅ Successfully created {len(bounding_boxes)} bounding boxes")
        if failed_clusters > 0:
            print(f"⚠️  Skipped {failed_clusters} clusters due to geometric issues")
        
        return bounding_boxes, object_info
    
    def create_object_tracking_visualization(self, pcd, bounding_boxes):
        """Create visualization with object tracking bounding boxes"""
        geometries = [pcd]
        
        # Add all bounding boxes
        for bbox_info in bounding_boxes:
            geometries.append(bbox_info['aabb'])  # Axis-aligned
            if bbox_info['obb'] is not None:
                geometries.append(bbox_info['obb'])   # Oriented
        
        return geometries
    
    def save_object_tracking_data(self, object_info, filename="object_tracking_data_20views.json"):
        """Save object tracking data to JSON file"""
        import json
        
        tracking_data = {
            'total_objects': len(object_info),
            'objects': object_info,
            'timestamp': str(np.datetime64('now')),
            'description': 'Object tracking data from 20-view desk reconstruction'
        }
        
        with open(filename, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"💾 Object tracking data saved to {filename}")

def load_and_parse_bboxes(mat_file_path):
    """Load and parse bounding box data from .mat file"""
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        bboxes = mat_data['bboxes']
        
        parsed_bboxes = []
        
        for i in range(bboxes.shape[1]):
            try:
                bbox_obj = bboxes[0, i]
                
                if bbox_obj is not None and len(bbox_obj) > 0:
                    bbox_data = bbox_obj[0]
                    
                    if len(bbox_data) == 6:
                        category = str(bbox_data[0][0]) if len(bbox_data[0]) > 0 else "unknown"
                        instance = int(bbox_data[1][0][0]) if bbox_data[1].size > 0 else 0
                        top = int(bbox_data[2][0][0]) if bbox_data[2].size > 0 else 0
                        bottom = int(bbox_data[3][0][0]) if bbox_data[3].size > 0 else 0
                        left = int(bbox_data[4][0][0]) if bbox_data[4].size > 0 else 0
                        right = int(bbox_data[5][0][0]) if bbox_data[5].size > 0 else 0
                        
                        parsed_bbox = {
                            'category': category,
                            'instance': instance,
                            'top': top,
                            'bottom': bottom,
                            'left': left,
                            'right': right,
                            'bbox_coords': [left, top, right, bottom]
                        }
                        
                        parsed_bboxes.append(parsed_bbox)
                        
            except Exception as e:
                continue
        
        return parsed_bboxes
        
    except Exception as e:
        print(f"❌ Error loading MATLAB file: {e}")
        return None

def main_robust_20_views():
    """Main function for robust 20-view processing with progress tracking"""
    
    # Configuration
    dataset_path = "desk_dataset/rgbd-scenes/desk/desk_2/"
    mat_file_path = "desk_dataset/rgbd-scenes/desk/desk_2.mat"
    total_views = 20
    camera_intrinsics = (525.0, 525.0, 319.5, 239.5)
    
    print("="*80)
    print("🚀 ROBUST 20-VIEW PROCESSING WITH PROGRESS TRACKING")
    print("="*80)
    
    # Load bounding box data
    print("\n1️⃣ Loading bounding box data from .mat file...")
    bbox_data = load_and_parse_bboxes(mat_file_path)
    if bbox_data:
        print(f"✅ Loaded {len(bbox_data)} bounding boxes")
    
    # Initialize generator
    generator = RobustMultiViewGenerator(dataset_path, bbox_data, total_views, camera_intrinsics)
    
    try:
        # Step 1: Generate point clouds for 20 selected views
        print("\n2️⃣ GENERATING 20 SELECTED POINT CLOUDS")
        print("-" * 60)
        selected_views = np.linspace(1, 95, 20, dtype=int)
        pointclouds = generator.generate_selected_pointclouds(selected_views)
        
        if not pointclouds:
            print("❌ No point clouds generated. Check your dataset path and files.")
            return None
        
        # Step 2: Perform optimized multiway registration
        print("\n3️⃣ PERFORMING OPTIMIZED MULTIWAY REGISTRATION")
        print("-" * 60)
        aligned_pointclouds = generator.optimized_multiway_registration(pointclouds)
        
        # Step 3: Combine all aligned point clouds
        print("\n4️⃣ COMBINING ALL 20 ALIGNED POINT CLOUDS")
        print("-" * 60)
        final_pcd = generator.combine_aligned_pointclouds(aligned_pointclouds)
        
        if final_pcd is None:
            print("❌ Failed to combine point clouds")
            return None
        
        # Step 4: Robust object detection and tracking
        print("\n5️⃣ ROBUST OBJECT DETECTION AND TRACKING")
        print("-" * 60)
        bounding_boxes, object_info = generator.detect_and_track_objects_robust(final_pcd)
        
        print(f"🎯 Detected {len(bounding_boxes)} objects in the scene")
        
        # Step 5: Save results
        print("\n6️⃣ SAVING RESULTS")
        print("-" * 60)
        
        # Save final combined model
        final_path = "final_robust_20_view_desk_model.ply"
        o3d.io.write_point_cloud(final_path, final_pcd)
        print(f"💾 Final combined model saved as: {final_path}")
        
        # Save object tracking data
        generator.save_object_tracking_data(object_info, "desk_object_tracking_robust_20views.json")
        
        # Create mesh (optional)
        try:
            print("🔄 Creating mesh from combined point cloud...")
            final_pcd.estimate_normals()
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(final_pcd, depth=9)
            mesh_path = "final_robust_20_view_desk_mesh.ply"
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"💾 Mesh saved as: {mesh_path}")
        except Exception as e:
            print(f"⚠️  Mesh creation failed: {e}")
        
        # Step 6: Visualization
        print("\n7️⃣ VISUALIZING RESULTS WITH OBJECT TRACKING")
        print("-" * 60)
        
        # Create visualization with bounding boxes
        geometries = generator.create_object_tracking_visualization(final_pcd, bounding_boxes)
        
        print("🎨 Opening 3D visualization...")
        o3d.visualization.draw_geometries(geometries,
                                        window_name="Robust 20-View Desk Model with Object Tracking",
                                        width=1400,
                                        height=1000,
                                        left=50,
                                        top=50)
        
        # Print object summary
        print("\n8️⃣ OBJECT TRACKING SUMMARY")
        print("-" * 60)
        for obj in object_info:
            print(f"🎯 Object {obj['id']}: {obj['points']} points, "
                  f"center at {[f'{x:.3f}' for x in obj['center']]}, "
                  f"size {[f'{x:.3f}' for x in obj['size']]}")
        
        print("\n" + "="*80)
        print("✅ ROBUST 20-VIEW PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Final model: {final_path}")
        print(f"📁 Object tracking data: desk_object_tracking_robust_20views.json")
        print(f"📁 Individual models: individual_desk_*.ply")
        print(f"📁 Aligned models: aligned_desk_*.ply")
        print(f"🎯 Total objects detected: {len(bounding_boxes)}")
        print(f"📊 Selected views: {selected_views}")
        
        return final_pcd, bounding_boxes, object_info
        
    except Exception as e:
        print(f"❌ Error in robust 20-view processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Required packages: pip install opencv-python open3d numpy matplotlib scipy tqdm
    
    print("🚀 Starting robust 20-view processing with progress tracking...")
    
    # Run main processing
    result = main_robust_20_views()
    
    if result is not None:
        final_pcd, bounding_boxes, object_info = result
        print("\n🎉 Robust 20-view processing completed successfully!")
        print("All Qhull errors have been handled gracefully!")
    else:
        print("\n❌ Processing failed. Please check the error messages above.")
