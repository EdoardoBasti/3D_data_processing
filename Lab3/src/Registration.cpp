#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  PointDistance(Eigen::Vector3d sour, Eigen::Vector3d targ)
              :sour(sour), targ(targ) {}

  template <typename T>
  bool operator()(const T* const tran,
                  T* residuals) const {
    
    //source
    T source[3];
    source[0] = T(sour[0]);
    source[1] = T(sour[1]);
    source[2] = T(sour[2]);

    //rotate source
    T  transformed_source[3];
    ceres::AngleAxisRotatePoint(tran, source, transformed_source);

    //translate source
    transformed_source[0] += tran[3];
    transformed_source[1] += tran[4];
    transformed_source[2] += tran[5];

    //compute residual
    residuals[0] = transformed_source[0] - T(targ[0]);
    residuals[1] = transformed_source[1] - T(targ[1]);
    residuals[2] = transformed_source[2] - T(targ[2]);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d sour, const Eigen::Vector3d targ) {
    return (new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(new PointDistance(sour, targ)));
  }

  Eigen::Vector3d sour, targ;
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //initialize parameters for previous iteration
  double previous_error = std::numeric_limits<double>::max();
  Eigen::Matrix4d previous_transformation = transformation_;

  //main icp loop
  for(int index_iter = 0; index_iter < max_iteration; index_iter++)
  {
    std::cout << "\niteration <" << index_iter << ">" << std::endl; 

    //find closest matches and rmse
    std::tuple<std::vector<size_t>, std::vector<size_t>, double> tuple_closest_rmse = find_closest_point(threshold);

    //current rmse
    double current_error = std::get<2>(tuple_closest_rmse);
    std::cout << "previous error: " << previous_error << std::endl;
    std::cout << "current error: " << current_error << std::endl;

    //convergence conditions -> stop
    if(previous_error < current_error) //then the error got worse -> return the previous transformation instead
    {
      transformation_ = previous_transformation;
      std::cout << "ICP convergence\n";
      break;
    }
    if(std::abs(previous_error - current_error) < relative_rmse) //then the error almost stays the same -> return current transformation
    {
      std::cout << "ICP convergence\n";
      break;
    }

    //get current estimate of R, t
    Eigen::Matrix4d current_transformation;
    
    //svd
    if(mode == "svd")
    {
      current_transformation = get_svd_icp_transformation(std::get<0>(tuple_closest_rmse), std::get<1>(tuple_closest_rmse));
    }
    //lm
    else
    {
      current_transformation = get_lm_icp_registration(std::get<0>(tuple_closest_rmse), std::get<1>(tuple_closest_rmse));
    }

    //keep in memory the last transformation in case the error gets worse in the next iteration
    previous_transformation = transformation_;

    //update previous_error
    previous_error = current_error;

    //update the transformation
    transformation_ = transformation_ * current_transformation;
  }
  //end loop

  return;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double rmse;

  //create a KD tree using the target point cloud
  open3d::geometry::KDTreeFlann target_kd_tree(target_);

  //clone source cloud
  open3d::geometry::PointCloud source_clone = source_;
  //tranform clone of source cloud
  source_clone.Transform(transformation_);

  //get number of points of the source point cloud
  int size_source = source_clone.points_.size();

  //vector with index of target correspondence
  std::vector<int> index_target(1);
  //square distance between corresponding points
  std::vector<double> dist2(1);

  //loop over the points of the source point cloud
  for(int index_s = 0; index_s < size_source; index_s++)
  {
    //current point of transformed source point cloud
    source_point = source_clone.points_[index_s];

    //find correspondence
    target_kd_tree.SearchKNN(source_point, 1, index_target, dist2);

    //distance
    double dist1 = sqrt(dist2[0]);

    //check distance
    if(dist1 <= threshold)
    {
      //save indices
      source_indices.push_back(index_s);
      target_indices.push_back(index_target[0]);

      //update rmse
      rmse = rmse*index_s/(index_s+1) + dist2[0]/(index_s+1);
    }
  }

  //rmse (root of mse)
  rmse = sqrt(rmse);
  
  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  //get a copy of the source point cloud
  open3d::geometry::PointCloud source_clone = source_;
  //tranform clone of source cloud
  source_clone.Transform(transformation_);

  //tranformed source and target centroids
  Eigen::Vector3d source_clone_centroid (0,0,0);
  Eigen::Vector3d target_centroid (0,0,0);

  //compute tranformed source centroid
  int size_source = source_clone.points_.size();
  for(int s_i = 0; s_i < size_source; s_i++)
  {
    source_clone_centroid += source_clone.points_[s_i];
  }
  source_clone_centroid = source_clone_centroid/size_source;

  //compute target centroid
  int size_target = target_.points_.size();
  for(int t_i = 0; t_i < size_target; t_i++)
  {
    target_centroid += target_.points_[t_i];
  }
  target_centroid = target_centroid/size_target;

  //matrices 3xN
  Eigen::MatrixXd subtracted_source(3, source_indices.size());
  Eigen::MatrixXd subtracted_target(3, source_indices.size());

  //loop to subtract the centroid from the points
  for(int index_matches = 0; index_matches < source_indices.size(); index_matches++)
  {
    //current point in the source point cloud
    Eigen::Vector3d source_point = source_clone.points_[source_indices[index_matches]];
    //subtract centroid and insert it in the matrix
    subtracted_source.col(index_matches) = source_point - source_clone_centroid;

    //current point in the target point cloud
    Eigen::Vector3d target_point = target_.points_[target_indices[index_matches]];
    //subtract centroid and insert it in the matrix
    subtracted_target.col(index_matches) = target_point - target_centroid;
  }
  
  //transpose source matrix (Nx3)
  Eigen::MatrixXd transposed_subtracted_source = subtracted_source.transpose();
  
  //multiply the 2 matrices to obtain matrix W (3x3)
  Eigen::Matrix3d W_matrix = subtracted_target * transposed_subtracted_source;

  //svd
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  
  //find matrix R
  Eigen::Matrix3d R_matrix = svd.matrixU() * svd.matrixV().transpose();

  //special case
  if(R_matrix.determinant() == -1)
  {
    //diagonal matrix (1, 1, -1)
    Eigen::Matrix3d inverse_last_column = Eigen::Matrix3d::Identity();
    inverse_last_column(2, 2) = -1;

    R_matrix = svd.matrixU() * inverse_last_column * svd.matrixV().transpose();
  }

  //compute t = centroid_target - R*centroid_source
  Eigen::Vector3d t_vec = target_centroid - R_matrix * source_clone_centroid;

  //create the transformation matrix using R and t
  transformation.block<3,3>(0, 0) = R_matrix; //R first 3x3 block starting from (0,0)
  transformation.block<3,1>(0, 3) = t_vec; //t first 3 elements of last column

  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();

  //tranformed source clone
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);

  //declare probelm and summary
  ceres::Problem problem;
  ceres::Solver::Summary summary;

  Eigen::Vector3d source_point;
  Eigen::Vector3d target_point;

  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
    //get current points
    source_point = source_clone.points_[source_indices[i]];
    target_point = target_.points_[target_indices[i]];

    ceres::CostFunction* cost_function = PointDistance::Create(source_point, target_point);
    problem.AddResidualBlock(cost_function, nullptr, transformation_arr.data());
  }

  Solve(options, &problem, &summary);

  //compute rotation matrix
  Eigen::Matrix3d Rot_matrix;
  Rot_matrix = Eigen::AngleAxisd(transformation_arr[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(transformation_arr[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(transformation_arr[2], Eigen::Vector3d::UnitZ());

  //compute translation vector
  Eigen::Vector3d t_vec(transformation_arr[3], transformation_arr[4], transformation_arr[5]);

  //create transformation matrix
  transformation.block<3, 3> (0, 0) = Rot_matrix;
  transformation.block<3, 1> (0, 3) = t_vec;

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}


