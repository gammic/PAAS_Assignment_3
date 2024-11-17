#include <nav_msgs/Odometry.h>
#include <chrono>
#include <ros/ros.h>
#include <iostream>
#include <math.h>
#include "particle/particle_filter.h"
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/transforms.h>
#include "Renderer.hpp"
#include <pcl/filters/voxel_grid.h>
#include <particle/helper_cloud.h>
/*
* TODO
* Define the proper number of particles
*/
#define NPARTICLES 10000
#define circleID "circle_id"
#define reflectorID "reflector_id"

using namespace std;
using namespace lidar_obstacle_detection;


Map map_mille;  
ParticleFilter pf;
bool init_odom=false;
Renderer renderer;
vector<Particle> best_particles;
std::ofstream myfile;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_particles(new pcl::PointCloud<pcl::PointXYZ>);

// Valori ideali: sigma_init [3] = {0.01, 0.01, 0.01} sigma_pos [3]  = {0.18, 0.18, 0.18} sigma_landmark [2] = {0.23, 0.23}
double sigma_init [3] = {0.01, 0.01, 0.01};  //[x,y,theta] initialization noise. 
double sigma_pos [3]  = {0.18, 0.18, 0.18}; //[x,y,theta] movement noise. Try values between [0.5 and 0.01]
double sigma_landmark [2] = {0.23, 0.23};     //[x,y] sensor measurement noise. Try values between [0.5 and 0.1]
std::vector<Color> colors = {Color(1,0,0), Color(1,1,0), Color(0,0,1), Color(1,0,1), Color(0,1,1)};
control_s odom;


// This function updates the position of the particles in the viewer
void showPCstatus(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,vector<Particle> particles){

    for (size_t i = 0; i < particles.size(); ++i) {
        cloud->points[i].x = particles[i].x;
        cloud->points[i].y = particles[i].y;
        
    }
    renderer.updatePointCloud(cloud,"particles");

}

// This function adds the observations from the LiDAR (reflectors) and updates its position
void updateViewerReflector(pcl::PointCloud<pcl::PointXYZI> reflectorCenter){
    for(int i = 0; i < reflectorCenter.size(); i++)
    {
        // Update the pose of the reflectors with respect the best particle
        pcl::PointXYZI pt;
        pt.x = reflectorCenter[i].x;
        pt.y = reflectorCenter[i].y;
        pt.z = 0.0f;
        float gx = pt.x*cos(best_particles.back().theta)-pt.y*sin(best_particles.back().theta)+best_particles.back().x;
        float gy = pt.x*sin(best_particles.back().theta)+pt.y*cos(best_particles.back().theta)+best_particles.back().y;
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() << gx, gy, 0;

        // Apply the transformation
        renderer.updatePose(reflectorID+std::to_string(i),transform);
        renderer.updateShape(reflectorID+std::to_string(i),1.0);
    }
}

// This functions processes the odometry from the forklift (Prediction phase)
void OdomCb(const nav_msgs::Odometry::ConstPtr& msg){
    //static double t_start=0.0, t_end=0.0;
    static std::chrono::time_point<std::chrono::high_resolution_clock> t_start ;
    static std::chrono::time_point<std::chrono::high_resolution_clock> t_end ;
    odom.velocity = msg->twist.twist.linear.x;  
    odom.yawrate = msg->twist.twist.angular.z; 
    //t_start=msg->header.stamp.toSec();
    t_start = std::chrono::high_resolution_clock::now();
    // Prediction phase
    if(!init_odom){
        pf.prediction(0, sigma_pos, odom.velocity, odom.yawrate);
        init_odom=true;
    }else{
        //double delta_t = t_start-t_end;
        double delta_t = (std::chrono::duration<double, std::milli>(t_start-t_end).count())/1000;
        pf.prediction(delta_t, sigma_pos, odom.velocity, odom.yawrate);
    }

    t_end = std::chrono::high_resolution_clock::now();
}

// This functions processes the point cloud (Update phase)
void PointCloudCb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){

    // Define the timers for loggin the execution time
    static std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end ;
    t_start = std::chrono::high_resolution_clock::now();
    
    // Convert to PCL data type
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    // Extract landmarks
    pcl::PointCloud<pcl::PointXYZI> reflectorCenter=extractReflectors(cloud);

    // Let's hide all the reflectors
    for(int i=0;i<nReflectors;i++)
        renderer.updateShape(reflectorID+std::to_string(i),0.0);

    // Update the observations and shows the reflectors (this can be improved, this line can be executed after computing the best particle)
    updateViewerReflector(reflectorCenter);

    // Receive noisy observation data 
    vector<LandmarkObs> noisy_observations;
    for(int i = 0; i < reflectorCenter.size(); i++)
    {
        LandmarkObs obs;
        obs.x = reflectorCenter[i].x;
        obs.y = reflectorCenter[i].y;
        noisy_observations.push_back(obs);
    }

    // Update the weights of the particle 
    pf.updateWeights(sigma_landmark, noisy_observations, map_mille);

    // Resample the particles
    pf.resample();

    // Calculate and output the average weighted error of the particle filter over all time steps so far.
    Particle best_particle;
    vector<Particle> particles = pf.particles;
    double highest_weight = -1.0;
    for (int i = 0; i < particles.size(); ++i) {
        if (particles[i].weight > highest_weight) {
            highest_weight = particles[i].weight;
            best_particle = particles[i];
        }
    }
    best_particles.push_back(best_particle);

    // Show the particles in the map
    showPCstatus(cloud_particles,particles);
    renderer.removeShape(circleID+std::to_string(NPARTICLES+1));
    renderer.addCircle(best_particles.back().x, best_particles.back().y, circleID+std::to_string(NPARTICLES+1), 0.3,1,0,0);

    // Log the execution time
    t_end = std::chrono::high_resolution_clock::now();
    double delta_t = (std::chrono::duration<double, std::milli>(t_end-t_start).count())/1000;
    // Write the results in a file
    myfile<< best_particle.x<<" "<< best_particle.y<<" "<<delta_t<<'\n';

    renderer.SpinViewerOnce();
}


int main(int argc,char **argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudMap (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudReflectors (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    // Create file with the map info
    pcl::io::loadPCDFile ("./data/map_reflector.pcd", *cloudReflectors); // cloud with just the reflectors
    pcl::io::loadPCDFile ("./data/map_pepperl.pcd", *cloudMap); // total cloud (used for rendering)

    remove("./res.txt");
    // This function locates the reflectors within the map and writes into the file
    createMap(cloudReflectors,"./data/map_data.txt",map_mille);

    // Read map data
    if (!read_map_data("./data/map_data.txt", map_mille)) {
        cout << "Error: Could not open map file" << endl;
        return -1;
    } 
 
    // Reduce the number of points in the map point cloud (for improving the performance of the rendering)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_map (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloudMap);
    vg.setLeafSize (1.0f, 1.0f, 1.0f); //Set the voxel grid 	
    vg.filter (*cloud_filtered_map);

    // Starts the rendering
    renderer.InitCamera(CameraAngle::XY);
    // Clear viewer
    renderer.ClearViewer();

    // Render the map and reflectors
    renderer.RenderPointCloud(cloud_filtered_map,"originalCloud",colors[2]);
    renderer.RenderPointCloud(cloudReflectors,"reflectorCloud",colors[0]);

    //Add the reflectors detected by the particles (you can ignore this)
    for(int i=0;i<nReflectors;i++)
        renderer.addCircle(0, 0, reflectorID+std::to_string(i), 0.2,1,1,1);

    // Initial position of the forklift
    double GPS_x = 2.37256; 
    double GPS_y = 1.70077;
    double GPS_theta = -1.68385;

    // Insert one particle in the best particle set
    Particle p(GPS_x,GPS_y,GPS_theta);
    best_particles.push_back(p);
    
    // Init the particle filter
    //pf.init(GPS_x, GPS_y, GPS_theta, sigma_init, NPARTICLES);
    pf.init_random(sigma_init,NPARTICLES);

    // Render all the particles
    for(int i=0;i<NPARTICLES;i++){
        pcl::PointXYZ point;
        point.x = pf.particles[i].x;
        point.y = pf.particles[i].y;
        point.z = 0;
        cloud_particles->push_back(point);
    }   
    renderer.RenderPointCloud(cloud_particles,"particles",colors[0]);

    //render the best initial guess as a circle 
    renderer.addCircle(GPS_x, GPS_y, circleID+std::to_string(NPARTICLES+1), 0.4,0,1,1); 
    renderer.SpinViewerOnce();

    // Start ROS node
    std::cout<<"Map loaded, waiting for the rosbag"<<std::endl;
    myfile.open("./res.txt", std::ios_base::app);

    ros::init(argc, argv, "Particle");
    ros::NodeHandle n;

    //Subscriber 
    ros::Subscriber odom_sub        = n.subscribe<nav_msgs::Odometry>("/auriga_id0_odom",1,&OdomCb); //average rate: 31.438hz
    ros::Subscriber pc_sub          = n.subscribe<sensor_msgs::PointCloud2>("/pepperl_id0_cloud",1,&PointCloudCb); //average rate: 10.728hz
    // To force 10hz replay use: rosbag play --clock --hz=10 out.bag
    ros::spin(); 
    myfile.close();

}

