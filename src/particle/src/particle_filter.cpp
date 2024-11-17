#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle/particle_filter.h"
using namespace std;

static  default_random_engine gen;

//Possibili tipi di resampling: str (stratificato),
//sis (sistematico), def (base).
#define RES_TYPE "sis"
//Possibili tipi di data association: NN (nearest neighbors),
//WNN (weighted nearest neighbors)
#define DATA_ASS "NN"
//Parametro di sensibilità della gaussiana
//Un valore piccolo comporta peso maggiore solo ai landmark molto vicini
//Un valore grande considera anche i landmark più lontani
#define SIGMA 10.0


void ParticleFilter::init_random(double std[],int nParticles) {
    num_particles=nParticles;
    normal_distribution<double> dist_x(-std[0],std[0]);
    normal_distribution<double> dist_y(-std[1],std[1]);
    normal_distribution<double> dist_theta(-std[2],std[2]);

    for (int i=0; i<num_particles; i++){
        Particle p;
        p.x=dist_x(gen);
        p.y=dist_y(gen);
        p.theta=dist_theta(gen);
        p.weight=1.0;
        particles.push_back(p);
    }
    is_initialized=true;
    
}


void ParticleFilter::init(double x, double y, double theta, double std[],int nParticles) {
    num_particles = nParticles;
    
    normal_distribution<double> dist_x(-std[0], std[0]); 
    normal_distribution<double> dist_y(-std[1], std[1]);
    normal_distribution<double> dist_theta(-std[2], std[2]);

    for (int i=0; i<num_particles; i++){
        Particle p;
        p.x=dist_x(gen)+x;
        p.y=dist_y(gen)+y;
        p.theta=dist_theta(gen)+theta;
        particles.push_back(p);
    }
    is_initialized=true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for (auto &particle :particles)
    {
        if (fabs(yaw_rate) < 0.00001) {
            particle.x += velocity * delta_t * cos(particle.theta);
            particle.y += velocity * delta_t * sin(particle.theta); 
            }
        else{
            particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y += velocity / yaw_rate * (cos(particle.theta)- cos(particle.theta + yaw_rate*delta_t));
            particle.theta+=yaw_rate*delta_t;
        } 
    
          
        normal_distribution<double> dist_x(0, std_pos[0]); 
        normal_distribution<double> dist_y(0, std_pos[1]);
        normal_distribution<double> dist_theta(0, std_pos[2]);
        particle.x+=dist_x(gen);
        particle.y+=dist_y(gen);
        particle.theta+=dist_theta(gen);
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> mapLandmark, std::vector<LandmarkObs>& observations) {
    if (DATA_ASS=="NN"){
        //Nearest Neighbor data association
        for(auto &obs:observations){
            double min_dist=numeric_limits<double>::max();
            int nearest_id=-1;

            for (const auto &landmark:mapLandmark){
                double distance=pow(obs.x - landmark.x, 2) + pow(obs.y - landmark.y, 2);
                if (distance<min_dist){
                    min_dist=distance;
                    nearest_id=landmark.id;
                }
            }
            obs.id=nearest_id;
        }
    }
    else if (DATA_ASS=="WNN")
    {
        //Weighted Nearest Neighbor data association
        for (auto &obs : observations) {
            double max_weight = 0.0; 
            int nearest_id = -1;     

            for (const auto &landmark : mapLandmark) {
                double distance = pow(obs.x - landmark.x, 2) + pow(obs.y - landmark.y, 2);
                
                double weight = exp(-distance / (2.0 * SIGMA * SIGMA));

                if (weight > max_weight) {
                    max_weight = weight;
                    nearest_id = landmark.id;
                }
            }
            obs.id = nearest_id;
        }
    }
    else{
        throw std::invalid_argument("Tipo di data association non valido. Valori validi: NN e WNN.");
    }
    
}

LandmarkObs transformation(LandmarkObs observation, Particle p){
    LandmarkObs global;
    
    global.id = observation.id;
    global.x = p.x+observation.x*cos(p.theta)-observation.y*sin(p.theta); 
    //TODO
    global.y = p.y + sin(p.theta)*observation.x+cos(p.theta)*observation.y; 
    //TODO

    return global;
}

void ParticleFilter::updateWeights(double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    std::vector<LandmarkObs> mapLandmark;
    for(int j=0;j<map_landmarks.landmark_list.size();j++){
        mapLandmark.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f});
    }
    for(auto& particle:particles){

        std::vector<LandmarkObs> transformed_observations;
        for (const auto&obs:observations){
            transformed_observations.push_back(transformation(obs,particle));
        }
        dataAssociation(mapLandmark,transformed_observations);
        particle.weight = 1.0;

        for(int k=0;k<transformed_observations.size();k++){
            double obs_x,obs_y,l_x,l_y;
            obs_x = transformed_observations[k].x;
            obs_y = transformed_observations[k].y;
            for (int p = 0; p < mapLandmark.size(); p++) {
                if (transformed_observations[k].id == mapLandmark[p].id) {
                    l_x = mapLandmark[p].x;
                    l_y = mapLandmark[p].y;
                }
            }	
            double w = exp( -( pow(l_x-obs_x,2)/(2*pow(std_landmark[0],2)) + pow(l_y-obs_y,2)/(2*pow(std_landmark[1],2)) ) ) / ( 2*M_PI*std_landmark[0]*std_landmark[1] );
            particle.weight *= w;
        }
    }
    
}


void ParticleFilter::resample() {
    vector<Particle> new_particles;
    
    
    vector<double> weights;
    
    for (const auto& particle : particles) {
        weights.push_back(particle.weight);
    }

    uniform_int_distribution<int> dist_index(0,num_particles-1);
    
    //Normalizzazione pesi
    double total_weight = accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& weight : weights) {
        weight /= total_weight;
    }

    if (RES_TYPE=="def"){
        //Resampling base:
        int index = dist_index(gen);
        double beta  = 0.0;
        double max_w = *max_element(weights.begin(), weights.end());
        uniform_real_distribution<double> dist_beta(0.0, 2.0*max_w);

        for(int i=0;i<num_particles;i++){
            beta+=dist_beta(gen);
            while(beta>weights[index]){
                beta-=weights[index];
                index=(index+1)%num_particles;
            }
            new_particles.push_back(particles[index]);
        }
        particles=new_particles;
    }
    else if (RES_TYPE=="sis")
    {
        //Resampling sistematico
        vector<int> integer_copies(num_particles, 0);
        vector<double> residual_weights(num_particles, 0.0);

        int num_copied = 0;
        for (int i = 0; i < num_particles; i++) {
            integer_copies[i] = floor(weights[i] * num_particles);
            residual_weights[i] = (weights[i] * num_particles) - integer_copies[i];
            num_copied += integer_copies[i];
        }

        for (int i = 0; i < num_particles; i++) {
            for (int j = 0; j < integer_copies[i]; j++) {
                new_particles.push_back(particles[i]);
            }
        }

        vector<double> normalized_residuals;
        for (int i = 0; i < num_particles; i++) {
            normalized_residuals.push_back(residual_weights[i] / (num_particles - num_copied));
        }

        vector<double> cumulative_weights(normalized_residuals.size(), 0.0);
        partial_sum(normalized_residuals.begin(), normalized_residuals.end(), cumulative_weights.begin());
        uniform_real_distribution<double> dist_random(0.0, 1.0 / (num_particles-num_copied));
        double random_offset = dist_random(gen);
        double step = 1.0 / (num_particles - num_copied);
        
        int index=0;
        for (int i = 0; i < (num_particles - num_copied); i++) {
            double threshold = random_offset + i * step;
            while (threshold > cumulative_weights[index]) {
                index++;
            }
            new_particles.push_back(particles[index]);
        }
        particles=new_particles;
    }
    else if (RES_TYPE=="str"){
        //Resampling stratificato

        vector<double> cumulative_weights(weights.size(), 0.0);
        partial_sum(weights.begin(), weights.end(), cumulative_weights.begin());
        uniform_real_distribution<double> dist_random(0.0, 1.0 / num_particles);

        for (int i = 0; i < num_particles; i++) {
            double random_offset = dist_random(gen);
            double threshold = (i + random_offset) / num_particles;

            auto it = lower_bound(cumulative_weights.begin(), cumulative_weights.end(), threshold);
            int index = it - cumulative_weights.begin();
            new_particles.push_back(particles[index]);
        }
        particles=new_particles;
    }
    else{
        throw std::invalid_argument("Tipo di resampling non valido. Valori validi: def, sis, str.");
    }
}
    
    
    
    