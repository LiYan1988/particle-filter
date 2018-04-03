/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

int ParticleFilter::getNumParticles() {
    return num_particles;
}

void ParticleFilter::init(double x, double y, double theta, double sigma_pos[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	double std_x        = sigma_pos[0];
	double std_y        = sigma_pos[1];
	double std_theta    = sigma_pos[2];

    default_random_engine gen;
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i=0; i<num_particles; i++) {
        Particle p;
        p.id        = i;
        p.x         = dist_x(gen);
        p.y         = dist_y(gen);
        p.theta     = dist_theta(gen);
        p.weight    = 1;
        particles.push_back(p);
        weights.push_back(1);
//        std::cout<<"Particle "<<p.id<<", x="<<p.x<<", y="<<p.y<<", theta="<<p.theta<<", weight="<<p.weight<<std::endl;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

//	std::cout<<"prediction begin"<<std::endl;
	double std_x        = std_pos[0];
	double std_y        = std_pos[1];
	double std_theta    = std_pos[2];

    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for (int i=0; i<num_particles; i++) {
        double x        = particles[i].x;
        double y        = particles[i].y;
        double theta    = particles[i].theta;

        double x_new;
        double y_new;
        double theta_new;

        if (abs(yaw_rate)>=0.0001) {
            x_new        = x + velocity * (sin(theta + yaw_rate * delta_t) - sin(theta)) / yaw_rate + dist_x(gen);
            y_new        = y + velocity * (cos(theta) - cos(theta + yaw_rate * delta_t)) / yaw_rate + dist_y(gen);
            theta_new    = theta + yaw_rate * delta_t + dist_theta(gen);
        }
        else {
            x_new        = x + velocity * cos(theta) * delta_t + dist_x(gen);
            y_new        = y + velocity * sin(theta) * delta_t + dist_y(gen);
            theta_new    = theta + dist_theta(gen);
        }


        particles[i].x      = x_new;
        particles[i].y      = y_new;
        particles[i].theta  = theta_new;
//        std::cout<<"Particle "<<particles[i].id<<", x="<<particles[i].x<<", y="<<particles[i].y<<", theta="<<particles[i].theta<<", weight="<<particles[i].weight<<std::endl;
    }
//    std::cout<<"prediction finish"<<std::endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

//	std::cout<<"association begin"<<std::endl;
    // for each observation
    for (int i=0; i<observations.size(); i++) {
        // compare with each predicted observation
        double x_observed       = observations[i].x;
        double y_observed       = observations[i].y;
        int id_matched          = 0; // this is the index in the predicted vector
        double x_predicted      = predicted[0].x;
        double y_predicted      = predicted[0].y;
        double dist_min         = sqrt(pow(x_observed - x_predicted, 2) + pow(y_observed - y_predicted, 2));
        double dist_tmp;
        for (int j=1; j<predicted.size(); j++) {
            // calculate the distance, and find the closest one
            x_predicted  = predicted[j].x;
            y_predicted  = predicted[j].y;
            dist_tmp     = sqrt(pow(x_observed - x_predicted, 2) + pow(y_observed - y_predicted, 2));
            if (dist_tmp < dist_min) {
                dist_min = dist_tmp;
                id_matched = j;
            }
        }
        observations[i].id = id_matched;
//        std::cout<<"observation ("<<x_observed<<", "<<y_observed<<") matches with landmark ("<<predicted[id_matched].x<<", "<<predicted[id_matched].y<<")"<<endl;
    }
//    std::cout<<"association finish"<<std::endl;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

//	std::cout<<"update weight begin"<<std::endl;

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
    double weights_sum = 0;

//    std::cout<<"Map data: ";
//    for (int i=0; i<map_landmarks.landmark_list.size(); i++) {
//        std::cout<<"id: "<<map_landmarks.landmark_list[i].id_i<<", x: "<<map_landmarks.landmark_list[i].x_f<<", y: "<<map_landmarks.landmark_list[i].y_f<<std::endl;
//    }
//    std::cout<<std::endl;
//    std::cout<<"#observations="<<observations.size()<<std::endl;
	// for each particle
	for (int i=0; i<num_particles; i++) {
        double x_particle       = particles[i].x;
        double y_particle       = particles[i].y;
        double theta_particle   = particles[i].theta;

        // for each observation, convert to map coordinate
        std::vector<LandmarkObs> observations_transformed;
        for (int j=0; j<observations.size(); j++) {
            LandmarkObs obs;
            double x_obs_car    = observations[j].x;
            double y_obs_car    = observations[j].y;

            obs.x   = x_particle + cos(theta_particle) * x_obs_car - sin(theta_particle) * y_obs_car;
            obs.y   = y_particle + sin(theta_particle) * x_obs_car + cos(theta_particle) * y_obs_car;
            obs.id  = observations[j].id;
            observations_transformed.push_back(obs);
        }

        // construct a vector of predicted landmarks within the sensor range
        std::vector<LandmarkObs> predicted;
        for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
            int id_landmark     = map_landmarks.landmark_list[j].id_i;
            double x_landmark   = map_landmarks.landmark_list[j].x_f;
            double y_landmark   = map_landmarks.landmark_list[j].y_f;
            double dist         = sqrt(pow(x_particle-x_landmark, 2) + pow(y_particle-y_landmark, 2));
            if (dist < sensor_range) {
                LandmarkObs landmark_tmp;
                landmark_tmp.id     = id_landmark;
                landmark_tmp.x      = x_landmark;
                landmark_tmp.y      = y_landmark;
                predicted.push_back(landmark_tmp);
            }
        }

//        for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
//            int id_landmark     = map_landmarks.landmark_list[j].id_i;
//            double x_landmark   = map_landmarks.landmark_list[j].x_f;
//            double y_landmark   = map_landmarks.landmark_list[j].y_f;
//            LandmarkObs landmark_tmp;
//            landmark_tmp.id     = id_landmark;
//            landmark_tmp.x      = x_landmark;
//            landmark_tmp.y      = y_landmark;
//            predicted.push_back(landmark_tmp);
//        }

        // data association
//        cout<<"Particle "<<i<<endl;
        dataAssociation(predicted, observations_transformed);

        // update weight
//        std::cout<<"#observations_transformed="<<observations_transformed.size()<<std::endl;
        double prob = 1;
        for (int j=0; j<observations_transformed.size(); j++) {
            int id_landmark         = observations_transformed[j].id;
            double x_observation    = observations_transformed[j].x;
            double y_observation    = observations_transformed[j].y;
            double x_landmark       = predicted[id_landmark].x;
            double y_landmark       = predicted[id_landmark].y;
            double exponent = -(pow(x_observation-x_landmark, 2) / (2 * pow(std_x, 2)) + pow(y_observation-y_landmark, 2) / (2 * pow(std_y, 2)));
//            std::cout<<exponent<<endl;
            prob *= exp(exponent) / (2 * M_PI * std_x * std_y);
        }
        particles[i].weight = prob;
        weights_sum += prob;
	}

//    cout<<"weights: \n";
	for (int i=0; i<num_particles; i++) {
        particles[i].weight /= weights_sum;
        weights[i] = particles[i].weight;
//        cout<<weights[i]<<", ";
	}
//	cout<<endl;
//    std::cout<<"update weight finish"<<std::endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
//    std::cout<<"resample begin"<<std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());

//    std::vector<double> weights;
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());

//    std::cout<<"weights: ";
//    for (int i=0; i<num_particles; i++) std::cout<<weights[i]<<", ";
//    std::cout<<endl;

    std::vector<Particle> particles_new;
    for (int i=0; i<num_particles; i++) {
        int index = distribution(gen);
        particles_new.push_back(particles[index]);
//        std::cout<<"sampled index "<<index;
    }
//    std::cout<<endl;

    particles = particles_new;
//    std::cout<<"resample finish"<<std::endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
