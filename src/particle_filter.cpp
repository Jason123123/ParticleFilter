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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 200;

  default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  //initialize each particle
  for(int i = 0; i < num_particles; ++i) {
    Particle p;
    //  Sample  and from these normal distrubtions like this:
    //  sample_x = dist_x(gen);
    //  where "gen" is the random engine initialized earlier.

    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }
  //initialize weights
  weights.resize(num_particles);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  // make the random engine static to avoid recreation
  static default_random_engine gen;
  // loop through all particles
  for(int i = 0; i < particles.size(); i++){
    // predict position and velocity without add gaussian noise
    // avoid dividing by zero
    if (fabs(yaw_rate) > 0.001) {
        particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
        particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
        particles[i].theta += yaw_rate * delta_t;
    }
    else {
        particles[i].x += velocity * delta_t * cos(particles[i].theta);
        particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    // add gaussian noise
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
  for(int i = 0; i < observations.size(); i++){
    int idx = -1;
    double min_dist = numeric_limits<double>::max();
    for(int j = 0; j < landmarks.size(); j++){
      double distance = dist(observations[i].x, observations[i].y, landmarks[j].x, landmarks[j].y);
      if(distance < min_dist){
        min_dist = distance;
        idx = landmarks[j].id;
      }
    }
    observations[i].id = idx;
  }
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

  // iterate over all particles
  for(int i = 0; i < num_particles; i++){

    // used the vector container to store landmarks within sensor range
    vector<LandmarkObs> landmarks;
    // push landmarks into the vector
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      // calculate the distance between
      double distance = dist(particles[i].x, particles[i].y, landmark_x, landmark_y);
      // if within sensor range
      // create new landmarkObs and add it to landmark list
      if(distance <= sensor_range){
        landmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    // transform observations from car coordinate system to map system
    vector<LandmarkObs> transformed_obs;
    for(int j = 0; j < observations.size(); j++){
      double x_m = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
      double y_m = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
      transformed_obs.push_back(LandmarkObs{observations[j].id, x_m, y_m});
    }

    //data association
    dataAssociation(landmarks, transformed_obs);
    vector<int> v_association;
    vector<double> v_sense_x;
    vector<double> v_sense_y;
    for(int j = 0; j < transformed_obs.size(); j++){
      v_association.push_back(transformed_obs[j].id);
      v_sense_x.push_back(transformed_obs[j].x);
      v_sense_y.push_back(transformed_obs[j].y);
    }
    SetAssociations(particles[i], v_association, v_sense_x, v_sense_y);

    //initiate weights
    particles[i].weight = 1.0;
    //loop over all observations to update weights
    for(int j = 0; j < transformed_obs.size(); j++){
      //place holder to store mu_x and mu_y
      double mu_x = 0;
      double mu_y = 0;
      double x_obs = transformed_obs[j].x;
      double y_obs = transformed_obs[j].y;
      //find the coorresponding landmark
      for(int k = 0; k < landmarks.size(); k++){
        if(transformed_obs[j].id == landmarks[k].id){
          mu_x = landmarks[k].x;
          mu_y = landmarks[k].y;
        }
      }
      //calculate weights
      double sig_x= std_landmark[0];
      double sig_y= std_landmark[1];

      double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));
      //double exponent = (pow(x_obs - mu_x)**2)/(2 * sig_x**2) + ((y_obs - mu_y)**2)/(2 * sig_y**2)
      double exponent = pow(x_obs - mu_x, 2) / (2 * sig_x * sig_x) + pow(y_obs - mu_y, 2) / (2 * sig_y * sig_y);
      double weight = gauss_norm * exp(-exponent);
      particles[i].weight *= weight;
    }
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  static default_random_engine gen;
  discrete_distribution<> dist_particles(weights.begin(), weights.end());
  vector<Particle> new_particles;
  new_particles.resize(num_particles);
  for (int i = 0; i < num_particles; i++) {
      new_particles[i] = particles[dist_particles(gen)];
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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

