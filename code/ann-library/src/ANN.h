#pragma once

#include <vector>
#include <stdexcept>
#include <random>

using std::vector;

class ANN
{
public:
	enum INIT {ZERO, UNIFORM, GAUSS, WEIGHTS};
	ANN() { 
		init_weights(INIT::ZERO);
	}
	ANN(size_t n, size_t s, size_t k, INIT init, 
		vector <double> weights_first_layer = {}, 
		vector <double> weights_second_layer = {}) 
		: n(n), s(s), k(k), weights_first_layer(weights_first_layer), 
		weights_second_layer(weights_second_layer) { 
		init_weights(init);	
	}

	ANN(size_t n, size_t s, size_t k, INIT init, double sigma) 
		: n(n), s(s), k(k), weights_first_layer(weights_first_layer), 
		weights_second_layer(weights_second_layer) { 
		init_weights(init, sigma);	
	}

	void set_learning_rate(double learning_rate_) { learning_rate = learning_rate_; }
	double get_learning_rate() { return learning_rate; }

	vector <double> compute(const vector <double> & input_) {
		input = input_;
		if (input.size() != n) throw std::length_error("Input hasn't length as number neurons of input layer");
		mult(input, weights_first_layer, layer);
		for (auto & q : layer) {
			q = tanh(q);
		}
		mult(layer, weights_second_layer, output);
		double sum = 0.0;
		for (auto & q : output) {
			q = exp(q);
			sum += q;
		}
		for (auto & q : output) {
			q /= sum;
		}
		return output;
	}

	void train(const vector <double> & target)
	{
		compute_gradients(target);
		update_weights(target);
	}

	double compute_error(const vector<double> & target)
	{
		double err = 0;
		for (size_t i = 0; i < target.size(); i++) {
			err += target[i] * log(output[i]) + (1 - target[i]) * log(1 - output[i]);
		}
		return -err;
	}
    
private:
	size_t n = 100, s = 50, k = 10;
	double learning_rate = 1.0;
	vector <double> weights_first_layer, weights_second_layer;
	vector <double> input, layer, output;
	vector <double> layer_gradients, output_gradients;

	void mult(const vector <double> & x, const vector <double> & A, vector <double> & y)
	{
		y.resize(A.size() / x.size());
		for (size_t i = 0; i < y.size(); i++) {
			y[i] = 0;
		}
		for (size_t i = 0; i < A.size(); i++) {
			y[i / x.size()] += A[i] * x[i % x.size()];
		}
	}

	void init_weights(INIT init, double sigma = 1.0)
	{
		if (init != INIT::WEIGHTS) {
			weights_first_layer.resize(n * s, 0);
			weights_second_layer.resize(s * k, 0);
		}
		if (weights_first_layer.size() != n * s) throw std::length_error("First layer hasn't length as number neurons pairs between first and second layers");
		if (weights_second_layer.size() != s * k) throw std::length_error("Second layer hasn't length as number neurons pairs between second and third layers");

		std::random_device rd;
		std::mt19937 gen(rd());

		switch (init) {
			case INIT::ZERO:
				break;
			case INIT::UNIFORM: 
			{

				std::uniform_real_distribution<double> distr_uniform(0, 1);
				for (auto & weight : weights_first_layer) {
					weight = distr_uniform(gen);
				}
				for (auto & weight : weights_second_layer) {
					weight = distr_uniform(gen);
				}
				break;
			}
			case INIT::GAUSS:
			{
				std::normal_distribution<double> distr_gauss(0, sigma);
				for (auto & weight : weights_first_layer) {
					weight = distr_gauss(gen);
				}
				for (auto & weight : weights_second_layer) {
					weight = distr_gauss(gen);
				}
				break;
			}
			case INIT::WEIGHTS:
				break;
			default:
				break;
		}
	}

	void compute_gradients(const vector<double> & target)
	{
		output_gradients.resize(k, 0);
		for (size_t i = 0; i < k; i++) {
			output_gradients[i] = target[i] - output[i];
		}
		layer_gradients.resize(s, 0);
		for (size_t i = 0; i < s; i++) {
			double derivative = (1 - layer[i]) * (1 + layer[i]);
			double sum = 0.0;
			for (size_t j = 0; j < k; j++) {
				sum += layer_gradients[j] * weights_second_layer[i * k + j];
			}
			layer_gradients[i] = derivative * sum;
		}
	}

	void update_weights(const vector<double> & target)
	{
		for (size_t j = 0; j < s; j++) {
			for (size_t i = 0; i < n; i++) {
				double delta = learning_rate * layer_gradients[j] * input[i];
				weights_first_layer[j * n + i] += delta;
			}
		}
		for (size_t j = 0; j < k; j++) {
			for (size_t i = 0; i < s; i++) {
				double delta = learning_rate * output_gradients[j] * layer[i];
				weights_second_layer[j * s + i] += delta;
			}
		}	
	}

};