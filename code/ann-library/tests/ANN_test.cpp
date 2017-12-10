#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "ANN.h"

TEST(ANN, can_construct)
{
	ANN simple_ann;
}

TEST(ANN, can_construct_with_params)
{
	int w = 10, h = 10, s = 50, k = 10;
	ANN simple_ann(w * h, s, k, ANN::INIT::ZERO);
}

TEST(ANN, can_set_get_learning_rate)
{
	ANN simple_ann;
	simple_ann.set_learning_rate(0.5);
	EXPECT_EQ(simple_ann.get_learning_rate(), 0.5);
}             	

TEST(ANN, can_compute_zero_output)
{
	int w = 1, h = 1, s = 1, k = 1;
	ANN simple_ann(w * h, s, k, ANN::INIT::ZERO);
	auto output = simple_ann.compute({0});
	EXPECT_EQ(output, vector <double> {1.0});	
}

TEST(ANN, is_computed_output_has_required_dimension)
{
	int w = 2, h = 1, s = 1, k = 2;
	ANN simple_ann(w * h, s, k, ANN::INIT::ZERO);
	auto output = simple_ann.compute({0, 1});
	EXPECT_EQ(output.size(), 2);	
}

TEST(ANN, is_computation_input_has_required_dimension)
{
	int w = 2, h = 1, s = 1, k = 1;
	ANN simple_ann(w * h, s, k, ANN::INIT::ZERO);
	EXPECT_THROW(simple_ann.compute({0}), std::length_error);
}

TEST(ANN, is_computation_correct_for_simple_net)
{
	int w = 2, h = 1, s = 1, k = 2;
	ANN simple_ann(w * h, s, k, ANN::INIT::ZERO);
	auto output = simple_ann.compute({0, 1});
	EXPECT_EQ(output, vector <double> ({0.5, 0.5}));
}

TEST(ANN, can_construct_from_weights)
{
	int w = 2, h = 1, s = 3, k = 4;
	std::vector <double> weights_first_layer (6, 0), weights_second_layer (12, 0);
	ANN simple_ann(w * h, s, k, ANN::INIT::WEIGHTS, weights_first_layer, weights_second_layer);
}

TEST(ANN, cannot_construct_from_weights_with_bad_length)
{
	int w = 2, h = 1, s = 1, k = 1;
	std::vector <double> weights_first_layer = {}, weights_second_layer = {};
	EXPECT_THROW(ANN (w * h, s, k, ANN::INIT::WEIGHTS, weights_first_layer, weights_second_layer), std::length_error);
}

TEST(ANN, can_construct_from_gauss_distribution)
{
	int w = 1, h = 1, s = 1, k = 1;
	ANN ann_simple(w * h, s, k, ANN::INIT::GAUSS);
}

TEST(ANN, can_construct_from_gauss_distribution_sigma)
{
	int w = 1, h = 1, s = 1, k = 1;
	ANN ann_simple(w * h, s, k, ANN::INIT::GAUSS, 2.0);
}

TEST(ANN, can_construct_from_uniform_distribution)
{
	int w = 1, h = 1, s = 1, k = 1;
	ANN ann_simple(w * h, s, k, ANN::INIT::UNIFORM);
}

TEST(ANN, is_computation_correct_for_complex_net)
{
	int w = 2, h = 1, s = 3, k = 4;
	std::vector <double> weights_first_layer = {1, 2, 0, 1, 1, 3}, 
						 weights_second_layer = {0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 1, 0};
	ANN simple_ann(w * h, s, k, ANN::INIT::WEIGHTS, weights_first_layer, weights_second_layer);
	vector <double> input = {2, 1};
	vector <double> expected_output = {0.00626323, 0.017004, 0.971798, 0.00493514}; //{0.0000166958, 0.000335344, 0.999648, 3.05795e-7}; //{5, 8, 16, 1};
	vector <double> output = simple_ann.compute(input);
	for (size_t i = 0; i < expected_output.size(); i++)
		EXPECT_NEAR(output[i], expected_output[i], 1e-6);
}

TEST(ANN, is_train_net_good)
{
	int w = 1, h = 1, s = 2, k = 2;
	ANN simple_ann(w * h, s, k, ANN::INIT::ZERO);
	vector <double> input = {1};
	vector <double> target = {0, 1};
	simple_ann.compute(input);
	simple_ann.train(target);
	vector <double> output = simple_ann.compute(input);
}