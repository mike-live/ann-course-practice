#include "ANN.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <random>

using std::vector;
using std::cout;
using std::string;
using std::ostream;
using std::pair;
using std::endl;
using std::max;
using std::fstream;
using std::ios;

#include <climits>

namespace std {
	template <typename T>
	T swap_endian(T u)
	{
	    static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");
	
	    union
	    {
	        T u;
	        unsigned char u8[sizeof(T)];
	    } source, dest;
	
	    source.u = u;
	
	    for (size_t k = 0; k < sizeof(T); k++)
		    dest.u8[k] = source.u8[sizeof(T) - k - 1];

	    return dest.u;
	}
};

class {
public:
	size_t num_epoch = 20, num_hidden_layers = 300;
	double learning_rate = 0.008;
	size_t width, height, num_labels = 10;
	vector <pair <vector <double>, vector <double>>> train_data;
	vector <pair <vector <double>, vector <double>>> test_data;
	string bin_path;
//	string path_data = "../../../../download/";
	string path_data = "../download/";
	void init(int argc, char ** argv) {
		bin_path = argv[0];
		cout << bin_path << endl;
		size_t pos1 = bin_path.find_last_of('/'), pos2 = bin_path.find_last_of('\\');
		size_t pos = pos1 == string::npos ? pos2 : (pos2 == string::npos ? pos1 : max(pos1, pos2));
		if (pos != string::npos) {
			bin_path = bin_path.substr(0, pos);
		}
		cout << bin_path << endl;
		path_data = bin_path + "/" + path_data;
		for (int i = 0; i < argc; i++) {
			string arg = string(argv[i]);
			size_t pos = arg.find('=');
			if (pos != string::npos) {
				arg.substr(0, pos);
			}
		}
		read();
	}
	
	struct file_labels
	{
		unsigned int magic, num_items;
		void swap_endian ()
		{
			magic = std::swap_endian(magic);
			num_items = std::swap_endian(num_items);
		}
	};

	struct file_images
	{
		unsigned int magic, num_items, height, width;
		void swap_endian ()
		{
			magic = std::swap_endian(magic);
			num_items = std::swap_endian(num_items);
			height = std::swap_endian(height);
			width = std::swap_endian(width);
		}
	};

	void read_dataset(string path_images, string path_labels, vector <pair <vector <double>, vector <double>>> & data)
	{
		FILE * bu = fopen(path_labels.c_str(), "rb");
		cout << path_labels << endl;
		file_labels fl;
		fread (&fl, sizeof(fl), 1, bu);
		fl.swap_endian();
		unsigned char * labels = new unsigned char[fl.num_items];
		fread (labels, 1, fl.num_items, bu);
		fclose(bu);
		
		bu = fopen(path_images.c_str(), "rb");
		file_images fi;
		fread (&fi, sizeof(fi), 1, bu);
		fi.swap_endian();
		unsigned char * images = new unsigned char[fi.num_items * fi.width * fi.height];
		fread (images, 1, fi.num_items * fi.width * fi.height, bu);
		fclose(bu);

		data.resize(fl.num_items);
		for (size_t i = 0; i < data.size(); i++) {
			vector <double> label (num_labels, 0.0), image(fi.width * fi.height);
			for (size_t j = 0; j < image.size(); j++) {
				image[j] = images[i * image.size() + j];
			}
			label[labels[i]] = 1.0;
			data[i] = {image, label};
		}
		width = fi.width;
		height = fi.height;
		cout << (data).size() << " " << data[0].first.size() << " " << data[0].second.size() << endl;
	}

	void read()
	{
		read_dataset(path_data + "train-images.idx3-ubyte", path_data + "train-labels.idx1-ubyte", train_data);
		read_dataset(path_data + "t10k-images.idx3-ubyte", path_data + "t10k-labels.idx1-ubyte", test_data);
	}
	
	void calc()
	{
		ANN ann_mnist(width * height, num_hidden_layers, num_labels, ANN::INIT::GAUSS, 0.0001);
		ann_mnist.set_learning_rate(learning_rate);
		clock_t be_calc = clock();
		vector <size_t> train_permutation(train_data.size());
		for (size_t i = 0; i < train_permutation.size(); i++) {
			train_permutation[i] = i;
		}
		std::random_device rd;
		std::mt19937 gen(rd());

		for (size_t epoch = 0; epoch < num_epoch; epoch++) {
			clock_t be_epoch = clock();
			std::shuffle(train_permutation.begin(), train_permutation.end(), gen);
			for (auto id : train_permutation) {
				auto & data = train_data[id];
				ann_mnist.compute(data.first);
				ann_mnist.train(data.second);
			}
			double accuracy = 0.0;
			for (auto data : test_data) {
				ann_mnist.compute(data.first);
				accuracy += ann_mnist.check(data.second);
			}	
			accuracy /= test_data.size();
			log_to_file(cout, epoch, accuracy, clock() - be_epoch, clock() - be_calc);
		}
	}

	void log_to_file(ostream & fout, size_t epoch, double accuracy, clock_t epoch_time, clock_t time_all)
	{
		fout << "Epoch: " << epoch << " Test accuracy: " << accuracy << " ";
		fout << "Calculation epoch time: " << epoch_time / CLOCKS_PER_SEC << " sec. ";
		fout << "All calculation time: " << time_all / CLOCKS_PER_SEC << " sec." << endl;
	}
} MNIST;

int main (int argc, char ** argv)
{
	MNIST.init(argc, argv);
	MNIST.calc();    
	return 0;
}