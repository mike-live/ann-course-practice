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
	double learning_rate = 0.008, sigma = 0.01;
	size_t width, height, num_labels = 10;
	vector <pair <vector <double>, vector <double>>> train_data;
	vector <pair <vector <double>, vector <double>>> test_data;
	string bin_path;
//	string path_data = "../../../../download/";
	string path_data = "../download/";
	bool is_relative_path = true;

	void init(int argc, char ** argv) {
		for (int i = 0; i < argc; i++) {
			string arg = string(argv[i]);
			size_t pos = arg.find('=');
			if (pos != string::npos) {
				string name = (arg.substr(0, pos));
				string param = arg.substr(pos + 1);
				if ((name).size() > 2 && name == string("learning_rate").substr(0, (name).size())) {
					learning_rate = stod(param);
				} else 
				if ((name).size() > 2 && name == string("num_epoch").substr(0, (name).size())) {
					num_epoch = stol(param);
				} else 
				if ((name).size() > 2 && name == string("num_hidden_layers").substr(0, (name).size())) {
					num_hidden_layers = stol(param);
				} else 
				if ((name).size() > 2 && name == string("sigma").substr(0, (name).size())) {
					sigma = stod(param);
				} else
				if ((name).size() > 2 && name == string("images_relative_dir").substr(0, (name).size())) {
					path_data = param;
					is_relative_path = true;
				} else
				if ((name).size() > 2 && name == string("images_dir").substr(0, (name).size())) {
					path_data = param;
					is_relative_path = false;
				} else {}
			}
		}
		if (is_relative_path) {
			bin_path = argv[0];
			size_t pos1 = bin_path.find_last_of('/'), pos2 = bin_path.find_last_of('\\');
			size_t pos = pos1 == string::npos ? pos2 : (pos2 == string::npos ? pos1 : max(pos1, pos2));
			if (pos != string::npos) {
				bin_path = bin_path.substr(0, pos);
			}
			path_data = bin_path + "/" + path_data;
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
		cout << "Path to labels: " << path_labels << endl;
		cout << "Path to images: " << path_images << endl;

		FILE * bu = fopen(path_labels.c_str(), "rb");
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
				image[j] = images[i * image.size() + j] / 255.0;
			}
			label[labels[i]] = 1.0;
			data[i] = {image, label};
		}
		width = fi.width;
		height = fi.height;

		cout << "Size: " << (data).size() << " Width: " << width << " Height: " << height << endl;
	}

	void read()
	{
		cout << "MNIST." << endl;
		cout << "Train data set:" << endl;
		read_dataset(path_data + "train-images.idx3-ubyte", path_data + "train-labels.idx1-ubyte", train_data);
		cout << "Test data set:" << endl;
		read_dataset(path_data + "t10k-images.idx3-ubyte", path_data + "t10k-labels.idx1-ubyte", test_data);
	}
	
	void calc()
	{
		ANN ann_mnist(width * height, num_hidden_layers, num_labels, ANN::INIT::GAUSS, sigma);
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
/*				printf("%d\n", id);
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < height; j++) {
						printf("%d ", data.first[i * height + j] > 0);
					}
					printf("\n");
				}*/
				ann_mnist.compute(data.first);
				ann_mnist.train(data.second);
//				ann_mnist.compute(data.first);
			}
			double accuracy_train = 0.0;
			for (auto data : train_data) {
				ann_mnist.compute(data.first);
				accuracy_train += ann_mnist.check(data.second);
			}	
			accuracy_train /= train_data.size();
			double accuracy_test = 0.0;
			for (auto data : test_data) {
				ann_mnist.compute(data.first);
				accuracy_test += ann_mnist.check(data.second);
			}	
			accuracy_test /= test_data.size();
			log_to_file(cout, epoch, accuracy_test, accuracy_train, clock() - be_epoch, clock() - be_calc);
		}
	}

	void log_to_file(ostream & fout, size_t epoch, double accuracy_test, double accuracy_train, clock_t epoch_time, clock_t time_all)
	{
		fout << "Epoch: " << epoch << " ";
		fout << "Test accuracy: " << accuracy_test << " ";
		fout << "Train accuracy: " << accuracy_train << " ";
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