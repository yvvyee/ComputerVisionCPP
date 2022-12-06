#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
	// libtorch 동작 테스트
	at::Tensor tensor = torch::rand({ 2,3 });
	cout << tensor << endl;

	string pt = argv[1];
	string img = argv[2];

	torch::jit::script::Module yolov7;
	try {
		yolov7 = torch::jit::load(pt);
		cout << "yolov7 model loaded" << endl;
	}
	catch (const c10::Error& e) {
		cerr << "cannot load the weight file" << endl;
		cerr << e.backtrace() << endl;
		return -1;
	}

	
	
	return 0;
}