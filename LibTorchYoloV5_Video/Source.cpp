#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

enum Det {
	tl_x = 0,
	tl_y = 1,
	br_x = 2,
	br_y = 3,
	score = 4,
	class_idx = 5
};

struct Detection {
	cv::Rect bbox;
	float score;
	int class_idx;
};

// YOLO 좌표를 OpenCV 좌표 형태로 변환
torch::Tensor xywh2xyxy(const torch::Tensor& x) {
	auto y = torch::zeros_like(x);
	// convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
	y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
	y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
	y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
	y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
	return y;
}

void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
	const at::TensorAccessor<float, 2>& det,
	std::vector<cv::Rect>& offset_box_vec,
	std::vector<float>& score_vec) {
	for (int i = 0; i < offset_boxes.size(0); i++) {
		offset_box_vec.emplace_back(
			cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
				cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
		);
		score_vec.emplace_back(det[i][Det::score]);
	}
}

// 좌표를 원본 이미지의 좌표로 재계산
void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
	float scale, const cv::Size& img_shape) {
	auto clip = [](float n, float lower, float upper) {
		return std::max(lower, std::min(n, upper));
	};

	std::vector<Detection> detections;
	for (auto& i : data) {
		float x1 = (i.bbox.tl().x - pad_w) / scale;  // x padding
		float y1 = (i.bbox.tl().y - pad_h) / scale;  // y padding
		float x2 = (i.bbox.br().x - pad_w) / scale;  // x padding
		float y2 = (i.bbox.br().y - pad_h) / scale;  // y padding

		x1 = clip(x1, 0, img_shape.width);
		y1 = clip(y1, 0, img_shape.height);
		x2 = clip(x2, 0, img_shape.width);
		y2 = clip(y2, 0, img_shape.height);

		i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
	}
}

// 이미지 크기 재설정 후 차이값을 계산
std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
	auto in_h = static_cast<float>(src.rows);
	auto in_w = static_cast<float>(src.cols);
	float out_h = out_size.height;
	float out_w = out_size.width;

	float scale = std::min(out_w / in_w, out_h / in_h);

	int mid_h = static_cast<int>(in_h * scale);
	int mid_w = static_cast<int>(in_w * scale);

	cv::resize(src, dst, cv::Size(mid_w, mid_h));

	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
	int left = (static_cast<int>(out_w) - mid_w) / 2;
	int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
	return pad_info;
}

int main(int argc, char** argv) {

	if (argc < 3) {
		cout << "Weight file and Image file are needed" << endl;
		return -1;
	}

	float conf_thres = 0.25; // 검출 정확도 기준
	float iou_thres = 0.45;  // BBox 정확도 기준

	// 클래스 이름
	std::vector<std::string> class_names = {
		"3_Diamond",
		"3_Heart",
		"4_Spade",
		"4_Clover",
		"4_Diamond",
		"4_Heart",
		"5_Spade",
		"5_Clover",
		"5_Diamond",
		"5_Heart",
		"6_Spade",
		"6_Clover",
		"6_Diamond",
		"6_Heart",
		"7_Spade",
		"7_Clover",
		"7_Diamond",
		"7_Heart",
		"8_Spade",
		"8_Clover",
		"8_Diamond",
		"8_Heart",
		"9_Spade",
		"9_Clover",
		"9_Diamond",
		"9_Heart",
		"10_Spade",
		"10_Clover",
		"10_Diamond",
		"10_Heart",
		"2_Spade",
		"2_Clover",
		"2_Diamond",
		"2_Heart",
		"3_Spade",
		"3_Clover"
	};

	torch::jit::script::Module model; // 파이토치 모델 객체
	try {
		model = torch::jit::load(argv[1]); // 첫번째 인자는 weight 파일 경로
		cout << "YOLO model loaded" << endl;
	}
	catch (const c10::Error& e) {
		cerr << "cannot load the weight file" << endl;
		cerr << e.backtrace() << endl;
		return -1;
	}

	VideoCapture cap(argv[2]);

	if (!cap.isOpened()) {
		return -1;
	}

	Mat img;
	while (true)
	{
		cap >> img;
		if (img.empty())
		{
			printf("Incorrect video!");
			return -1;
		}
		// imshow("Driving Video", frame);
		if (waitKey(1) == 27) break;

		cv::Mat img_input = img.clone(); // 복사

		// 이미지 크기를 640 x 640 크기로 일괄 변경
		std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
		const float pad_w = pad_info[0];
		const float pad_h = pad_info[1];
		const float scale = pad_info[2];

		cv::resize(img_input, img_input, cv::Size(640, 640)); // 크기 변경
		cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB); // BGR -> RGB 컬러 채널 순서 변경
		img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f); // normalization 1/255

		// 텐서 객체 생성
		auto tensor_img = torch::from_blob(img_input.data,
			{ 1, img_input.rows, img_input.cols, img_input.channels() }, torch::kFloat);
		tensor_img = tensor_img.permute({ 0, 3, 1, 2 }).contiguous(); // 차원 순서 변경, BHWC -> BCHW (Batch, Channel, Height, Width)

		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(tensor_img);

		// 추론
		torch::jit::IValue output = model.forward(inputs);

		auto detections = output.toTuple()->elements()[0].toTensor();

		constexpr int item_attr_size = 5;
		int batch_size = detections.size(0);	// 배치 크기

		auto num_classes = detections.size(2) - item_attr_size; // 클래스 수

		// 검출 결과 중에서 confidence > threshold 조건에 해당하는 후보만 남김
		auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);
		auto det = torch::masked_select(detections[0], conf_mask[0]).view({ -1, num_classes + item_attr_size });

		if (0 == det.size(0)) { // det 크기가 0 이면 검출된 결과가 존재하는 않는 것
			return -1;
		}

		std::vector<std::vector<Detection>> result;
		result.reserve(batch_size);

		// 스코어 계산 = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
		det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);

		// 박스 좌표 계산 (center x, center y, width, height) to (x1, y1, x2, y2)
		torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));

		// 검출 결과에서 같은 클래스는 스코어가 가장 높은 것을 기준으로 함
		std::tuple<torch::Tensor, torch::Tensor> max_classes =
			torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

		// class score
		auto max_conf_score = std::get<0>(max_classes);
		// index
		auto max_conf_index = std::get<1>(max_classes);

		max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
		max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

		// shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
		det = torch::cat({ box.slice(1, 0, 4), max_conf_score, max_conf_index }, 1);

		// NMS 계산을 하기 위해 검출된 박스 결과를 배치 단위화
		constexpr int max_wh = 4096;
		auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
		auto offset_box = det.slice(1, 0, 4) + c;

		std::vector<cv::Rect> offset_box_vec;
		std::vector<float> score_vec;

		// copy data back to cpu
		auto offset_boxes_cpu = offset_box.cpu();
		auto det_cpu = det.cpu();
		const auto& det_cpu_array = det_cpu.accessor<float, 2>();

		// 텐서 객체를 Detection 객체로 변경
		Tensor2Detection(offset_boxes_cpu.accessor<float, 2>(), det_cpu_array, offset_box_vec, score_vec);

		// NMS: Non-Maximum Suppression 실행
		std::vector<int> nms_indices;
		cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

		// NMS 계산 후의 박스들의 위치 좌표 저장
		std::vector<Detection> det_vec;
		for (int index : nms_indices) {
			Detection t;
			const auto& b = det_cpu_array[index];
			t.bbox =
				cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
					cv::Point(b[Det::br_x], b[Det::br_y]));
			t.score = det_cpu_array[index][Det::score];
			t.class_idx = det_cpu_array[index][Det::class_idx];
			det_vec.emplace_back(t);
		}

		// 원본 이미지 크기에 맞추어 좌표를 재계산
		ScaleCoordinates(det_vec, pad_w, pad_h, scale, img.size());
		result.emplace_back(det_vec);

		// 이미지에 검출 결과를 표시
		for (const auto& detection : result[0]) {
			const auto& box = detection.bbox;
			float score = detection.score;
			int class_idx = detection.class_idx;

			cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << score;
			std::string s = class_names[class_idx] + " " + ss.str();

			auto font_face = cv::FONT_HERSHEY_DUPLEX;
			auto font_scale = 1.0;
			int thickness = 1;
			int baseline = 0;
			auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
			cv::rectangle(img,
				cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
				cv::Point(box.tl().x + s_size.width, box.tl().y),
				cv::Scalar(0, 0, 255), -1);
			cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
				font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
		}

		// 화면 출력
		cv::namedWindow("Result", WINDOW_AUTOSIZE);
		cv::imshow("Result", img);
		//cv::waitKey(0);
	}

	return 0;
}