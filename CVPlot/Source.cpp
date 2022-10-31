#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
using namespace cv;
using namespace plot;

int main(int argc, char* argv) {
	
    std::vector<double> data_x;
    std::vector<double> data_y;

    data_x.push_back(0);
    data_y.push_back(0);
    std::cout << "data_x : " << data_x[0] << std::endl;
    std::cout << "data_y : " << data_y[0] << std::endl;


    for (int i = 1; i < 1000; i++) {

        data_x.push_back(i * 0.01);
        data_y.push_back(std::sin(data_x.back()));
        Mat plot_result;
        Ptr<plot::Plot2d> plot = plot::Plot2d::create(data_x, data_y);
        plot->render(plot_result);

        imshow("The plot rendered with default visualization options", plot_result);

        plot->setShowText(false);
        plot->setShowGrid(false);
        plot->setPlotBackgroundColor(Scalar(255, 200, 200));
        plot->setPlotLineColor(Scalar(255, 0, 0));
        plot->setPlotLineWidth(2);
        plot->setInvertOrientation(true);
        plot->render(plot_result);

        imshow("The plot rendered with some of custom visualization options",
            plot_result);
        waitKey(3);
    }
    cv::waitKey();
    {
        char tmp;
        std::cin >> tmp;
    }
	return 0;
}