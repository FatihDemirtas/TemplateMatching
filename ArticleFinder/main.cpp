#include <iostream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <chrono>
//#include "Overlay.h"
#include "Overlay.h"



int main() {

    int ruler;
    cv::Point maxLoc = cv::Point(0,0) ;
    cv::cuda::GpuMat d_image;
    double maxVal;

for(int i = 1;  i<4000; i++)
{

    std::ostringstream ss;
    ss << std::setw(10) << std::setfill('0') << i;
    std::string str = ss.str();

    std::string frame_filename = "/home/fatihdemirtas/sahin/21-Images/" + str + ".jpg";

//  FOR video-21
    if(i < 1284)
    {
        ruler = 3;
    }
    else if(i > 1283 && i < 2064)
    {
        ruler = 2;
    }
    else if(i > 2063 && i < 2410)
    {
        ruler = 4;
    }
    else if(i > 2409 && i < 2559)
    {
        ruler = 2;
    }
    else if(i > 2558 && i < 3028)
    {
        ruler = 4;
    }
    else if( i > 3027 && i <= 3600)
    {
        ruler = 2;
    }
    else if(i >= 13000)
    {
        ruler = 1;
    }


    //  FOR video-1
//    if(i >= 2057 && i < 2080)
//    {
//        ruler = 3;
//    }
//    else if(i>= 2080 && i < 4000)
//    {
//        ruler = 3;
//    }
//    else if(i > 1283 && i < 2064)
//    {
//        ruler = 2;
//    }
//    else if(i > 2063 && i < 2410)
//    {
//        ruler = 4;
//    }
//    else if(i > 2409 && i < 2559)
//    {
//        ruler = 2;
//    }
//    else if(i > 2558 && i < 3028)
//    {
//        ruler = 4;
//    }
//    else if( i > 3027)
//    {
//        ruler = 2;
//    }


    TemplateResults minMaxProp;
    cv::Mat image = cv::imread(frame_filename, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);


    ArticleDetector articleDetector = ArticleDetector();

    auto start = std::chrono::system_clock::now();
    articleDetector.Process(image, ruler,  minMaxProp);
    auto end = std::chrono::system_clock::now();
    int64_t  elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed Time : " << elapsed_ms << std::endl;
    std::cout << " i : " + to_string(i) << std::endl;
    std::cout << std::endl;

}

    return 0;
}
