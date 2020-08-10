//
// Created by fatihdemirtas on 16.07.2020.
//

#ifndef UNTITLED_OVERLAY_H
#define UNTITLED_OVERLAY_H

#include "opencv2/opencv.hpp"
#include <gflags/gflags.h>

using namespace std;

struct TemplateResults
{
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    int width;
    int heigth;
};


class ArticleDetector
{

public:
    ArticleDetector()
    {
        isRead = ReadTemplates();
    }
    bool Process(cv::Mat& gray_cropped, int ruler, TemplateResults &minMaxProp);
    cv::Size2f templateSize;

private:
    bool ReadTemplates();
    bool isRead;

    cv::Ptr<cv::cuda::TemplateMatching> TemplateAlg;
    cv::cuda::Stream stream;

    //Template Variables
    cv::Mat template_1,
            template_2,
            template_3,
            template_4,
            template_5;

    //Template Mask Variables
    cv::Mat mask_1,
            mask_2,
            mask_3,
            mask_4,
            mask_5;

    bool Apply(const cv::Mat &gray, const cv::Mat &templ, const cv::Mat &mask, TemplateResults &results);

    void GetRelevantTemplate(const int ruler, cv::Mat &relevantTemplate, cv::Mat &relevantMask);

    int DetermineInterval(int ruler);
};

#endif //UNTITLED_OVERLAY_H
