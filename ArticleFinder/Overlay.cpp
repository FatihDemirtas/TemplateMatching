//
// Created by fatihdemirtas on 10.07.2020.
//

#include "Overlay.h"
#include <gflags/gflags.h>
#include <chrono>

#define searchArea cv::Rect(600, 200, 600, 600) // Possible area of article

DEFINE_string(template_path_1, "../Articles/common_1.png", "Template path for (Size of template = cv::Size(315,315))");
DEFINE_string(template_path_2, "../Articles/common_2.png", "Template path for (Size of template = cv::Size(160,160))");
DEFINE_string(template_path_3, "../Articles/common_3.png", "Template path for (Size of template = cv::Size(110,110))");
DEFINE_string(template_path_4, "../Articles/common_4.png", "Template path for (Size of template = cv::Size(067,067))");
DEFINE_string(template_path_5, "../Articles/common_5.png", "Template path for (Size of template = cv::Size(050,050))");

DEFINE_string(template_path_mask_1, "../Articles/common_1_mask.png", "Mask path for template 1");
DEFINE_string(template_path_mask_2, "../Articles/common_2_mask.png", "Mask path for template 2");
DEFINE_string(template_path_mask_3, "../Articles/common_3_mask.png", "Mask path for template 3");
DEFINE_string(template_path_mask_4, "../Articles/common_4_mask.png", "Mask path for template 4");
DEFINE_string(template_path_mask_5, "../Articles//common_5_mask.png", "Mask path for template 5");

DEFINE_int32(temp_match_method, 3, "TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED");

DEFINE_bool(template_debug_mode, true, "Show template matching results or not!");

DEFINE_int32(enlarge, 40, "Enlarge related area to recheck article!");

/**
 * @brief Main function of template matching
 *
 * @param gray_cropped         Frame which should be gray-scale image or one channel image
 * @param ruler        Represents the intervals of ruler on the image
 * @param minMaxProp   Template matching result (minValue, maxValue, minLocation, maxLocation)
 *
 * @return TRUE if template matching done
 *         FALSE otherwise
 */
bool ArticleDetector::Process(cv::Mat& gray, int ruler, TemplateResults &minMaxProp)
{
   bool isDone;

   // If templates and masks are not read, Leave!
   if(!isRead)
   {
       isDone = false;
       return isDone;
   }
   else
   {
       cv::Mat templ;
       cv::Mat mask;
       cv::Mat gray_cropped;
       TemplateResults minMaxPropCropped;

       int interval;

       // Search article in only possible area to save computational cost
       gray_cropped = gray(searchArea);

       // Convert given image to GRAY if image has 3 channels
       if(gray_cropped.channels() == 3)
       {
           cv::cvtColor(gray_cropped, gray_cropped, cv::COLOR_RGB2GRAY);
       }
       else
       {
           // DO NOTHING
       }

       //TODO : Uncomment while available ruler value is given
//       interval = DetermineInterval(ruler);
       interval = ruler;
       GetRelevantTemplate(interval, templ, mask);

       isDone = Apply(gray_cropped, templ, mask, minMaxPropCropped);

       if(minMaxPropCropped.maxVal < 0.8)
       {
           cv::Mat upperScaleTemp;
           cv::Mat upperScaleMask;
           TemplateResults minMaxPropUp;

           GetRelevantTemplate(interval - 1 , upperScaleTemp, upperScaleMask);

           cv::Rect enlarged = cv::Rect(minMaxPropCropped.maxLoc.x - FLAGS_enlarge, minMaxPropCropped.maxLoc.y - FLAGS_enlarge,
                                        templ.rows + FLAGS_enlarge*2, templ.cols + FLAGS_enlarge*2);


           Apply(gray_cropped(enlarged), upperScaleTemp, upperScaleMask, minMaxPropUp);

           cv::Mat lowerScaleTemp;
           cv::Mat lowerScaleMask;
           TemplateResults minMaxPropDown;

           GetRelevantTemplate(interval + 1 , lowerScaleTemp, lowerScaleMask);

           Apply(gray_cropped(enlarged), lowerScaleTemp, lowerScaleMask, minMaxPropDown);

            // TODO : Some arrangment needed!
           if (std::max(minMaxPropUp.maxVal, minMaxPropDown.maxVal) > minMaxPropCropped.maxVal && std::max(minMaxPropUp.maxVal, minMaxPropDown.maxVal) > 0.8)
           {
               if (minMaxPropUp.maxVal > minMaxPropDown.maxVal)
               {
                   minMaxProp = minMaxPropUp;
                   minMaxProp.maxLoc.x = searchArea.x + enlarged.x + minMaxPropUp.maxLoc.x;
                   minMaxProp.maxLoc.y = searchArea.y + enlarged.y + minMaxPropUp.maxLoc.y;
                   minMaxProp.minLoc.x = searchArea.x + enlarged.x + minMaxPropUp.minLoc.x;
                   minMaxProp.minLoc.y = searchArea.y + enlarged.y + minMaxPropUp.minLoc.y;
                   minMaxProp.width = upperScaleTemp.rows;
                   minMaxProp.heigth = upperScaleTemp.cols;

                   if(FLAGS_template_debug_mode)
                   {
                       cv::Mat grayTemp;
                       gray_cropped(enlarged).copyTo(grayTemp);

                       cv::rectangle(grayTemp, minMaxPropUp. maxLoc,
                                     cv::Point(minMaxPropUp.maxLoc.x + upperScaleTemp.rows, minMaxPropUp.maxLoc.y + upperScaleTemp.cols),
                                     CV_RGB(255, 0, 0), 2);

                       cv::putText(grayTemp, "MAX VALUE : " + to_string(minMaxPropUp.maxVal), cv::Point(50,50),
                                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

                       cv::imshow("UP", grayTemp);
                       cv::waitKey(0);
                   }
               }
               else
               {
                   minMaxProp = minMaxPropDown;
                   minMaxProp.maxLoc.x = searchArea.x + enlarged.x + minMaxPropDown.maxLoc.x;
                   minMaxProp.maxLoc.y = searchArea.y + enlarged.y + minMaxPropDown.maxLoc.y;
                   minMaxProp.minLoc.x = searchArea.x + enlarged.x + minMaxPropDown.minLoc.x;
                   minMaxProp.minLoc.y = searchArea.y + enlarged.y + minMaxPropDown.minLoc.y;
                   minMaxProp.width = lowerScaleTemp.rows;
                   minMaxProp.heigth = lowerScaleTemp.cols;

                   if(FLAGS_template_debug_mode)
                   {
                       cv::Mat grayTemp;
                       gray_cropped(enlarged).copyTo(grayTemp);

                       cv::rectangle(grayTemp, minMaxPropDown.maxLoc,
                                     cv::Point(minMaxPropDown.maxLoc.x + lowerScaleTemp.rows, minMaxPropDown.maxLoc.y + lowerScaleTemp.cols),
                                     CV_RGB(255, 0, 0), 2);

                       cv::putText(grayTemp, "MAX VALUE : " + to_string(minMaxPropDown.maxVal), cv::Point(50,50),
                                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

                       cv::imshow("DOWN", grayTemp);
                   }
               }
           }
           else
           {
               minMaxProp = minMaxPropCropped;
               minMaxProp.maxLoc.x = searchArea.x + minMaxPropCropped.maxLoc.x;
               minMaxProp.maxLoc.y = searchArea.y + minMaxPropCropped.maxLoc.y;
               minMaxProp.minLoc.x = searchArea.x + minMaxPropCropped.minLoc.x;
               minMaxProp.minLoc.y = searchArea.y + minMaxPropCropped.minLoc.y;
               minMaxProp.width = templ.rows;
               minMaxProp.heigth = templ.cols;
           }

       }
       else
       {
           minMaxProp = minMaxPropCropped;
           minMaxProp.maxLoc.x = searchArea.x + minMaxPropCropped.maxLoc.x;
           minMaxProp.maxLoc.y = searchArea.y + minMaxPropCropped.maxLoc.y;
           minMaxProp.minLoc.x = searchArea.x + minMaxPropCropped.minLoc.x;
           minMaxProp.minLoc.y = searchArea.y + minMaxPropCropped.minLoc.y;
           minMaxProp.width = templ.rows;
           minMaxProp.heigth = templ.cols;
       }

       if (isDone)
       {
           if(FLAGS_template_debug_mode)
           {
               cv::Mat grayTemp;
               gray.copyTo(grayTemp);

               cv::rectangle(grayTemp, minMaxProp.maxLoc,
                             cv::Point(minMaxProp.maxLoc.x + minMaxProp.width, minMaxProp.maxLoc.y + minMaxProp.heigth),
                             CV_RGB(255, 0, 0), 2);

               cv::putText(grayTemp, "MAX VALUE : " + to_string(minMaxProp.maxVal), cv::Point(50, 50),
                           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

               cv::imshow("Maximum Template Matching Result", grayTemp);
               cv::waitKey(10);
           }
       }
   }
   return isDone;
}

/**
 * @brief Apply template matching to given frame.
 *
 * @param gray        Frame which should be gray-scale image or one channel image
 * @param templ       Template which also should be gray-scale image  or one channel image
 * @param mask        Template mask which removes backround pixels from template
 * @param minMaxProp  Template matching result (minValue, maxValue, minLocation, maxLocation)
 *
 * @return TRUE if any error is not occured!
 *         FALSE if otherwise
 */
bool ArticleDetector::Apply(const cv::Mat& gray, const cv::Mat& templ, const cv::Mat& mask, TemplateResults& minMaxProp)
{
    cv::Mat result;
    bool isApplied;

    try
    {
        // Apply template matching with given mask and method
        cv::matchTemplate(gray, templ, result, FLAGS_temp_match_method, mask);

        // Find Maximum correlation value and location
        cv::cuda::minMaxLoc(result, &minMaxProp.minVal, &minMaxProp.maxVal, &minMaxProp.minLoc, &minMaxProp.maxLoc,
                            cv::Mat());

        isApplied = true;
    }
    catch(cv::Exception& e)
    {
        std::cout << "Error occured while template matching is :: " << e.what() << std::endl;
        isApplied = false;
    }
    return isApplied;
}

/**
 * @brief Read templates and masks
 *
 * @return Return TRUE if all templates and masks are read succesfully
 *         Return FALSE if any template or mask is not read!
 */
bool ArticleDetector::ReadTemplates()
{
    bool isRead;
    try
    {
    template_1 = cv::imread(FLAGS_template_path_1, cv::IMREAD_GRAYSCALE);
    template_2 = cv::imread(FLAGS_template_path_2, cv::IMREAD_GRAYSCALE);
    template_3 = cv::imread(FLAGS_template_path_3, cv::IMREAD_GRAYSCALE);
    template_4 = cv::imread(FLAGS_template_path_4, cv::IMREAD_GRAYSCALE);

    mask_1 = cv::imread(FLAGS_template_path_mask_1, cv::IMREAD_GRAYSCALE);
    mask_2 = cv::imread(FLAGS_template_path_mask_2, cv::IMREAD_GRAYSCALE);
    mask_3 = cv::imread(FLAGS_template_path_mask_3, cv::IMREAD_GRAYSCALE);
    mask_4 = cv::imread(FLAGS_template_path_mask_4, cv::IMREAD_GRAYSCALE);

    isRead = true;
    }
    catch(cv::Exception & e)
    {
        std::cout << "ERROR occured while reading templates is :: " << e.what() << std::endl;
        isRead = false;
    }
    return isRead;
}

/**
 * @brief According to ruler value, get relevant template.
 *        If ruler is between 01-14 (1), template size should be Size(315,315)
 *                 is between 15-24 (2), template size should be Size(160,160)
 *                 is between 25-39 (3), template size should be Size(110,110)
 *                 is between 40-70 (4), template size should be Size(067,067)
 *                 is others        (5), template size should be smaller than Size(067,067)
 *
 * @param ruler            Represents the intervals
 * @param relevantTemplate Output which includes relevant template
 * @param relevantMask     Output which includes relevant mask
 * */
void ArticleDetector::GetRelevantTemplate(const int ruler, cv::Mat &relevantTemplate, cv::Mat &relevantMask)
{
    if (ruler == 1)
    {
        relevantTemplate = template_1;
        relevantMask = mask_1;
    }
    else if(ruler == 2)
    {
        relevantTemplate = template_2;
        relevantMask = mask_2;
    }
    else if(ruler == 3)
    {
        relevantTemplate = template_3;
        relevantMask = mask_3;
    }
    else if(ruler == 4)
    {
        relevantTemplate = template_4;
        relevantMask = mask_4;
    }
    else if(ruler == 5)
    {
        relevantTemplate = template_5;
        relevantMask = mask_5;
    }
}

/**
 * @brief Determine the interval of ruler
 * @param ruler Real value of ruler on the image
 * @return Interval
 */
int ArticleDetector::DetermineInterval(const int ruler)
{
    int interval;

    if (ruler > 0 && ruler <=14)
    {
        interval = 1;
    }
    else if(ruler > 14 && ruler <=24)
    {
        interval = 2;
    }
    else if(ruler > 24 && ruler <=38)
    {
        interval = 3;
    }
    else if(ruler > 38 && ruler <=60)
    {
        interval = 4;
    }
    else
    {
        interval = 5;
    }

    return interval;
}

