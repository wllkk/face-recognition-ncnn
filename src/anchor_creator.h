#ifndef __ANCHOR_CREATOR_H__
#define __ANCHOR_CREATOR_H__

#include <stdio.h>
#include <vector>
#include <config.h>
#include <opencv2/opencv.hpp>
#include "net.h"

#define landmark 1

class CRet2f
{
public:
    CRet2f(float x1, float x2, float x3, float x4)
    {
        val[0] = x1;
        val[1] = x2;
        val[2] = x3;
        val[3] = x4;

    }

    float operator[](int i) const
    {
        return val[i];
    }

    float& operator[](int i) 
    {
        return val[i];
    }

    float val[4];

    void print()
    {
        printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
    }
};

class Anchor
{
public:
    Anchor(){}
    
    bool operator<(const Anchor &t) const
    {
        return sorce < t.sorce;
    }

    bool operator>(const Anchor &t) const
    {
        return sorce > t.sorce;
    }

    float& operator[](int i)
    {
        assert(i < 4);
        if(i == 0)
            return finalbox.x;
        if(i == 1)
            return finalbox.y;
        if(i == 2)
            return finalbox.width;
        if(i == 3)
            return finalbox.height;
    }

    float operator[](int i) const
    {
        assert(i < 4);
        if(i == 0)
            return finalbox.x;
        if(i == 1)
            return finalbox.y;
        if(i == 2)
            return finalbox.width;
        if(i == 3)
            return finalbox.height;
    }

    void print()
    {
        printf("Rect %f %f %f %f Sorce: %f\n", finalbox.x, finalbox.y, finalbox.width, finalbox.height, sorce);
    }
    
    cv::Rect_<float> finalbox; //use to save finalbox (after adding the delta)
    cv::Rect_<float> anchor;  // use to save orignal anchor  
    float sorce;              // use to save the anchor sorce
    std::vector<cv::Point2f> pts;  // use to save the landmark point
    cv::Point center;
};


class AnchorCreator
{
public:
    int init(int stride, const AnchorCfg& cfg, bool dense_anchor);
    void FilterAnchor(ncnn::Mat& cls, ncnn::Mat& reg, ncnn::Mat& pts, std::vector<Anchor>& proposals);

private:
    void _ratio_enum(const CRet2f& base_anchor, const std::vector<float>& ratio, std::vector<CRet2f>& ratio_anchor);
    void _scale_enum(const std::vector<CRet2f>& ratio_anchor, const std::vector<float> scales, std::vector<CRet2f>& scale_anchor);
    void _box_pred(const CRet2f& per_anc, const CRet2f& delta, cv::Rect_<float>& box);
    void _landmark_pred(const CRet2f& box, const std::vector<cv::Point2f>& pts_delta, std::vector<cv::Point2f>& landmark_pre);

    std::vector<CRet2f> pre_anchor;
    int anchor_stride;
    int anchor_num;
};

#endif