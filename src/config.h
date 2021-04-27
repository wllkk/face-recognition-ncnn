#ifndef __CONFIG_H__
#define __CONFIG_H__

#include<iostream>
#include <map>
#include <vector>
#include "net.h"

#define fmc 3

class AnchorCfg
{
public:
    AnchorCfg(){}
    std::vector<float> SCALES;
    std::vector<float> RATIOS;
    int BASE_SIZE;

    AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size)
    {
        SCALES = s;
        RATIOS = r;
        BASE_SIZE = size;
    }

};

extern std::vector<int> _feat_stride_fpn;
extern std::map<int, AnchorCfg> anchor_config;
extern float cls_threshold;
extern float nms_threshold;

#endif