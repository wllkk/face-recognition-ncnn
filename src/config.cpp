#include "config.h"

#if fmc == 3
std::vector<int> _feat_stride_fpn = {32,16,8};

std::map<int, AnchorCfg> anchor_config = {
    {32, AnchorCfg(std::vector<float>{32,16}, std::vector<float>{1}, 16)},
    {16, AnchorCfg(std::vector<float>{8,4}, std::vector<float>{1}, 16)},
    {8, AnchorCfg(std::vector<float>{2,1}, std::vector<float>{1}, 16)}
};

float cls_threshold = 0.8;
float nms_threshold = 0.4;

#endif

