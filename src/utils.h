#ifndef __UIILS_H__
#define __UTILS_H__

#include "anchor_creator.h"

void box_nms_cpu(std::vector<Anchor>& boxs, const float threshold, std::vector<Anchor>& res, int img_size);
float calc_similarity_with_cos(const std::vector<float>& feat1, const std::vector<float>& feat2);

#endif