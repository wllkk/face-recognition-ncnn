#include "utils.h"
#include <iostream>
#include <omp.h>
#ifdef __ARM_NEON
    #include "arm_neon.h"
#endif

void box_nms_cpu(std::vector<Anchor>& boxs, const float threshold, std::vector<Anchor>& res, int target_size)
{
    res.clear();
    if(boxs.size() == 0)
        return;
    std::vector<size_t> indexs(boxs.size());
    for(size_t i = 0; i < indexs.size(); ++i)
    {
        indexs[i] = i;
    }

    std::sort(boxs.begin(), boxs.end(), std::greater<Anchor>());

    while(indexs.size() > 0)
    {
        int good_idx = indexs[0];
        res.push_back(boxs[good_idx]);

        std::vector<size_t> tmp = indexs;
        indexs.clear();
        for(size_t i = 1; i < tmp.size(); ++i)
        {
            size_t tmp_i = tmp[i];
            float inter_x1 = std::max(boxs[good_idx][0], boxs[tmp_i][0]);
            float inter_y1 = std::max(boxs[good_idx][1], boxs[tmp_i][1]);
            float inter_x2 = std::min(boxs[good_idx][2], boxs[tmp_i][2]);
            float inter_y2 = std::min(boxs[good_idx][3], boxs[tmp_i][3]);

            float w = std::max(inter_x2 - inter_x1 + 1, 0.0f);
            float h = std::max(inter_y2 - inter_y1 + 1, 0.0f);

            float inter = w * h;

            float area1 = (boxs[good_idx][2] - boxs[good_idx][0] + 1) * (boxs[good_idx][3] - boxs[good_idx][1] + 1);
            float area2 = (boxs[tmp_i][2] - boxs[tmp_i][0] + 1) * (boxs[tmp_i][3] - boxs[tmp_i][1] + 1);

            float iou = inter / (area1 + area2 - inter);

            if(iou <= threshold)
            {
                indexs.push_back(tmp_i);
            }
        }
    }

    //clip
    for(size_t i = 0; i < res.size(); ++i)
    {
        res[i].finalbox.x = std::max(std::min(res[i].finalbox.x, (float)(target_size - 1)), 0.f);
        res[i].finalbox.y = std::max(std::min(res[i].finalbox.y, (float)(target_size - 1)), 0.f);
        res[i].finalbox.width = std::max(std::min(res[i].finalbox.width, (float)(target_size - 1)), 0.f);
        res[i].finalbox.height = std::max(std::min(res[i].finalbox.height, (float)(target_size - 1)), 0.f);
    }
}

#ifdef __ARM_NEON
float calc_innerProduct_neon(const std::vector<float>& feat1, const std::vector<float>& feat2)
{
    int block = feat1.size() >> 3;
    //128 % 8 = 0
    const float* ptr_feat1 = &feat1[0];
    const float* ptr_feat2 = &feat2[0];
    float temp = 0.f;
    float* res = &temp;
    
#ifdef __aarch64__
    if(block > 0)
    {
        __asm__ volatile(
            "prfm             pldl1keep,        [%0, #128]      \n"
            "ld1              {v0.4s, v1.4s},   [%0]            \n"
            "add              %0, %0, #32                       \n"
            "prfm             pldl1keep,        [%1, #128]      \n"
            "ld1              {v2.4s, v3.4s},   [%1]            \n"
            "add              %1, %1, #32                       \n"
            "fmul             v4.4s, v0.4s, v2.4s               \n"
            "fmla             v4.4s, v1.4s, v3.4s               \n"
            "subs             %2, %2, #1                        \n"

            "0:                                                 \n"
            "ld1              {v0.4s, v1.4s},   [%0]            \n"
            "add              %0, %0, #32                       \n"
            "ld1              {v2.4s, v3.4s},   [%1]            \n"
            "add              %1, %1, #32                       \n"
            "fmla             v4.4s, v0.4s, v2.4s               \n"
            "fmla             v4.4s, v1.4s, v3.4s               \n"

            "subs             %2, %2, #1                        \n"
            "bne              0b                                \n"

            "dup              v5.2d,v4.d[0]                     \n"
            "dup              v6.2d,v4.d[1]                     \n"
            "fadd             v7.2s, v5.2s, v6.2s               \n"
            "faddp            v7.2s, v7.2s, v7.2s               \n"
            "dup              v8.2s, v7.s[0]                    \n"
            "st1              {v8.s}[0], [%3]                   \n"

            : "=r"(ptr_feat1),
            "=r"(ptr_feat2),
            "=r"(block),
            "=r"(res)
            : "0"(ptr_feat1),
            "1"(ptr_feat2),
            "2"(block),
            "3"(res)
            :"cc", "memory", "v0", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
        );
    }
    return *res;
#else
    float32x4_t _sum = {0.0f};
    for(; block > 0; --block)
    {
        float32x4_t _f1 = vld1q_f32(ptr_feat1);
        ptr_feat1+=4;
        float32x4_t _f2 = vld1q_f32(ptr_feat2);
         ptr_feat2+=4;
        float32x4_t _f3 = vld1q_f32(ptr_feat1);
         ptr_feat1+=4;
        float32x4_t _f4 = vld1q_f32(ptr_feat2);
         ptr_feat2+=4;
        _sum = vmlaq_f32(_sum, _f1, _f2);
        _sum = vmlaq_f32(_sum, _f3, _f4);
    }
    float32x2_t _res = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
    float32x2_t _ress = vpadd_f32(_res, _res);
    return vget_lane_f32(_ress, 0);
#endif
}

#else
float calc_innerProduct(const std::vector<float>& feat1, const std::vector<float>& feat2)
{
    float res = 0.f;
//#pragma omp parallel for reduction(+:res)
    for(size_t i = 0; i < feat1.size(); ++i)
    {
        res += feat1[i] * feat2[i];
    }
    return res;
}
#endif

float calc_similarity_with_cos(const std::vector<float>& feat1, const std::vector<float>& feat2)
{
    // feat1* feat2 / (||feat1|| * ||feat2 * feat2||)
    float res = 0.f;
#ifdef __ARM_NEON
    res = calc_innerProduct_neon(feat1, feat2) / (sqrt(calc_innerProduct_neon(feat1, feat1)) * sqrt(calc_innerProduct_neon(feat2, feat2)));
#else
    res = calc_innerProduct(feat1, feat2) / (sqrt(calc_innerProduct(feat1, feat1)) * sqrt(calc_innerProduct(feat2, feat2)));
#endif
    return res;
}
