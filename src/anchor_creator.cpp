#include "anchor_creator.h"
#include "utils.h"

int AnchorCreator::init(int stride, const AnchorCfg& cfg, bool dense_anchor)
{
    CRet2f base_anchor(0,0, cfg.BASE_SIZE - 1, cfg.BASE_SIZE - 1);
    std::vector<CRet2f> ratio_anchors;
    
    _ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
    _scale_enum(ratio_anchors, cfg.SCALES, pre_anchor);

    if(dense_anchor)
    {

    }

    anchor_stride = stride;
    anchor_num = pre_anchor.size();

    // for(int i = 0; i < anchor_num; ++i)
    // {
    //     pre_anchor[i].print();
    // }
    return anchor_num;
}

void AnchorCreator::FilterAnchor(ncnn::Mat& cls, ncnn::Mat& reg, ncnn::Mat& pts, std::vector<Anchor>& proposals)
{
    int w = cls.w;
    int h = cls.h;
    
    int pts_length = pts.c / anchor_num / 2;

    for(int i = 0; i < h; ++i)
    {
        for(int j = 0; j < w; ++j)
        {
            int index = i * w + j;
            for(int a = 0; a < anchor_num;++a)
            {
                // cls.channel(0 | 1) 存放的是两个anchor 分类为不是人脸的概率  cls.channel(2 | 3) 是 分类为人脸的概率
                if(cls.channel(anchor_num + a)[index] >= cls_threshold)  
                {
                    //printf("%f\n", cls.channel(anchor_num + a)[index]);
                    CRet2f pre_anchor_real(j * anchor_stride + pre_anchor[a][0],
                    i * anchor_stride + pre_anchor[a][1],
                    j * anchor_stride + pre_anchor[a][2],
                    i * anchor_stride + pre_anchor[a][3]);

                    CRet2f delta(reg.channel(a * 4 + 0)[index],
                    reg.channel(a * 4 + 1)[index],
                    reg.channel(a * 4 + 2)[index],
                    reg.channel(a * 4 + 3)[index]);

                    Anchor res;
                    res.anchor = cv::Rect_<float>(pre_anchor_real[0], pre_anchor_real[1], pre_anchor_real[2], pre_anchor_real[3]);
                    res.sorce = cls.channel(a + anchor_num)[index];
                    res.center = cv::Point(j, i);

                    _box_pred(pre_anchor_real, delta, res.finalbox);

                    
                    std::vector<cv::Point2f> pts_delta(pts_length);
                    for(int p  = 0; p < pts_length; ++p)
                    {
                        pts_delta[p].x = pts.channel(a * pts_length * 2 + p * 2)[index];
                        pts_delta[p].y = pts.channel(a * pts_length * 2 + p * 2 + 1)[index];
                    }

                    _landmark_pred(pre_anchor_real, pts_delta, res.pts);

                    proposals.push_back(res);
                }
            }
        }
    }
}

void AnchorCreator::_ratio_enum(const CRet2f& base_anchor, const std::vector<float>& ratio, std::vector<CRet2f>& ratio_anchor)
{
    float w = base_anchor[2] - base_anchor[0] + 1;
    float h = base_anchor[3] - base_anchor[1] + 1;
    float x_center = base_anchor[0] + 0.5 * (w - 1);
    float y_center = base_anchor[1] + 0.5 * (h - 1);

    ratio_anchor.clear();
    float orign_size = w * h;
    for(size_t i = 0; i < ratio.size(); i++)
    {
        float r = ratio[i];
        float ratio_size = orign_size / r;

        float w_r = std::sqrt(ratio_size);
        float h_r = w_r;

        ratio_anchor.push_back(CRet2f(x_center - 0.5 * (w_r - 1),
        y_center - 0.5 * (h_r - 1),
        x_center + 0.5 * (w_r - 1),
        y_center + 0.5 * (h_r - 1)));
    }
}

void AnchorCreator::_scale_enum(const std::vector<CRet2f>& ratio_anchor, const std::vector<float> scales, std::vector<CRet2f>& scale_anchor)
{
    for(size_t r = 0; r < ratio_anchor.size(); ++r)
    {
        CRet2f rat_an = ratio_anchor[r];
        float w = rat_an[2] - rat_an[0] + 1;
        float h = rat_an[3] - rat_an[1] + 1;
        float x_center = rat_an[0] + 0.5 * (w - 1);
        float y_center = rat_an[1] + 0.5 * (h - 1);

        for(size_t i = 0; i < scales.size(); ++i)
        {
            float w_s = w * scales[i];
            float h_s = h * scales[i];
            
            scale_anchor.push_back(CRet2f(x_center - 0.5 * (w_s - 1),
            y_center - 0.5 * (h_s - 1),
            x_center + 0.5 * (w_s - 1),
            y_center + 0.5 * (h_s -1 )));
        }
    }
}


void AnchorCreator::_box_pred(const CRet2f& per_anc, const CRet2f& delta, cv::Rect_<float>& box)
{
    float w = per_anc[2] - per_anc[0] + 1;
    float h = per_anc[3] - per_anc[1] + 1;
    float x_center = per_anc[0] + 0.5 * (w - 1);
    float y_center = per_anc[1] + 0.5 * (h - 1);

    float x_d  = delta[0];
    float y_d  = delta[1];
    float w_d  = delta[2];
    float h_d  = delta[3];

    float real_x_center = x_d * w + x_center;
    float real_y_center = y_d * h + y_center;
    float real_w = std::exp(w_d) * w;
    float real_h = std::exp(h_d) * h;

    float w_transform = 0.5 * (real_w - 1.0);
    float h_transform = 0.5 * (real_h - 1.0);
    box = cv::Rect_<float>(real_x_center - w_transform,
    real_y_center - h_transform,
    real_x_center + w_transform,
    real_y_center + h_transform);
}

void AnchorCreator::_landmark_pred(const CRet2f& box, const std::vector<cv::Point2f>& pts_delta, std::vector<cv::Point2f>& landmark_pre)
{
    float w = box[2] - box[0] + 1;
    float h = box[3] - box[1] + 1;
    float x_center = box[0] + 0.5 * (w - 1);
    float y_center = box[1] + 0.5 * (h - 1);

    landmark_pre.resize(pts_delta.size());

    for(size_t i = 0; i < pts_delta.size(); ++i)
    {
        landmark_pre[i].x = pts_delta[i].x * w + x_center;
        landmark_pre[i].y = pts_delta[i].y * h + y_center;
    }
}