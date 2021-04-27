
#include "net.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#if CV_MAJOR_VERSION >= 3
#include "opencv2/video/video.hpp"
#endif

#include <vector>
#include "config.h"
#include "anchor_creator.h"
#include "utils.h"
#include "benchmark.h"

static int init_retinaface(ncnn::Net* retinaface, const int target_size)
{
    int ret = 0;
    retinaface->opt.num_threads = 8;
    retinaface->opt.use_winograd_convolution = true;
    retinaface->opt.use_sgemm_convolution = true;

    const char* model_param = "../models/retinaface.param";
    const char* model_model = "../models/retinaface.bin";
    
    ret = retinaface->load_param(model_param);
    if(ret)
    {
        return ret;
    }
    ret = retinaface->load_model(model_model);
    if(ret)
    {
        return ret;
    }

    return 0;
}

static int init_mbv2facenet(ncnn::Net* mbv2facenet, const int target_size)
{
    int ret = 0;

    mbv2facenet->opt.num_threads = 8;
    mbv2facenet->opt.use_sgemm_convolution = 1;
    mbv2facenet->opt.use_winograd_convolution = 1;

    const char* model_param = "../models/mbv2facenet.param";
    const char* model_bin = "../models/mbv2facenet.bin";

    ret = mbv2facenet->load_param(model_param);
    if(ret)
    {
        return ret;
    }

    ret = mbv2facenet->load_model(model_bin);
    if(ret)
    {
        return ret;
    }

    return 0;
}

void detect_retinaface(ncnn::Net* retinaface, cv::Mat& img, const int target_size, std::vector<cv::Mat>& face_det)
{
    int img_w = img.cols;
    int img_h = img.rows;
    
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1, 1, 1};

    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
    //ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB,img_w, img_h, target_size, target_size);
    input.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = retinaface->create_extractor();
 
    std::vector<AnchorCreator> ac(_feat_stride_fpn.size());
    for(size_t i = 0; i < _feat_stride_fpn.size(); i++)
    {
        int stride = _feat_stride_fpn[i];
        ac[i].init(stride, anchor_config[stride], false);
    }
    
    ex.input("data", input);

    std::vector<Anchor> proposals;

    for(size_t i = 0; i < _feat_stride_fpn.size(); i++)
    {
        ncnn::Mat cls;
        ncnn::Mat reg;
        ncnn::Mat pts;

        char cls_name[100];
        char reg_name[100];
        char pts_name[100];
        sprintf(cls_name, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        sprintf(reg_name, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        sprintf(pts_name, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        
        ex.extract(cls_name,cls);
        ex.extract(reg_name,reg);
        ex.extract(pts_name,pts);

        printf("cls: %d %d %d\n", cls.c, cls.h, cls.w);
        printf("reg: %d %d %d\n", reg.c, reg.h, reg.w);
        printf("pts: %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        // for(size_t p = 0; p < proposals.size(); ++p)
        // {
        //     proposals[p].print();
        // }
    }

    std::vector<Anchor> finalres;
    box_nms_cpu(proposals, nms_threshold, finalres, target_size);
    //cv::resize(img, img, cv::Size(target_size, target_size));
    for(size_t i = 0; i < finalres.size(); ++i)
    {
        finalres[i].print();
        cv::Mat face = img(cv::Range((int)finalres[i].finalbox.y, (int)finalres[i].finalbox.height),cv::Range((int)finalres[i].finalbox.x, (int)finalres[i].finalbox.width)).clone();
        face_det.push_back(face);
        
        cv::rectangle(img, cv::Point((int)finalres[i].finalbox.x, (int)finalres[i].finalbox.y), cv::Point((int)finalres[i].finalbox.width, (int)finalres[i].finalbox.height),cv::Scalar(255,255,0), 2, 8, 0);
        for(size_t l = 0; l < finalres[i].pts.size(); ++l)
        {
            cv::circle(img, cv::Point((int)finalres[i].pts[l].x, (int)finalres[i].pts[l].y), 1, cv::Scalar(255, 255, 0), 2, 8, 0);
        }
    }
}

void run_mbv2facenet(ncnn::Net* mbv2facenet, std::vector<cv::Mat>& img, int target_size, std::vector<std::vector<float>>& res)
{
    for(size_t i = 0; i < img.size(); ++i)
    {
        ncnn::Extractor ex = mbv2facenet->create_extractor();
        //网络结构中的前两层已经做了归一化和均值处理， 在输入的时候不用处理了
        ncnn::Mat input = ncnn::Mat::from_pixels_resize(img[i].data, ncnn::Mat::PIXEL_BGR2RGB, img[i].cols, img[i].rows, target_size, target_size);
        ex.input("data", input);
        
        ncnn::Mat feat;

        ex.extract("fc1", feat);

        printf("c: %d h: %d w: %d\n", feat.c, feat.h, feat.w);
        std::vector<float> tmp;
        for(int i = 0; i < feat.w; ++i)
        {
            //printf("%f ", feat.channel(0)[i]);
            tmp.push_back(feat.channel(0)[i]);
        }
        res.push_back(tmp);
        //printf("\n");
    }
}   

int main(int argc, char** argv)
{
    int target_size = 300;
    int facenet_size = 112;
    ncnn::Net retinaface;
    ncnn::Net mbv2facenet;
    
    int ret = 0;

    if(argc < 3)
    {
        fprintf(stderr, "Usage: %s [input image1 input image2]", argv[0]);
        return -1;
    }
    
    cv::Mat img1 = cv::imread(argv[1], 1);
    cv::Mat img2 = cv::imread(argv[2], 1);
    if(img1.empty() || img2.empty())
    {
        fprintf(stderr, "Failed to read image from %s or %s.\n", argv[1], argv[2]);
        return -1;
    }

    ret = init_retinaface(&retinaface, target_size);
    if(ret)
    {
        fprintf(stderr, "Failed to load retinaface param or bin, error code %d.\n", ret);
        return -1;
    }

    ret = init_mbv2facenet(&mbv2facenet, facenet_size);
    if(ret)
    {
        fprintf(stderr, "Failed to load mbv2facenet param or bin, error code %d.\n", ret);
        return -1;
    }

    cv::resize(img1, img1, cv::Size(target_size, target_size));
    cv::resize(img2, img2, cv::Size(target_size, target_size));


    std::vector<cv::Mat> face_det;
    std::vector<std::vector<float>> feature_face;
    detect_retinaface(&retinaface, img1, target_size, face_det); //pic1 dect
    detect_retinaface(&retinaface, img2, target_size, face_det);//pic2 dect
    for(size_t i = 0; i < face_det.size(); ++i)
    {
        char name[30];
        sprintf(name, "../output_pic/face_%d.jpg", i);
        cv::imwrite(name, face_det[i]);
    }
    if(face_det.size() == 2) //只有两张人脸的时候才进行人脸识别
    {
        run_mbv2facenet(&mbv2facenet, face_det, facenet_size, feature_face);

        //余弦距离sim值越接近1，代表两个向量的夹角越接近0，则两个向量越相似。反之，越接近0，代表两个向量夹角趋于90°，两个向量差异越大。
        //相似阈值可以取 > 0.3
        float sim = calc_similarity_with_cos(feature_face[0], feature_face[1]);

       //将两种输入合成一张
        cv::Mat des;
        des.create(target_size, 2 * target_size, img1.type());
        cv::Mat r1 = des(cv::Rect(0, 0, target_size, target_size));
        cv::Mat r2 = des(cv::Rect(target_size, 0, target_size, target_size));     
        
        img1.copyTo(r1);
        img2.copyTo(r2);

        char text[50];
        if(sim >= 0.3)
        {
            sprintf(text, "face similarity: %f same person", sim);
        }
        else
        {
            sprintf(text, "face similarity: %f unsame person", sim);
        }
        cv::putText(des, text, cv::Point(target_size - 250, target_size - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        cv::imwrite("../output_pic/des.jpg", des);
        printf("face_sim: %f\n", sim);
    }  
    return 0;
}