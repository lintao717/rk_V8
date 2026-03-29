// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "yolov8_pose.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifndef RKNPU1
#include <Float16.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <set>
#include <vector>

#define LABEL_NALE_TXT_PATH "/root/rknn_yolov8_pose_demo/model/yolov8_pose_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];

// Cross-module diagnostic flags (read in src/main.cpp).
std::atomic<int> g_pose_diag_semantic_mismatch(0);
std::atomic<int> g_pose_diag_nms_misselection(0);

static const bool POSE_DIAG_ENABLE = true;
static const int POSE_DIAG_PRINT_EVERY_N_FRAMES = 10;
static const float POSE_DIAG_SEMANTIC_OOR_RATIO_THRESH = 0.30f;
static const float POSE_DIAG_NMS_KPT_MEAN_GAP_THRESH = 0.15f;
static const float POSE_DIAG_NMS_IOU_THRESH = 0.50f;

struct PosePostDiagAgg {
    uint64_t frame_count = 0;
    uint64_t candidates_before_nms_sum = 0;
    uint64_t candidates_after_nms_sum = 0;
    uint64_t raw_conf_count = 0;
    uint64_t raw_conf_neg = 0;
    uint64_t raw_conf_in_01 = 0;
    uint64_t raw_conf_gt1 = 0;
    double raw_conf_sum = 0.0;
    uint64_t nms_misselection_events = 0;
};

static PosePostDiagAgg g_pose_post_diag_agg;
static std::mutex g_pose_post_diag_mutex;

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale);

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL;

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL;
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = (int)buff_len;

    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 5 + 0];
            float ymin0 = outputLocations[n * 5 + 1];
            float xmax0 = outputLocations[n * 5 + 0] + outputLocations[n * 5 + 2];
            float ymax0 = outputLocations[n * 5 + 1] + outputLocations[n * 5 + 3];

            float xmin1 = outputLocations[m * 5 + 0];
            float ymin1 = outputLocations[m * 5 + 1];
            float xmax1 = outputLocations[m * 5 + 0] + outputLocations[m * 5 + 2];
            float ymax1 = outputLocations[m * 5 + 1] + outputLocations[m * 5 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static float unsigmoid(float y) { return -1.0f * logf((1.0f / y) - 1.0f); }

static float normalize_kpt_conf_for_diag(float conf)
{
    if (!std::isfinite(conf))
    {
        return 0.0f;
    }
    if (conf < 0.0f || conf > 1.0f)
    {
        return sigmoid(conf);
    }
    return conf;
}

static float read_kpt_raw_conf(rknn_app_context_t *app_ctx,
#if defined(RV1106_1103)
                               rknn_tensor_mem **_outputs,
#else
                               rknn_output *_outputs,
#endif
                               int keypoint_count, int kpt_idx, int keypoints_index)
{
    if (keypoints_index < 0)
    {
        keypoints_index = 0;
    }
    if (keypoints_index >= keypoint_count)
    {
        keypoints_index = keypoint_count - 1;
    }

    if (app_ctx->is_quant)
    {
#if defined(RV1106_1103)
        return deqnt_affine_u8_to_f32(((uint8_t *)_outputs[3].virt_addr)[kpt_idx * 3 * keypoint_count + 2 * keypoint_count + keypoints_index],
                                      app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale);
#elif defined(RKNPU1)
        return deqnt_affine_u8_to_f32(((uint8_t *)_outputs[3].buf)[kpt_idx * 3 * keypoint_count + 2 * keypoint_count + keypoints_index],
                                      app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale);
#else
        return (float)((rknpu2::float16 *)_outputs[3].buf)[kpt_idx * 3 * keypoint_count + 2 * keypoint_count + keypoints_index];
#endif
    }
    else
    {
#if defined(RV1106_1103)
        return ((float *)_outputs[3].virt_addr)[kpt_idx * 3 * keypoint_count + 2 * keypoint_count + keypoints_index];
#else
        return ((float *)_outputs[3].buf)[kpt_idx * 3 * keypoint_count + 2 * keypoint_count + keypoints_index];
#endif
    }
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void softmax(float *input, int size)
{
    float max_val = input[0];
    for (int i = 1; i < size; ++i)
    {
        if (input[i] > max_val)
        {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        sum_exp += expf(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i)
    {
        input[i] = expf(input[i] - max_val) / sum_exp;
    }
}

static int process_i8(int8_t *input, int grid_h, int grid_w, int stride, std::vector<float> &boxes,
                      std::vector<float> &boxScores, std::vector<int> &classId, float threshold, int32_t zp, float scale,
                      int index)
{
    static const int input_loc_len = 64;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            for (int a = 0; a < OBJ_CLASS_NUM; a++)
            {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_i8)
                {
                    float box_conf_f32 = sigmoid(deqnt_affine_to_f32(
                        input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w], zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i)
                    {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i)
                    {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl)
                    {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0] = (w + 0.5f) - xywh_[0];
                    xywh_[1] = (h + 0.5f) - xywh_[1];
                    xywh_[2] = (w + 0.5f) + xywh_[2];
                    xywh_[3] = (h + 0.5f) + xywh_[3];
                    xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                    xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                    xywh[0] = xywh[0] - xywh[2] / 2;
                    xywh[1] = xywh[1] - xywh[3] / 2;
                    boxes.push_back(xywh[0]);
                    boxes.push_back(xywh[1]);
                    boxes.push_back(xywh[2]);
                    boxes.push_back(xywh[3]);
                    boxes.push_back((float)(index + (h * grid_w) + w)); // keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_u8(uint8_t *input, int grid_h, int grid_w, int stride, std::vector<float> &boxes,
                      std::vector<float> &boxScores, std::vector<int> &classId, float threshold, int32_t zp, float scale,
                      int index)
{
    static const int input_loc_len = 64;
    int validCount = 0;

    uint8_t thres_i8 = qnt_f32_to_affine_u8(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            for (int a = 0; a < OBJ_CLASS_NUM; a++)
            {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_i8)
                {
                    float box_conf_f32 = sigmoid(deqnt_affine_u8_to_f32(
                        input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w], zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i)
                    {
                        loc[i] = deqnt_affine_u8_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i)
                    {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl)
                    {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0] = (w + 0.5f) - xywh_[0];
                    xywh_[1] = (h + 0.5f) - xywh_[1];
                    xywh_[2] = (w + 0.5f) + xywh_[2];
                    xywh_[3] = (h + 0.5f) + xywh_[3];
                    xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                    xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                    xywh[0] = xywh[0] - xywh[2] / 2;
                    xywh[1] = xywh[1] - xywh[3] / 2;
                    boxes.push_back(xywh[0]);
                    boxes.push_back(xywh[1]);
                    boxes.push_back(xywh[2]);
                    boxes.push_back(xywh[3]);
                    boxes.push_back((float)(index + (h * grid_w) + w)); // keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_fp32(float *input, int grid_h, int grid_w, int stride, std::vector<float> &boxes,
                        std::vector<float> &boxScores, std::vector<int> &classId, float threshold, int32_t zp,
                        float scale, int index)
{
    (void)zp;
    (void)scale;
    static const int input_loc_len = 64;
    int validCount = 0;
    float thres_fp = unsigmoid(threshold);
    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            for (int a = 0; a < OBJ_CLASS_NUM; a++)
            {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_fp)
                {
                    float box_conf_f32 = sigmoid(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w]);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i)
                    {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i)
                    {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl)
                    {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0] = (w + 0.5f) - xywh_[0];
                    xywh_[1] = (h + 0.5f) - xywh_[1];
                    xywh_[2] = (w + 0.5f) + xywh_[2];
                    xywh_[3] = (h + 0.5f) + xywh_[3];
                    xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                    xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                    xywh[0] = xywh[0] - xywh[2] / 2;
                    xywh[1] = xywh[1] - xywh[3] / 2;
                    boxes.push_back(xywh[0]);
                    boxes.push_back(xywh[1]);
                    boxes.push_back(xywh[2]);
                    boxes.push_back(xywh[3]);
                    boxes.push_back((float)(index + (h * grid_w) + w)); // keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold,
                 object_detect_result_list *od_results)
{
#if defined(RV1106_1103)
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)outputs;
#else
    rknn_output *_outputs = (rknn_output *)outputs;
#endif
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    if (app_ctx->io_num.n_output <= 3)
    {
        printf("pose post_process expects 4 outputs, but got %d\n", app_ctx->io_num.n_output);
        return -1;
    }
    int keypoint_count = app_ctx->output_attrs[3].n_elems / (17 * 3);
    if (keypoint_count <= 0)
    {
        keypoint_count = 8400;
    }
    memset(od_results, 0, sizeof(object_detect_result_list));
    int index = 0;

#ifdef RKNPU1
    for (int i = 0; i < 3; i++)
    {
        grid_h = app_ctx->output_attrs[i].dims[1];
        grid_w = app_ctx->output_attrs[i].dims[0];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant)
        {
            validCount += process_u8((uint8_t *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs, classId,
                                     conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        else
        {
            validCount += process_fp32((float *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs, classId,
                                       conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        index += grid_h * grid_w;
    }
#else
    for (int i = 0; i < 3; i++)
    {
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant)
        {
            validCount += process_i8((int8_t *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs, classId,
                                     conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        else
        {
            validCount += process_fp32((float *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs, classId,
                                       conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        index += grid_h * grid_w;
    }
#endif

    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }

    std::vector<float> candidate_kpt_mean_norm(validCount, 0.0f);
    uint64_t frame_raw_conf_count = 0;
    uint64_t frame_raw_conf_neg = 0;
    uint64_t frame_raw_conf_in_01 = 0;
    uint64_t frame_raw_conf_gt1 = 0;
    double frame_raw_conf_sum = 0.0;
    for (int n = 0; n < validCount; ++n)
    {
        int keypoints_index = (int)filterBoxes[n * 5 + 4];
        float kpt_conf_sum_norm = 0.0f;
        for (int j = 0; j < 17; ++j)
        {
            float raw_conf = read_kpt_raw_conf(app_ctx, _outputs, keypoint_count, j, keypoints_index);
            if (raw_conf < 0.0f)
            {
                frame_raw_conf_neg++;
            }
            else if (raw_conf <= 1.0f)
            {
                frame_raw_conf_in_01++;
            }
            else
            {
                frame_raw_conf_gt1++;
            }
            frame_raw_conf_count++;
            frame_raw_conf_sum += raw_conf;
            kpt_conf_sum_norm += normalize_kpt_conf_for_diag(raw_conf);
        }
        candidate_kpt_mean_norm[n] = kpt_conf_sum_norm / 17.0f;
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    std::vector<int> sorted_before_nms = indexArray;

    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (std::set<int>::iterator it = class_set.begin(); it != class_set.end(); ++it)
    {
        nms(validCount, filterBoxes, classId, indexArray, *it, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];
        float x1 = filterBoxes[n * 5 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 5 + 1] - letter_box->y_pad;
        float w = filterBoxes[n * 5 + 2];
        float h = filterBoxes[n * 5 + 3];
        int keypoints_index = (int)filterBoxes[n * 5 + 4];
        if (keypoints_index < 0)
        {
            keypoints_index = 0;
        }
        if (keypoints_index >= keypoint_count)
        {
            keypoints_index = keypoint_count - 1;
        }

        for (int j = 0; j < 17; ++j)
        {
            if (app_ctx->is_quant)
            {
#ifdef RKNPU1
                od_results->results[last_count].keypoints[j][0] =
                    (deqnt_affine_u8_to_f32(((uint8_t *)_outputs[3].buf)[j * 3 * keypoint_count + 0 * keypoint_count + keypoints_index],
                                            app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale) -
                     letter_box->x_pad) /
                    letter_box->scale;
                od_results->results[last_count].keypoints[j][1] =
                    (deqnt_affine_u8_to_f32(((uint8_t *)_outputs[3].buf)[j * 3 * keypoint_count + 1 * keypoint_count + keypoints_index],
                                            app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale) -
                     letter_box->y_pad) /
                    letter_box->scale;
                od_results->results[last_count].keypoints[j][2] =
                    deqnt_affine_u8_to_f32(((uint8_t *)_outputs[3].buf)[j * 3 * keypoint_count + 2 * keypoint_count + keypoints_index],
                                           app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale);
#else
                od_results->results[last_count].keypoints[j][0] =
                    (((float)((rknpu2::float16 *)_outputs[3].buf)[j * 3 * keypoint_count + 0 * keypoint_count + keypoints_index]) -
                     letter_box->x_pad) /
                    letter_box->scale;
                od_results->results[last_count].keypoints[j][1] =
                    (((float)((rknpu2::float16 *)_outputs[3].buf)[j * 3 * keypoint_count + 1 * keypoint_count + keypoints_index]) -
                     letter_box->y_pad) /
                    letter_box->scale;
                od_results->results[last_count].keypoints[j][2] =
                    (float)((rknpu2::float16 *)_outputs[3].buf)[j * 3 * keypoint_count + 2 * keypoint_count + keypoints_index];
#endif
            }
            else
            {
                od_results->results[last_count].keypoints[j][0] =
                    (((float *)_outputs[3].buf)[j * 3 * keypoint_count + 0 * keypoint_count + keypoints_index] - letter_box->x_pad) /
                    letter_box->scale;
                od_results->results[last_count].keypoints[j][1] =
                    (((float *)_outputs[3].buf)[j * 3 * keypoint_count + 1 * keypoint_count + keypoints_index] - letter_box->y_pad) /
                    letter_box->scale;
                od_results->results[last_count].keypoints[j][2] =
                    ((float *)_outputs[3].buf)[j * 3 * keypoint_count + 2 * keypoint_count + keypoints_index];
            }
        }

        int id = classId[n];
        float obj_conf = objProbs[i];
        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x1 + w, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y1 + h, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;

    bool frame_nms_misselection = false;
    float best_kept_obj_conf = -1.0f;
    float best_kept_kpt_mean = 0.0f;
    float best_supp_kpt_mean = 0.0f;
    float best_supp_obj_conf = 0.0f;
    int best_kept_n = -1;
    int best_supp_n = -1;
    for (int i = 0; i < validCount; ++i)
    {
        int n = indexArray[i];
        if (n == -1)
        {
            continue;
        }
        if (objProbs[i] > best_kept_obj_conf)
        {
            best_kept_obj_conf = objProbs[i];
            best_kept_kpt_mean = candidate_kpt_mean_norm[n];
            best_kept_n = n;
        }
    }
    if (best_kept_n >= 0)
    {
        const float kx1 = filterBoxes[best_kept_n * 5 + 0];
        const float ky1 = filterBoxes[best_kept_n * 5 + 1];
        const float kw = filterBoxes[best_kept_n * 5 + 2];
        const float kh = filterBoxes[best_kept_n * 5 + 3];
        const float kx2 = kx1 + kw;
        const float ky2 = ky1 + kh;
        for (int i = 0; i < validCount; ++i)
        {
            if (indexArray[i] != -1)
            {
                continue;
            }
            int n = sorted_before_nms[i];
            if (n < 0 || classId[n] != classId[best_kept_n])
            {
                continue;
            }
            float sx1 = filterBoxes[n * 5 + 0];
            float sy1 = filterBoxes[n * 5 + 1];
            float sw = filterBoxes[n * 5 + 2];
            float sh = filterBoxes[n * 5 + 3];
            float sx2 = sx1 + sw;
            float sy2 = sy1 + sh;
            float iou = CalculateOverlap(kx1, ky1, kx2, ky2, sx1, sy1, sx2, sy2);
            if (iou < POSE_DIAG_NMS_IOU_THRESH)
            {
                continue;
            }
            if (candidate_kpt_mean_norm[n] > best_supp_kpt_mean)
            {
                best_supp_kpt_mean = candidate_kpt_mean_norm[n];
                best_supp_obj_conf = objProbs[i];
                best_supp_n = n;
            }
        }
        if (best_supp_n >= 0 && (best_supp_kpt_mean - best_kept_kpt_mean) > POSE_DIAG_NMS_KPT_MEAN_GAP_THRESH)
        {
            frame_nms_misselection = true;
            g_pose_diag_nms_misselection.store(1);
        }
    }

    float frame_raw_oor_ratio = 0.0f;
    if (frame_raw_conf_count > 0)
    {
        frame_raw_oor_ratio = (float)(frame_raw_conf_neg + frame_raw_conf_gt1) / (float)frame_raw_conf_count;
    }
    if (frame_raw_oor_ratio > POSE_DIAG_SEMANTIC_OOR_RATIO_THRESH)
    {
        g_pose_diag_semantic_mismatch.store(1);
    }

    if (POSE_DIAG_ENABLE)
    {
        std::lock_guard<std::mutex> lock(g_pose_post_diag_mutex);
        g_pose_post_diag_agg.frame_count++;
        g_pose_post_diag_agg.candidates_before_nms_sum += validCount;
        g_pose_post_diag_agg.candidates_after_nms_sum += last_count;
        g_pose_post_diag_agg.raw_conf_count += frame_raw_conf_count;
        g_pose_post_diag_agg.raw_conf_neg += frame_raw_conf_neg;
        g_pose_post_diag_agg.raw_conf_in_01 += frame_raw_conf_in_01;
        g_pose_post_diag_agg.raw_conf_gt1 += frame_raw_conf_gt1;
        g_pose_post_diag_agg.raw_conf_sum += frame_raw_conf_sum;
        if (frame_nms_misselection)
        {
            g_pose_post_diag_agg.nms_misselection_events++;
        }

        const bool print_now = (g_pose_post_diag_agg.frame_count % POSE_DIAG_PRINT_EVERY_N_FRAMES == 0);
        if (print_now)
        {
            float frame_raw_mean = (frame_raw_conf_count > 0) ? (float)(frame_raw_conf_sum / (double)frame_raw_conf_count) : 0.0f;
            printf("[POSE_DIAG][POST][frame=%llu] pre_nms=%d post_nms=%d raw_conf: neg=%llu in01=%llu gt1=%llu mean=%.4f oor_ratio=%.3f\n",
                   (unsigned long long)g_pose_post_diag_agg.frame_count,
                   validCount, last_count,
                   (unsigned long long)frame_raw_conf_neg,
                   (unsigned long long)frame_raw_conf_in_01,
                   (unsigned long long)frame_raw_conf_gt1,
                   frame_raw_mean, frame_raw_oor_ratio);
            for (int i = 0; i < validCount; ++i)
            {
                int n = indexArray[i];
                if (n == -1)
                {
                    continue;
                }
                printf("[POSE_DIAG][POST][kept] rank=%d obj=%.3f kpt_mean=%.3f box=(%.1f %.1f %.1f %.1f)\n",
                       i, objProbs[i], candidate_kpt_mean_norm[n],
                       filterBoxes[n * 5 + 0], filterBoxes[n * 5 + 1],
                       filterBoxes[n * 5 + 2], filterBoxes[n * 5 + 3]);
            }
            if (frame_nms_misselection)
            {
                printf("[POSE_DIAG][POST][nms_misselection] kept_obj=%.3f kept_kpt_mean=%.3f supp_obj=%.3f supp_kpt_mean=%.3f gap=%.3f\n",
                       best_kept_obj_conf, best_kept_kpt_mean, best_supp_obj_conf, best_supp_kpt_mean,
                       best_supp_kpt_mean - best_kept_kpt_mean);
            }
            float agg_raw_oor_ratio = 0.0f;
            float agg_raw_mean = 0.0f;
            if (g_pose_post_diag_agg.raw_conf_count > 0)
            {
                agg_raw_oor_ratio = (float)(g_pose_post_diag_agg.raw_conf_neg + g_pose_post_diag_agg.raw_conf_gt1) /
                                    (float)g_pose_post_diag_agg.raw_conf_count;
                agg_raw_mean = (float)(g_pose_post_diag_agg.raw_conf_sum / (double)g_pose_post_diag_agg.raw_conf_count);
            }
            printf("[POSE_DIAG][POST][agg] frames=%llu pre_nms_avg=%.2f post_nms_avg=%.2f raw_conf_mean=%.4f raw_oor_ratio=%.3f nms_misselection_events=%llu\n",
                   (unsigned long long)g_pose_post_diag_agg.frame_count,
                   (g_pose_post_diag_agg.frame_count > 0)
                       ? (double)g_pose_post_diag_agg.candidates_before_nms_sum / (double)g_pose_post_diag_agg.frame_count
                       : 0.0,
                   (g_pose_post_diag_agg.frame_count > 0)
                       ? (double)g_pose_post_diag_agg.candidates_after_nms_sum / (double)g_pose_post_diag_agg.frame_count
                       : 0.0,
                   agg_raw_mean, agg_raw_oor_ratio,
                   (unsigned long long)g_pose_post_diag_agg.nms_misselection_events);
        }
    }

    return 0;
}

int init_post_process()
{
    int ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{
    if (cls_id >= OBJ_CLASS_NUM)
    {
        return (char *)"null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return (char *)"null";
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != NULL)
        {
            free(labels[i]);
            labels[i] = NULL;
        }
    }
}
