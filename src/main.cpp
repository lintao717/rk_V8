/****************************************************************
 * @file main.c
 * @brief 基于泰山派的AI网络摄像头 - 实时视频采集、推理、编码、推流
 *
 * 本项目参考了以下开源项目和库：
 * - luckfox_pico_rkmpi_example
 *   - 项目地址：https://github.com/LuckfoxTECH/luckfox_pico_rkmpi_example
 *   - 许可证：未发现
 *   - 用途：未发现
 * - ZLMediaKit
 *   - 项目地址：https://github.com/ZLMediaKit/ZLMediaKit
 *   - 许可证：MIT
 *   - 用途：用于实现高性能的流媒体服务器功能，支持多种协议（如 RTSP、RTMP、HLS、HTTP-FLV、WebSocket-FLV 等）。
 * - rknn_model_zoo
 *   - 项目地址：https://github.com/airockchip/rknn_model_zoo
 *   - 许可证：Apache License 2.0
 *   - 用途：用于 RKNN 模型的部署示例，支持多种主流算法和模型，包括分类、目标检测、图像分割等。
 * - rockit
 * 
 * @copyright Copyright (C) 2025 Code0bug
 * @license GNU Affero General Public License v3.0 (AGPL-3.0)
 *
 * 本项目采用 AGPL-3.0 许可证。你可以自由地使用、修改和分发本项目，
 * 但必须遵循以下条款：
 * - 如果你对本项目进行了修改或基于本项目创建了衍生作品，
 *   你必须将修改后的代码开源，并采用相同的 AGPL-3.0 许可证。
 * - 如果你通过网络服务（如 SaaS）提供基于本项目的功能，
 *   你必须公开修改后的源代码。
 * - 保留原始代码中的版权声明和许可信息。
 *
 * 更多详情请参阅 AGPL-3.0 许可证：
 * https://www.gnu.org/licenses/agpl-3.0.html
 *
 * 如果你有任何疑问或是建议，请联系 peng0808@mail.nwpu.edu.cn
 *****************************************************************/
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include "yolov8_pose.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "mk_mediakit.h"
#include "ThreadPool.h"
#include "rk_mpi.h"


// rknn 相关变量
rknn_app_context_t rknn_app_ctx;
const char *model_path = "/root/rknn_yolov8_pose_demo/model/yolov8n-pose.rknn";

// mpi 相关变量
int width    = DISP_WIDTH;
int height   = DISP_HEIGHT;
VENC_STREAM_S stFrame;	 	    // 编码码流结构体，用于存储读取到的编码流
VIDEO_FRAME_INFO_S h264_frame;  // 编码帧信息结构体
unsigned char *data = nullptr;  // 指向缓存块的指针

// 用于fps显示
char fps_text[16];
float fps = 0;
int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
                    7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
static const int KEYPOINT_NUM = 17;
static const float DRAW_BOX_THRESH = 0.82f;
static const float DRAW_POINT_THRESH = 0.05f;
static const float DRAW_LINE_THRESH = 0.03f;
static const int DRAW_MIN_KPT_NUM = 1;
static const float DRAW_MIN_BOX_AREA_RATIO = 0.015f;
static const float TRACK_MATCH_IOU_THRESH = 0.30f;
static const float TRACK_PREV_KPT_MIN_CONF = 0.05f;
static const float TRACK_EMA_ALPHA = 0.60f;
static const int TRACK_MAX_MISSED_FOR_REPAIR = 5;
static const int TRACK_MAX_MISSED_KEEP = 8;
static const int TRACK_FRAME_MS = 66;

// 线程池
ThreadPool rknnPool(2);
ThreadPool h264encPool(1);

// ZLMediaKit 媒体变量
mk_media media;

struct PoseTrack {
	uint64_t track_id;
	image_rect_t box;
	float keypoints[KEYPOINT_NUM][3];
	int missed_frames;
	uint64_t last_ts_ms;
};

std::vector<PoseTrack> g_pose_tracks;
std::mutex g_pose_track_mutex;
std::atomic<uint64_t> g_next_track_id(1);

static uint64_t get_now_ms() {
	auto now = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
	return (uint64_t)duration.count();
}

static int calc_missed_frames(uint64_t now_ms, uint64_t last_ms) {
	if (now_ms <= last_ms) {
		return 0;
	}
	return (int)((now_ms - last_ms) / TRACK_FRAME_MS);
}

static float calc_iou(const image_rect_t &a, const image_rect_t &b) {
	int left = std::max(a.left, b.left);
	int top = std::max(a.top, b.top);
	int right = std::min(a.right, b.right);
	int bottom = std::min(a.bottom, b.bottom);
	int inter_w = std::max(0, right - left);
	int inter_h = std::max(0, bottom - top);
	float inter_area = (float)(inter_w * inter_h);
	float area_a = (float)(std::max(0, a.right - a.left) * std::max(0, a.bottom - a.top));
	float area_b = (float)(std::max(0, b.right - b.left) * std::max(0, b.bottom - b.top));
	float union_area = area_a + area_b - inter_area;
	if (union_area <= 0.0f) {
		return 0.0f;
	}
	return inter_area / union_area;
}

static void cleanup_pose_tracks_locked(uint64_t now_ms) {
	g_pose_tracks.erase(std::remove_if(g_pose_tracks.begin(), g_pose_tracks.end(),
									   [now_ms](PoseTrack &track) {
										   track.missed_frames = calc_missed_frames(now_ms, track.last_ts_ms);
										   return track.missed_frames > TRACK_MAX_MISSED_KEEP;
									   }),
					 g_pose_tracks.end());
}

static int find_track_index_by_id_locked(uint64_t track_id) {
	for (size_t i = 0; i < g_pose_tracks.size(); ++i) {
		if (g_pose_tracks[i].track_id == track_id) {
			return (int)i;
		}
	}
	return -1;
}

static int find_best_track_index_locked(const image_rect_t &box, uint64_t now_ms, uint64_t *matched_track_id) {
	float best_iou = 0.0f;
	int best_idx = -1;
	*matched_track_id = 0;
	for (size_t i = 0; i < g_pose_tracks.size(); ++i) {
		PoseTrack &track = g_pose_tracks[i];
		track.missed_frames = calc_missed_frames(now_ms, track.last_ts_ms);
		if (track.missed_frames > TRACK_MAX_MISSED_KEEP) {
			continue;
		}
		float iou = calc_iou(track.box, box);
		if (iou >= TRACK_MATCH_IOU_THRESH && iou > best_iou) {
			best_iou = iou;
			best_idx = (int)i;
			*matched_track_id = track.track_id;
		}
	}
	return best_idx;
}

static void repair_keypoints_from_track(float keypoints[KEYPOINT_NUM][3], const PoseTrack &track) {
	if (track.missed_frames > TRACK_MAX_MISSED_FOR_REPAIR) {
		return;
	}
	for (int j = 0; j < KEYPOINT_NUM; ++j) {
		float curr_conf = keypoints[j][2];
		float prev_conf = track.keypoints[j][2];
		if (curr_conf >= DRAW_POINT_THRESH || prev_conf < TRACK_PREV_KPT_MIN_CONF) {
			continue;
		}
		if (curr_conf > 0.0f) {
			keypoints[j][0] = TRACK_EMA_ALPHA * keypoints[j][0] + (1.0f - TRACK_EMA_ALPHA) * track.keypoints[j][0];
			keypoints[j][1] = TRACK_EMA_ALPHA * keypoints[j][1] + (1.0f - TRACK_EMA_ALPHA) * track.keypoints[j][1];
		} else {
			keypoints[j][0] = track.keypoints[j][0];
			keypoints[j][1] = track.keypoints[j][1];
		}
		keypoints[j][2] = std::max(curr_conf, prev_conf * 0.9f);
	}
}

static void upsert_pose_track_locked(uint64_t matched_track_id, const image_rect_t &box,
									 float keypoints[KEYPOINT_NUM][3], uint64_t now_ms) {
	int idx = -1;
	if (matched_track_id != 0) {
		idx = find_track_index_by_id_locked(matched_track_id);
	}
	if (idx < 0) {
		PoseTrack track;
		track.track_id = g_next_track_id.fetch_add(1);
		track.box = box;
		track.missed_frames = 0;
		track.last_ts_ms = now_ms;
		for (int j = 0; j < KEYPOINT_NUM; ++j) {
			track.keypoints[j][0] = keypoints[j][0];
			track.keypoints[j][1] = keypoints[j][1];
			track.keypoints[j][2] = keypoints[j][2];
		}
		g_pose_tracks.push_back(track);
		return;
	}
	g_pose_tracks[idx].box = box;
	g_pose_tracks[idx].missed_frames = 0;
	g_pose_tracks[idx].last_ts_ms = now_ms;
	for (int j = 0; j < KEYPOINT_NUM; ++j) {
		g_pose_tracks[idx].keypoints[j][0] = keypoints[j][0];
		g_pose_tracks[idx].keypoints[j][1] = keypoints[j][1];
		g_pose_tracks[idx].keypoints[j][2] = keypoints[j][2];
	}
}

/*********************************************
 * h264编码任务，包含rtsp推流
 * 1.给图像添加fps信息
 * 2.发送视频帧给VENC进行编码
 * 3.读取编码结果并进行推流
 *********************************************/
void encode_task(cv::Mat frame) {

	sprintf(fps_text, "fps = %.2f", fps);		
	cv::putText(frame, fps_text,
					cv::Point(40, 40),
					cv::FONT_HERSHEY_SIMPLEX,1,
					cv::Scalar(0,255,0),2);


	memcpy(data, frame.data, width * height * 3);
		
	/**************************
	 * 向VENC发送原始图像进行编码
	 * 0为编码通道号
	 * h264_frame 为原始图像信息
	 * -1 表示阻塞，发送成功后释放
	 **************************/	
	RK_MPI_VENC_SendFrame(0, &h264_frame, -1);

	/**************************
	 * 获取编码码流
	 * 0为编码通道号
	 * stFrame为码流结构体指针 
	 * -1 表示阻塞，获取编码流后释放
	 **************************/
	RK_S32 s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);	
	
	if(s32Ret == RK_SUCCESS) {

		void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
		uint32_t len = stFrame.pstPack->u32Len;

		// 推流
		static int64_t time_last = 0;
		// 获取当前时间点
		auto now = std::chrono::system_clock::now();
		// 转换为自1970年1月1日以来的毫秒数
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
		// 获取毫秒级时间戳
		int64_t timestamp = duration.count();

		int64_t fps_time = timestamp - time_last;
		fps = 1000 / fps_time;

		time_last = timestamp;

		mk_frame frame = mk_frame_create(MKCodecH264, timestamp, timestamp, (char*)pData, (size_t)len, NULL, NULL);
		mk_media_input_frame(media, frame);
		mk_frame_unref(frame);
	}

	s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
	if (s32Ret != RK_SUCCESS) {
		RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
	}



}

/*********************************************
 * rknn推理任务
 * 1.执行yolov8推理并画框
 * 2.转为Mat类型，利用opencv将NV12格式转为RGB888
 * 3.转换后的Mat类型帧，提交给encode_task
 * 4.推理完成后释放VI帧（保证RGA使用期间fd有效）
 *********************************************/
void rknn_task(VIDEO_FRAME_INFO_S stViFrame) {

	// 直接使用VI帧的DMA内存，fd有效，RGA可正常工作
	image_buffer_t src_image;
	src_image.width  = stViFrame.stVFrame.u32Width;
	src_image.height = stViFrame.stVFrame.u32Height;
	src_image.format = IMAGE_FORMAT_YUV420SP_NV12;
	src_image.size   = RK_MPI_MB_GetSize(stViFrame.stVFrame.pMbBlk);
	src_image.fd     = RK_MPI_MB_Handle2Fd(stViFrame.stVFrame.pMbBlk);
	src_image.virt_addr = reinterpret_cast<unsigned char*>(
	                          RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk));

	// 执行推理
	object_detect_result_list od_results;
	int ret = inference_yolov8_pose_model(&rknn_app_ctx, &src_image, &od_results);
	if (ret != 0)
	{
		printf("inference_yolov8_pose_model fail! ret=%d\n", ret);
	}

	// 画框和概率
	char text[256];
	for (int i = 0; i < od_results.count; i++)
	{
		object_detect_result *det_result = &(od_results.results[i]);
		if (det_result->prop < DRAW_BOX_THRESH)
		{
			continue;
		}

		printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
				det_result->box.left, det_result->box.top,
				det_result->box.right, det_result->box.bottom,
				det_result->prop);
		int x1 = det_result->box.left;
		int y1 = det_result->box.top;
		int x2 = det_result->box.right;
		int y2 = det_result->box.bottom;
		int box_w = x2 - x1;
		int box_h = y2 - y1;
		if (box_w <= 0 || box_h <= 0)
		{
			continue;
		}

		float box_area_ratio = (float)(box_w * box_h) / (float)(src_image.width * src_image.height);
		if (box_area_ratio < DRAW_MIN_BOX_AREA_RATIO)
		{
			continue;
		}

		float draw_keypoints[KEYPOINT_NUM][3];
		for (int j = 0; j < KEYPOINT_NUM; ++j)
		{
			draw_keypoints[j][0] = det_result->keypoints[j][0];
			draw_keypoints[j][1] = det_result->keypoints[j][1];
			draw_keypoints[j][2] = det_result->keypoints[j][2];
		}

		uint64_t matched_track_id = 0;
		{
			uint64_t now_ms = get_now_ms();
			std::lock_guard<std::mutex> lock(g_pose_track_mutex);
			cleanup_pose_tracks_locked(now_ms);
			int matched_idx = find_best_track_index_locked(det_result->box, now_ms, &matched_track_id);
			if (matched_idx >= 0)
			{
				repair_keypoints_from_track(draw_keypoints, g_pose_tracks[matched_idx]);
			}
		}

		int valid_kpt_num = 0;
		for (int j = 0; j < KEYPOINT_NUM; ++j)
		{
			if (draw_keypoints[j][2] >= DRAW_POINT_THRESH)
			{
				valid_kpt_num++;
			}
		}
		if (valid_kpt_num < DRAW_MIN_KPT_NUM)
		{
			continue;
		}

		draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

		sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
		draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);

		for (int j = 0; j < 38 / 2; ++j)
		{
			int p1 = skeleton[2 * j] - 1;
			int p2 = skeleton[2 * j + 1] - 1;
			if (draw_keypoints[p1][2] < DRAW_LINE_THRESH || draw_keypoints[p2][2] < DRAW_LINE_THRESH)
			{
				continue;
			}

			draw_line(&src_image, (int)(draw_keypoints[skeleton[2 * j] - 1][0]),
					  (int)(draw_keypoints[skeleton[2 * j] - 1][1]),
					  (int)(draw_keypoints[skeleton[2 * j + 1] - 1][0]),
					  (int)(draw_keypoints[skeleton[2 * j + 1] - 1][1]), COLOR_ORANGE, 3);
		}

		for (int j = 0; j < KEYPOINT_NUM; ++j)
		{
			if (draw_keypoints[j][2] < DRAW_POINT_THRESH)
			{
				continue;
			}

			draw_circle(&src_image, (int)(draw_keypoints[j][0]),
						(int)(draw_keypoints[j][1]), 1, COLOR_YELLOW, 1);
		}

		{
			uint64_t now_ms = get_now_ms();
			std::lock_guard<std::mutex> lock(g_pose_track_mutex);
			cleanup_pose_tracks_locked(now_ms);
			upsert_pose_track_locked(matched_track_id, det_result->box, draw_keypoints, now_ms);
		}
	}

	// 转换为RGB888格式
	cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, src_image.virt_addr);
	cv::Mat frame(height, width, CV_8UC3);
	cv::cvtColor(yuv420sp, frame, cv::COLOR_YUV420sp2BGR);

	h264encPool.enqueue(encode_task, frame);

	// 推理和绘制完成后再释放VI帧
	RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
}

/*************************************
 * 视频捕获任务
 * 1.捕获视频帧以VIDEO_FRAME_INFO_S类型存储
 * 2.将帧直接提交给rknn_task，由其在完成后释放
 *   （保证RGA转换期间DMA内存有效）
 *************************************/
void capture_task() {

	VIDEO_FRAME_INFO_S stViFrame;
	RK_S32 s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
	if(s32Ret == RK_SUCCESS) {
		// 直接入队，VI帧由rknn_task负责释放
		rknnPool.enqueue(rknn_task, stViFrame);
	} else {
		RK_LOGE("RK_MPI_VI_GetChnFrame fail %x", s32Ret);
	}
}


int main(int argc, char *argv[]) {

	memset(fps_text, 0, 16);

	/******************************* rknn 相关初始化 *******************************/
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    init_post_process();
	    int ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
	    if (ret != 0)
	    {
	        printf("init_yolov8_pose_model fail! ret=%d model_path=%s\n", ret, model_path);
	    }

	/******************************* mpi 相关初始化 *******************************/
	RK_S32 s32Ret = 0; 

	stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));

	// 内存缓存池配置
	MB_POOL_CONFIG_S PoolCfg;
	memset(&PoolCfg, 0, sizeof(MB_POOL_CONFIG_S));
	PoolCfg.u64MBSize = width * height * 3 ;  		   // 缓存块大小 
	PoolCfg.u32MBCnt = 1;                              // 内存缓存池中缓存块个数
	PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;           // 申请内存类型
	//PoolCfg.bPreAlloc = RK_FALSE;
	MB_POOL src_Pool = RK_MPI_MB_CreatePool(&PoolCfg); // 创建内存缓存池
	printf("Create Pool success !\n");	

	// 从内存缓存池中获取一个缓存块，返回该缓存块的地址
	MB_BLK src_Blk = RK_MPI_MB_GetMB(src_Pool, width * height * 3, RK_TRUE);
	
	// 配置编码帧信息
	h264_frame.stVFrame.u32Width = width;
	h264_frame.stVFrame.u32Height = height;
	h264_frame.stVFrame.u32VirWidth = width;
	h264_frame.stVFrame.u32VirHeight = height;
	h264_frame.stVFrame.enPixelFormat =  RK_FMT_RGB888; 
	h264_frame.stVFrame.u32FrameFlag = 160;
	h264_frame.stVFrame.pMbBlk = src_Blk;   // 将帧地址关联到缓存块
	// 将缓存块地址转为虚拟地址，并创建Mat类型变量指向该地址
	data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);

	// rkmpi init 
	if (RK_MPI_SYS_Init() != RK_SUCCESS) {
		RK_LOGE("rk mpi sys init fail!");
		// return -1;
	}

	// vi init
	vi_dev_init();
	vi_chn_init(0, width, height);

	// venc init
	RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
	venc_init(0, width, height, enCodecType);

	/******************************* ZLMediaKit相关初始化 *******************************/
	char *ini_path = mk_util_get_exe_dir("./config.ini");
	mk_config config = {
		0,				// thread_num
		0,				// log_level
		LOG_CONSOLE,	// log_mask
		NULL,			// log_file_path
		0,				// log_file_days
		1,				// ini_is_path
		ini_path,		// ini 
		1,				// ssl_is_path
		NULL,			// ssl
		NULL			// ssl_pwd
	};
	mk_env_init(&config);
	mk_free(ini_path);
	// 创建rtsp服务器 8554为端口号
	mk_rtsp_server_start(8554, 0);
	// 监听事件
	mk_events events = {
            .on_mk_media_changed = NULL,
            .on_mk_media_publish = NULL,
            .on_mk_media_play = NULL,
            .on_mk_media_not_found = NULL,
            .on_mk_media_no_reader = NULL,
            .on_mk_http_request = NULL,
            .on_mk_http_access = NULL,
            .on_mk_http_before_access = NULL,
            .on_mk_rtsp_get_realm = NULL,
            .on_mk_rtsp_auth = NULL,
            .on_mk_record_mp4 = NULL,
            .on_mk_shell_login = NULL,
            .on_mk_flow_report = NULL
    };
    mk_events_listen(&events);
	// 创建媒体源  
	// 对应URL：rtsp://ip:8554/live/camera
	media = mk_media_create("__defaultVhost__", "live", "camera", 0, 0, 0);
	// 添加视频轨道
    codec_args v_args = {0};
    mk_track v_track = mk_track_create(MKCodecH264, &v_args);
    // 初始化媒体源的视频轨道
    mk_media_init_track(media, v_track);
    // 完成轨道添加
    mk_media_init_complete(media);
    // 释放资源
    mk_track_unref(v_track);

	while (1) {
		// 主线程执行视频捕获任务
		capture_task();

	}

	// 销毁资源
	mk_media_release(media);
    mk_stop_all_server();


	// Destory MB
	RK_MPI_MB_ReleaseMB(src_Blk);
	// Destory Pool
	RK_MPI_MB_DestroyPool(src_Pool);

	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableDev(0);

	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);

	free(stFrame.pstPack);
	
	RK_MPI_SYS_Exit();

	deinit_post_process();
	release_yolov8_pose_model(&rknn_app_ctx);

	return 0;
}
