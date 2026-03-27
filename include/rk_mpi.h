#ifndef _rk_mpi_h
#define _rk_mpi_h
#include <stdio.h>
#include <sys/poll.h>
#include <errno.h>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <pthread.h>

#include "rk_defines.h"
#include "rk_debug.h"
#include "rk_mpi_vi.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"
#include "rk_mpi_venc.h"
#include "rk_mpi_vpss.h"
#include "rk_mpi_vo.h"
#include "rk_mpi_rgn.h"
#include "rk_common.h"
#include "rk_comm_rgn.h"
#include "rk_comm_vi.h"
#include "rk_comm_vo.h"
#include "test_common.h"
#include "test_comm_utils.h"
#include "test_comm_argparse.h"
#include "rk_mpi_cal.h"
#include "rk_mpi_mmz.h"

#define DISP_WIDTH  640
#define DISP_HEIGHT 640

int vi_dev_init();
int vi_chn_init(int channelId, int width, int height);
int venc_init(int chnId, int width, int height, RK_CODEC_ID_E enType);

#endif