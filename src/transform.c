/*****************************************************************************
 * This file is part of Kvazaar HEVC encoder.
 *
 * Copyright (C) 2013-2015 Tampere University of Technology and others (see
 * COPYING file).
 *
 * Kvazaar is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation; either version 2.1 of the License, or (at your
 * option) any later version.
 *
 * Kvazaar is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with Kvazaar.  If not, see <http://www.gnu.org/licenses/>.
 ****************************************************************************/

#include "transform.h"

#include "image.h"
#include "kvazaar.h"
#include "rdo.h"
#include "strategies/strategies-dct.h"
#include "strategies/strategies-quant.h"
#include "strategies/strategies-picture.h"
#include "tables.h"

#include "xdct.h"
#if HW_TYPE == HW_16X16_2D

//#include "ximage_filter.c"
//#include "ximage_filter_linux.c"

#include "xdct.c"
#include "xdct_linux.c"

XDct doDCT;
u64 *src_base_address;

#include <threads.h>//
extern double DCT_16x16_real_time;
extern long count_16x16;
#endif

/**
 * \brief RDPCM direction.
 */
typedef enum rdpcm_dir {
  RDPCM_VER = 0, // vertical
  RDPCM_HOR = 1, // horizontal
} rdpcm_dir;

//////////////////////////////////////////////////////////////////////////
// INITIALIZATIONS
//


const uint8_t kvz_g_chroma_scale[58]=
{
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,
  17,18,19,20,21,22,23,24,25,26,27,28,29,29,30,31,32,
  33,33,34,34,35,35,36,36,37,37,38,39,40,41,42,43,44,
  45,46,47,48,49,50,51
};

//////////////////////////////////////////////////////////////////////////
// FUNCTIONS
//

/**
 * \brief Bypass transform and quantization.
 *
 * Copies the reference pixels directly to reconstruction and the residual
 * directly to coefficients. Used when cu_transquant_bypass_flag is set.
 * Parameters pred_in and rec_out may be aliased.
 *
 * \param width       Transform width.
 * \param in_stride   Stride for ref_in and pred_in
 * \param out_stride  Stride for rec_out.
 * \param ref_in      Reference pixels.
 * \param pred_in     Predicted pixels.
 * \param rec_out     Returns the reconstructed pixels.
 * \param coeff_out   Returns the coefficients used for reconstruction of rec_out.
 *
 * \returns  Whether coeff_out contains any non-zero coefficients.
 */
static bool bypass_transquant(const int width,
                              const int in_stride,
                              const int out_stride,
                              const kvz_pixel *const ref_in,
                              const kvz_pixel *const pred_in,
                              kvz_pixel *rec_out,
                              coeff_t *coeff_out)
{
  bool nonzero_coeffs = false;

  for (int y = 0; y < width; ++y) {
    for (int x = 0; x < width; ++x) {
      int32_t in_idx    = x + y * in_stride;
      int32_t out_idx   = x + y * out_stride;
      int32_t coeff_idx = x + y * width;

      // The residual must be computed before writing to rec_out because
      // pred_in and rec_out may point to the same array.
      coeff_t coeff        = (coeff_t)(ref_in[in_idx] - pred_in[in_idx]);
      coeff_out[coeff_idx] = coeff;
      rec_out[out_idx]     = ref_in[in_idx];

      nonzero_coeffs |= (coeff != 0);
    }
  }

  return nonzero_coeffs;
}

/**
 * Apply DPCM to residual.
 *
 * \param width   width of the block
 * \param dir     RDPCM direction
 * \param coeff   coefficients (residual) to filter
 */
static void rdpcm(const int width,
                  const rdpcm_dir dir,
                  coeff_t *coeff)
{
  const int offset = (dir == RDPCM_HOR) ? 1 : width;
  const int min_x  = (dir == RDPCM_HOR) ? 1 : 0;
  const int min_y  = (dir == RDPCM_HOR) ? 0 : 1;

  for (int y = width - 1; y >= min_y; y--) {
    for (int x = width - 1; x >= min_x; x--) {
      const int index = x + y * width;
      coeff[index] -= coeff[index - offset];
    }
  }
}

/**
 * \brief Get scaled QP used in quantization
 *
 */
int32_t kvz_get_scaled_qp(int8_t type, int8_t qp, int8_t qp_offset)
{
  int32_t qp_scaled = 0;
  if(type == 0) {
    qp_scaled = qp + qp_offset;
  } else {
    qp_scaled = CLIP(-qp_offset, 57, qp);
    if(qp_scaled < 0) {
      qp_scaled = qp_scaled + qp_offset;
    } else {
      qp_scaled = kvz_g_chroma_scale[qp_scaled] + qp_offset;
    }
  }
  return qp_scaled;
}

/**
 * \brief NxN inverse transform (2D)
 * \param coeff input data (transform coefficients)
 * \param block output data (residual)
 * \param block_size input data (width of transform)
 */
void kvz_transformskip(const encoder_control_t * const encoder, int16_t *block,int16_t *coeff, int8_t block_size)
{
  uint32_t log2_tr_size =  kvz_g_convert_to_bit[block_size] + 2;
  int32_t  shift = MAX_TR_DYNAMIC_RANGE - encoder->bitdepth - log2_tr_size;
  int32_t  j,k;
  for (j = 0; j < block_size; j++) {
    for(k = 0; k < block_size; k ++) {
      coeff[j * block_size + k] = block[j * block_size + k] << shift;
    }
  }
}

/**
 * \brief inverse transform skip
 * \param coeff input data (transform coefficients)
 * \param block output data (residual)
 * \param block_size width of transform
 */
void kvz_itransformskip(const encoder_control_t * const encoder, int16_t *block,int16_t *coeff, int8_t block_size)
{
  uint32_t log2_tr_size =  kvz_g_convert_to_bit[block_size] + 2;
  int32_t  shift = MAX_TR_DYNAMIC_RANGE - encoder->bitdepth - log2_tr_size;
  int32_t  j,k;
  int32_t offset;
  offset = (1 << (shift -1)); // For rounding
  for ( j = 0; j < block_size; j++ ) {
    for(k = 0; k < block_size; k ++) {
      block[j * block_size + k] =  (coeff[j * block_size + k] + offset) >> shift;
    }
  }
}

/**
 * \brief forward transform (2D)
 * \param block input residual
 * \param coeff transform coefficients
 * \param block_size width of transform
 */
#if HW_TYPE == HW_16X16_2D
const char name[] = "dct";

int DCT_2D_HW_init(void)
{
	//int i, j;
	int status;
	//int shift_1st = 1;
	//int shift_2nd = 8;
	//int getshift;
	int fd_src;
	unsigned int size;
	//int16_t x[]={-16, -18, -18, -18,
	//                -19, -20, -19, -19,
	//                -21, -20, -19, -21,
	//                -20, -19, -19, -20};
	/* Step 1, open the UIO device file to allow access to control the device */

	//fd_src = open("/dev/uio0", O_RDWR);
	//if (fd_src < 1) {
	//					printf("Unable to open UIO device file \n");
	//					return -1;
	//                }

	printf("DCT Initiliaze \n");
	fd_src = open("/dev/mem", O_RDWR| O_SYNC);
	if (fd_src < 1) {
		printf("Unable to open BRAM controller  \n");
		return -1;
	}


	size = 0x8000;
    off_t bram_1_pbase = 0xA0010000; // physical base address of BRAM 1   // SG_Mod


	/* Step 4, map the device memory into the process address space so that it can be
 	 * accessed
 	 */

	//src_base_address =        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_src, 0);
      src_base_address = (u64 *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_src, bram_1_pbase);    // SG_Mod
	printf("Test DCT 16x16 HW \n");
	status = XDct_Initialize(&doDCT, name);

	if(status != XST_SUCCESS)
	{
		printf("Error initializing DCT core \n");
		return -1;
	}
	return 0;
}

#endif
void kvz_transform2d(const encoder_control_t * const encoder,
                     int16_t *block,
                     int16_t *coeff,
                     int8_t block_size,
                     color_t color,
                     cu_type_t type)
{
  dct_func *dct_func = kvz_get_dct_func(block_size, color, type);
  dct_func(encoder->bitdepth, block, coeff);
}

#if HW_TYPE == HW_16X16_2D
void kvz_transform2d16X16_hw(int16_t *block, int16_t *coeff)
{
	  KVZ_CLOCK_T encoding_DCT_start_time;
	  KVZ_CLOCK_T encoding_DCT_end_time;
	  count_16x16++;
		int i,j;
		int32_t src[128];
	    int32_t src_reg;
		int16_t reg1;
		int16_t reg0;
		j=0;
		for( i=0 ; i < 128 ; i++)
		{
			//65535 is decimal value of (0000ffff)
			src[i] = ((block[j] & 65535) << 16) | (block[j+1] & 65535);
			j += 2;
		}
		KVZ_GET_TIME(&encoding_DCT_start_time);
	//	printf("b");
	//	printf("\n");
		//pthread_mutex_lock(&count_mutex);
		for (i = 0; i < 128; i++) {
			XDct_WriteReg(src_base_address,i,src[i]);
		}
//		memcpy(src_base_address,block,256);
		//XPartial_butterfly_32_generic_try_Set_shift(&doDCT,shift);
		XDct_Start(&doDCT);
	//	printf("a ");
	//	printf("\n");
		while(!XDct_IsDone(&doDCT));
	//	printf("d ");
	//			printf("\n");
//		for(i=0; i < 1024; i++)
//			XDct_WriteReg(src_base_address,i,XDct_ReadReg(dst_base_address, i));
//		}
		j=0;
		for(i=0; i < 128; i++)
		{
			src_reg = XDct_ReadReg(src_base_address, i);
			reg1 = src_reg;
			reg0 = src_reg >> 16;
			coeff[j] = reg0;
			coeff[j + 1] = reg1;
			j += 2;
		}
//		memcpy(coeff,src_base_address,256);
	//	printf("d ");
	//	printf("\n");
		//pthread_mutex_unlock(&count_mutex);
		KVZ_GET_TIME(&encoding_DCT_end_time);
		DCT_16x16_real_time +=  KVZ_CLOCK_T_AS_DOUBLE(encoding_DCT_end_time) - KVZ_CLOCK_T_AS_DOUBLE(encoding_DCT_start_time);
}
#endif

void kvz_itransform2d(const encoder_control_t * const encoder,
                      int16_t *block,
                      int16_t *coeff,
                      int8_t block_size,
                      color_t color,
                      cu_type_t type)
{
  dct_func *idct_func = kvz_get_idct_func(block_size, color, type);
  idct_func(encoder->bitdepth, coeff, block);
}

/**
 * \brief Like kvz_quantize_residual except that this uses trskip if that is better.
 *
 * Using this function saves one step of quantization and inverse quantization
 * compared to doing the decision separately from the actual operation.
 *
 * \param width  Transform width.
 * \param color  Color.
 * \param scan_order  Coefficient scan order.
 * \param trskip_out  Whether transform skip is used.
 * \param stride  Stride for ref_in, pred_in and rec_out.
 * \param ref_in  Reference pixels.
 * \param pred_in  Predicted pixels.
 * \param rec_out  Reconstructed pixels.
 * \param coeff_out  Coefficients used for reconstruction of rec_out.
 *
 * \returns  Whether coeff_out contains any non-zero coefficients.
 */
int kvz_quantize_residual_trskip(
    encoder_state_t *const state,
    const cu_info_t *const cur_cu, const int width, const color_t color,
    const coeff_scan_order_t scan_order, int8_t *trskip_out, 
    const int in_stride, const int out_stride,
    const kvz_pixel *const ref_in, const kvz_pixel *const pred_in, 
    kvz_pixel *rec_out, coeff_t *coeff_out)
{
  struct {
    kvz_pixel rec[4*4];
    coeff_t coeff[4*4];
    uint32_t cost;
    int has_coeffs;
  } skip, noskip, *best;

  const int bit_cost = (int)(state->lambda + 0.5);
  
  noskip.has_coeffs = kvz_quantize_residual(
      state, cur_cu, width, color, scan_order,
      0, in_stride, 4,
      ref_in, pred_in, noskip.rec, noskip.coeff);
  noskip.cost = kvz_pixels_calc_ssd(ref_in, noskip.rec, in_stride, 4, 4);
  noskip.cost += kvz_get_coeff_cost(state, noskip.coeff, 4, 0, scan_order) * bit_cost;

  skip.has_coeffs = kvz_quantize_residual(
    state, cur_cu, width, color, scan_order,
    1, in_stride, 4,
    ref_in, pred_in, skip.rec, skip.coeff);
  skip.cost = kvz_pixels_calc_ssd(ref_in, skip.rec, in_stride, 4, 4);
  skip.cost += kvz_get_coeff_cost(state, skip.coeff, 4, 0, scan_order) * bit_cost;

  if (noskip.cost <= skip.cost) {
    *trskip_out = 0;
    best = &noskip;
  } else {
    *trskip_out = 1;
    best = &skip;
  }

  if (best->has_coeffs || rec_out != pred_in) {
    // If there is no residual and reconstruction is already in rec_out, 
    // we can skip this.
    kvz_pixels_blit(best->rec, rec_out, width, width, 4, out_stride);
  }
  copy_coeffs(best->coeff, coeff_out, width);

  return best->has_coeffs;
}

/**
 * Calculate the residual coefficients for a single TU.
 */
static void quantize_tr_residual(encoder_state_t * const state,
                                 const color_t color,
                                 const int32_t x,
                                 const int32_t y,
                                 const uint8_t depth,
                                 cu_info_t *cur_pu,
                                 lcu_t* lcu)
{
  const kvz_config *cfg    = &state->encoder_control->cfg;
  const int32_t shift      = color == COLOR_Y ? 0 : 1;
  const vector2d_t lcu_px  = { SUB_SCU(x) >> shift, SUB_SCU(y) >> shift };

  // If luma is 4x4, do chroma for the 8x8 luma area when handling the top
  // left PU because the coordinates are correct.
  bool handled_elsewhere = color != COLOR_Y &&
                           depth > MAX_DEPTH &&
                           (lcu_px.x % 4 != 0 || lcu_px.y % 4 != 0);
  if (handled_elsewhere) {
    return;
  }

  // Clear coded block flag structures for depths lower than current depth.
  // This should ensure that the CBF data doesn't get corrupted if this function
  // is called more than once.
  cbf_clear(&cur_pu->cbf, depth, color);

  int32_t tr_width;
  if (color == COLOR_Y) {
    tr_width = LCU_WIDTH >> depth;
  } else {
    const int chroma_depth = (depth == MAX_PU_DEPTH ? depth - 1 : depth);
    tr_width = LCU_WIDTH_C >> chroma_depth;
  }
  const int32_t lcu_width = LCU_WIDTH >> shift;
  const int8_t mode =
    (color == COLOR_Y) ? cur_pu->intra.mode : cur_pu->intra.mode_chroma;
  const coeff_scan_order_t scan_idx =
    kvz_get_scan_order(cur_pu->type, mode, depth);
  const int offset = lcu_px.x + lcu_px.y * lcu_width;
  const int z_index = xy_to_zorder(lcu_width, lcu_px.x, lcu_px.y);

  // Pointers to current location in arrays with prediction. The
  // reconstruction will be written to this array.
  kvz_pixel *pred = NULL;
  // Pointers to current location in arrays with reference.
  const kvz_pixel *ref = NULL;
  // Pointers to current location in arrays with quantized coefficients.
  coeff_t *coeff = NULL;

  switch (color) {
    case COLOR_Y:
      pred  = &lcu->rec.y[offset];
      ref   = &lcu->ref.y[offset];
      coeff = &lcu->coeff.y[z_index];
      break;
    case COLOR_U:
      pred = &lcu->rec.u[offset];
      ref  = &lcu->ref.u[offset];
      coeff = &lcu->coeff.u[z_index];
      break;
    case COLOR_V:
      pred = &lcu->rec.v[offset];
      ref  = &lcu->ref.v[offset];
      coeff = &lcu->coeff.v[z_index];
      break;
  }

  const bool can_use_trskip = tr_width == 4 &&
                              color == COLOR_Y &&
                              cfg->trskip_enable;

  bool has_coeffs;

  if (cfg->lossless) {
    has_coeffs = bypass_transquant(tr_width,
                                   lcu_width, // in stride
                                   lcu_width, // out stride
                                   ref,
                                   pred,
                                   pred,
                                   coeff);
    if (cfg->implicit_rdpcm && cur_pu->type == CU_INTRA) {
      // implicit rdpcm for horizontal and vertical intra modes
      if (mode == 10) {
        rdpcm(tr_width, RDPCM_HOR, coeff);
      } else if (mode == 26) {
        rdpcm(tr_width, RDPCM_VER, coeff);
      }
    }

  } else if (can_use_trskip) {
    int8_t tr_skip = 0;

    // Try quantization with trskip and use it if it's better.
    has_coeffs = kvz_quantize_residual_trskip(state,
                                              cur_pu,
                                              tr_width,
                                              color,
                                              scan_idx,
                                              &tr_skip,
                                              lcu_width,
                                              lcu_width,
                                              ref,
                                              pred,
                                              pred,
                                              coeff);
    cur_pu->tr_skip = tr_skip;
  } else {
    has_coeffs = kvz_quantize_residual(state,
                                       cur_pu,
                                       tr_width,
                                       color,
                                       scan_idx,
                                       false, // tr skip
                                       lcu_width,
                                       lcu_width,
                                       ref,
                                       pred,
                                       pred,
                                       coeff);
  }

  if (has_coeffs) {
    cbf_set(&cur_pu->cbf, depth, color);
  }
}

/**
 * This function calculates the residual coefficients for a region of the LCU
 * (defined by x, y and depth) and updates the reconstruction with the
 * kvantized residual. Processes the TU tree recursively.
 *
 * Inputs are:
 * - lcu->rec  pixels after prediction for the area
 * - lcu->ref  reference pixels for the area
 * - lcu->cu   for the area
 *
 * Outputs are:
 * - lcu->rec               reconstruction after quantized residual
 * - lcu->coeff             quantized coefficients for the area
 * - lcu->cbf               coded block flags for the area
 * - lcu->cu.intra.tr_skip  tr skip flags for the area (in case of luma)
 */
void kvz_quantize_lcu_residual(encoder_state_t * const state,
                               const bool luma,
                               const bool chroma,
                               const int32_t x,
                               const int32_t y,
                               const uint8_t depth,
                               cu_info_t *cur_pu,
                               lcu_t* lcu)
{
  const int32_t width = LCU_WIDTH >> depth;
  const vector2d_t lcu_px  = { SUB_SCU(x), SUB_SCU(y) };

  if (cur_pu == NULL) {
    cur_pu = LCU_GET_CU_AT_PX(lcu, lcu_px.x, lcu_px.y);
  }

  // Tell clang-analyzer what is up. For some reason it can't figure out from
  // asserting just depth.
  assert(width ==  4 ||
         width ==  8 ||
         width == 16 ||
         width == 32 ||
         width == 64);

  if (depth == 0 || cur_pu->tr_depth > depth) {
    // Split transform and increase depth
    const int offset = width / 2;
    const int32_t x2 = x + offset;
    const int32_t y2 = y + offset;

    kvz_quantize_lcu_residual(state, luma, chroma, x,  y,  depth + 1, NULL, lcu);
    kvz_quantize_lcu_residual(state, luma, chroma, x2, y,  depth + 1, NULL, lcu);
    kvz_quantize_lcu_residual(state, luma, chroma, x,  y2, depth + 1, NULL, lcu);
    kvz_quantize_lcu_residual(state, luma, chroma, x2, y2, depth + 1, NULL, lcu);

    // Propagate coded block flags from child CUs to parent CU.
    uint16_t child_cbfs[3] = {
      LCU_GET_CU_AT_PX(lcu, lcu_px.x + offset, lcu_px.y         )->cbf,
      LCU_GET_CU_AT_PX(lcu, lcu_px.x,          lcu_px.y + offset)->cbf,
      LCU_GET_CU_AT_PX(lcu, lcu_px.x + offset, lcu_px.y + offset)->cbf,
    };

    if (depth <= MAX_DEPTH) {
      cbf_set_conditionally(&cur_pu->cbf, child_cbfs, depth, COLOR_Y);
      cbf_set_conditionally(&cur_pu->cbf, child_cbfs, depth, COLOR_U);
      cbf_set_conditionally(&cur_pu->cbf, child_cbfs, depth, COLOR_V);
    }

  } else {
    // Process a leaf TU.
    if (luma) {
      quantize_tr_residual(state, COLOR_Y, x, y, depth, cur_pu, lcu);
    }
    if (chroma) {
      quantize_tr_residual(state, COLOR_U, x, y, depth, cur_pu, lcu);
      quantize_tr_residual(state, COLOR_V, x, y, depth, cur_pu, lcu);
    }
  }
}
