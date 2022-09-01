// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenColorIO Project.

#include "Lut3DOpCPU_SSE2.h"

#if OCIO_USE_SSE2

#include <immintrin.h>

namespace OCIO_NAMESPACE
{
namespace {
// Macros for alignment declarations
#define SSE2_SIMD_BYTES 16
#if defined( _MSC_VER )
#define SSE2_ALIGN(decl) __declspec(align(SSE2_SIMD_BYTES)) decl
#elif ( __APPLE__ )

#define SSE2_ALIGN(decl) decl
#else
#define SSE2_ALIGN(decl) decl __attribute__((aligned(SSE2_SIMD_BYTES)))
#endif

typedef struct {
    const float *lut;
    __m128 lutmax;
    __m128 lutsize;
    __m128 lutsize2;
} Lut3DContextSSE2;

typedef struct rgbvec_sse2 {
    __m128 r, g, b;
} rgbvec_sse2;

typedef struct rgabvec_sse2 {
    __m128 r, g, b, a;
} m128_rgbavec;

#define gather_rgb_sse2(src, idx)             \
    _mm_store_si128((__m128i *)indices, idx); \
    row0 = _mm_loadu_ps(src + indices[0]);    \
    row1 = _mm_loadu_ps(src + indices[1]);    \
    row2 = _mm_loadu_ps(src + indices[2]);    \
    row3 = _mm_loadu_ps(src + indices[3]);    \
    tmp0 = _mm_unpacklo_ps(row0, row1);       \
    tmp2 = _mm_unpacklo_ps(row2, row3);       \
    tmp1 = _mm_unpackhi_ps(row0, row1);       \
    tmp3 = _mm_unpackhi_ps(row2, row3);       \
    sample_r = _mm_movelh_ps(tmp0, tmp2);     \
    sample_g = _mm_movehl_ps(tmp2, tmp0);     \
    sample_b = _mm_movelh_ps(tmp1, tmp3)

static inline __m128 floor_ps_sse2(__m128 v)
{
    // NOTE: using truncate cvtt
    return _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
}

static inline __m128 blendv_ps_sse2(__m128 a, __m128 b, __m128 mask)
{
    return _mm_xor_ps(_mm_and_ps(_mm_xor_ps(a, b), mask), a);
}

static inline __m128 fmadd_ps_sse2(__m128 a, __m128 b, __m128 c)
{
    return  _mm_add_ps(_mm_mul_ps(a, b), c);
}

static inline rgbvec_sse2 interp_tetrahedral_sse2(const Lut3DContextSSE2 *ctx, __m128 r, __m128 g, __m128 b)
{
    SSE2_ALIGN(uint32_t indices[4]);

    __m128 x0, x1, x2;
    __m128 cxxxa;
    __m128 cxxxb;
    __m128 mask;

    __m128 tmp0, tmp1, tmp2, tmp3;
    __m128 row0, row1, row2, row3;
    __m128 sample_r, sample_g, sample_b;

    rgbvec_sse2 result;

    __m128 lut_max  = ctx->lutmax;
    __m128 lutsize  = ctx->lutsize;
    __m128 lutsize2 = ctx->lutsize2;

    __m128 one_f   = _mm_set1_ps(1.0f);
    __m128 four_f  = _mm_set1_ps(4.0f);

    __m128 prev_r = floor_ps_sse2(r);
    __m128 prev_g = floor_ps_sse2(g);
    __m128 prev_b = floor_ps_sse2(b);

    // rgb delta values
    __m128 d_r = _mm_sub_ps(r, prev_r);
    __m128 d_g = _mm_sub_ps(g, prev_g);
    __m128 d_b = _mm_sub_ps(b, prev_b);

    __m128 next_r = _mm_min_ps(lut_max, _mm_add_ps(prev_r, one_f));
    __m128 next_g = _mm_min_ps(lut_max, _mm_add_ps(prev_g, one_f));
    __m128 next_b = _mm_min_ps(lut_max, _mm_add_ps(prev_b, one_f));

    // prescale indices
    prev_r = _mm_mul_ps(prev_r, lutsize2);
    next_r = _mm_mul_ps(next_r, lutsize2);

    prev_g = _mm_mul_ps(prev_g, lutsize);
    next_g = _mm_mul_ps(next_g, lutsize);

    prev_b = _mm_mul_ps(prev_b, four_f);
    next_b = _mm_mul_ps(next_b, four_f);

    // This is the tetrahedral blend equation
    // red = (1-x0) * c000.r + (x0-x1) * cxxxa.r + (x1-x2) * cxxxb.r + x2 * c111.r;
    // The x values are the rgb delta values sorted, x0 >= x1 >= x2
    // c### are samples from the lut, which are indices made with prev_(r,g,b) and next_(r,g,b) values
    // 0 = use prev, 1 = use next
    // c### = (prev_r or next_r) * (lutsize * lutsize) + (prev_g or next_g) * lutsize + (prev_b or next_b)

    // cxxxa
    // always uses 1 next and 2 prev and next is largest delta
    // r> == c100 == (r>g && r>b) == (!b>r && r>g)
    // g> == c010 == (g>r && g>b) == (!r>g && g>b)
    // b> == c001 == (b>r && b>g) == (!g>b && b>r)

    // cxxxb
    // always uses 2 next and 1 prev and prev is smallest delta
    // r< == c011 == (r<=g && r<=b) == (!r>g && b>r)
    // g< == c101 == (g<=r && g<=b) == (!g>b && r>g)
    // b< == c110 == (b<=r && b<=g) == (!b>r && g>b)

    // c000 and c111 are const (prev,prev,prev) and (next,next,next)

    __m128 gt_r = _mm_cmpgt_ps(d_r, d_g); // r>g
    __m128 gt_g = _mm_cmpgt_ps(d_g, d_b); // g>b
    __m128 gt_b = _mm_cmpgt_ps(d_b, d_r); // b>r

    // r> !b>r && r>g
    mask  = _mm_andnot_ps(gt_b, gt_r);
    cxxxa = blendv_ps_sse2(prev_r, next_r, mask);

    // r< !r>g && b>r
    mask  = _mm_andnot_ps(gt_r, gt_b);
    cxxxb = blendv_ps_sse2(next_r, prev_r, mask);

    // g> !r>g && g>b
    mask  = _mm_andnot_ps(gt_r, gt_g);
    cxxxa = _mm_add_ps(cxxxa, blendv_ps_sse2(prev_g, next_g, mask));

    // g< !g>b && r>g
    mask  = _mm_andnot_ps(gt_g, gt_r);
    cxxxb = _mm_add_ps(cxxxb, blendv_ps_sse2(next_g, prev_g, mask));

    // b> !g>b && b>r
    mask  = _mm_andnot_ps(gt_g, gt_b);
    cxxxa = _mm_add_ps(cxxxa, blendv_ps_sse2(prev_b, next_b, mask));

    // b< !b>r && g>b
    mask  = _mm_andnot_ps(gt_b, gt_g);
    cxxxb = _mm_add_ps(cxxxb, blendv_ps_sse2(next_b, prev_b, mask));

    __m128 c000 = _mm_add_ps(_mm_add_ps(prev_r, prev_g), prev_b);
    __m128 c111 = _mm_add_ps(_mm_add_ps(next_r, next_g), next_b);

    // sort delta r,g,b x0 >= x1 >= x2
    __m128 rg_min = _mm_min_ps(d_r, d_g);
    __m128 rg_max = _mm_max_ps(d_r, d_g);

    x2         = _mm_min_ps(rg_min, d_b);
    __m128 mid = _mm_max_ps(rg_min, d_b);

    x0 = _mm_max_ps(rg_max, d_b);
    x1 = _mm_min_ps(rg_max, mid);

    // convert indices to int
    __m128i c000_idx  = _mm_cvttps_epi32(c000);
    __m128i cxxxa_idx = _mm_cvttps_epi32(cxxxa);
    __m128i cxxxb_idx = _mm_cvttps_epi32(cxxxb);
    __m128i c111_idx  = _mm_cvttps_epi32(c111);

    gather_rgb_sse2(ctx->lut, c000_idx);

    // (1-x0) * c000
    __m128 v = _mm_sub_ps(one_f, x0);
    result.r = _mm_mul_ps(sample_r, v);
    result.g = _mm_mul_ps(sample_g, v);
    result.b = _mm_mul_ps(sample_b, v);

    gather_rgb_sse2(ctx->lut, cxxxa_idx);

    // (x0-x1) * cxxxa
    v = _mm_sub_ps(x0, x1);
    result.r = fmadd_ps_sse2(v, sample_r, result.r);
    result.g = fmadd_ps_sse2(v, sample_g, result.g);
    result.b = fmadd_ps_sse2(v, sample_b, result.b);

    gather_rgb_sse2(ctx->lut, cxxxb_idx);

    // (x1-x2) * cxxxb
    v = _mm_sub_ps(x1, x2);
    result.r = fmadd_ps_sse2(v, sample_r, result.r);
    result.g = fmadd_ps_sse2(v, sample_g, result.g);
    result.b = fmadd_ps_sse2(v, sample_b, result.b);

    gather_rgb_sse2(ctx->lut, c111_idx);

    // x2 * c111
    result.r = fmadd_ps_sse2(x2, sample_r, result.r);
    result.g = fmadd_ps_sse2(x2, sample_g, result.g);
    result.b = fmadd_ps_sse2(x2, sample_b, result.b);

    return result;
}

static inline m128_rgbavec rgba_transpose_4x4_sse2(__m128 row0, __m128 row1, __m128 row2, __m128 row3)
{
    m128_rgbavec result;
    __m128 tmp0 = _mm_unpacklo_ps(row0, row1);
    __m128 tmp2 = _mm_unpacklo_ps(row2, row3);
    __m128 tmp1 = _mm_unpackhi_ps(row0, row1);
    __m128 tmp3 = _mm_unpackhi_ps(row2, row3);
    result.r    = _mm_movelh_ps(tmp0, tmp2);
    result.g    = _mm_movehl_ps(tmp2, tmp0); // Note movhlps swaps b with a which is different than unpckhpd
    result.b    = _mm_movelh_ps(tmp1, tmp3);
    result.a    = _mm_movehl_ps(tmp3, tmp1);
    return result;
}

} // anonymous namespace

void applyTetrahedralSSE2(const float *lut3d, int dim, const float *src, float *dst, int total_pixel_count)
{
    m128_rgbavec c;
    rgbvec_sse2 c2;

    Lut3DContextSSE2 ctx;

    float lutmax = (float)dim - 1;
    __m128 scale_r = _mm_set1_ps(lutmax);
    __m128 scale_g = _mm_set1_ps(lutmax);
    __m128 scale_b = _mm_set1_ps(lutmax);
    __m128 zero    = _mm_setzero_ps();

    ctx.lut      =  lut3d;
    ctx.lutmax   = _mm_set1_ps(lutmax);
    ctx.lutsize  = _mm_set1_ps((float)dim * 4);
    ctx.lutsize2 = _mm_set1_ps((float)dim * dim * 4);

    int pixel_count = total_pixel_count / 4 * 4;
    int remainder = total_pixel_count - pixel_count;

    for (int i = 0; i < pixel_count; i += 4 ) {
        __m128 rgba0 = _mm_loadu_ps(src +  0);
        __m128 rgba1 = _mm_loadu_ps(src +  4);
        __m128 rgba2 = _mm_loadu_ps(src +  8);
        __m128 rgba3 = _mm_loadu_ps(src + 12);
        c = rgba_transpose_4x4_sse2(rgba0, rgba1, rgba2, rgba3);

         // scale and clamp values
        c.r = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.r, scale_r)));
        c.g = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.g, scale_g)));
        c.b = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.b, scale_b)));

        c2 = interp_tetrahedral_sse2(&ctx, c.r, c.g, c.b);

        c = rgba_transpose_4x4_sse2(c2.r, c2.g, c2.b, c.a);
        _mm_storeu_ps(dst +  0, c.r);
        _mm_storeu_ps(dst +  4, c.g);
        _mm_storeu_ps(dst +  8, c.b);
        _mm_storeu_ps(dst + 12, c.a);

        src += 16;
        dst += 16;
    }

    // handler leftovers pixels
    if (remainder) {
        SSE2_ALIGN(float r[4]);
        SSE2_ALIGN(float g[4]);
        SSE2_ALIGN(float b[4]);
        SSE2_ALIGN(float a[4]);

        for (int i = 0; i < remainder; i++) {
            r[i] = src[0];
            g[i] = src[1];
            b[i] = src[2];
            a[i] = src[3];
            src += 4;
        }

        c.r = _mm_load_ps(r);
        c.g = _mm_load_ps(g);
        c.b = _mm_load_ps(b);

        // scale and clamp values
        c.r = _mm_mul_ps(c.r, scale_r);
        c.g = _mm_mul_ps(c.g, scale_g);
        c.b = _mm_mul_ps(c.b, scale_b);

        c.r = _mm_max_ps(c.r, zero);
        c.g = _mm_max_ps(c.g, zero);
        c.b = _mm_max_ps(c.b, zero);

        c.r = _mm_min_ps(c.r, ctx.lutmax);
        c.g = _mm_min_ps(c.g, ctx.lutmax);
        c.b = _mm_min_ps(c.b, ctx.lutmax);

        c2 = interp_tetrahedral_sse2(&ctx, c.r, c.g, c.b);

        _mm_store_ps(r, c2.r);
        _mm_store_ps(g, c2.g);
        _mm_store_ps(b, c2.b);

        for (int i = 0; i < remainder; i++) {
            dst[0] = r[i];
            dst[1] = g[i];
            dst[2] = b[i];
            dst[3] = a[i];
            dst += 4;
        }
    }
}

} // OCIO_NAMESPACE

#endif // OCIO_USE_SSE2