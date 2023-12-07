#define ENABLE_DP
//@#define ENABLE_DP
#define LOG2VECTLENDP 1
//@#define LOG2VECTLENDP 1
#define VECTLENDP (1 << LOG2VECTLENDP)
//@#define VECTLENDP (1 << LOG2VECTLENDP)
#define ENABLE_FMA_DP
//@#define ENABLE_FMA_DP

#define ENABLE_SP
//@#define ENABLE_SP
#define LOG2VECTLENSP (LOG2VECTLENDP+1)
//@#define LOG2VECTLENSP (LOG2VECTLENDP+1)
#define VECTLENSP (1 << LOG2VECTLENSP)
//@#define VECTLENSP (1 << LOG2VECTLENSP)
#define ENABLE_FMA_SP
//@#define ENABLE_FMA_SP

#define FULL_FP_ROUNDING
//@#define FULL_FP_ROUNDING
#define ACCURATE_SQRT
//@#define ACCURATE_SQRT

#define ISANAME "lsx"
#define DFTPRIORITY 25


#include <lsxintrin.h>

#include <stdint.h>
#include "misc.h"


typedef __m128i vmask;
typedef __m128i vopmask;

typedef __m128d vdouble;
typedef __m128i vint;

typedef __m128 vfloat;
typedef __m128i vint2;

typedef __m128i vint64;
typedef __m128i vuint64;

typedef struct {
  vmask x, y;
} vquad;

typedef vquad vargquad;


static INLINE int vavailability_i(int name) { return 3; }

static INLINE int vtestallones_i_vo32(vopmask g) {  return __lsx_bnz_w(g); }
static INLINE int vtestallones_i_vo64(vopmask g) {  return __lsx_bnz_d(g); }


static INLINE vdouble vcast_vd_d(double d) { return (vdouble){d, d}; }
static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (vopmask) vd; }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return (vdouble) vm; }


static INLINE vint2 vloadu_vi2_p(int32_t *p) { return __lsx_vld((void const *)p, 0);}
static INLINE void vstoreu_v_p_vi2(int32_t *p, vint2 v) { *((vint2*)p) = v;}
static INLINE vint vloadu_vi_p(int32_t *p) {return __lsx_vld((void const *)p, 0);}
static INLINE void vstoreu_v_p_vi(int32_t *p, vint v) { *((vint*)p) = v;}
static INLINE vfloat vload_vf_p(const float *ptr) {return (vfloat)__lsx_vld((void const *)ptr, 0); }
static INLINE vfloat vloadu_vf_p(const float *ptr) {return (vfloat)__lsx_vld((void const *)ptr, 0);}
static INLINE void vstore_v_p_vf(float *ptr, vfloat v) { *((vfloat*)ptr) = v; }
static INLINE void vstoreu_v_p_vf(float *ptr, vfloat v) { *((vfloat*)ptr) = v; }
static INLINE vdouble vload_vd_p(const double *ptr) { return (vdouble) __lsx_vld((void const *)ptr, 0);}
static INLINE vdouble vloadu_vd_p(const double *ptr) { return (vdouble) __lsx_vld((void const *)ptr, 0);}
static INLINE void vstore_v_p_vd(double *ptr, vdouble v) { *((vdouble*)ptr) = v;}
static INLINE void vstoreu_v_p_vd(double *ptr, vdouble v) { *((vdouble*)ptr) = v;}

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return __lsx_vand_v(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return __lsx_vandn_v(x, y); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return __lsx_vor_v(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return __lsx_vxor_v(x, y); }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return __lsx_vand_v(x, y); }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return __lsx_vandn_v(x, y); }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return __lsx_vor_v(x, y); }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return __lsx_vxor_v(x, y); }

static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) { return __lsx_vand_v(x, y); }
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) { return __lsx_vandn_v(x, y); }
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) { return __lsx_vor_v(x, y); }
static INLINE vmask vxor_vm_vo64_vm(vopmask x, vmask y) { return __lsx_vxor_v(x, y); }

static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) { return __lsx_vand_v(x, y); }
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) { return __lsx_vandn_v(x, y); }
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) { return __lsx_vor_v(x, y); }
static INLINE vmask vxor_vm_vo32_vm(vopmask x, vmask y) { return __lsx_vxor_v(x, y); }

static INLINE vopmask vcast_vo32_vo64(vopmask o) { return __lsx_vshuf4i_w(o,8);}

static INLINE vopmask vcast_vo64_vo32(vopmask o) { return __lsx_vshuf4i_w(o,80);}

static INLINE vopmask vcast_vo_i(int i) {
  return __lsx_vreplgr2vr_d(i ? -1 : 0);
}

//towards the nearest even
static INLINE vdouble vrint_vd_vd(vdouble vd) { return (vdouble)__lsx_vfrintrne_d(vd); }
static INLINE vint vrint_vi_vd(vdouble vd) { return  __lsx_vftintrne_w_d(vd,vd);}
static INLINE vfloat vrint_vf_vf(vfloat vd) { return (vfloat)__lsx_vfrintrne_s(vd); }

//towards zero
static INLINE vint vtruncate_vi_vd(vdouble vd) { return __lsx_vftintrz_w_d(vd,vd);}
static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return (vdouble)__lsx_vfrintrz_d(vd); }
static INLINE vfloat vtruncate_vf_vf(vfloat vf) { return (vfloat)__lsx_vfrintrz_s(vf); }

static INLINE vdouble vcast_vd_vi(vint vi) { return  __lsx_vffint_d_l(__lsx_vsllwil_d_w(vi, 0));}
static INLINE vint vcast_vi_i(int i) { return __lsx_vreplgr2vr_w(i); }

static INLINE vmask vcastu_vm_vi(vint vi) {  return __lsx_vslli_d(__lsx_vsllwil_du_wu(vi, 0), 32);}
static INLINE vint vcastu_vi_vm(vmask vi) { return __lsx_vshuf4i_w(vi, 221);}
static INLINE vmask vcast_vm_i_i(int i0, int i1) {
   return __lsx_vilvl_w(__lsx_vreplgr2vr_w(i0), __lsx_vreplgr2vr_w(i1));
}

static INLINE vmask vcast_vm_i64(int64_t i) { return __lsx_vreplgr2vr_d(i); }
static INLINE vmask vcast_vm_u64(uint64_t i) { return __lsx_vreplgr2vr_d((uint64_t)i); }

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) { return __lsx_vseq_d(x, y); }
static INLINE vmask vadd64_vm_vm_vm(vmask x, vmask y) { return __lsx_vadd_d(x, y); }


static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return __lsx_vfadd_d(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return __lsx_vfsub_d(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return __lsx_vfmul_d(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return __lsx_vfdiv_d(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return  __lsx_vfrecip_d(x); }
static INLINE vdouble vsqrt_vd_vd(vdouble x) { return __lsx_vfsqrt_d(x); }
static INLINE vdouble vabs_vd_vd(vdouble d) { return (vdouble) __lsx_vandn_v((vopmask)vcast_vd_d(-0.0), (vopmask)d);}
static INLINE vdouble vneg_vd_vd(vdouble d) { return (vdouble)__lsx_vxor_v((vopmask)vcast_vd_d(-0.0), (vopmask)d); }
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfmadd_d(x, y, z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfmsub_d(x, y, z); }
static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfnmsub_d(x, y, z); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return __lsx_vfmax_d(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return __lsx_vfmin_d(x, y); }
static INLINE vdouble vlogb_vd_vd(vdouble d)  { return __lsx_vflogb_d(d);}

static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfmadd_d(x, y, z); }
static INLINE vdouble vfmapp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfmadd_d(x, y, z); }
static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfmsub_d(x, y, z); }
static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfnmsub_d(x, y, z); }
static INLINE vdouble vfmann_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lsx_vfnmadd_d(x, y, z); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return __lsx_vfcmp_ceq_d(x, y); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return __lsx_vfcmp_cne_d(x, y); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return __lsx_vfcmp_clt_d(x, y); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return __lsx_vfcmp_cle_d(x, y); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return __lsx_vfcmp_clt_d(y, x);}
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return __lsx_vfcmp_clt_d(y, x); }


static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return __lsx_vadd_w(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return __lsx_vsub_w(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return __lsx_vneg_w(e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return __lsx_vand_v(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return __lsx_vandn_v(x, y); }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return __lsx_vor_v(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return __lsx_vxor_v(x, y); }

static INLINE vint vandnot_vi_vo_vi(vopmask m, vint y) { return __lsx_vandn_v(m, y); }
static INLINE vint vand_vi_vo_vi(vopmask m, vint y) { return __lsx_vand_v(m, y); }

static INLINE vint vsll_vi_vi_i(vint x, int c) { return __lsx_vslli_w(x, c); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return __lsx_vsrli_w(x, c); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return __lsx_vsrai_w(x, c); }

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return __lsx_vseq_w(x, y); }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return __lsx_vslt_w(y,x);}

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) { return   __lsx_vseq_w(x, y); }
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) { return __lsx_vslt_w(y, x);}

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) { return __lsx_vbitsel_v(y, x, m); }

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask o, vdouble x, vdouble y) { return (vdouble) __lsx_vbitsel_v((vopmask)y, (vopmask)x, o); }
static INLINE vdouble vsel_vd_vo_d_d(vopmask o, double v1, double v0)
{
  return vsel_vd_vo_vd_vd(o, (vdouble){v1, v1}, (vdouble){v0, v0});
}

static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3)
{
   return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_vd_vd(o1, vcast_vd_d(d1), vsel_vd_vo_d_d(o2,d2, d3)));
}
static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_d_d(o1, d1, d2));
}

static INLINE vopmask visinf_vo_vd(vdouble d) {
  return __lsx_vfcmp_ceq_d(vabs_vd_vd(d), vcast_vd_d(SLEEF_INFINITY));
}

static INLINE vopmask vispinf_vo_vd(vdouble d) {
  return __lsx_vfcmp_ceq_d(d, vcast_vd_d(SLEEF_INFINITY));
}

static INLINE vopmask visminf_vo_vd(vdouble d) {
  return __lsx_vfcmp_ceq_d(d, vcast_vd_d(-SLEEF_INFINITY));
}

static INLINE vopmask visnan_vo_vd(vdouble d) {
  return __lsx_vfcmp_sune_d(d,d);
}

static INLINE vdouble vgather_vd_p_vi(const double *ptr, vint vi) {
  int a[VECTLENDP];
  vstoreu_v_p_vi(a, vi);
  return (vdouble){ptr[a[0]], ptr[a[1]]};
}

static INLINE vint2 vcast_vi2_vm(vmask vm) { return vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return vi; }

static INLINE vint2 vrint_vi2_vf(vfloat vf) { return vcast_vi2_vm(__lsx_vftintrne_w_s(vf)); }
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vcast_vi2_vm(__lsx_vftintrz_w_s(vf)); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return __lsx_vffint_s_w(vcast_vm_vi2(vi)); }
static INLINE vfloat vcast_vf_f(float f) { return (vfloat){f, f, f, f}; }
static INLINE vint2 vcast_vi2_i(int i) { return __lsx_vreplgr2vr_w(i); }

static INLINE vmask vreinterpret_vm_vf(vfloat vf)  { return (vopmask)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm)  { return (vfloat)vm; }
static INLINE vfloat vreinterpret_vf_vi2(vint2 vi) { return (vfloat)vi; }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return (vopmask)vf; }

static INLINE vdouble vreinterpret_vd_vf(vfloat vf) { return (vdouble)vf; }
static INLINE vfloat vreinterpret_vf_vd(vdouble vd) { return (vfloat)vd; }


static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return __lsx_vfadd_s(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return __lsx_vfsub_s(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return __lsx_vfmul_s(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return __lsx_vfdiv_s(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return __lsx_vfrecip_s(x); }
static INLINE vfloat vsqrt_vf_vf(vfloat x) { return __lsx_vfsqrt_s(x); }
static INLINE vfloat vabs_vf_vf(vfloat f) { return  (vfloat)__lsx_vandn_v((vopmask)vcast_vf_f(-0.0f), (vopmask)f); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return  (vfloat)__lsx_vxor_v((vopmask)vcast_vf_f(-0.0f), (vopmask)d); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return __lsx_vfmax_s(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return __lsx_vfmin_s(x, y); }

static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfmadd_s(x, y, z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfnmsub_s(x, y, z); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfmsub_s(x, y, z); }
static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfmadd_s(x, y, z); }
static INLINE vfloat vfmapp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfmadd_s(x, y, z); }
static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfmsub_s(x, y, z); }
static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfnmsub_s(x, y, z); }
static INLINE vfloat vfmann_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lsx_vfnmadd_s(x, y, z); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return __lsx_vfcmp_ceq_s(x, y); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return  __lsx_vfcmp_cne_s(x, y); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return  __lsx_vfcmp_clt_s(x, y); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return __lsx_vfcmp_cle_s(x, y); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) {  return  __lsx_vfcmp_clt_s(y, x);}
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return __lsx_vfcmp_clt_s(y, x);}

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vadd_w(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vsub_w(x, y); }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return __lsx_vneg_w(e); }
static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vand_v(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vandn_v(x, y); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vor_v(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vxor_v(x, y); }

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) { return vand_vi2_vi2_vi2(vcast_vi2_vm(x), y); }
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) { return vandnot_vi2_vi2_vi2(vcast_vi2_vm(x), y); }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return __lsx_vslli_w(x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return __lsx_vsrli_w(x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return __lsx_vsrai_w(x, c); }

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return __lsx_vseq_w(x, y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return __lsx_vslt_w(y, x); }
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vseq_w(x, y); }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return __lsx_vslt_w(y, x); }

static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) { return __lsx_vbitsel_v(y, x, m);}

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask o, vfloat x, vfloat y) { return (vfloat) __lsx_vbitsel_v((vopmask)y, (vopmask)x, o); }

// At this point, the following three functions are implemented in a generic way,
// but I will try target-specific optimization later on.
static INLINE CONST vfloat vsel_vf_vo_f_f(vopmask o, float v1, float v0) {
  return vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0));
}

static INLINE vfloat vsel_vf_vo_vo_f_f_f(vopmask o0, vopmask o1, float d0, float d1, float d2) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_f_f(o1, d1, d2));
}

static INLINE vfloat vsel_vf_vo_vo_vo_f_f_f_f(vopmask o0, vopmask o1, vopmask o2, float d0, float d1, float d2, float d3) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_vf_vf(o1, vcast_vf_f(d1), vsel_vf_vo_f_f(o2, d2, d3)));
}

static INLINE vopmask visinf_vo_vf(vfloat d) { return veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(SLEEF_INFINITYf)); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(SLEEF_INFINITYf)); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(-SLEEF_INFINITYf)); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return __lsx_vfcmp_sune_s(d,d); }

#ifdef _MSC_VER
// This function is needed when debugging on MSVC.
static INLINE float vcast_f_vf(vfloat v) {

  return v[0];
}
#endif

static INLINE vfloat vgather_vf_p_vi2(const float *ptr, vint2 vi2) {
  int a[VECTLENSP];
  vstoreu_v_p_vi2(a, vi2);
  return (vfloat){ptr[a[0]], ptr[a[1]], ptr[a[2]], ptr[a[3]]};
}

#define PNMASK ((vdouble) { +0.0, -0.0})
#define NPMASK ((vdouble) { -0.0, +0.0})
#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f})
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f})

static INLINE vdouble vposneg_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(PNMASK))); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(NPMASK))); }
static INLINE vfloat vposneg_vf_vf(vfloat d) { return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(PNMASKf))); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(NPMASKf))); }

static INLINE vdouble vsubadd_vd_vd_vd(vdouble x, vdouble y) { return __lsx_vfadd_d(x, vnegpos_vd_vd(y)); }
static INLINE vfloat vsubadd_vf_vf_vf(vfloat x, vfloat y) { return __lsx_vfadd_s(x, vnegpos_vf_vf(y)); }

static INLINE vdouble vmlsubadd_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vmla_vd_vd_vd_vd(x, y, vnegpos_vd_vd(z)); }
static INLINE vfloat vmlsubadd_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vmla_vf_vf_vf_vf(x, y, vnegpos_vf_vf(z)); }

static INLINE vmask vsel_vm_vo64_vm_vm(vopmask o, vmask x, vmask y) { return vor_vm_vm_vm(vand_vm_vm_vm(o, x), vandnot_vm_vm_vm(o, y)); }

static INLINE vmask vsub64_vm_vm_vm(vmask x, vmask y) { return __lsx_vsub_d(x, y); }
static INLINE vmask vneg64_vm_vm(vmask x) { return __lsx_vneg_d(x); }
static INLINE vopmask vgt64_vo_vm_vm(vmask x, vmask y) {return __lsx_vslt_d(y,x); } // signed compare

#define vsll64_vm_vm_i(x, c) __lsx_vslli_d(x, c)
#define vsrl64_vm_vm_i(x, c) __lsx_vsrli_d(x, c)
//@#define vsll64_vm_vm_i(x, c)__lsx_vslli_d(x, c)
//@#define vsrl64_vm_vm_i(x, c) __lsx_vslli_d(x, c)

static INLINE vmask vcast_vm_vi(vint vi) { return vi; } // signed 32-bit => 64-bit
static INLINE vint vcast_vi_vm(vmask vm) { return vm;  }

static INLINE vmask vreinterpret_vm_vi64(vint64 v) { return v; }
static INLINE vint64 vreinterpret_vi64_vm(vmask m) { return m; }
static INLINE vmask vreinterpret_vm_vu64(vuint64 v) { return v; }
static INLINE vuint64 vreinterpret_vu64_vm(vmask m) { return m; }
