/*
 * MixFuncTable.cpp
 * ----------------
 * Purpose: Table containing all mixer functions.
 * Notes  : The Visual Studio project settings for this file have been adjusted
 *          to force function inlining, so that the mixer has a somewhat acceptable
 *          performance in debug mode. If you need to debug anything here, be sure
 *          to disable those optimizations if needed.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "MixFuncTable.h"
#include "Mixer.h"
#include "ModChannel.h"
#include "Snd_defs.h"

#ifdef MPT_INTMIXER
#include "IntMixer.h"
#else
#include "FloatMixer.h"
#endif // MPT_INTMIXER

OPENMPT_NAMESPACE_BEGIN

namespace MixFuncTable
{
#ifdef MPT_INTMIXER
using I8M = Int8MToIntS;
using I16M = Int16MToIntS;
using I8S = Int8SToIntS;
using I16S = Int16SToIntS;
using F32M = Float32MToIntS;
using F32S = Float32SToIntS;
// Float64 samples are not supported with the integer mixer — reads double*
// through float* traits. Placing a static_assert here as a reminder.
static_assert(sizeof(somefloat32) == sizeof(double), "MPT_INTMIXER does not support Float64 samples");
using F64M = Float32MToIntS;
using F64S = Float32SToIntS;
#else
using I8M = Int8MToFloatS;
using I16M = Int16MToFloatS;
using I8S = Int8SToFloatS;
using I16S = Int16SToFloatS;
using F32M = Float32MToFloatS;
using F32S = Float32SToFloatS;
using F64M = Float64MToFloatS;
using F64S = Float64SToFloatS;
#endif // MPT_INTMIXER

// Build mix function table for given resampling, filter and ramping settings.
#define BuildMixFuncTableRamp(resampling, filter, ramp) \
	SampleLoop<I8M, resampling<I8M>, filter<I8M>, MixMono ## ramp<I8M> >, \
	SampleLoop<I16M, resampling<I16M>, filter<I16M>, MixMono ## ramp<I16M> >, \
	SampleLoop<I8S, resampling<I8S>, filter<I8S>, MixStereo ## ramp<I8S> >, \
	SampleLoop<I16S, resampling<I16S>, filter<I16S>, MixStereo ## ramp<I16S> >, \
	SampleLoop<F32M, resampling<F32M>, filter<F32M>, MixMono ## ramp<F32M> >, \
	SampleLoop<F64M, resampling<F64M>, filter<F64M>, MixMono ## ramp<F64M> >, \
	SampleLoop<F32S, resampling<F32S>, filter<F32S>, MixStereo ## ramp<F32S> >, \
	SampleLoop<F64S, resampling<F64S>, filter<F64S>, MixStereo ## ramp<F64S> >

// Build mix function table for given resampling, filter settings: With and without ramping
#define BuildMixFuncTableFilter(resampling, filter) \
	BuildMixFuncTableRamp(resampling, filter, NoRamp), \
	BuildMixFuncTableRamp(resampling, filter, Ramp)

// Build mix function table for given resampling settings: With and without filter
#define BuildMixFuncTable(resampling) \
	BuildMixFuncTableFilter(resampling, NoFilter), \
	BuildMixFuncTableFilter(resampling, ResonantFilter)

const MixFuncInterface Functions[8 * 32] =
{
	BuildMixFuncTable(NoInterpolation),        // No SRC
	BuildMixFuncTable(LinearInterpolation),    // Linear SRC
	BuildMixFuncTable(FastSincInterpolation),  // Fast Sinc (Cubic Spline) SRC
	BuildMixFuncTable(PolyphaseInterpolation), // Kaiser SRC
	BuildMixFuncTable(FIRFilterInterpolation), // FIR SRC
	BuildMixFuncTable(AmigaBlepInterpolation), // Amiga emulation
	BuildMixFuncTable(Aniso64Interpolation),   // 64-tap anisotropic sinc (Quinlight)
	BuildMixFuncTable(CatmullRomInterpolation), // Centripetal Catmull-Rom with β-shear (Quinlight)
};

#undef BuildMixFuncTableRamp
#undef BuildMixFuncTableFilter
#undef BuildMixFuncTable


ResamplingIndex ResamplingModeToMixFlags(ResamplingMode resamplingMode)
{
	switch(resamplingMode)
	{
	case SRCMODE_NEAREST: return ndxNoInterpolation;
	case SRCMODE_LINEAR:  return ndxLinear;
	case SRCMODE_CUBIC:   return ndxFastSinc;
	case SRCMODE_SINC8LP: return ndxKaiser;
	case SRCMODE_SINC8:   return ndxFIRFilter;
	case SRCMODE_AMIGA:   return ndxAmigaBlep;
	case SRCMODE_ANISO64: return ndxAniso64;
	case SRCMODE_CATMULL: return ndxCatmullRom;
	default:              MPT_ASSERT_NOTREACHED();
	}
	return ndxNoInterpolation;
}

} // namespace MixFuncTable

OPENMPT_NAMESPACE_END
