/*
 * SampleMipChain.h
 * ----------------
 * Purpose: Builder for the per-sample Aniso-64 data mip pyramid.
 * Notes  : Layout and offset helpers live in SampleMip (ModSample.h).
 *          Mip bodies are the true signal decimated through a cascaded
 *          Kaiser half-band; loop correctness comes from strip windows
 *          holding the decimation of the periodically extended signal
 *          around each loop boundary, so decimation never blends key-off
 *          tail content into the loop body.
 * Authors: Quinlight Audio
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "openmpt/all/BuildSettings.hpp"
#include "Snd_defs.h"

OPENMPT_NAMESPACE_BEGIN

struct ModSample;
class CSoundFile;

// Build all mip bodies and loop strips for the sample's current waveform and
// loop points, and set smp.nMipLevelsBuilt. Dispatches on the sample's
// runtime format internally. Called from ModSample::PrecomputeLoops; cost is
// roughly one 63-tap filter pass over the sample per octave (~30-80 ms per
// million stereo frames), so it runs at load / sample-replace time, not in
// the mixer.
void BuildSampleMips(ModSample &smp, const CSoundFile &sndFile);

OPENMPT_NAMESPACE_END
