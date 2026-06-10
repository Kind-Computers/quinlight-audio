/*
 * SampleMipChain.cpp
 * ------------------
 * Purpose: Builder for the per-sample Aniso-64 data mip pyramid.
 * Notes  : (currently none)
 * Authors: Quinlight Audio
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "SampleMipChain.h"
#include "ModSample.h"
#include "Sndfile.h"
#include "mpt/base/numbers.hpp"
#include "mpt/base/saturate_round.hpp"

#include <new>
#include <vector>
#include <cmath>


OPENMPT_NAMESPACE_BEGIN

namespace
{

// Modified Bessel function of the first kind, order 0 (power series).
// Small and local on purpose — coefficients are built once per process.
double BesselI0(double x)
{
	double sum = 1.0, term = 1.0;
	const double halfX = x * 0.5;
	for(int k = 1; k < 64; k++)
	{
		term *= (halfX / k) * (halfX / k);
		sum += term;
		if(term < sum * 1e-21)
			break;
	}
	return sum;
}


// 63-tap Kaiser-windowed half-band decimator (beta = 9, ~-90 dB stopband).
// Built from a half-band prototype (center 0.5, even offsets zero), then
// normalized so the tap sum is exactly 1.0 — unity DC gain keeps the mip
// ladder level-matched by construction. (The normalization slightly rescales
// the center and odd taps, so the result is no longer an exact half-band;
// the even-offset zeros survive and the deviation is far below the stopband.)
struct Halfband
{
	static constexpr int Radius = 31;
	static constexpr int NumTaps = 2 * Radius + 1;
	double h[NumTaps];

	Halfband()
	{
		const double beta = 9.0;
		const double i0Beta = BesselI0(beta);
		double sum = 0.0;
		for(int n = -Radius; n <= Radius; n++)
		{
			const double t = static_cast<double>(n) / Radius;
			const double window = BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - t * t))) / i0Beta;
			double sinc;
			if(n == 0)
				sinc = 0.5;
			else if(n % 2 == 0)
				sinc = 0.0;  // exact half-band zeros
			else
				sinc = std::sin(0.5 * mpt::numbers::pi * n) / (mpt::numbers::pi * n);
			h[n + Radius] = sinc * window;
			sum += h[n + Radius];
		}
		for(double &tap : h)
			tap /= sum;
	}
};


const Halfband &GetHalfband()
{
	static const Halfband instance;
	return instance;
}


template <typename T>
double FromSample(T v) noexcept
{
	return static_cast<double>(v);
}

template <typename T>
T ToSample(double v) noexcept
{
	return static_cast<T>(v);
}

template <>
int8 ToSample<int8>(double v) noexcept
{
	return mpt::saturate_round<int8>(v);
}

template <>
int16 ToSample<int16>(double v) noexcept
{
	return mpt::saturate_round<int16>(v);
}


// One 2:1 half-band decimation pass over interleaved frames, with edge-hold
// extension at the buffer ends (same semantics as the existing interpolation
// margins). dst frame n is computed from src frames around 2n, so frame n of
// the output corresponds exactly to frame 2n of the input — the dyadic grid
// is preserved with no half-sample offset.
void DecimateOnce(const double *src, SmpLength srcFrames, double *dst, SmpLength dstFrames, int numChannels)
{
	const Halfband &hb = GetHalfband();
	const int64 maxSrc = static_cast<int64>(srcFrames) - 1;
	for(SmpLength n = 0; n < dstFrames; n++)
	{
		for(int c = 0; c < numChannels; c++)
		{
			double acc = 0.0;
			for(int m = -Halfband::Radius; m <= Halfband::Radius; m++)
			{
				int64 pos = static_cast<int64>(n) * 2 + m;
				if(pos < 0)
					pos = 0;
				else if(pos > maxSrc)
					pos = maxSrc;
				acc += hb.h[m + Halfband::Radius] * src[pos * numChannels + c];
			}
			dst[n * numChannels + c] = acc;
		}
	}
}


// Loop-position walk, ported from PrecomputeLoop::CopyLoop (ModSample.cpp).
// Generates the same read-position sequence as the existing mip-0 wrap
// buffers — forward wrap, ping-pong reflection (with the boundary-repeat
// behavior CopyLoop produces), and IT ping-pong mode — but over arbitrary
// distances, so the mip strips agree with the level-0 loop semantics.
struct LoopWalk
{
	SmpLength loopLen;
	SmpLength rp;  // loop-relative position, 0..loopLen-1
	int dir;
	bool pingpong;
	bool itMode;

	void Step()
	{
		if(rp == loopLen - 1 && dir > 0)
		{
			if(pingpong)
			{
				dir = -1;
				if(itMode && rp > 0)
					rp--;
			} else
			{
				rp = 0;
			}
		} else if(rp == 0 && dir < 0)
		{
			if(pingpong)
				dir = 1;
			else
				rp = loopLen - 1;
		} else
		{
			rp = static_cast<SmpLength>(static_cast<int64>(rp) + dir);
		}
	}
};


struct LoopInfo
{
	SmpLength start, end;
	bool pingpong;
	bool itMode;
};


// Fill a scratch buffer (interleaved doubles) with mip-0 content for the
// mip-0 position range [x0, x0 + numFrames), extended with the loop's
// periodic continuation around the given boundary.
//
// atEnd == true: the loop-end window. Every position on both sides of the
// boundary is the periodized loop signal — this is what steady looping
// plays, and it is what keeps key-off tail content out of the decimated
// strip.
//
// atEnd == false: the loop-start window (read after the loop has wrapped at
// least once). Positions >= loop start inside the loop are the real signal;
// positions before the loop start are the periodic continuation backwards
// (i.e. end-of-loop content); positions at/after the loop end wrap forward
// again (steady looping never plays past the end).
template <typename T>
void FillPeriodizedScratch(double *scratch, int64 x0, SmpLength numFrames, const T *sampleData, const LoopInfo &loop, int numChannels, bool atEnd)
{
	const SmpLength loopLen = loop.end - loop.start;
	auto writeFrame = [&](SmpLength idx, SmpLength srcFrame)
	{
		for(int c = 0; c < numChannels; c++)
			scratch[idx * numChannels + c] = FromSample(sampleData[static_cast<size_t>(srcFrame) * numChannels + c]);
	};

	// Anchor the walks at the boundary sample.
	const int64 anchorPos = atEnd ? (static_cast<int64>(loop.end) - 1) : static_cast<int64>(loop.start);
	const SmpLength anchorRp = atEnd ? (loopLen - 1) : 0;
	const int64 anchorIdx64 = anchorPos - x0;
	MPT_ASSERT(anchorIdx64 >= 0 && anchorIdx64 < static_cast<int64>(numFrames));
	const SmpLength anchorIdx = static_cast<SmpLength>(anchorIdx64);

	if(atEnd)
	{
		// Backward leg: periodized content before the boundary.
		LoopWalk back{loopLen, anchorRp, -1, loop.pingpong, loop.itMode};
		writeFrame(anchorIdx, loop.start + anchorRp);
		for(int64 idx = anchorIdx64 - 1; idx >= 0; idx--)
		{
			back.Step();
			writeFrame(static_cast<SmpLength>(idx), loop.start + back.rp);
		}
		// Forward leg: periodized continuation past the boundary.
		LoopWalk fwd{loopLen, anchorRp, 1, loop.pingpong, loop.itMode};
		for(int64 idx = anchorIdx64 + 1; idx < static_cast<int64>(numFrames); idx++)
		{
			fwd.Step();
			writeFrame(static_cast<SmpLength>(idx), loop.start + fwd.rp);
		}
	} else
	{
		// In-loop region (and forward periodization past the loop end).
		LoopWalk fwd{loopLen, anchorRp, 1, loop.pingpong, loop.itMode};
		writeFrame(anchorIdx, loop.start + anchorRp);
		for(int64 idx = anchorIdx64 + 1; idx < static_cast<int64>(numFrames); idx++)
		{
			fwd.Step();
			writeFrame(static_cast<SmpLength>(idx), loop.start + fwd.rp);
		}
		// Periodized content before the loop start (the post-wrap case).
		LoopWalk back{loopLen, anchorRp, -1, loop.pingpong, loop.itMode};
		for(int64 idx = anchorIdx64 - 1; idx >= 0; idx--)
		{
			back.Step();
			writeFrame(static_cast<SmpLength>(idx), loop.start + back.rp);
		}
	}
}


// Build all mip bodies for one sample: a cascaded 2:1 half-band over a
// double-precision working copy (so integer formats do not accumulate
// per-level rounding), with the result of each level stored in the sample's
// own format. Bodies are the *true* signal decimated (edge-hold extension) —
// loop periodization is confined to the strips, which keeps play-through
// across a sustain-loop end seamless.
template <typename T>
void BuildMipBodies(ModSample &smp, int maxLevel)
{
	const SmpLength len = smp.nLength;
	const int numChannels = smp.GetNumChannels();
	const T *base = static_cast<const T *>(smp.samplev());
	T *baseMut = static_cast<T *>(smp.samplev());

	std::vector<double> work(static_cast<size_t>(len) * numChannels);
	for(size_t i = 0; i < work.size(); i++)
		work[i] = FromSample(base[i]);
	std::vector<double> next;

	SmpLength srcFrames = len;
	for(int k = 1; k <= maxLevel; k++)
	{
		const SmpLength bodyFrames = SampleMip::BodyFrames(len, k);
		next.assign(static_cast<size_t>(bodyFrames) * numChannels, 0.0);
		DecimateOnce(work.data(), srcFrames, next.data(), bodyFrames, numChannels);

		T *body = baseMut + SampleMip::BodyOffsetFrames(len, k) * numChannels;
		for(size_t i = 0; i < next.size(); i++)
			body[i] = ToSample<T>(next[i]);

		// Pre/post hold margins (edge hold, like the level-0 margins).
		T *pre = baseMut + SampleMip::LevelOffsetFrames(len, k) * numChannels;
		for(SmpLength i = 0; i < SampleMip::kMipPre; i++)
			for(int c = 0; c < numChannels; c++)
				pre[i * numChannels + c] = body[c];
		T *post = body + static_cast<size_t>(bodyFrames) * numChannels;
		for(SmpLength i = 0; i < SampleMip::kMipPost; i++)
			for(int c = 0; c < numChannels; c++)
				post[i * numChannels + c] = body[(static_cast<size_t>(bodyFrames) - 1) * numChannels + c];

		work.swap(next);
		srcFrames = bodyFrames;
	}
}


// Build the four strip windows for one loop boundary pair at every level.
// Each strip is the k-stage cascade run over a freshly periodized mip-0
// scratch segment — the same composite filter as the body cascade, so strip
// and body content agree away from the boundary and the handoff between them
// (64 mip-k frames before the boundary, kernel reach +-32, cascade
// contamination radius < 32) is seam-free.
template <typename T>
void BuildLoopStrips(ModSample &smp, const LoopInfo &loop, int maxLevel, int endWindow, int startWindow)
{
	const SmpLength len = smp.nLength;
	const int numChannels = smp.GetNumChannels();
	const T *base = static_cast<const T *>(smp.samplev());
	T *baseMut = static_cast<T *>(smp.samplev());

	// 32 mip-k frames of cascade margin on each side of the 256-frame window.
	// A k-stage cascade contaminates Radius·(2^k − 1) mip-0 frames from each
	// scratch edge, so the margin must satisfy kMargin·2^k ≥ Radius·(2^k − 1),
	// i.e. kMargin ≥ Radius — with only one mip-k frame to spare at k = 7.
	constexpr SmpLength kMargin = 32;
	static_assert(kMargin >= Halfband::Radius, "strip margin must cover the halfband cascade contamination radius");
	constexpr SmpLength kSpan = SampleMip::kMipStripWin + 2 * kMargin;  // 320 mip-k frames

	std::vector<double> scratch, half;
	for(int k = 1; k <= maxLevel; k++)
	{
		for(int window = 0; window < 2; window++)
		{
			const bool atEnd = (window == 0);
			const SmpLength boundary = atEnd ? loop.end : loop.start;
			const int64 center = static_cast<int64>(boundary >> k);
			// (center - 160) * 2^k — multiply, not <<, as it may be negative.
			const int64 x0 = (center - static_cast<int64>(kSpan / 2)) * (int64(1) << k);
			const SmpLength frames0 = kSpan << k;

			scratch.resize(static_cast<size_t>(frames0) * numChannels);
			FillPeriodizedScratch(scratch.data(), x0, frames0, base, loop, numChannels, atEnd);

			SmpLength curFrames = frames0;
			for(int stage = 1; stage <= k; stage++)
			{
				const SmpLength dstFrames = curFrames / 2;
				half.assign(static_cast<size_t>(dstFrames) * numChannels, 0.0);
				DecimateOnce(scratch.data(), curFrames, half.data(), dstFrames, numChannels);
				scratch.swap(half);
				curFrames = dstFrames;
			}
			MPT_ASSERT(curFrames == kSpan);

			// Store the central 256 frames; the margins only absorbed edge
			// effects. Strip frame i corresponds to mip-k position
			// (boundary >> k) - 128 + i.
			T *strip = baseMut + SampleMip::StripOffsetFrames(len, k, atEnd ? endWindow : startWindow) * numChannels;
			const double *srcCenter = scratch.data() + static_cast<size_t>(kMargin) * numChannels;
			for(size_t i = 0; i < static_cast<size_t>(SampleMip::kMipStripWin) * numChannels; i++)
				strip[i] = ToSample<T>(srcCenter[i]);
		}
	}
}


template <typename T>
void BuildSampleMipsImpl(ModSample &smp, const CSoundFile &sndFile)
{
	const int maxLevel = SampleMip::MaxLevels(smp.nLength);
	if(maxLevel < 1)
		return;

	BuildMipBodies<T>(smp, maxLevel);

	const bool itMode = sndFile.m_playBehaviour[kITPingPongMode];
	if(smp.uFlags[CHN_LOOP] && smp.nLoopEnd > smp.nLoopStart && smp.nLoopEnd <= smp.nLength)
	{
		const LoopInfo loop{smp.nLoopStart, smp.nLoopEnd, smp.HasPingPongLoop(), itMode};
		BuildLoopStrips<T>(smp, loop, maxLevel, SampleMip::kStripLoopEnd, SampleMip::kStripLoopStart);
	}
	if(smp.uFlags[CHN_SUSTAINLOOP] && smp.nSustainEnd > smp.nSustainStart && smp.nSustainEnd <= smp.nLength)
	{
		const LoopInfo loop{smp.nSustainStart, smp.nSustainEnd, smp.HasPingPongSustainLoop(), itMode};
		BuildLoopStrips<T>(smp, loop, maxLevel, SampleMip::kStripSustainEnd, SampleMip::kStripSustainStart);
	}

	smp.nMipLevelsBuilt = static_cast<uint8>(maxLevel);
}

}  // unnamed namespace


void BuildSampleMips(ModSample &smp, const CSoundFile &sndFile)
{
	smp.nMipLevelsBuilt = 0;
	if(!smp.HasSampleData())
		return;
	// The allocation falls back to the pre-mip layout when the pyramid would
	// not fit the size budget (huge float64 stereo samples) — there is no mip
	// region to write to in that case.
	if(!ModSample::CanFitMips(smp.nLength, smp.GetBytesPerSample()))
		return;

	try
	{
		if(smp.GetElementarySampleSize() == 2)
			BuildSampleMipsImpl<int16>(smp, sndFile);
		else if(smp.GetElementarySampleSize() == 1)
			BuildSampleMipsImpl<int8>(smp, sndFile);
		else if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float64)
			BuildSampleMipsImpl<double>(smp, sndFile);
		else if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float32)
			BuildSampleMipsImpl<somefloat32>(smp, sndFile);
	} catch(const std::exception &)
	{
		// The double-precision working buffers are transient but can reach
		// several GiB for maximum-length samples; besides bad_alloc, vector
		// can throw length_error when the request exceeds max_size() (32-bit
		// size_t). PrecomputeLoops must not throw — degrade to the
		// level-0-only gather instead.
		smp.nMipLevelsBuilt = 0;
	}
}


OPENMPT_NAMESPACE_END
