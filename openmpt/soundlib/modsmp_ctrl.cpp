/*
 * modsmp_ctrl.cpp
 * ---------------
 * Purpose: Basic sample editing code.
 * Notes  : This is a legacy namespace. Some of this stuff is not required in libopenmpt (but stuff in soundlib/ still depends on it). The rest could be merged into struct ModSample.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "modsmp_ctrl.h"
#include "AudioCriticalSection.h"
#include "Sndfile.h"

OPENMPT_NAMESPACE_BEGIN

namespace ctrlSmp
{

static void ReverseSampleImpl(std::byte *pStart, const SmpLength length, const std::size_t frameSize)
{
	for(SmpLength i = 0; i < length / 2; i++)
	{
		std::byte *a = pStart + i * frameSize;
		std::byte *b = pStart + (length - 1 - i) * frameSize;
		for(std::size_t j = 0; j < frameSize; j++)
		{
			std::swap(a[j], b[j]);
		}
	}
}

// Reverse sample data
bool ReverseSample(ModSample &smp, SmpLength start, SmpLength end, CSoundFile &sndFile)
{
	if(!smp.HasSampleData()) return false;
	if(end == 0 || start > smp.nLength || end > smp.nLength)
	{
		start = 0;
		end   = smp.nLength;
	}

	if(end - start < 2) return false;

	ReverseSampleImpl(smp.sampleb() + start * smp.GetBytesPerSample(), end - start, smp.GetBytesPerSample());

	smp.PrecomputeLoops(sndFile, false);
	return true;
}


template <class T>
static void InvertSampleImpl(T *pStart, const SmpLength length)
{
	for(SmpLength i = 0; i < length; i++)
	{
		pStart[i] = ~pStart[i];
	}
}

template <>
void InvertSampleImpl<somefloat32>(somefloat32 *pStart, const SmpLength length)
{
	for(SmpLength i = 0; i < length; i++)
	{
		pStart[i] = -pStart[i];
	}
}

template <>
void InvertSampleImpl<double>(double *pStart, const SmpLength length)
{
	for(SmpLength i = 0; i < length; i++)
	{
		pStart[i] = -pStart[i];
	}
}

// Invert sample data (flip by 180 degrees)
bool InvertSample(ModSample &smp, SmpLength start, SmpLength end, CSoundFile &sndFile)
{
	if(!smp.HasSampleData()) return false;
	if(end == 0 || start > smp.nLength || end > smp.nLength)
	{
		start = 0;
		end = smp.nLength;
	}
	start *= smp.GetNumChannels();
	end *= smp.GetNumChannels();
	if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float64)
		InvertSampleImpl(smp.sampled() + start, end - start);
	else if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float32)
		InvertSampleImpl(smp.samplef() + start, end - start);
	else if(smp.GetElementarySampleSize() == 2)
		InvertSampleImpl(smp.sample16() + start, end - start);
	else if(smp.GetElementarySampleSize() == 1)
		InvertSampleImpl(smp.sample8() + start, end - start);
	else
		return false;

	smp.PrecomputeLoops(sndFile, false);
	return true;
}


template <class T>
static void XFadeSampleImpl(const T *srcIn, const T *srcOut, T *output, const SmpLength fadeLength, double e)
{
	const double length = 1.0 / static_cast<double>(fadeLength);
	for(SmpLength i = 0; i < fadeLength; i++, srcIn++, srcOut++, output++)
	{
		double fact1 = std::pow(i * length, e);
		double fact2 = std::pow((fadeLength - i) * length, e);
		int32 val = static_cast<int32>(
			static_cast<double>(*srcIn) * fact1 +
			static_cast<double>(*srcOut) * fact2);
		*output = mpt::saturate_cast<T>(val);
	}
}

template <>
void XFadeSampleImpl<somefloat32>(const somefloat32 *srcIn, const somefloat32 *srcOut, somefloat32 *output, const SmpLength fadeLength, double e)
{
	const double length = 1.0 / static_cast<double>(fadeLength);
	for(SmpLength i = 0; i < fadeLength; i++, srcIn++, srcOut++, output++)
	{
		double fact1 = std::pow(i * length, e);
		double fact2 = std::pow((fadeLength - i) * length, e);
		*output = static_cast<somefloat32>(
			static_cast<double>(*srcIn) * fact1 +
			static_cast<double>(*srcOut) * fact2);
	}
}

template <>
void XFadeSampleImpl<double>(const double *srcIn, const double *srcOut, double *output, const SmpLength fadeLength, double e)
{
	const double length = 1.0 / static_cast<double>(fadeLength);
	for(SmpLength i = 0; i < fadeLength; i++, srcIn++, srcOut++, output++)
	{
		double fact1 = std::pow(i * length, e);
		double fact2 = std::pow((fadeLength - i) * length, e);
		*output = *srcIn * fact1 + *srcOut * fact2;
	}
}

// X-Fade sample data to create smooth loop transitions
bool XFadeSample(ModSample &smp, SmpLength fadeLength, int fadeLaw, bool afterloopFade, bool useSustainLoop, CSoundFile &sndFile)
{
	if(!smp.HasSampleData()) return false;
	const auto [loopStart, loopEnd] = useSustainLoop ? smp.GetSustainLoop() : smp.GetLoop();
	
	if(loopEnd <= loopStart || loopEnd > smp.nLength) return false;
	if(loopStart < fadeLength) return false;

	const SmpLength start = (loopStart - fadeLength) * smp.GetNumChannels();
	const SmpLength end = (loopEnd - fadeLength) * smp.GetNumChannels();
	const SmpLength afterloopStart = loopStart * smp.GetNumChannels();
	const SmpLength afterloopEnd = loopEnd * smp.GetNumChannels();
	const SmpLength afterLoopLength = std::min(smp.nLength - loopEnd, fadeLength) * smp.GetNumChannels();
	fadeLength *= smp.GetNumChannels();

	// e=0.5: constant power crossfade (for uncorrelated samples), e=1.0: constant volume crossfade (for perfectly correlated samples)
	const double e = 1.0 - fadeLaw / 200000.0;

	if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float64)
	{
		XFadeSampleImpl(smp.sampled() + start, smp.sampled() + end, smp.sampled() + end, fadeLength, e);
		if(afterloopFade) XFadeSampleImpl(smp.sampled() + afterloopEnd, smp.sampled() + afterloopStart, smp.sampled() + afterloopEnd, afterLoopLength, e);
	} else if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float32)
	{
		XFadeSampleImpl(smp.samplef() + start, smp.samplef() + end, smp.samplef() + end, fadeLength, e);
		if(afterloopFade) XFadeSampleImpl(smp.samplef() + afterloopEnd, smp.samplef() + afterloopStart, smp.samplef() + afterloopEnd, afterLoopLength, e);
	} else if(smp.GetElementarySampleSize() == 2)
	{
		XFadeSampleImpl(smp.sample16() + start, smp.sample16() + end, smp.sample16() + end, fadeLength, e);
		if(afterloopFade) XFadeSampleImpl(smp.sample16() + afterloopEnd, smp.sample16() + afterloopStart, smp.sample16() + afterloopEnd, afterLoopLength, e);
	} else if(smp.GetElementarySampleSize() == 1)
	{
		XFadeSampleImpl(smp.sample8() + start, smp.sample8() + end, smp.sample8() + end, fadeLength, e);
		if(afterloopFade) XFadeSampleImpl(smp.sample8() + afterloopEnd, smp.sample8() + afterloopStart, smp.sample8() + afterloopEnd, afterLoopLength, e);
	} else
		return false;

	smp.PrecomputeLoops(sndFile, true);
	return true;
}


template <class T>
static void ConvertStereoToMonoMixImpl(T *pDest, const SmpLength length)
{
	const T *pEnd = pDest + length;
	for(T *pSource = pDest; pDest != pEnd; pDest++, pSource += 2)
	{
		*pDest = static_cast<T>(mpt::rshift_signed(pSource[0] + pSource[1] + 1, 1));
	}
}

template <>
void ConvertStereoToMonoMixImpl<somefloat32>(somefloat32 *pDest, const SmpLength length)
{
	const somefloat32 *pEnd = pDest + length;
	for(somefloat32 *pSource = pDest; pDest != pEnd; pDest++, pSource += 2)
	{
		*pDest = (pSource[0] + pSource[1]) * 0.5f;
	}
}

template <>
void ConvertStereoToMonoMixImpl<double>(double *pDest, const SmpLength length)
{
	const double *pEnd = pDest + length;
	for(double *pSource = pDest; pDest != pEnd; pDest++, pSource += 2)
	{
		*pDest = (pSource[0] + pSource[1]) * 0.5;
	}
}


template <class T>
static void ConvertStereoToMonoOneChannelImpl(T *pDest, const T *pSource, const SmpLength length)
{
	for(const T *pEnd = pDest + length; pDest != pEnd; pDest++, pSource += 2)
	{
		*pDest = *pSource;
	}
}


// Convert a multichannel sample to mono (currently only implemented for stereo)
bool ConvertToMono(ModSample &smp, CSoundFile &sndFile, StereoToMonoMode conversionMode)
{
	if(!smp.HasSampleData() || smp.GetNumChannels() != 2) return false;

	// Note: Sample is overwritten in-place! Unused data is not deallocated!
	if(conversionMode == mixChannels)
	{
		if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float64)
			ConvertStereoToMonoMixImpl(smp.sampled(), smp.nLength);
		else if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float32)
			ConvertStereoToMonoMixImpl(smp.samplef(), smp.nLength);
		else if(smp.GetElementarySampleSize() == 2)
			ConvertStereoToMonoMixImpl(smp.sample16(), smp.nLength);
		else if(smp.GetElementarySampleSize() == 1)
			ConvertStereoToMonoMixImpl(smp.sample8(), smp.nLength);
		else
			return false;
	} else
	{
		if(conversionMode == splitSample)
		{
			conversionMode = onlyLeft;
		}
		if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float64)
			ConvertStereoToMonoOneChannelImpl(smp.sampled(), smp.sampled() + (conversionMode == onlyLeft ? 0 : 1), smp.nLength);
		else if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float32)
			ConvertStereoToMonoOneChannelImpl(smp.samplef(), smp.samplef() + (conversionMode == onlyLeft ? 0 : 1), smp.nLength);
		else if(smp.GetElementarySampleSize() == 2)
			ConvertStereoToMonoOneChannelImpl(smp.sample16(), smp.sample16() + (conversionMode == onlyLeft ? 0 : 1), smp.nLength);
		else if(smp.GetElementarySampleSize() == 1)
			ConvertStereoToMonoOneChannelImpl(smp.sample8(), smp.sample8() + (conversionMode == onlyLeft ? 0 : 1), smp.nLength);
		else
			return false;
	}

	CriticalSection cs;
	smp.uFlags.reset(CHN_STEREO);
	for(auto &chn : sndFile.m_PlayState.Chn)
	{
		if(chn.pModSample == &smp)
		{
			chn.dwFlags.reset(CHN_STEREO);
		}
	}

	smp.PrecomputeLoops(sndFile, false);
	return true;
}


template <class T>
static void SplitStereoImpl(void *destL, void *destR, const T *source, SmpLength length)
{
	T *l = static_cast<T *>(destL), *r = static_cast<T*>(destR);
	while(length--)
	{
		*(l++) = source[0];
		*(r++) = source[1];
		source += 2;
	}
}


// Converts a stereo sample into two mono samples. Source sample will not be deleted.
bool SplitStereo(const ModSample &source, ModSample &left, ModSample &right, CSoundFile &sndFile)
{
	if(!source.HasSampleData() || source.GetNumChannels() != 2 || &left == &right)
		return false;
	const bool sourceIsLeft = &left == &source, sourceIsRight = &right == &source;
	if(left.HasSampleData() && !sourceIsLeft)
		return false;
	if(right.HasSampleData() && !sourceIsRight)
		return false;

	void *leftData  = sourceIsLeft ? left.samplev() : ModSample::AllocateSample(source.nLength, source.GetElementarySampleSize());
	void *rightData = sourceIsRight ? right.samplev() : ModSample::AllocateSample(source.nLength, source.GetElementarySampleSize());
	if(!leftData || !rightData)
	{
		if(!sourceIsLeft)
			ModSample::FreeSample(leftData);
		if(!sourceIsRight)
			ModSample::FreeSample(rightData);
		return false;
	}

	if(source.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float64)
		SplitStereoImpl(leftData, rightData, source.sampled(), source.nLength);
	else if(source.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float32)
		SplitStereoImpl(leftData, rightData, source.samplef(), source.nLength);
	else if(source.GetElementarySampleSize() == 2)
		SplitStereoImpl(leftData, rightData, source.sample16(), source.nLength);
	else if(source.GetElementarySampleSize() == 1)
		SplitStereoImpl(leftData, rightData, source.sample8(), source.nLength);
	else
		MPT_ASSERT_NOTREACHED();

	CriticalSection cs;
	left = source;
	left.uFlags.reset(CHN_STEREO);
	left.pData.pSample = leftData;

	right = source;
	right.uFlags.reset(CHN_STEREO);
	right.pData.pSample = rightData;

	for(auto &chn : sndFile.m_PlayState.Chn)
	{
		if(chn.pModSample == &left || chn.pModSample == &right)
			chn.dwFlags.reset(CHN_STEREO);
	}

	left.PrecomputeLoops(sndFile, false);
	right.PrecomputeLoops(sndFile, false);
	return true;
}


template <class T>
static void ConvertMonoToStereoImpl(const T *MPT_RESTRICT src, T *MPT_RESTRICT dst, SmpLength length)
{
	while(length--)
	{
		dst[0] = *src;
		dst[1] = *src;
		dst += 2;
		src++;
	}
}


// Convert a multichannel sample to mono (currently only implemented for stereo)
bool ConvertToStereo(ModSample &smp, CSoundFile &sndFile)
{
	if(!smp.HasSampleData() || smp.GetNumChannels() != 1) return false;

	void *newSample = ModSample::AllocateSample(smp.nLength, smp.GetBytesPerSample() * 2);
	if(newSample == nullptr)
	{
		return 0;
	}

	if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float64)
		ConvertMonoToStereoImpl(smp.sampled(), static_cast<double *>(newSample), smp.nLength);
	else if(smp.GetRuntimeSampleFormat() == ModSample::RuntimeSampleFormat::Float32)
		ConvertMonoToStereoImpl(smp.samplef(), static_cast<somefloat32 *>(newSample), smp.nLength);
	else if(smp.GetElementarySampleSize() == 2)
		ConvertMonoToStereoImpl(smp.sample16(), (int16 *)newSample, smp.nLength);
	else if(smp.GetElementarySampleSize() == 1)
		ConvertMonoToStereoImpl(smp.sample8(), (int8 *)newSample, smp.nLength);
	else
		return false;

	CriticalSection cs;
	smp.uFlags.set(CHN_STEREO);
	smp.ReplaceWaveform(newSample, smp.nLength, sndFile);

	smp.PrecomputeLoops(sndFile, false);
	return true;
}


} // namespace ctrlSmp



namespace ctrlChn
{

namespace
{

static PitchT ScaleLivePeriod(PitchT period, FreqT oldC5Speed, FreqT newC5Speed, bool periodIsFreq)
{
	if(period <= 0 || oldC5Speed == 0 || newC5Speed == 0 || oldC5Speed == newC5Speed)
		return period;
	if(periodIsFreq)
		return period * newC5Speed / oldC5Speed;
	return period * oldC5Speed / newC5Speed;
}

static SamplePosition ScaleLiveSamplePosition(SamplePosition position, uint32 oldC5Speed, uint32 newC5Speed)
{
	if(oldC5Speed == 0 || newC5Speed == 0 || oldC5Speed == newC5Speed || position.IsZero())
	{
		return position;
	}

	const int64 raw = position.GetRaw();
	const bool negative = raw < 0;
	const uint64 absRaw = negative ? static_cast<uint64>(-raw) : static_cast<uint64>(raw);
	const uint64 intPart = absRaw >> 32;
	const uint64 fracPart = absRaw & SamplePosition::fractMax;
	const uint64 scaledInt = (intPart * newC5Speed) / oldC5Speed;
	const uint64 scaledIntRemainder = (intPart * newC5Speed) % oldC5Speed;
	const uint64 scaledFrac = ((scaledIntRemainder << 32) + fracPart * newC5Speed) / oldC5Speed;
	const uint64 carry = scaledFrac >> 32;
	const uint32 outFrac = static_cast<uint32>(scaledFrac);
	int64 outRaw = (static_cast<int64>(scaledInt + carry) << 32) | outFrac;
	if(negative)
	{
		outRaw = -outRaw;
	}
	return SamplePosition(outRaw);
}

} // namespace

void ReplaceSample( CSoundFile &sndFile,
					const ModSample &sample,
					const void * const pNewSample,
					const SmpLength newLength,
					FlagSet<ChannelFlags> setFlags,
					FlagSet<ChannelFlags> resetFlags)
{
	const bool periodIsFreq = sndFile.PeriodsAreFrequencies();
	for(auto &chn : sndFile.m_PlayState.Chn)
	{
		if(chn.pModSample == &sample)
		{
			const uint32 oldC5Speed = chn.nC5Speed > 0 ? static_cast<uint32>(chn.nC5Speed) : 0u;
			const int32 oldPeriod = chn.nPeriod;
			const int32 oldPortamentoDest = chn.nPortamentoDest;
			if(chn.pCurrentSample != nullptr)
				chn.pCurrentSample = pNewSample;
			// Scale play position to maintain temporal alignment after rate change.
			// Without this, an 8kHz sample at position 5000/8000 (625ms) replaced
			// with 48kHz data would continue at 5000/48000 (104ms) — a large
			// temporal jump causing an audible click.
			if(oldC5Speed > 0 && sample.nC5Speed > 0 && oldC5Speed != static_cast<uint32>(sample.nC5Speed))
			{
				chn.position = ScaleLiveSamplePosition(chn.position, oldC5Speed, sample.nC5Speed);
				if(chn.position.GetUInt() >= newLength)
					chn.position.Set(0);
			} else if(chn.position.GetUInt() > newLength)
			{
				chn.position.Set(0);
			}
			if(chn.nLength > 0)
				LimitMax(chn.nLength, newLength);
			if(chn.InSustainLoop())
			{
				chn.nLoopStart = sample.nSustainStart;
				chn.nLoopEnd = sample.nSustainEnd;
			} else
			{
				chn.nLoopStart = sample.nLoopStart;
				chn.nLoopEnd = sample.nLoopEnd;
			}
			chn.dwFlags.set(setFlags);
			chn.dwFlags.reset(resetFlags);

			if(chn.HasCustomTuning())
			{
				chn.nC5Speed = sample.nC5Speed;
				chn.cachedPeriod = 0;
				chn.glissandoPeriod = 0;
				chn.m_CalculateFreq = true;
				continue;
			}

			if(sndFile.GetType() == MOD_TYPE_XM)
			{
				// XM live replacement keeps the current period and portamento target,
				// but restores the channel's runtime transpose / finetune to the
				// sample's original note math. The actual waveform-rate change is
				// applied later in GetChannelIncrement().
				const auto [playbackTranspose, playbackFineTune] = sample.GetPlaybackTransposeFineTune(sndFile.GetType());
				chn.nTranspose = playbackTranspose;
				chn.nFineTune = playbackFineTune;
			} else if(sndFile.GetType() == MOD_TYPE_MOD)
			{
				// MOD: keep period and portamento target unchanged.
				// GetFreqFromPeriod for MOD returns AMIGA_CLOCK / period with no
				// C5Speed dependency (Snd_fx.cpp:6605-6609), and the Quinlight
				// rate scaling in GetChannelIncrement (Sndmix.cpp:2062-2068)
				// multiplies the frequency by nC5Speed / nC5SpeedOriginal.
				// Scaling the period here would double-count the rate change
				// and pitch-shift continuing notes by the rate ratio — e.g. an
				// additional 6x (~2.6 octaves) for an 8 kHz → 48 kHz remaster.
				// Keep this behaviour in lock-step with the XM branch above.
			} else
			{
				if(oldC5Speed && sample.nC5Speed)
				{
					// When SONG_LINEARSLIDES is active with non-hertz periods (or MDL/DTM),
					// GetPeriodFromNote returns C5Speed-independent table values and
					// GetFreqFromPeriod applies C5Speed in its own formula (c5speed * K / period).
					// Scaling the period would double-count the rate change — updating
					// chn.nC5Speed alone provides the correct frequency ratio.
					const bool freqUsesC5Speed =
						!periodIsFreq
						&& (sndFile.m_SongFlags[SONG_LINEARSLIDES]
							|| sndFile.GetType() == MOD_TYPE_DTM
							|| sndFile.GetType() == MOD_TYPE_MDL);
					if(!freqUsesC5Speed)
					{
						chn.nPeriod = ScaleLivePeriod(oldPeriod, oldC5Speed, sample.nC5Speed, periodIsFreq);
						chn.nPortamentoDest = ScaleLivePeriod(oldPortamentoDest, oldC5Speed, sample.nC5Speed, periodIsFreq);
					}
				}
			}
			chn.nC5Speed = sample.GetPlaybackC5Speed(sndFile.GetType());
			chn.cachedPeriod = 0;
			chn.glissandoPeriod = 0;
			chn.m_CalculateFreq = true;
			// Immediately update the mixer increment so that any partial tick
			// remaining in the current audio buffer uses the correct playback
			// speed.  Without this, the old increment (designed for the old
			// sample rate) drives playback through the new data until the next
			// ReadNote() call — up to one full tick (~20 ms) of wrong pitch.
			chn.increment = sndFile.GetChannelIncrement(chn, chn.nPeriod, 0).first;
		}
	}
}


void RefreshChannelsForSample(CSoundFile &sndFile, const ModSample &sample)
{
	for(auto &chn : sndFile.m_PlayState.Chn)
	{
		if(chn.pModSample != &sample)
			continue;

		// IIR filter memory: history from the old rate causes a brief resonant
		// zing when fed new-rate data through coefficients meant for the old rate.
		chn.nFilter_Y[0][0] = 0;
		chn.nFilter_Y[0][1] = 0;
		chn.nFilter_Y[1][0] = 0;
		chn.nFilter_Y[1][1] = 0;
		chn.nFilter_Y2[0][0] = 0;
		chn.nFilter_Y2[0][1] = 0;
		chn.nFilter_Y2[1][0] = 0;
		chn.nFilter_Y2[1][1] = 0;
		chn.nFilter_HP = 0;

		// β-shear resampler acceleration history: the 2nd-order interpolator
		// assumes a continuous increment, which a rate change breaks. Seed with
		// the current increment and zero delta so the next tick starts flat.
		chn.prevIncrement = chn.increment;
		chn.prevDeltaIncrement.Set(0);

		// Output DC-offset buffer: residue from the old sample at the old rate.
		chn.nROfs = 0;
		chn.nLOfs = 0;

		// Volume ramp: any in-flight ramp was sized for the old rate; snap the
		// ramping volume to the current target and cancel the ramp.
		chn.leftRamp = 0;
		chn.rightRamp = 0;
		chn.rampLeftVol = chn.leftVol;
		chn.rampRightVol = chn.rightVol;
		chn.nRampLength = 0;
		chn.nRampPosition = 0;

		// Re-sync loop state from the now-updated sample. ReplaceSample
		// (called from inside ReplaceWaveform) copies channel-side boundaries
		// and flags from the sample BEFORE the loop points are scaled or the
		// sample's flags are rewritten, so we must cover both here:
		//  1. Boundaries — channel holds stale nLoopStart/nLoopEnd, which
		//     makes the mixer wrap at the wrong positions and skip the
		//     interpolation-lookahead fast path (Fastmix.cpp:79, 160-169, 234).
		//  2. Flag semantics — channel holds the OLD CHN_LOOP /
		//     CHN_PINGPONGLOOP bits, so a mode change (looped→one-shot,
		//     forward→ping-pong, sustain toggles) plays with stale flags.
		//  3. Effective length — for looped channels the mixer's end-of-
		//     sample bound is the active loop end, not the full waveform;
		//     only non-looped playback uses sample.nLength.
		// Mirrors ModSample::UpdateLoopPointsInActiveChannels.
		bool looped = false, bidi = false;
		if(sample.nSustainStart < sample.nSustainEnd
			&& sample.nSustainEnd <= sample.nLength
			&& sample.uFlags[CHN_SUSTAINLOOP]
			&& !chn.dwFlags[CHN_KEYOFF])
		{
			chn.nLoopStart = sample.nSustainStart;
			chn.nLoopEnd   = sample.nSustainEnd;
			chn.nLength    = sample.nSustainEnd;
			looped = true;
			bidi   = sample.uFlags[CHN_PINGPONGSUSTAIN];
		} else if(sample.nLoopStart < sample.nLoopEnd
			&& sample.nLoopEnd <= sample.nLength
			&& sample.uFlags[CHN_LOOP])
		{
			chn.nLoopStart = sample.nLoopStart;
			chn.nLoopEnd   = sample.nLoopEnd;
			chn.nLength    = sample.nLoopEnd;
			looped = true;
			bidi   = sample.uFlags[CHN_PINGPONGLOOP];
		} else
		{
			chn.nLength = sample.nLength;
		}
		chn.dwFlags.set(CHN_LOOP, looped);
		chn.dwFlags.set(CHN_PINGPONGLOOP, looped && bidi);
		if(!bidi)
			chn.dwFlags.reset(CHN_PINGPONGFLAG);

		// Snap stale play positions into the loop body. Also handles the
		// engine-discovered-loop case where the rate-scaled position lands
		// past the new (possibly smaller) loop end.
		if(chn.position.GetUInt() > chn.nLength)
		{
			chn.position.Set(chn.nLoopStart);
			chn.dwFlags.reset(CHN_PINGPONGFLAG);
		}
	}
}

} // namespace ctrlChn


OPENMPT_NAMESPACE_END
