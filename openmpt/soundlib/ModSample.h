/*
 * ModSample.h
 * -----------
 * Purpose: Module Sample header class and helpers
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "openmpt/all/BuildSettings.hpp"
#include "Snd_defs.h"

OPENMPT_NAMESPACE_BEGIN

class CSoundFile;

// Sample Struct
struct ModSample
{
	enum class RuntimeSampleFormat : uint8
	{
		Auto = 0,
		Int8,
		Int16,
		Float32,
		Float64,
	};

	SmpLength nLength;						// In frames
	SmpLength nLoopStart, nLoopEnd;			// Ditto
	SmpLength nSustainStart, nSustainEnd;	// Ditto
	union
	{
		void  *pSample;						// Pointer to sample data
		int8  *pSample8;					// Pointer to 8-bit sample data
		int16 *pSample16;					// Pointer to 16-bit sample data
		somefloat32 *pSampleFloat;			// Pointer to float32 sample data
		double *pSampleDouble;				// Pointer to float64 sample data
	} pData;
	FreqT nC5Speed;							// Frequency of middle-C, in Hz (for IT/S3M/MPTM)
	FreqT nC5SpeedOriginal = 0;				// Original C5Speed before Quinlight resampling (0 = unmodified)
	SmpLength nLoopStartOriginal = 0, nLoopEndOriginal = 0;
	SmpLength nSustainStartOriginal = 0, nSustainEndOriginal = 0;
	uint16 nPan;							// Default sample panning (if pan flag is set), 0...256
	uint16 nVolume;							// Default volume, 0...256 (ignored if uFlags[SMP_NODEFAULTVOLUME] is set)
	uint16 nGlobalVol;						// Global volume (sample volume is multiplied by this), 0...64
	SampleFlags uFlags;						// Sample flags (see ChannelFlags enum)
	int8   RelativeTone;					// Relative note to middle c (for MOD/XM)
	int8   nFineTune;						// Finetune period (for MOD/XM), -128...127, unit is 1/128th of a semitone
	int8   RelativeToneOriginal = 0;		// Original XM relative tone before Quinlight resampling
	int8   nFineTuneOriginal = 0;			// Original XM finetune before Quinlight resampling
	VibratoType nVibType;					// Auto vibrato type
	uint8  nVibSweep;						// Auto vibrato sweep (i.e. how long it takes until the vibrato effect reaches its full depth)
	uint8  nVibDepth;						// Auto vibrato depth
	uint8  nVibRate;						// Auto vibrato rate (speed)
	uint8  rootNote;						// For multisample import
	RuntimeSampleFormat runtimeFormat = RuntimeSampleFormat::Auto;
	uint8 saveBitsPerSample = 8;

	//char name[MAX_SAMPLENAME];			// Maybe it would be nicer to have sample names here, but that would require some refactoring.
	mpt::charbuf<MAX_SAMPLEFILENAME> filename;
	std::string GetFilename() const { return filename; }

	union
	{
		std::array<SmpLength, 9> cues;
		OPLPatch adlib;
	};

	ModSample(MODTYPE type = MOD_TYPE_NONE)
	{
		pData.pSample = nullptr;
		Initialize(type);
	}

	bool HasSampleData() const noexcept
	{
		MPT_ASSERT(!pData.pSample || (pData.pSample && nLength > 0));  // having sample pointer implies non-zero sample length
		return pData.pSample != nullptr && nLength != 0;
	}

	MPT_FORCEINLINE const void *samplev() const noexcept
	{
		return pData.pSample;
	}
	MPT_FORCEINLINE void *samplev() noexcept
	{
		return pData.pSample;
	}
	MPT_FORCEINLINE const std::byte *sampleb() const noexcept
	{
		return mpt::void_cast<const std::byte*>(pData.pSample);
	}
	MPT_FORCEINLINE std::byte *sampleb() noexcept
	{
		return mpt::void_cast<std::byte*>(pData.pSample);
	}
	MPT_FORCEINLINE const int8 *sample8() const noexcept
	{
		MPT_ASSERT(GetElementarySampleSize() == sizeof(int8));
		return pData.pSample8;
	}
	MPT_FORCEINLINE int8 *sample8() noexcept
	{
		MPT_ASSERT(GetElementarySampleSize() == sizeof(int8));
		return pData.pSample8;
	}
	MPT_FORCEINLINE const int16 *sample16() const noexcept
	{
		MPT_ASSERT(GetElementarySampleSize() == sizeof(int16));
		return pData.pSample16;
	}
	MPT_FORCEINLINE int16 *sample16() noexcept
	{
		MPT_ASSERT(GetElementarySampleSize() == sizeof(int16));
		return pData.pSample16;
	}
	MPT_FORCEINLINE const somefloat32 *samplef() const noexcept
	{
		MPT_ASSERT(GetRuntimeSampleFormat() == RuntimeSampleFormat::Float32);
		return pData.pSampleFloat;
	}
	MPT_FORCEINLINE somefloat32 *samplef() noexcept
	{
		MPT_ASSERT(GetRuntimeSampleFormat() == RuntimeSampleFormat::Float32);
		return pData.pSampleFloat;
	}
	MPT_FORCEINLINE const double *sampled() const noexcept
	{
		MPT_ASSERT(GetRuntimeSampleFormat() == RuntimeSampleFormat::Float64);
		return pData.pSampleDouble;
	}
	MPT_FORCEINLINE double *sampled() noexcept
	{
		MPT_ASSERT(GetRuntimeSampleFormat() == RuntimeSampleFormat::Float64);
		return pData.pSampleDouble;
	}
	template <typename Tsample>
	MPT_FORCEINLINE const Tsample *sample() const noexcept = delete;
	template <typename Tsample>
	MPT_FORCEINLINE Tsample *sample() noexcept = delete;

	RuntimeSampleFormat GetRuntimeSampleFormat() const noexcept
	{
		switch(runtimeFormat)
		{
		case RuntimeSampleFormat::Int8: return RuntimeSampleFormat::Int8;
		case RuntimeSampleFormat::Int16: return RuntimeSampleFormat::Int16;
		case RuntimeSampleFormat::Float32: return RuntimeSampleFormat::Float32;
		case RuntimeSampleFormat::Float64: return RuntimeSampleFormat::Float64;
		case RuntimeSampleFormat::Auto:
		default:
			return uFlags[CHN_16BIT] ? RuntimeSampleFormat::Int16 : RuntimeSampleFormat::Int8;
		}
	}
	void SetRuntimeSampleFormat(RuntimeSampleFormat format) noexcept { runtimeFormat = format; }
	void SetSaveBitsPerSample(uint8 bits) noexcept { saveBitsPerSample = bits >= 16 ? 16 : 8; }
	uint8 GetSaveBitsPerSample() const noexcept { return saveBitsPerSample >= 16 ? 16 : 8; }
	uint8 GetSavedElementarySampleSize() const noexcept { return GetSaveBitsPerSample() >= 16 ? 2 : 1; }
	uint8 GetSavedBytesPerSample() const noexcept { return GetSavedElementarySampleSize() * GetNumChannels(); }

	// Return the size of one (elementary) sample in bytes.
	uint8 GetElementarySampleSize() const noexcept
	{
		switch(GetRuntimeSampleFormat())
		{
		case RuntimeSampleFormat::Float64: return sizeof(double);
		case RuntimeSampleFormat::Float32: return sizeof(somefloat32);
		case RuntimeSampleFormat::Int16: return sizeof(int16);
		case RuntimeSampleFormat::Int8:
		default:
			return sizeof(int8);
		}
	}

	// Return the number of channels in the sample.
	uint8 GetNumChannels() const noexcept { return (uFlags & CHN_STEREO) ? 2 : 1; }

	// Return the number of bytes per frame (Channels * Elementary Sample Size)
	uint8 GetBytesPerSample() const noexcept { return GetElementarySampleSize() * GetNumChannels(); }

	// Return the size which pSample is at least.
	SmpLength GetSampleSizeInBytes() const noexcept { return nLength * GetBytesPerSample(); }

	// Returns sample rate of the sample. The argument is needed because
	// the sample rate is obtained differently for different module types.
	uint32 GetSampleRate(const MODTYPE type) const;

	// Translate sample properties between two given formats.
	void Convert(MODTYPE fromType, MODTYPE toType);

	// Initialize sample slot with default values.
	void Initialize(MODTYPE type = MOD_TYPE_NONE);

	// Copies sample data from another sample slot and ensures that the 16-bit/stereo flags are set accordingly.
	bool CopyWaveform(const ModSample &smpFrom);

	// Replace waveform with given data, keeping the currently chosen format of the sample slot.
	void ReplaceWaveform(void *newWaveform, const SmpLength newLength, CSoundFile &sndFile);

	// Allocate sample based on a ModSample's properties.
	// Returns number of bytes allocated, 0 on failure.
	size_t AllocateSample();
	// Allocate sample memory. On sucess, a pointer to the silenced sample buffer is returned. On failure, nullptr is returned.
	static void *AllocateSample(SmpLength numFrames, size_t bytesPerSample);
	// Compute sample buffer size in bytes, including any overhead introduced by pre-computed loops and such. Returns 0 if sample is too big.
	static size_t GetRealSampleBufferSize(SmpLength numSamples, size_t bytesPerSample);

	void FreeSample();
	static void FreeSample(void *samplePtr);

	// Set loop points and update loop wrap-around buffer
	void SetLoop(SmpLength start, SmpLength end, bool enable, bool pingpong, CSoundFile &sndFile);
	// Set sustain loop points and update loop wrap-around buffer
	void SetSustainLoop(SmpLength start, SmpLength end, bool enable, bool pingpong, CSoundFile &sndFile);
	// Retrieve the normal loop points
	std::pair<SmpLength, SmpLength> GetLoop() const noexcept { return std::make_pair(nLoopStart, nLoopEnd); }
	// Retrieve the sustain loop points
	std::pair<SmpLength, SmpLength> GetSustainLoop() const noexcept { return std::make_pair(nSustainStart, nSustainEnd); }
	// Update loop wrap-around buffer
	void PrecomputeLoops(CSoundFile &sndFile, bool updateChannels = true);
	bool ConvertStoredDataToFloat();

	// Propagate loop point changes to player
	bool UpdateLoopPointsInActiveChannels(CSoundFile &sndFile);
	somefloat32 ReadSampleAsFloat(SmpLength frame, uint8 channel = 0) const noexcept;
	int8 ReadSampleAsInt8(SmpLength frame, uint8 channel = 0) const noexcept;

	constexpr bool HasLoop() const noexcept { return uFlags[CHN_LOOP] && nLoopEnd > nLoopStart; }
	constexpr bool HasSustainLoop() const noexcept { return uFlags[CHN_SUSTAINLOOP] && nSustainEnd > nSustainStart; }
	constexpr bool HasPingPongLoop() const noexcept { return uFlags.test_all(CHN_LOOP | CHN_PINGPONGLOOP) && nLoopEnd > nLoopStart; }
	constexpr bool HasPingPongSustainLoop() const noexcept { return uFlags.test_all(CHN_SUSTAINLOOP | CHN_PINGPONGSUSTAIN) && nSustainEnd > nSustainStart; }

	// Remove loop points if they're invalid.
	void SanitizeLoops();

	// Transpose <-> Frequency conversions
	static uint32 TransposeToFrequency(int transpose, int finetune = 0);
	void TransposeToFrequency();
	static std::pair<int8, int8> FrequencyToTranspose(uint32 freq);
	void FrequencyToTranspose();
	constexpr bool HasQuinlightRateChange() const noexcept
	{
		return nC5SpeedOriginal != 0 && nC5Speed != 0 && nC5SpeedOriginal != nC5Speed;
	}
	MPT_FORCEINLINE FreqT GetPlaybackC5Speed(MODTYPE type) const noexcept
	{
		return (type == MOD_TYPE_XM && HasQuinlightRateChange()) ? nC5SpeedOriginal : nC5Speed;
	}
	MPT_FORCEINLINE std::pair<int8, int8> GetPlaybackTransposeFineTune(MODTYPE type) const
	{
		if(type == MOD_TYPE_XM && HasQuinlightRateChange())
			return {RelativeToneOriginal, nFineTuneOriginal};
		return {RelativeTone, nFineTune};
	}

	// Transpose the sample by amount specified in octaves (i.e. amount=1 transposes one octave up)
	void Transpose(double amount);

	// Check if the sample has any valid cue points
	bool HasAnyCuePoints() const;
	// Check if the sample's cue points are the default cue point set.
	bool HasCustomCuePoints() const;
	void SetDefaultCuePoints();
	// Set cue points so that they are suitable for regular offset command extension
	void Set16BitCuePoints();
	void RemoveAllCuePoints();

	void SetAdlib(bool enable, OPLPatch patch = OPLPatch{{}});
};

template <>
MPT_FORCEINLINE const int8 *ModSample::sample<int8>() const noexcept
{
	MPT_ASSERT(GetElementarySampleSize() == sizeof(int8));
	return pData.pSample8;
}
template <>
MPT_FORCEINLINE int8 *ModSample::sample<int8>() noexcept
{
	MPT_ASSERT(GetElementarySampleSize() == sizeof(int8));
	return pData.pSample8;
}
template <>
MPT_FORCEINLINE const int16 *ModSample::sample<int16>() const noexcept
{
	MPT_ASSERT(GetElementarySampleSize() == sizeof(int16));
	return pData.pSample16;
}
template <>
MPT_FORCEINLINE int16 *ModSample::sample<int16>() noexcept
{
	MPT_ASSERT(GetElementarySampleSize() == sizeof(int16));
	return pData.pSample16;
}
template <>
MPT_FORCEINLINE const somefloat32 *ModSample::sample<somefloat32>() const noexcept
{
	MPT_ASSERT(GetRuntimeSampleFormat() == RuntimeSampleFormat::Float32);
	return pData.pSampleFloat;
}
template <>
MPT_FORCEINLINE somefloat32 *ModSample::sample<somefloat32>() noexcept
{
	MPT_ASSERT(GetRuntimeSampleFormat() == RuntimeSampleFormat::Float32);
	return pData.pSampleFloat;
}

OPENMPT_NAMESPACE_END
