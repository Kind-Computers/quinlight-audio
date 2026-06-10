/*
 * FloatMixer.h
 * ------------
 * Purpose: Floating point mixer classes
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "openmpt/all/BuildSettings.hpp"

#include "MixerInterface.h"
#include "ModSample.h"
#include "Resampler.h"
#include "mpt/base/arithmetic_shift.hpp"

OPENMPT_NAMESPACE_BEGIN

template<int channelsOut, int channelsIn, typename out, typename in, int int2float>
struct IntToFloatTraits : public MixerTraits<channelsOut, channelsIn, out, in>
{
	using base_t = MixerTraits<channelsOut, channelsIn, out, in>;
	using input_t = typename base_t::input_t;
	using output_t = typename base_t::output_t;

	static_assert(std::numeric_limits<input_t>::is_integer, "Input must be integer");
	static_assert(!std::numeric_limits<output_t>::is_integer, "Output must be floating point");

	static MPT_CONSTEXPRINLINE output_t Convert(const input_t x)
	{
		return static_cast<output_t>(x) * (static_cast<output_t>(1) / static_cast<output_t>(int2float));
	}
};

using Int8MToFloatS = IntToFloatTraits<2, 1, mixsample_t, int8,  -int8_min>;
using Int16MToFloatS = IntToFloatTraits<2, 1, mixsample_t, int16, -int16_min>;
using Int8SToFloatS = IntToFloatTraits<2, 2, mixsample_t, int8,  -int8_min>;
using Int16SToFloatS  = IntToFloatTraits<2, 2, mixsample_t, int16, -int16_min>;

// Float32 sample input → float accumulator (identity conversion)
template<int channelsOut, int channelsIn, typename out>
struct FloatToFloatTraits : public MixerTraits<channelsOut, channelsIn, out, somefloat32>
{
	using base_t = MixerTraits<channelsOut, channelsIn, out, somefloat32>;
	using input_t = typename base_t::input_t;
	using output_t = typename base_t::output_t;

	static MPT_CONSTEXPRINLINE output_t Convert(input_t x)
	{
		return static_cast<output_t>(x);
	}
};

using Float32MToFloatS = FloatToFloatTraits<2, 1, mixsample_t>;
using Float32SToFloatS = FloatToFloatTraits<2, 2, mixsample_t>;

// Float64 sample input → float accumulator (identity: both are double)
template<int channelsOut, int channelsIn, typename out>
struct Float64ToFloatTraits : public MixerTraits<channelsOut, channelsIn, out, double>
{
	using base_t = MixerTraits<channelsOut, channelsIn, out, double>;
	using input_t = typename base_t::input_t;
	using output_t = typename base_t::output_t;

	static MPT_CONSTEXPRINLINE output_t Convert(input_t x)
	{
		return x;  // double → double: identity
	}
};

using Float64MToFloatS = Float64ToFloatTraits<2, 1, mixsample_t>;
using Float64SToFloatS = Float64ToFloatTraits<2, 2, mixsample_t>;


//////////////////////////////////////////////////////////////////////////
// Interpolation templates


template<class Traits>
struct AmigaBlepInterpolation
{
	SamplePosition subIncrement;
	Paula::State &paula;
	const Paula::BlepArray &WinSincIntegral;
	const int numSteps;
	unsigned int remainingSamples = 0;

	MPT_FORCEINLINE AmigaBlepInterpolation(ModChannel &chn, const CResampler &resampler, unsigned int numSamples)
		: paula{chn.paulaState}
		, WinSincIntegral{resampler.blepTables.GetAmigaTable(resampler.m_Settings.emulateAmiga, chn.dwFlags[CHN_AMIGAFILTER])}
		, numSteps{chn.paulaState.numSteps}
	{
		if(numSteps)
		{
			subIncrement = chn.increment / numSteps;
			const int32 targetPos = (chn.position + chn.increment * numSamples).GetInt();
			if(static_cast<SmpLength>(targetPos) > chn.nLength)
				remainingSamples = numSamples;
		}
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const MPT_RESTRICT inBuffer, const uint32 posLo)
	{
		if(--remainingSamples == 0)
			subIncrement = {};

		SamplePosition pos(0, posLo);
		for(int step = numSteps; step > 0; step--)
		{
			typename Traits::output_t inSample = 0;
			int32 posInt = pos.GetInt() * Traits::numChannelsIn;
			for(int32 i = 0; i < Traits::numChannelsIn; i++)
				inSample += Traits::Convert(inBuffer[posInt + i]);
			paula.InputSample(static_cast<int16>(inSample / (4 * Traits::numChannelsIn)));
			paula.Clock(Paula::MINIMUM_INTERVAL);
			pos += subIncrement;
		}
		paula.remainder += paula.stepRemainder;

		uint32 remainClocks = paula.remainder.GetInt();
		if(remainClocks)
		{
			typename Traits::output_t inSample = 0;
			int32 posInt = pos.GetInt() * Traits::numChannelsIn;
			for(int32 i = 0; i < Traits::numChannelsIn; i++)
				inSample += Traits::Convert(inBuffer[posInt + i]);
			paula.InputSample(static_cast<int16>(inSample / (4 * Traits::numChannelsIn)));
			paula.Clock(remainClocks);
			paula.remainder.RemoveInt();
		}

		auto out = paula.OutputSample(WinSincIntegral);
		for(int i = 0; i < Traits::numChannelsOut; i++)
			outSample[i] = out;
	}
};


template<class Traits>
struct LinearInterpolation
{
	const typename Traits::output_t *sincA;
	const typename Traits::output_t *sincB;
	typename Traits::output_t blendFactor;
	int32 betaPhaseShift;

	MPT_FORCEINLINE LinearInterpolation(const ModChannel &chn, const CResampler &resampler, unsigned int)
	{
		const int64 absInc = std::abs(chn.increment.GetRaw());
		const double ratio = std::max(1.0, static_cast<double>(absInc) / 4294967296.0);
		const double mipLevel = std::log2(ratio);
		int mipA = static_cast<int>(mipLevel);
		if(mipA < 0) mipA = 0;
		if(mipA >= MIP_LEVELS - 1)
		{
			mipA = MIP_LEVELS - 1;
			sincA = resampler.gLinearMip[mipA];
			sincB = sincA;
			blendFactor = 0;
		}
		else
		{
			sincA = resampler.gLinearMip[mipA];
			sincB = resampler.gLinearMip[mipA + 1];
			blendFactor = static_cast<typename Traits::output_t>(mipLevel - mipA);
		}

		betaPhaseShift = 0;
		if(!chn.prevIncrement.IsZero())
		{
			const int64 dInc = chn.increment.GetRaw() - chn.prevIncrement.GetRaw();
			if(dInc != 0)
			{
				constexpr double k_beta = 0.25;
				constexpr double maxShift = 1073741824.0;
				constexpr double blendCeiling = 0.4;
				constexpr double blendSensitivity = 1.0 / 0x04000000;

				const double dIncD = static_cast<double>(dInc);
				betaPhaseShift = static_cast<int32>(
					std::clamp(k_beta * dIncD, -maxShift, maxShift));

				if(blendFactor == 0)
				{
					sincB = sincA;
					blendFactor = static_cast<typename Traits::output_t>(
						std::min(blendCeiling, std::abs(dIncD) * blendSensitivity * blendCeiling));
				}
			}
		}
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const inBuffer, const uint32 posLo)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");
		const uint32 phase = ((posLo >> (32 - TAP2_PHASES_BITS)) & TAP2_MASK) * TAP2_WIDTH;
		const typename Traits::output_t *lutA = sincA + phase;

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t out =
				  lutA[0] * Traits::Convert(inBuffer[i])
				+ lutA[1] * Traits::Convert(inBuffer[i + Traits::numChannelsIn]);

			if(blendFactor != 0)
			{
				const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
				const uint32 phaseB = ((posLoB >> (32 - TAP2_PHASES_BITS)) & TAP2_MASK) * TAP2_WIDTH;
				const typename Traits::output_t *lutB = sincB + phaseB;
				typename Traits::output_t outB =
					  lutB[0] * Traits::Convert(inBuffer[i])
					+ lutB[1] * Traits::Convert(inBuffer[i + Traits::numChannelsIn]);
				out = out + blendFactor * (outB - out);
			}

			outSample[i] = out;
		}
	}
};


template<class Traits>
struct FastSincInterpolation
{
	const typename Traits::output_t *sincA;
	const typename Traits::output_t *sincB;
	typename Traits::output_t blendFactor;
	int32 betaPhaseShift;

	MPT_FORCEINLINE FastSincInterpolation(const ModChannel &chn, const CResampler &resampler, unsigned int)
	{
		const int64 absInc = std::abs(chn.increment.GetRaw());
		const double ratio = std::max(1.0, static_cast<double>(absInc) / 4294967296.0);
		const double mipLevel = std::log2(ratio);
		int mipA = static_cast<int>(mipLevel);
		if(mipA < 0) mipA = 0;
		if(mipA >= MIP_LEVELS - 1)
		{
			mipA = MIP_LEVELS - 1;
			sincA = resampler.g4TapMip[mipA];
			sincB = sincA;
			blendFactor = 0;
		}
		else
		{
			sincA = resampler.g4TapMip[mipA];
			sincB = resampler.g4TapMip[mipA + 1];
			blendFactor = static_cast<typename Traits::output_t>(mipLevel - mipA);
		}

		betaPhaseShift = 0;
		if(!chn.prevIncrement.IsZero())
		{
			const int64 dInc = chn.increment.GetRaw() - chn.prevIncrement.GetRaw();
			if(dInc != 0)
			{
				constexpr double k_beta = 0.4;
				constexpr double maxShift = 1073741824.0;
				constexpr double blendCeiling = 0.45;
				constexpr double blendSensitivity = 1.0 / 0x04000000;

				const double dIncD = static_cast<double>(dInc);
				betaPhaseShift = static_cast<int32>(
					std::clamp(k_beta * dIncD, -maxShift, maxShift));

				if(blendFactor == 0)
				{
					sincB = sincA;
					blendFactor = static_cast<typename Traits::output_t>(
						std::min(blendCeiling, std::abs(dIncD) * blendSensitivity * blendCeiling));
				}
			}
		}
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const inBuffer, const uint32 posLo)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");
		const uint32 phase = ((posLo >> (32 - TAP4_PHASES_BITS)) & TAP4_MASK) * TAP4_WIDTH;
		const typename Traits::output_t *lutA = sincA + phase;

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t out =
				  lutA[0] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
				+ lutA[1] * Traits::Convert(inBuffer[i])
				+ lutA[2] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
				+ lutA[3] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn]);

			if(blendFactor != 0)
			{
				const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
				const uint32 phaseB = ((posLoB >> (32 - TAP4_PHASES_BITS)) & TAP4_MASK) * TAP4_WIDTH;
				const typename Traits::output_t *lutB = sincB + phaseB;
				typename Traits::output_t outB =
					  lutB[0] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
					+ lutB[1] * Traits::Convert(inBuffer[i])
					+ lutB[2] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
					+ lutB[3] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn]);
				out = out + blendFactor * (outB - out);
			}

			outSample[i] = out;
		}
	}
};


template<class Traits>
struct CatmullRomInterpolation
{
	const typename Traits::output_t *sincA;
	const typename Traits::output_t *sincB;
	typename Traits::output_t blendFactor;
	typename Traits::output_t betaBlendWeight;  // Pre-computed beta-shear blend weight (used in pure-CR path)
	typename Traits::output_t catmullWeight;    // 1.0 = pure Catmull-Rom, 0.0 = pure sinc mip
	int32 betaPhaseShift;

	MPT_FORCEINLINE CatmullRomInterpolation(const ModChannel &chn, const CResampler &resampler, unsigned int)
	{
		const int64 absInc = std::abs(chn.increment.GetRaw());
		const double ratio = static_cast<double>(absInc) / 4294967296.0;

		// Hybrid crossfade: Catmull-Rom character at near-unity, sinc AA at high ratios.
		// ratio <= 1.0:  pure Catmull-Rom
		// 1.0 - 2.0:    crossfade Catmull-Rom → sinc mip
		// ratio > 2.0:  pure sinc mip with LOD selection
		if(ratio <= 1.0)
		{
			catmullWeight = 1;
			sincA = resampler.g4TapMip[0];
			sincB = sincA;
			blendFactor = 0;
		}
		else
		{
			const double mipLevel = std::log2(ratio);
			catmullWeight = static_cast<typename Traits::output_t>(
				std::max(0.0, 1.0 - mipLevel));

			int mipA = static_cast<int>(mipLevel);
			if(mipA < 0) mipA = 0;
			if(mipA >= MIP_LEVELS - 1)
			{
				mipA = MIP_LEVELS - 1;
				sincA = resampler.g4TapMip[mipA];
				sincB = sincA;
				blendFactor = 0;
			}
			else
			{
				sincA = resampler.g4TapMip[mipA];
				sincB = resampler.g4TapMip[mipA + 1];
				blendFactor = static_cast<typename Traits::output_t>(mipLevel - mipA);
			}
		}

		betaPhaseShift = 0;
		betaBlendWeight = 0;
		if(!chn.prevIncrement.IsZero())
		{
			const int64 dInc = chn.increment.GetRaw() - chn.prevIncrement.GetRaw();
			if(dInc != 0)
			{
				constexpr double k_beta = 0.5;
				constexpr double maxShift = 1073741824.0;
				constexpr double blendCeiling = 0.5;
				constexpr double blendSensitivity = 1.0 / 0x04000000;

				const double dIncD = static_cast<double>(dInc);
				betaPhaseShift = static_cast<int32>(
					std::clamp(k_beta * dIncD, -maxShift, maxShift));
				betaBlendWeight = static_cast<typename Traits::output_t>(
					std::min(blendCeiling, std::abs(dIncD) * blendSensitivity * blendCeiling));

				if(blendFactor == 0 && catmullWeight < 1)
				{
					sincB = sincA;
					blendFactor = betaBlendWeight;
				}
			}
		}
	}

	// Evaluate centripetal Catmull-Rom (α=0.5) at a single fractional position
	// using the Barry-Goldman recursive formula.
	static MPT_FORCEINLINE typename Traits::output_t EvalCatmullRom(
		typename Traits::output_t P0, typename Traits::output_t P1,
		typename Traits::output_t P2, typename Traits::output_t P3,
		typename Traits::output_t frac)
	{
		using T = typename Traits::output_t;
		constexpr T epsilon = static_cast<T>(1e-7);

		const T d01 = std::sqrt(std::abs(P1 - P0)) + epsilon;
		const T d12 = std::sqrt(std::abs(P2 - P1)) + epsilon;
		const T d23 = std::sqrt(std::abs(P3 - P2)) + epsilon;

		const T t1 = d01;
		const T t2 = d01 + d12;
		const T t3 = d01 + d12 + d23;
		const T t = t1 + frac * d12;

		const T A1 = (t1 - t) / t1       * P0 + t       / t1       * P1;
		const T A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2;
		const T A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3;

		const T B1 = (t2 - t) / t2       * A1 + t       / t2       * A2;
		const T B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3;

		return    (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2;
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const inBuffer, const uint32 posLo)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");
		const typename Traits::output_t frac = posLo / static_cast<typename Traits::output_t>(0x100000000);

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			const auto P0 = Traits::Convert(inBuffer[i - Traits::numChannelsIn]);
			const auto P1 = Traits::Convert(inBuffer[i]);
			const auto P2 = Traits::Convert(inBuffer[i + Traits::numChannelsIn]);
			const auto P3 = Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn]);

			typename Traits::output_t out;

			if(catmullWeight >= 1)
			{
				// Pure Catmull-Rom (upsampling / near-unity)
				out = EvalCatmullRom(P0, P1, P2, P3, frac);

				if(betaBlendWeight != 0)
				{
					const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
					const typename Traits::output_t fracB = posLoB / static_cast<typename Traits::output_t>(0x100000000);
					auto outB = EvalCatmullRom(P0, P1, P2, P3, fracB);
					out += betaBlendWeight * (outB - out);
				}
			}
			else
			{
				// Sinc mip path (downsampling)
				const uint32 phase = ((posLo >> (32 - TAP4_PHASES_BITS)) & TAP4_MASK) * TAP4_WIDTH;
				const typename Traits::output_t *lutA = sincA + phase;
				typename Traits::output_t outSinc =
					  lutA[0] * P0 + lutA[1] * P1 + lutA[2] * P2 + lutA[3] * P3;

				if(blendFactor != 0)
				{
					const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
					const uint32 phaseB = ((posLoB >> (32 - TAP4_PHASES_BITS)) & TAP4_MASK) * TAP4_WIDTH;
					const typename Traits::output_t *lutB = sincB + phaseB;
					typename Traits::output_t outB =
						  lutB[0] * P0 + lutB[1] * P1 + lutB[2] * P2 + lutB[3] * P3;
					outSinc = outSinc + blendFactor * (outB - outSinc);
				}

				if(catmullWeight > 0)
				{
					// Crossfade zone: blend Catmull-Rom and sinc
					auto outCR = EvalCatmullRom(P0, P1, P2, P3, frac);
					out = catmullWeight * outCR + (1 - catmullWeight) * outSinc;
				}
				else
				{
					out = outSinc;
				}
			}

			outSample[i] = out;
		}
	}
};


template<class Traits>
struct PolyphaseInterpolation
{
	const typename Traits::output_t *sincA;
	const typename Traits::output_t *sincB;  // Second kernel for trilinear blend (== sincA when not blending)
	typename Traits::output_t blendFactor;   // 0.0 = pure A, >0.0 = blend toward B
	int32 betaPhaseShift;                    // Anisotropic β-shear: phase offset for B kernel (posLo units)

	MPT_FORCEINLINE PolyphaseInterpolation(const ModChannel &chn, const CResampler &resampler, unsigned int)
	{
		// Mip LOD selection: log2(ratio) → adjacent mip levels + trilinear blend.
		// Replaces the old 3-kernel threshold system with continuous LOD.
		const int64 absInc = std::abs(chn.increment.GetRaw());
		const double ratio = std::max(1.0, static_cast<double>(absInc) / 4294967296.0);
		const double mipLevel = std::log2(ratio);
		int mipA = static_cast<int>(mipLevel);
		if(mipA < 0) mipA = 0;
		if(mipA >= MIP_LEVELS - 1)
		{
			mipA = MIP_LEVELS - 1;
			sincA = resampler.gPolyphaseMip[mipA];
			sincB = sincA;
			blendFactor = 0;
		}
		else
		{
			sincA = resampler.gPolyphaseMip[mipA];
			sincB = resampler.gPolyphaseMip[mipA + 1];
			blendFactor = static_cast<typename Traits::output_t>(mipLevel - mipA);
		}

		// Anisotropic β-shear: shift B kernel's phase along pitch trajectory.
		betaPhaseShift = 0;
		if(!chn.prevIncrement.IsZero())
		{
			const int64 dInc = chn.increment.GetRaw() - chn.prevIncrement.GetRaw();
			if(dInc != 0)
			{
				constexpr double k_beta = 0.5;
				constexpr double maxShift = 1073741824.0;
				constexpr double blendCeiling = 0.5;
				constexpr double blendSensitivity = 1.0 / 0x04000000;

				const double dIncD = static_cast<double>(dInc);
				betaPhaseShift = static_cast<int32>(
					std::clamp(k_beta * dIncD, -maxShift, maxShift));

				if(blendFactor == 0)
				{
					sincB = sincA;
					blendFactor = static_cast<typename Traits::output_t>(
						std::min(blendCeiling, std::abs(dIncD) * blendSensitivity * blendCeiling));
				}
			}
		}
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const inBuffer, const uint32 posLo)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");
		const uint32 phase = ((posLo >> (32 - SINC_PHASES_BITS)) & SINC_MASK) * SINC_WIDTH;
		const typename Traits::output_t *lutA = sincA + phase;

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t out =
				  lutA[ 0] * Traits::Convert(inBuffer[i - 7 * Traits::numChannelsIn])
				+ lutA[ 1] * Traits::Convert(inBuffer[i - 6 * Traits::numChannelsIn])
				+ lutA[ 2] * Traits::Convert(inBuffer[i - 5 * Traits::numChannelsIn])
				+ lutA[ 3] * Traits::Convert(inBuffer[i - 4 * Traits::numChannelsIn])
				+ lutA[ 4] * Traits::Convert(inBuffer[i - 3 * Traits::numChannelsIn])
				+ lutA[ 5] * Traits::Convert(inBuffer[i - 2 * Traits::numChannelsIn])
				+ lutA[ 6] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
				+ lutA[ 7] * Traits::Convert(inBuffer[i])
				+ lutA[ 8] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
				+ lutA[ 9] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn])
				+ lutA[10] * Traits::Convert(inBuffer[i + 3 * Traits::numChannelsIn])
				+ lutA[11] * Traits::Convert(inBuffer[i + 4 * Traits::numChannelsIn])
				+ lutA[12] * Traits::Convert(inBuffer[i + 5 * Traits::numChannelsIn])
				+ lutA[13] * Traits::Convert(inBuffer[i + 6 * Traits::numChannelsIn])
				+ lutA[14] * Traits::Convert(inBuffer[i + 7 * Traits::numChannelsIn])
				+ lutA[15] * Traits::Convert(inBuffer[i + 8 * Traits::numChannelsIn]);

			if(blendFactor != 0)
			{
				// Trilinear blend (+ anisotropic β-shear): evaluate second kernel at
				// shifted phase and lerp.  When betaPhaseShift != 0, the B kernel
				// probes along the pitch trajectory instead of at the same phase as A.
				const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
				const uint32 phaseB = ((posLoB >> (32 - SINC_PHASES_BITS)) & SINC_MASK) * SINC_WIDTH;
				const typename Traits::output_t *lutB = sincB + phaseB;
				typename Traits::output_t outB =
					  lutB[ 0] * Traits::Convert(inBuffer[i - 7 * Traits::numChannelsIn])
					+ lutB[ 1] * Traits::Convert(inBuffer[i - 6 * Traits::numChannelsIn])
					+ lutB[ 2] * Traits::Convert(inBuffer[i - 5 * Traits::numChannelsIn])
					+ lutB[ 3] * Traits::Convert(inBuffer[i - 4 * Traits::numChannelsIn])
					+ lutB[ 4] * Traits::Convert(inBuffer[i - 3 * Traits::numChannelsIn])
					+ lutB[ 5] * Traits::Convert(inBuffer[i - 2 * Traits::numChannelsIn])
					+ lutB[ 6] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
					+ lutB[ 7] * Traits::Convert(inBuffer[i])
					+ lutB[ 8] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
					+ lutB[ 9] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn])
					+ lutB[10] * Traits::Convert(inBuffer[i + 3 * Traits::numChannelsIn])
					+ lutB[11] * Traits::Convert(inBuffer[i + 4 * Traits::numChannelsIn])
					+ lutB[12] * Traits::Convert(inBuffer[i + 5 * Traits::numChannelsIn])
					+ lutB[13] * Traits::Convert(inBuffer[i + 6 * Traits::numChannelsIn])
					+ lutB[14] * Traits::Convert(inBuffer[i + 7 * Traits::numChannelsIn])
					+ lutB[15] * Traits::Convert(inBuffer[i + 8 * Traits::numChannelsIn]);
				out = out + blendFactor * (outB - out);
			}

			outSample[i] = out;
		}
	}
};


// Forward declarations for SIMD kernels — each compiled in its own TU with ISA-specific flags.
// All instantiated from Aniso64Kernel.h templates. Runtime CPUID selects the best path.

// SSE2 (baseline x86-64, no special flags)
extern "C" double aniso64_dot_mono_sse2(const double * __restrict__ kernel, const double * __restrict__ samples);
extern "C" void aniso64_dot_stereo_sse2(const double * __restrict__ kernel, const double * __restrict__ samples, double * __restrict__ outL, double * __restrict__ outR);

// AVX (-mavx, no FMA — vmulpd + vaddpd)
extern "C" double aniso64_dot_mono_avx(const double * __restrict__ kernel, const double * __restrict__ samples);
extern "C" void aniso64_dot_stereo_avx(const double * __restrict__ kernel, const double * __restrict__ samples, double * __restrict__ outL, double * __restrict__ outR);

// AVX2+FMA3 (-mavx2 -mfma — single-instruction vfmadd)
extern "C" double aniso64_dot_mono_avx2(const double * __restrict__ kernel, const double * __restrict__ samples);
extern "C" void aniso64_dot_stereo_avx2(const double * __restrict__ kernel, const double * __restrict__ samples, double * __restrict__ outL, double * __restrict__ outR);

// AVX-512F+VL (-mavx512f -mavx512vl — 8 doubles/vector)
extern "C" double aniso64_dot_mono_avx512(const double * __restrict__ kernel, const double * __restrict__ samples);
extern "C" void aniso64_dot_stereo_avx512(const double * __restrict__ kernel, const double * __restrict__ samples, double * __restrict__ outL, double * __restrict__ outR);


// 64-tap anisotropic sinc interpolation with SIMD acceleration — the full
// sheared-separable gather of "Anisotropic Audio Filters" (Quinlight paper,
// March 2026), Eq. 13/14, over TRUE DATA MIPS (the paper's W[q,m]):
//
//   * Each sample carries an octave-decimated data pyramid (see
//     SampleMipChain.cpp), exactly like a GPU texture mip chain. Slice j
//     reads mip level j by absolute position (position raw >> j — dyadic,
//     so 64 taps at level j span 64·2^j original samples: properly
//     bandlimited at any downsampling ratio).
//   * The footprint spans up to MaxSlices mip levels around the continuous
//     level μ, widened by R = 1 + k_r·|μ̇| (Eq. 15) when the pitch is moving.
//     Slice weights are a stretched tent w_j ∝ max(0, 1 − |j−μ|/R),
//     normalized — at R = 1 this reduces EXACTLY to the classic (1−ε, ε)
//     trilinear blend, i.e. ordinary GPU-style interpolation at steady pitch.
//   * Each slice's kernel is picked from a fractional one-octave family by
//     its residual ratio r_j = ratio / 2^j ∈ [0.5, 2) (nearest eighth-octave
//     cutoff; the data-mip blend covers the octave axis).
//   * Each slice's phase taps are sheared by β·(j−μ) where
//     β = k_β·İ + k_β²·Ï (tempo-invariant per-output-sample derivatives of
//     the increment, computed in Sndmix.cpp) — the per-slice tap tilt of
//     Eq. 14. The shear is added in mip-0 raw units BEFORE the >> j shift,
//     so its effective magnitude scales correctly per level.
//   * Near designated loop boundaries, mip slices switch to per-level strip
//     windows holding the decimation of the periodically extended signal —
//     the mip-level analog of the level-0 loop wrap-around buffers — so
//     loop seams stay continuous and key-off tails never bleed into loops.
template<class Traits>
struct Aniso64Interpolation
{
	static constexpr int MaxSlices = 4;
	// Per-slice gather state. Level-0 slices read through the inBuffer
	// pointer handed in by the caller, leaving Fastmix's loop-lookahead
	// pointer swap (and its quirks) fully intact; slices at level >= 1 read
	// the data mip pyramid by absolute position.
	const typename Traits::output_t *sliceKernel[MaxSlices];
	typename Traits::output_t sliceWeight[MaxSlices];
	int64 sliceShearRaw[MaxSlices];  // β·(j−μ) in mip-0 32.32 raw units
	const typename Traits::input_t *sliceBody[MaxSlices];        // mip body, indexable by int_j (level >= 1)
	const typename Traits::input_t *sliceStripEnd[MaxSlices];    // loop-end strip, bias-adjusted to int_j
	const typename Traits::input_t *sliceStripStart[MaxSlices];  // loop-start strip, bias-adjusted to int_j
	int32 sliceEndLo[MaxSlices];     // engage end strip when int_j >= this (int32_max = never)
	int32 sliceStartHi[MaxSlices];   // engage start strip when int_j < this (int32_min = never)
	uint8 sliceLevel[MaxSlices];
	int numSlices;
	int64 posRaw;  // absolute 32.32 position, advanced in lockstep with SampleLoop's smpPos
	int64 incRaw;
	bool useAVX512;
	bool useAVX2;
	bool useAVX;

	// Kernel level for the residual ratio r = ratio / 2^level ∈ [0.5, 2):
	// nearest eighth-octave step. r only exceeds 2 when the slice range was
	// clamped at the sample's deepest built mip — the lowest-cutoff kernel
	// is then the best (if under-bandlimited) choice available.
	static MPT_FORCEINLINE int KernelIndex(double ratio, int level)
	{
		const double r = ratio / static_cast<double>(int64(1) << level);
		if(r <= 1.0)
			return 0;
		const int idx = static_cast<int>(8.0 * std::log2(r) + 0.5);
		return std::min(idx, MIP_LEVELS - 1);
	}

	MPT_FORCEINLINE Aniso64Interpolation(const ModChannel &chn, const CResampler &resampler, unsigned int)
	{
		static_assert(SampleMip::kMaxLevels == MIP_LEVELS - 1);

		// Continuous mip level: ratio = increment / unity (>1 = downsampling).
		const int64 absInc = std::abs(chn.increment.GetRaw());
		const double ratio = std::max(1.0, static_cast<double>(absInc) / 4294967296.0);

		// Absolute-position tracking: the ctor runs before SampleLoop copies
		// c.position, and smpPos advances by one raw add per output sample —
		// posRaw += incRaw in operator() replicates it bit-exactly.
		posRaw = chn.position.GetRaw();
		incRaw = chn.increment.GetRaw();

		// Depth of the data mip pyramid built for this sample. 0 (no
		// pyramid yet) degrades to a level-0-only gather. CanFitMips guards
		// against a stale nMipLevelsBuilt carried along by a ModSample struct
		// copy whose buffer was later swapped without PrecomputeLoops (e.g.
		// the tracker-only LoadExternalSample path): if the current
		// length/format cannot contain a pyramid, never read past the body.
		const ModSample *smp = chn.pModSample;
		const int maxLevel = (smp != nullptr && smp->HasSampleData() && ModSample::CanFitMips(smp->nLength, smp->GetBytesPerSample()))
			? std::min(static_cast<int>(smp->nMipLevelsBuilt), SampleMip::MaxLevels(smp->nLength))
			: 0;
		const double mu = std::min(std::log2(ratio), static_cast<double>(maxLevel));

		const double k_beta = resampler.m_Settings.aniso64_k_beta;
		const double k_beta2 = resampler.m_Settings.aniso64_k_beta2;
		const double k_r = resampler.m_Settings.aniso64_k_r;

		// chn.anisoIncDot/anisoIncDotDot are per-output-sample; the legacy
		// defaults (k_β = 0.65, k_β² = 0.15) were tuned against raw per-tick
		// deltas at ≈1024-sample ticks, so velNorm restores those magnitudes
		// while keeping the shear tempo-invariant.
		constexpr double velNorm = 1024.0;
		constexpr double maxShift = 1073741824.0; // ±0x40000000 = ±0.25 sample
		const double shear = k_beta * chn.anisoIncDot * velNorm
			+ k_beta2 * chn.anisoIncDotDot * velNorm * velNorm;
		const double beta = std::clamp(shear, -maxShift, maxShift);

		// Footprint width R = 1 + k_r·|μ̇| (Eq. 15). μ̇ is mip-levels per
		// output sample — tiny — so scale to per-4096-samples to keep the ctl
		// O(1). R is clamped to [1, 2] (at most 4 slices).
		const double muDotScaled = chn.anisoMuDot * 4096.0;
		const double r = std::clamp(1.0 + k_r * std::abs(muDotScaled), 1.0, 2.0);

		int lo = static_cast<int>(std::floor(mu - (r - 1.0)));
		int hi = static_cast<int>(std::ceil(mu + (r - 1.0)));
		lo = std::clamp(lo, 0, maxLevel);
		hi = std::clamp(hi, lo, maxLevel);
		if(hi - lo >= MaxSlices)
			lo = hi - (MaxSlices - 1); // keep the most-bandlimited end (§10.3)

		// Strip gating, mirroring MixLoopState::UpdateLookaheadPointers: the
		// strips describe the sample's stored loop, so they only apply when
		// the channel's active loop matches it (no custom-loop preview).
		bool stripsOk = false, useSustainStrips = false, wrapped = false;
		SmpLength loopStart = 0, loopEnd = 0;
		if(smp != nullptr && chn.dwFlags[CHN_LOOP])
		{
			const bool inSustain = chn.InSustainLoop() && chn.nLoopStart == smp->nSustainStart && chn.nLoopEnd == smp->nSustainEnd;
			if(inSustain || (chn.nLoopStart == smp->nLoopStart && chn.nLoopEnd == smp->nLoopEnd))
			{
				stripsOk = true;
				useSustainStrips = inSustain;
				loopStart = chn.nLoopStart;
				loopEnd = chn.nLoopEnd;
				wrapped = chn.dwFlags[CHN_WRAPPED_LOOP];
			}
		}

		numSlices = 0;
		double weights[MaxSlices];
		double wsum = 0.0;
		for(int j = lo; j <= hi && numSlices < MaxSlices; j++)
		{
			const double w = std::max(0.0, 1.0 - std::abs(static_cast<double>(j) - mu) / r);
			if(w <= 0.0)
				continue;
			weights[numSlices] = w;
			InitSlice(numSlices, j, KernelIndex(ratio, j),
				static_cast<int64>(std::clamp(beta * (static_cast<double>(j) - mu), -maxShift, maxShift)),
				resampler, smp, loopStart, loopEnd, stripsOk, useSustainStrips, wrapped);
			wsum += w;
			numSlices++;
		}
		if(numSlices == 0)
		{
			// Unreachable (the slice nearest μ always has w > 0), but never
			// leave the gather empty.
			const int j = std::clamp(static_cast<int>(mu + 0.5), 0, maxLevel);
			InitSlice(0, j, KernelIndex(ratio, j), 0, resampler, smp, loopStart, loopEnd, stripsOk, useSustainStrips, wrapped);
			sliceWeight[0] = static_cast<typename Traits::output_t>(1);
			numSlices = 1;
			wsum = 0.0; // weights already final
		}
		if(wsum > 0.0)
		{
			for(int s = 0; s < numSlices; s++)
				sliceWeight[s] = static_cast<typename Traits::output_t>(weights[s] / wsum);
		}

		// Velocity blur at an integral mip level with R = 1 (typically mip 0
		// during upsampled playback, where μ̇ == 0 by construction because the
		// log is floored at unity ratio): blend toward a sheared copy of the
		// SAME slice so fast linear pitch motion still widens the effective
		// phase footprint — the §9.1 behavior this gather grew out of.
		if(numSlices == 1 && beta != 0.0)
		{
			constexpr double blendCeiling = 0.6;
			constexpr double blendSensitivity = 1.0 / 0x04000000;
			const double blend = std::min(
				blendCeiling, std::abs(chn.anisoIncDot * velNorm) * blendSensitivity * blendCeiling);
			if(blend > 0.0)
			{
				CopySlice(1, 0);
				sliceShearRaw[1] = static_cast<int64>(beta);
				sliceWeight[0] = static_cast<typename Traits::output_t>(1.0 - blend);
				sliceWeight[1] = static_cast<typename Traits::output_t>(blend);
				numSlices = 2;
			}
		}

		useAVX512 = resampler.m_hasAVX512;
		useAVX2 = resampler.m_hasAVX2;
		useAVX = resampler.m_hasAVX;
	}

	MPT_FORCEINLINE void InitSlice(int s, int level, int kernelIdx, int64 shearRaw, const CResampler &resampler,
		const ModSample *smp, SmpLength loopStart, SmpLength loopEnd, bool stripsOk, bool useSustainStrips, bool wrapped)
	{
		sliceKernel[s] = resampler.gAniso64Mip[kernelIdx];
		sliceShearRaw[s] = shearRaw;
		sliceLevel[s] = static_cast<uint8>(level);
		sliceBody[s] = nullptr;
		sliceStripEnd[s] = nullptr;
		sliceStripStart[s] = nullptr;
		sliceEndLo[s] = int32_max;
		sliceStartHi[s] = int32_min;
		if(level < 1)
			return;
		const SmpLength len = smp->nLength;
		constexpr int nch = Traits::numChannelsIn;
		const auto *base = static_cast<const typename Traits::input_t *>(smp->samplev());
		sliceBody[s] = base + SampleMip::BodyOffsetFrames(len, level) * nch;
		if(stripsOk)
		{
			const int endWin = useSustainStrips ? SampleMip::kStripSustainEnd : SampleMip::kStripLoopEnd;
			const int32 endC = static_cast<int32>(loopEnd >> level);
			const int32 startC = static_cast<int32>(loopStart >> level);
			// Strip frame i holds mip-level position C - 128 + i; bias-adjust
			// the pointers so they are indexable by int_j directly.
			sliceStripEnd[s] = base + SampleMip::StripOffsetFrames(len, level, endWin) * nch - static_cast<int64>(endC - 128) * nch;
			sliceStripStart[s] = base + SampleMip::StripOffsetFrames(len, level, endWin + 1) * nch - static_cast<int64>(startC - 128) * nch;
			// Mirror Fastmix's lookaheadStart = max(loopStart, loopEnd - 64):
			// inside tiny loops the (fully periodized) end strip serves the
			// whole loop body.
			sliceEndLo[s] = std::max(startC, endC - static_cast<int32>(SampleMip::kLookahead));
			if(wrapped)
				sliceStartHi[s] = startC + static_cast<int32>(SampleMip::kLookahead);
		}
	}

	MPT_FORCEINLINE void CopySlice(int dst, int src)
	{
		sliceKernel[dst] = sliceKernel[src];
		sliceShearRaw[dst] = sliceShearRaw[src];
		sliceLevel[dst] = sliceLevel[src];
		sliceBody[dst] = sliceBody[src];
		sliceStripEnd[dst] = sliceStripEnd[src];
		sliceStripStart[dst] = sliceStripStart[src];
		sliceEndLo[dst] = sliceEndLo[src];
		sliceStartHi[dst] = sliceStartHi[src];
	}

	// Resolve the data pointer and kernel phase for a level >= 1 slice at
	// the given (shear-adjusted) absolute raw position: mip body, or one of
	// the loop strips near a designated boundary. The end strip has
	// priority — for tiny loops both zones overlap and the end strip (fully
	// periodized on both sides) is the correct one.
	MPT_FORCEINLINE const typename Traits::input_t *SliceData(int s, int64 raw, uint32 &phase) const
	{
		const int j = sliceLevel[s];
		const int32 intk = static_cast<int32>(mpt::rshift_signed(raw, 32 + j));
		const uint32 frac = static_cast<uint32>(static_cast<uint64>(raw) >> j);
		phase = ((frac >> (32 - SINC_PHASES_64_BITS)) & SINC_MASK_64) * SINC_WIDTH_64;
		constexpr int nch = Traits::numChannelsIn;
		if(intk >= sliceEndLo[s])
			return sliceStripEnd[s] + intk * nch;
		if(intk < sliceStartHi[s])
			return sliceStripStart[s] + intk * nch;
		return sliceBody[s] + intk * nch;
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const inBuffer, const uint32 posLo)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");
		MPT_ASSERT(static_cast<uint32>(static_cast<uint64>(posRaw) & 0xFFFFFFFFu) == posLo);

		// SIMD stereo fast path: both channels computed simultaneously via
		// deinterleave. Dispatch cascade: AVX-512 → AVX2 → AVX → SSE2.
		if(std::is_same<typename Traits::input_t, double>::value && Traits::numChannelsIn == 2)
		{
			double accL = 0.0, accR = 0.0;
			for(int s = 0; s < numSlices; s++)
			{
				const double *samples;
				uint32 phase;
				if(sliceLevel[s] == 0)
				{
					const uint32 posLoS = static_cast<uint32>(static_cast<int32>(posLo) + static_cast<int32>(sliceShearRaw[s]));
					phase = ((posLoS >> (32 - SINC_PHASES_64_BITS)) & SINC_MASK_64) * SINC_WIDTH_64;
					samples = reinterpret_cast<const double *>(inBuffer) - 31 * 2;
				} else
				{
					samples = reinterpret_cast<const double *>(SliceData(s, posRaw + sliceShearRaw[s], phase)) - 31 * 2;
				}
				const typename Traits::output_t *lut = sliceKernel[s] + phase;
				double outL, outR;
				if(useAVX512)
					aniso64_dot_stereo_avx512(lut, samples, &outL, &outR);
				else if(useAVX2)
					aniso64_dot_stereo_avx2(lut, samples, &outL, &outR);
				else if(useAVX)
					aniso64_dot_stereo_avx(lut, samples, &outL, &outR);
				else
					aniso64_dot_stereo_sse2(lut, samples, &outL, &outR);
				const double w = static_cast<double>(sliceWeight[s]);
				accL += w * outL;
				accR += w * outR;
			}
			outSample[0] = static_cast<typename Traits::output_t>(accL);
			outSample[1] = static_cast<typename Traits::output_t>(accR);
			posRaw += incRaw;
			return;
		}

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t acc = 0;
			for(int s = 0; s < numSlices; s++)
			{
				uint32 phase;
				const typename Traits::input_t *data;
				if(sliceLevel[s] == 0)
				{
					const uint32 posLoS = static_cast<uint32>(static_cast<int32>(posLo) + static_cast<int32>(sliceShearRaw[s]));
					phase = ((posLoS >> (32 - SINC_PHASES_64_BITS)) & SINC_MASK_64) * SINC_WIDTH_64;
					data = inBuffer;
				} else
				{
					data = SliceData(s, posRaw + sliceShearRaw[s], phase);
				}
				const typename Traits::output_t *lut = sliceKernel[s] + phase;

				typename Traits::output_t out;
				// SIMD mono fast path: Float64 mono samples are contiguous
				// doubles — zero conversion.
				if(std::is_same<typename Traits::input_t, double>::value && Traits::numChannelsIn == 1)
				{
					const double *samples = reinterpret_cast<const double *>(data + i) - 31;
					if(useAVX512)
						out = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx512(lut, samples));
					else if(useAVX2)
						out = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx2(lut, samples));
					else if(useAVX)
						out = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx(lut, samples));
					else
						out = static_cast<typename Traits::output_t>(aniso64_dot_mono_sse2(lut, samples));
				}
				else
				{
					// Scalar fallback: non-double formats and stereo stride.
					out = 0;
					for(int t = 0; t < SINC_WIDTH_64; t++)
					{
						out += lut[t] * Traits::Convert(data[i + (t - 31) * Traits::numChannelsIn]);
					}
				}
				acc += sliceWeight[s] * out;
			}
			outSample[i] = acc;
		}
		posRaw += incRaw;
	}
};

template<class Traits>
struct FIRFilterInterpolation
{
	const typename Traits::output_t *sincA;
	const typename Traits::output_t *sincB;
	typename Traits::output_t blendFactor;
	int32 betaPhaseShift;

	MPT_FORCEINLINE FIRFilterInterpolation(const ModChannel &chn, const CResampler &resampler, unsigned int)
	{
		const int64 absInc = std::abs(chn.increment.GetRaw());
		const double ratio = std::max(1.0, static_cast<double>(absInc) / 4294967296.0);
		const double mipLevel = std::log2(ratio);
		int mipA = static_cast<int>(mipLevel);
		if(mipA < 0) mipA = 0;
		if(mipA >= MIP_LEVELS - 1)
		{
			mipA = MIP_LEVELS - 1;
			sincA = resampler.gSinc8Mip[mipA];
			sincB = sincA;
			blendFactor = 0;
		}
		else
		{
			sincA = resampler.gSinc8Mip[mipA];
			sincB = resampler.gSinc8Mip[mipA + 1];
			blendFactor = static_cast<typename Traits::output_t>(mipLevel - mipA);
		}

		betaPhaseShift = 0;
		if(!chn.prevIncrement.IsZero())
		{
			const int64 dInc = chn.increment.GetRaw() - chn.prevIncrement.GetRaw();
			if(dInc != 0)
			{
				constexpr double k_beta = 0.45;
				constexpr double maxShift = 1073741824.0;
				constexpr double blendCeiling = 0.5;
				constexpr double blendSensitivity = 1.0 / 0x04000000;

				const double dIncD = static_cast<double>(dInc);
				betaPhaseShift = static_cast<int32>(
					std::clamp(k_beta * dIncD, -maxShift, maxShift));

				if(blendFactor == 0)
				{
					sincB = sincA;
					blendFactor = static_cast<typename Traits::output_t>(
						std::min(blendCeiling, std::abs(dIncD) * blendSensitivity * blendCeiling));
				}
			}
		}
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const inBuffer, const uint32 posLo)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");
		const uint32 phase = ((posLo >> (32 - SINC8_PHASES_BITS)) & SINC8_MASK) * SINC8_WIDTH;
		const typename Traits::output_t *lutA = sincA + phase;

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t out =
				  lutA[0] * Traits::Convert(inBuffer[i - 3 * Traits::numChannelsIn])
				+ lutA[1] * Traits::Convert(inBuffer[i - 2 * Traits::numChannelsIn])
				+ lutA[2] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
				+ lutA[3] * Traits::Convert(inBuffer[i])
				+ lutA[4] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
				+ lutA[5] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn])
				+ lutA[6] * Traits::Convert(inBuffer[i + 3 * Traits::numChannelsIn])
				+ lutA[7] * Traits::Convert(inBuffer[i + 4 * Traits::numChannelsIn]);

			if(blendFactor != 0)
			{
				const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
				const uint32 phaseB = ((posLoB >> (32 - SINC8_PHASES_BITS)) & SINC8_MASK) * SINC8_WIDTH;
				const typename Traits::output_t *lutB = sincB + phaseB;
				typename Traits::output_t outB =
					  lutB[0] * Traits::Convert(inBuffer[i - 3 * Traits::numChannelsIn])
					+ lutB[1] * Traits::Convert(inBuffer[i - 2 * Traits::numChannelsIn])
					+ lutB[2] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
					+ lutB[3] * Traits::Convert(inBuffer[i])
					+ lutB[4] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
					+ lutB[5] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn])
					+ lutB[6] * Traits::Convert(inBuffer[i + 3 * Traits::numChannelsIn])
					+ lutB[7] * Traits::Convert(inBuffer[i + 4 * Traits::numChannelsIn]);
				out = out + blendFactor * (outB - out);
			}

			outSample[i] = out;
		}
	}
};


//////////////////////////////////////////////////////////////////////////
// Mixing templates (add sample to stereo mix)

template<class Traits>
struct NoRamp
{
	typename Traits::output_t lVol, rVol;

	MPT_FORCEINLINE NoRamp(const ModChannel &chn)
	{
		lVol = static_cast<typename Traits::output_t>(chn.leftVol) * (1.0 / 4096.0);
		rVol = static_cast<typename Traits::output_t>(chn.rightVol) * (1.0 / 4096.0);
	}
};


struct Ramp
{
	ModChannel &channel;
	double startLVol, startRVol;  // Volume at ramp position 0
	double endLVol, endRVol;      // Target volume (newLeftVol/newRightVol)
	uint32 rampPos;               // Current position in the ramp
	uint32 totalLen;              // Total ramp length (position + remaining)

	MPT_FORCEINLINE Ramp(ModChannel &chn)
		: channel{chn}
		, rampPos{chn.nRampPosition}
		, totalLen{chn.nRampPosition + chn.nRampLength}
	{
		endLVol = chn.newLeftVol;
		endRVol = chn.newRightVol;
		// Recover start volume: newVol - linearDelta * totalLen = originalStartVol
		if(totalLen > 0)
		{
			startLVol = endLVol - chn.leftRamp * static_cast<double>(totalLen);
			startRVol = endRVol - chn.rightRamp * static_cast<double>(totalLen);
		} else
		{
			startLVol = endLVol;
			startRVol = endRVol;
		}
	}

	MPT_FORCEINLINE ~Ramp()
	{
		// Compute current volume at rampPos using smoothstep
		double vol_l, vol_r;
		if(totalLen > 0)
		{
			double t = static_cast<double>(rampPos) / static_cast<double>(totalLen);
			if(t > 1.0) t = 1.0;
			double blend = t * t * (3.0 - 2.0 * t);  // Hermite smoothstep
			vol_l = startLVol + blend * (endLVol - startLVol);
			vol_r = startRVol + blend * (endRVol - startRVol);
		} else
		{
			vol_l = endLVol;
			vol_r = endRVol;
		}
		channel.rampLeftVol = vol_l;
		channel.rampRightVol = vol_r;
		channel.leftVol = vol_l;
		channel.rightVol = vol_r;
	}
};


// Legacy optimization: If chn.nLeftVol == chn.nRightVol, save one multiplication instruction
template<class Traits>
struct MixMonoFastNoRamp : public NoRamp<Traits>
{
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &chn, typename Traits::output_t * const outBuffer)
	{
		typename Traits::output_t vol = outSample[0] * this->lVol;
		for(int i = 0; i < Traits::numChannelsOut; i++)
		{
			outBuffer[i] += vol;
		}
	}
};


template<class Traits>
struct MixMonoNoRamp : public NoRamp<Traits>
{
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &, typename Traits::output_t * const outBuffer)
	{
		outBuffer[0] += outSample[0] * this->lVol;
		outBuffer[1] += outSample[0] * this->rVol;
	}
};


template<class Traits>
struct MixMonoRamp : public Ramp
{
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &, typename Traits::output_t * const outBuffer)
	{
		rampPos++;
		double t = static_cast<double>(rampPos) / static_cast<double>(std::max(totalLen, 1u));
		if(t > 1.0) t = 1.0;
		double blend = t * t * (3.0 - 2.0 * t);  // Hermite smoothstep
		double lVol = (startLVol + blend * (endLVol - startLVol)) * (1.0 / 4096.0);
		double rVol = (startRVol + blend * (endRVol - startRVol)) * (1.0 / 4096.0);
		outBuffer[0] += outSample[0] * lVol;
		outBuffer[1] += outSample[0] * rVol;
	}
};


template<class Traits>
struct MixStereoNoRamp : public NoRamp<Traits>
{
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &, typename Traits::output_t * const outBuffer)
	{
		outBuffer[0] += outSample[0] * this->lVol;
		outBuffer[1] += outSample[1] * this->rVol;
	}
};


template<class Traits>
struct MixStereoRamp : public Ramp
{
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &, typename Traits::output_t * const outBuffer)
	{
		rampPos++;
		double t = static_cast<double>(rampPos) / static_cast<double>(std::max(totalLen, 1u));
		if(t > 1.0) t = 1.0;
		double blend = t * t * (3.0 - 2.0 * t);  // Hermite smoothstep
		double lVol = (startLVol + blend * (endLVol - startLVol)) * (1.0 / 4096.0);
		double rVol = (startRVol + blend * (endRVol - startRVol)) * (1.0 / 4096.0);
		outBuffer[0] += outSample[0] * lVol;
		outBuffer[1] += outSample[1] * rVol;
	}
};


//////////////////////////////////////////////////////////////////////////
// Filter templates


template<class Traits>
struct NoFilter
{
	MPT_FORCEINLINE NoFilter(const ModChannel &) { }

	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &, const ModChannel &) { }
};


// Resonant filter — always 4-pole (IT resonance stage + Butterworth post-filter)
template<class Traits>
struct ResonantFilter
{
	ModChannel &channel;
	// Stage 1 (IT resonance) history
	typename Traits::output_t fy[Traits::numChannelsIn][2];
	// Stage 2 (Butterworth post-filter) history
	typename Traits::output_t fy2[Traits::numChannelsIn][2];

	MPT_FORCEINLINE ResonantFilter(ModChannel &chn)
		: channel{chn}
	{
		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			fy[i][0] = chn.nFilter_Y[i][0];
			fy[i][1] = chn.nFilter_Y[i][1];
			fy2[i][0] = chn.nFilter_Y2[i][0];
			fy2[i][1] = chn.nFilter_Y2[i][1];
		}
	}

	MPT_FORCEINLINE ~ResonantFilter()
	{
		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			channel.nFilter_Y[i][0] = fy[i][0];
			channel.nFilter_Y[i][1] = fy[i][1];
			channel.nFilter_Y2[i][0] = fy2[i][0];
			channel.nFilter_Y2[i][1] = fy2[i][1];
		}
	}

	// Filter values are clipped to double the input range
#define ClipFilter(x) Clamp(x, static_cast<typename Traits::output_t>(-2.0f), static_cast<typename Traits::output_t>(2.0f))

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const ModChannel &chn)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			// Stage 1: IT resonance filter (preserves original character)
			typename Traits::output_t val = outSample[i] * chn.nFilter_A0 + ClipFilter(fy[i][0]) * chn.nFilter_B0 + ClipFilter(fy[i][1]) * chn.nFilter_B1;
			fy[i][1] = fy[i][0];
			fy[i][0] = val - (outSample[i] * chn.nFilter_HP);

			// Stage 2: Butterworth post-filter (steepens rolloff to 24 dB/oct)
			typename Traits::output_t val2 = val * chn.nFilter2_A0 + ClipFilter(fy2[i][0]) * chn.nFilter2_B0 + ClipFilter(fy2[i][1]) * chn.nFilter2_B1;
			fy2[i][1] = fy2[i][0];
			fy2[i][0] = val2;

			outSample[i] = val2;
		}
	}

#undef ClipFilter
};


OPENMPT_NAMESPACE_END
