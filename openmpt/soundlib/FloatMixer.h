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
#include "Resampler.h"

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


// 64-tap anisotropic sinc interpolation with AVX2 SIMD acceleration.
// Enhanced β-shear with 2nd-order acceleration tracking.
// See: "Anisotropic Audio Filters" (Quinlight paper, March 2026).
template<class Traits>
struct Aniso64Interpolation
{
	const typename Traits::output_t *sincA;
	const typename Traits::output_t *sincB;
	typename Traits::output_t blendFactor;
	int32 betaPhaseShift;
	bool useAVX512;
	bool useAVX2;
	bool useAVX;

	MPT_FORCEINLINE Aniso64Interpolation(const ModChannel &chn, const CResampler &resampler, unsigned int)
	{
		// Mipmap-style anti-aliasing: select octave-spaced sinc tables via log2(ratio).
		// Trilinear blend between adjacent mip levels — zero aliasing at any ratio.
		//
		//   ratio = increment / unity  (>1 = downsampling, <1 = upsampling)
		//   mipLevel = log2(ratio)     (0.0 = unity, 1.0 = 2x, 5.13 = 35x, etc.)
		//   sincA = mip[floor(level)], sincB = mip[ceil(level)]
		//   blendFactor = frac(level)
		//
		const int64 absInc = std::abs(chn.increment.GetRaw());
		const double ratio = std::max(1.0, static_cast<double>(absInc) / 4294967296.0);
		const double mipLevel = std::log2(ratio);
		int mipA = static_cast<int>(mipLevel);
		if(mipA < 0) mipA = 0;
		if(mipA >= MIP_LEVELS - 1)
		{
			mipA = MIP_LEVELS - 1;
			sincA = resampler.gAniso64Mip[mipA];
			sincB = sincA;
			blendFactor = 0;
		}
		else
		{
			sincA = resampler.gAniso64Mip[mipA];
			sincB = resampler.gAniso64Mip[mipA + 1];
			blendFactor = static_cast<typename Traits::output_t>(mipLevel - mipA);
		}

		// Enhanced anisotropic β-shear: 2nd-order (acceleration-aware)
		// k_beta raised from 0.5 to 0.65 — the 64-tap kernel resolves finer time structure
		// k_beta2 = 0.15 adds acceleration term for rapid pitch bends
		betaPhaseShift = 0;
		if(!chn.prevIncrement.IsZero())
		{
			const int64 dInc = chn.increment.GetRaw() - chn.prevIncrement.GetRaw();
			const int64 d2Inc = dInc - chn.prevDeltaIncrement.GetRaw();

			if(dInc != 0 || d2Inc != 0)
			{
				const double k_beta = resampler.m_Settings.aniso64_k_beta;
				const double k_beta2 = resampler.m_Settings.aniso64_k_beta2;
				constexpr double maxShift = 1073741824.0;              // 0x40000000
				constexpr double blendCeiling = 0.6;                   // Higher ceiling — 64-tap makes blend smoother
				constexpr double blendSensitivity = 1.0 / 0x04000000;

				const double dIncD = static_cast<double>(dInc);
				const double d2IncD = static_cast<double>(d2Inc);
				const double shear = k_beta * dIncD + k_beta2 * d2IncD;
				betaPhaseShift = static_cast<int32>(
					std::clamp(shear, -maxShift, maxShift));

				if(blendFactor == 0)
				{
					sincB = sincA;
					blendFactor = static_cast<typename Traits::output_t>(
						std::min(blendCeiling, std::abs(dIncD) * blendSensitivity * blendCeiling));
				}
			}
		}

		useAVX512 = resampler.m_hasAVX512;
		useAVX2 = resampler.m_hasAVX2;
		useAVX = resampler.m_hasAVX;
	}

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const inBuffer, const uint32 posLo)
	{
		static_assert(static_cast<int>(Traits::numChannelsIn) <= static_cast<int>(Traits::numChannelsOut), "Too many input channels");
		const uint32 phase = ((posLo >> (32 - SINC_PHASES_64_BITS)) & SINC_MASK_64) * SINC_WIDTH_64;
		const typename Traits::output_t *lutA = sincA + phase;

		// SIMD stereo fast path: both channels computed simultaneously via deinterleave
		// Dispatch cascade: AVX-512 → AVX2 → AVX → SSE2 → scalar
		if(std::is_same<typename Traits::input_t, double>::value && Traits::numChannelsIn == 2)
		{
			const double *samples = reinterpret_cast<const double *>(inBuffer) - 31 * 2;
			double outL, outR;
			if(useAVX512)
				aniso64_dot_stereo_avx512(lutA, samples, &outL, &outR);
			else if(useAVX2)
				aniso64_dot_stereo_avx2(lutA, samples, &outL, &outR);
			else if(useAVX)
				aniso64_dot_stereo_avx(lutA, samples, &outL, &outR);
			else
				aniso64_dot_stereo_sse2(lutA, samples, &outL, &outR);

			if(blendFactor != 0)
			{
				const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
				const uint32 phaseB = ((posLoB >> (32 - SINC_PHASES_64_BITS)) & SINC_MASK_64) * SINC_WIDTH_64;
				const typename Traits::output_t *lutB = sincB + phaseB;
				double outBL, outBR;
				if(useAVX512)
					aniso64_dot_stereo_avx512(lutB, samples, &outBL, &outBR);
				else if(useAVX2)
					aniso64_dot_stereo_avx2(lutB, samples, &outBL, &outBR);
				else if(useAVX)
					aniso64_dot_stereo_avx(lutB, samples, &outBL, &outBR);
				else
					aniso64_dot_stereo_sse2(lutB, samples, &outBL, &outBR);
				outL = outL + static_cast<double>(blendFactor) * (outBL - outL);
				outR = outR + static_cast<double>(blendFactor) * (outBR - outR);
			}
			outSample[0] = static_cast<typename Traits::output_t>(outL);
			outSample[1] = static_cast<typename Traits::output_t>(outR);
			return;
		}

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t out;

			// SIMD mono fast path: Float64 mono samples are contiguous doubles — zero conversion
			// Dispatch cascade: AVX-512 → AVX2 → AVX → SSE2 → scalar
			if(std::is_same<typename Traits::input_t, double>::value && Traits::numChannelsIn == 1)
			{
				const double *samples = reinterpret_cast<const double *>(inBuffer + i) - 31;
				if(useAVX512)
					out = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx512(lutA, samples));
				else if(useAVX2)
					out = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx2(lutA, samples));
				else if(useAVX)
					out = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx(lutA, samples));
				else
					out = static_cast<typename Traits::output_t>(aniso64_dot_mono_sse2(lutA, samples));
			}
			else
			{
				// Scalar fallback: non-double formats (int8, int16, float32) and stereo stride
				out = 0;
				for(int t = 0; t < SINC_WIDTH_64; t++)
				{
					out += lutA[t] * Traits::Convert(inBuffer[i + (t - 31) * Traits::numChannelsIn]);
				}
			}

			if(blendFactor != 0)
			{
				// Trilinear blend with β-sheared kernel B
				const uint32 posLoB = static_cast<uint32>(static_cast<int32>(posLo) + betaPhaseShift);
				const uint32 phaseB = ((posLoB >> (32 - SINC_PHASES_64_BITS)) & SINC_MASK_64) * SINC_WIDTH_64;
				const typename Traits::output_t *lutB = sincB + phaseB;

				typename Traits::output_t outB;
				if(std::is_same<typename Traits::input_t, double>::value && Traits::numChannelsIn == 1)
				{
					const double *samples = reinterpret_cast<const double *>(inBuffer + i) - 31;
					if(useAVX512)
						outB = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx512(lutB, samples));
					else if(useAVX2)
						outB = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx2(lutB, samples));
					else if(useAVX)
						outB = static_cast<typename Traits::output_t>(aniso64_dot_mono_avx(lutB, samples));
					else
						outB = static_cast<typename Traits::output_t>(aniso64_dot_mono_sse2(lutB, samples));
				}
				else
				{
					outB = 0;
					for(int t = 0; t < SINC_WIDTH_64; t++)
					{
						outB += lutB[t] * Traits::Convert(inBuffer[i + (t - 31) * Traits::numChannelsIn]);
					}
				}
				out = out + blendFactor * (outB - out);
			}

			outSample[i] = out;
		}
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
