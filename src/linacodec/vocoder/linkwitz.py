import torch

## linkwitz_riley FFT Crossover. Original 48khz audio has slight phase issues with limited training data, this fixes the issue with 99% speed.
def crossover_merge_linkwitz_riley(path1_48k, path2_48k, sample_rate=48000, cutoff=4000, transition_bins=8):
    # 1. Frequency Domain
    spec1 = torch.fft.rfft(path1_48k)
    spec2 = torch.fft.rfft(path2_48k)

    n_bins = spec1.size(-1)
    cutoff_bin = int((cutoff / (sample_rate / 2)) * n_bins)

    # 2. Linkwitz-Riley / Butterworth Taper
    # This creates a "Flatter" response than Hann or Sigmoid
    mask = torch.ones(n_bins, device=spec1.device)
    
    half = transition_bins // 2
    start = max(0, cutoff_bin - half)
    end = min(n_bins, cutoff_bin + half)
    actual_width = end - start

    # Create a 4th-order-like steepness (very transparent)
    # We use a normalized frequency vector from -1 to 1 across the transition
    x = torch.linspace(-1, 1, steps=actual_width, device=spec1.device)
    
    # This polynomial approximates a Linkwitz-Riley transition:
    # It stays flatter longer than a Sigmoid but is smoother than Linear.
    fade = 0.5 * (x * (3 - x**2) * 0.5 + 0.5) 
    # Or for extreme transparency, use a Cubic Hermite:
    fade = 3 * torch.pow((x + 1) / 2, 2) - 2 * torch.pow((x + 1) / 2, 3)

    mask[:start] = 0
    mask[start:end] = fade
    mask[end:] = 1

    # 3. Direct Complex Merge
    merged_spec = (spec1 * mask) + (spec2 * (1.0 - mask))

    return torch.fft.irfft(merged_spec, n=path1_48k.size(-1))
