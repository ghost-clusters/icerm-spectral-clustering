from .spectral_clustering import similarity_matrix, laplacian_matrix
import numpy as np
import scipy.signal

def wave_clustering(data, k, T_max=1000, c=1.41, metric=None, kernel=None, **kwargs):
    '''
    Args:
        data (np.ndarray): (n,d) numpy array consisting of n d-valued points
        k (integer): desired number of clusters. Must be a power of 2.
        T_max: number of iterations of wave propagations to simulate. A larger value is slower but provides more accurate
            estimation of laplacian eigenvectors.
        c: wave propagation speed. 0 < c < sqrt(2) is required for guaranteed stability.
        metric: one of ["g", "e", "k"] - choose a metric for the data:
            "g" for Gaussian: d(x, y) = exp(-|x-y|^2 / 2 (s^2)). The scale `s` controls standard deviation.
            "e" for Exponential: d(x, y) = exp(-|x-y|/s). The scale `s` is the parameter of the exponential.
            "eps" for Epsilon neighbors: d(x, y) = |x-y| if less than epsilon, 0 otherwise
            "k" for Kernel: use an arbitrary kernel, given by the `kernel` argument.
        kernel: K(x, y, s) -> \R+ - an arbitrary distance function between two points of the same dimension with a given scale.

    Returns:
        A list of (n,) integers of the cluster assignments of each data point.
    '''
    k_ = int(np.log2(k))
    assert k_ == np.log2(k), "k must be a power of 2"

    data_sim = similarity_matrix(data, metric=metric, kernel=kernel, **kwargs)
    n, _ = data_sim.shape
    laplacian, degree = laplacian_matrix(data_sim)
    rw_laplacian = np.linalg.inv(deg) @ laplacian

    wave = np.zeros((n, T_max + 1))

    wave[:, 0] = np.random.uniform(size=n)
    wave[:, 1] = wave[:, 0]
    for i in range(2, T_max + 1):
        wave[:, i] = 2 * wave[:, i-1] - wave[:, i-2] - (c**2) * (rw_laplacian @ wave[:, i-1])

    wave_fft = np.fft.rfft(wave[:, 1:], axis=1)
    wave_cos_coeffs = cos_coeffs(wave_fft)

    freq_peaks = [scipy.signal.find_peaks(np.abs(wave_cos_coeffs[idx, 1:]))[0] for idx in range(n)]
    truncated_freq_peaks = np.stack([f[:k_] for f in freq_peaks])

    cluster_assns = np.zeros((n,))
    for i in range(n):
        for j, freq_peak in enumerate(truncated_freq_peaks[i]):
            if wave_cos_coeffs[i][freq_peak] > 0:
                cluster_assns[i] += 2**j
    return cluster_assns


def gimme_cos(n, k):
    '''
    Return a length n column vector whose components are cosine waves of frequency k/(n-1)
    '''
    return np.cos(2 * np.pi * k * np.arange(n) / n)

def gimme_sin(n, k):
    '''
    Return a length n column vector whose components are cosine waves of frequency k/(n-1)
    '''
    return np.sin(2 * np.pi * k * np.arange(n) / n)

def cos_coeffs(fourier_coeffs):
    '''
    Given a sequence of two-sided fourier coefficients (negative freqs, positive freqs) return the corresponding two-sided cos coeffs
    '''
    return np.real(fourier_coeffs)

def sin_coeffs(fourier_coeffs):
    '''
    Given a sequence of two-sided fourier coefficients (negative freqs, positive freqs) return the corresponding two-sided sin coeffs
    '''
    return -np.imag(fourier_coeffs)

if __name__ == "__main__":
    def assert_fourier_matches_cossin(f):
        # Note: by default, Numpy's fourier transform has no scaling and its inverse transform has 1/n scaling.
        # So all fourier coefficients should be scaled by 1/n prior to inverse.
        n = len(f)
        K = np.fft.fftfreq(n) # Note: numpy is positive frequencies then negative frequencies. If n odd, the positive frequencies get the extra freq
        fourier_coeffs = np.fft.fft(f)
        cos, sin = cos_coeffs(fourier_coeffs), sin_coeffs(fourier_coeffs)
        cos_parts = [alpha_n * np.cos(2*np.pi*omega_n*np.arange(n)) for alpha_n, omega_n in zip(cos, K)]
        sin_parts = [alpha_n * np.sin(2*np.pi*omega_n*np.arange(n)) for alpha_n, omega_n in zip(sin, K)]
        f_cossin = (np.sum(cos_parts, axis=0) + np.sum(sin_parts, axis=0)) / n
        assert np.allclose(f, f_cossin)

    assert_fourier_matches_cossin(gimme_cos(20, 3) + gimme_cos(20, 5) + gimme_sin(20, 8))
    assert_fourier_matches_cossin(np.random.uniform((50,)))

    data = np.arange(200).reshape((-1, 1))

    def kernel(x, y, s):
        if((x == 100 and y == 101) or (x == 101 and y == 100)):
            return 0.01
        elif (x == y + 1 or y == x + 1):
                return 1
        else:
            return 0
    print(wave_clustering(data, T_max=1000, c=1.41, k=2, metric="k", kernel=kernel))