// MLX 0.32.3 GPU FFT (fft.metal / fft.h ported verbatim from the vendored tree) specialised
// for the front end's real FFT of one 400-sample frame: n = 400 = 2 * 5 * 5 * 8,
// elems_per_thread 5, 80 threads per FFT, two RFFTs per threadgroup (tg memory 512).
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/fft.h"

instantiate_kernel("enc_fft_400", fft, 512, float, float2)
