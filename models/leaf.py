#https://github.com/denfed/leaf-audio-pytorch
import math
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

# Melfilters.
def mel_to_hertz(mel_values):
    """Converts frequencies in `mel_values` from the mel scale to linear scale."""
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        np.exp(np.array(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0)

def hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (np.array(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ))

def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=16000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0):
    """Returns a matrix to warp linear scale spectrograms to the mel scale.
    Adapted from tf.contrib.signal.linear_to_mel_weight_matrix with a minimum
    band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
    we compute the matrix at float64 precision and then cast to `dtype`
    at the end. This function can be constant folded by graph optimization
    since there are no Tensor inputs.
    Args:
      num_mel_bins: Int, number of output frequency dimensions.
      num_spectrogram_bins: Int, number of input frequency dimensions.
      sample_rate: Int, sample rate of the audio.
      lower_edge_hertz: Float, lowest frequency to consider.
      upper_edge_hertz: Float, highest frequency to consider.
    Returns:
      Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].
    Raises:
      ValueError: Input argument in the wrong range.
    """
    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
    if num_spectrogram_bins <= 0:
        raise ValueError(
            'num_spectrogram_bins must be positive. Got: %s' % num_spectrogram_bins)
    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                         (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                         'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                         % (upper_edge_hertz, sample_rate))

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:, np.newaxis]
    # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2)

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]

    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    for i in range(0, num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
        if upper_hz - lower_hz < freq_th:
            rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
            dm = _MEL_HIGH_FREQUENCY_Q * np.log(rhs + np.sqrt(1.0 + rhs**2))
            lower_edge_mel[i] = center_mel[i] - dm
            upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[np.newaxis, :]
    center_hz = mel_to_hertz(center_mel)[np.newaxis, :]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[np.newaxis, :]

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (linear_frequencies - lower_edge_hz) / (
        center_hz - lower_edge_hz)
    upper_slopes = (upper_edge_hz - linear_frequencies) / (
        upper_edge_hz - center_hz)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]],
                                'constant')
    return mel_weights_matrix

class Gabor:
    """This class creates gabor filters designed to match mel-filterbanks.

      Attributes:
        n_filters: number of filters
        min_freq: minimum frequency spanned by the filters
        max_freq: maximum frequency spanned by the filters
        sample_rate: samplerate (samples/s)
        window_len: window length in samples
        n_fft: number of frequency bins to compute mel-filters
        normalize_energy: boolean, True means that all filters have the same energy,
          False means that the higher the center frequency of a filter, the higher
          its energy
      """

    def __init__(self,
                 n_filters: int = 40,
                 min_freq: float = 0.,
                 max_freq: float = 8000.,
                 sample_rate: int = 16000,
                 window_len: int = 401,
                 n_fft: int = 512,
                 normalize_energy: bool = False):
        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy

    @property
    def gabor_params_from_mels(self):
        """Retrieves center frequencies and standard deviations of gabor filters."""
        coeff = np.sqrt(2. * np.log(2.)) * self.n_fft
        sqrt_filters = torch.sqrt(self.mel_filters)
        center_frequencies = torch.argmax(sqrt_filters, dim=1).type(torch.float32)
        peaks, indices = torch.max(sqrt_filters, dim=1)
        half_magnitudes = torch.div(peaks, 2.)
        fwhms = torch.sum((sqrt_filters >= half_magnitudes.unsqueeze(1)).type(torch.float32), dim=1)
        return torch.stack(
            [center_frequencies * 2 * np.pi / self.n_fft, coeff / (np.pi * fwhms)],
            dim=1)

    def _mel_filters_areas(self, filters):
        """Area under each mel-filter."""
        peaks, indices = torch.max(filters, dim=1)
        return peaks * (torch.sum((filters > 0).type(torch.float32), dim=1) + 2) * np.pi / self.n_fft

    @property
    def mel_filters(self):
        """Creates a bank of mel-filters."""
        # build mel filter matrix
        mel_filters = linear_to_mel_weight_matrix(
            num_mel_bins=self.n_filters,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.min_freq,
            upper_edge_hertz=self.max_freq)
        mel_filters = np.transpose(mel_filters)
        if self.normalize_energy:
            mel_filters = mel_filters / self._mel_filters_areas(torch.from_numpy(mel_filters)).unsqueeze(1)
            return mel_filters
        return torch.from_numpy(mel_filters)


# Initializer classes for each layer of the learnable frontend.
def PreempInit(tensor: Tensor, alpha: float = 0.97) -> Tensor:
    """Pytorch initializer for the pre-emphasis.

    Modifies conv weight Tensor to initialize the pre-emphasis layer of a Leaf instance.

    Attributes:
        alpha: parameter that controls how much high frequencies are emphasized by
        the following formula output[n] = input[n] - alpha*input[n-1] with 0 <
        alpha < 1 (higher alpha boosts high frequencies)
    """

    shape = tensor.shape
    assert shape == (1, 1, 2), f"Cannot initialize preemp layer of size {shape}"

    with torch.no_grad():
        tensor[0, 0, 0] = -alpha
        tensor[0, 0, 1] = 1

        return tensor

def GaborInit(tensor: Tensor, **kwargs) -> Tensor:
    kwargs.pop('n_filters', None)

    shape = tensor.shape

    n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
    window_len = 401 if len(shape) == 2 else shape[0]
    gabor_filters = Gabor(
        n_filters=n_filters, window_len=window_len, **kwargs)
    if len(shape) == 2:
        with torch.no_grad():
            tensor = gabor_filters.gabor_params_from_mels
            return tensor
    else:
        # TODO: FINISH
        pass

def ConstantInit(tensor: Tensor) -> Tensor:
    tensor[:, :, :, :] = 0.4
    return tensor

# Impulse responses
def gabor_impulse_response(t: Tensor, center: Tensor, fwhm: Tensor) -> Tensor:
    """Computes the gabor impulse response."""
    denominator = 1.0 / (np.sqrt(2.0 * math.pi) * fwhm)
    gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm**2), -t**2, dims=0))  # TODO: validate the dims here
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(
        1j * torch.tensordot(center_frequency_complex, t_complex, dims=0))
    #continue down
    # denominator = tf.cast(denominator, dtype=tf.complex64)[:, tf.newaxis]
    denominator = denominator.type(torch.complex64).unsqueeze(1)  # this should be the above line
    gaussian = gaussian.type(torch.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401, t_tensor=None) -> Tensor:
    """Computes the gabor filters from its parameters for a given size.

    Args:
    kernel: Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.

    Returns:
    A Tensor<float>[filters, size].
    """
    return gabor_impulse_response(
        t_tensor,
        center=kernel[:, 0], fwhm=kernel[:, 1])


def gaussian_lowpass(sigma: Tensor, filter_size: int, t_tensor: Tensor):
    """Generates gaussian windows centered in zero, of std sigma.

    Args:
    sigma: tf.Tensor<float>[1, 1, C, 1] for C filters.
    filter_size: length of the filter.

    Returns:
    A tf.Tensor<float>[1, filter_size, C, 1].
    """

    sigma = torch.clamp(sigma, (2. / filter_size), 0.5)
    t = t_tensor.view(1, filter_size, 1, 1)
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return torch.exp(-0.5 * (numerator / denominator)**2)

# Convolution
class GaborConstraint(nn.Module):
    """Constraint mu and sigma, in radians.

    Mu is constrained in [0,pi], sigma s.t full-width at half-maximum of the
    gaussian response is in [1,pi/2]. The full-width at half maximum of the
    Gaussian response is 2*sqrt(2*log(2))/sigma . See Section 2.2 of
    https://arxiv.org/pdf/1711.01161.pdf for more details.
    """

    def __init__(self, kernel_size):
        """Initialize kernel size.

        Args:
        kernel_size: the length of the filter, in samples.
        """
        super(GaborConstraint, self).__init__()
        self._kernel_size = kernel_size

    def forward(self, kernel):
        mu_lower = 0.
        mu_upper = math.pi
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi
        clipped_mu = torch.clamp(kernel[:, 0], mu_lower, mu_upper)
        clipped_sigma = torch.clamp(kernel[:, 1], sigma_lower, sigma_upper)
        return torch.stack([clipped_mu, clipped_sigma], dim=1)


class GaborConv1D(nn.Module):
    """Implements a convolution with filters defined as complex Gabor wavelets.

    These filters are parametrized only by their center frequency and
    the full-width at half maximum of their frequency response.
    Thus, for n filters, there are 2*n parameters to learn.
    """

    def __init__(self, filters, kernel_size, strides, padding, use_bias,
                 input_shape, kernel_initializer, kernel_regularizer, name,
                 trainable, sort_filters=False):
        super(GaborConv1D, self).__init__()
        self._filters = filters // 2
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._sort_filters = sort_filters

        initialized_kernel = kernel_initializer(torch.zeros(self._filters, 2), sample_rate=16000, min_freq=60.0, max_freq=7800.0)
        self._kernel = nn.Parameter(initialized_kernel, requires_grad=trainable)
        # TODO: implement kernel regularizer here
        self._kernel_constraint = GaborConstraint(self._kernel_size)
        if self._use_bias:
            self._bias = nn.Parameter(torch.zeros(self.filters * 2,), requires_grad=trainable)  # TODO: validate that requires grad is the same as trainable

        # Register an initialization tensor here for creating the gabor impulse response to automatically handle cpu/gpu
        # device selection.
        self.register_buffer("gabor_filter_init_t",
                             torch.arange(-(self._kernel_size // 2), (self._kernel_size + 1) // 2, dtype=torch.float32))

    def forward(self, x):
        kernel = self._kernel_constraint(self._kernel)
        if self._sort_filters:
            # TODO: validate this
            filter_order = torch.argsort(kernel[:, 0])
            kernel = torch.gather(kernel, dim=0, index=filter_order)

        filters = gabor_filters(kernel, self._kernel_size, self.gabor_filter_init_t)
        real_filters = torch.real(filters)
        img_filters = torch.imag(filters)
        stacked_filters = torch.stack([real_filters, img_filters], dim=1)
        stacked_filters = stacked_filters.view(2 * self._filters, self._kernel_size)
        stacked_filters = stacked_filters.unsqueeze(1)
        output = F.conv1d(x, stacked_filters,
                          bias=self._bias if self._use_bias else None, stride=self._strides,
                          padding=self._padding)
        return output

# pooling
class GaussianLowpass(nn.Module):
    """Depthwise pooling (each input filter has its own pooling filter).

    Pooling filters are parametrized as zero-mean Gaussians, with learnable
    std. They can be initialized with tf.keras.initializers.Constant(0.4)
    to approximate a Hanning window.
    We rely on depthwise_conv2d as there is no depthwise_conv1d in Keras so far.
    """

    def __init__(
            self,
            kernel_size,
            strides=1,
            filter_size=40,
            padding=0,
            use_bias=True,
            kernel_initializer=nn.init.xavier_uniform_,
            kernel_regularizer=None,
            trainable=False):

        super(GaussianLowpass, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.filter_size = filter_size
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.trainable = trainable

        initialized_kernel = self.kernel_initializer(torch.zeros(1, 1, self.filter_size, 1).type(torch.float32))
        self._kernel = nn.Parameter(initialized_kernel, requires_grad=self.trainable)

        # Register an initialization tensor here for creating the gaussian lowpass impulse response to automatically
        # handle cpu/gpu device selection.
        self.register_buffer("gaussian_lowpass_init_t", torch.arange(0, self.kernel_size, dtype=torch.float32))

    def forward(self, x):
        kernel = gaussian_lowpass(self._kernel, self.kernel_size, self.gaussian_lowpass_init_t)
        kernel = kernel.squeeze(3)
        kernel = kernel.permute(2, 0, 1)

        outputs = F.conv1d(x, kernel, stride=self.strides, groups=self.filter_size, padding=self.padding)
        return outputs

# postprocessing
def log_compression(inputs: Tensor, epsilon=1e-6) -> Tensor:
    "Log compression"
    return torch.log(inputs + epsilon)


class ExponentialMovingAverage(nn.Module):
    """Computes of an exponential moving average of an sequential input."""

    def __init__(self,
                 coeff_init: float,
                 per_channel: bool = False, trainable: bool = False):
        """Initializes the ExponentialMovingAverage.

        Args:
          coeff_init: the value of the initial coeff.
          per_channel: whether the smoothing should be different per channel.
          trainable: whether the smoothing should be trained or not.
        """
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self._trainable = trainable

    def build(self, num_channels):
        ema_tensor = torch.zeros((num_channels)).type(torch.float32) if self._per_channel else torch.zeros((1))
        ema_tensor[:] = self._coeff_init
        self._weights = nn.Parameter(ema_tensor, requires_grad=self._trainable)

    def forward(self, x, initial_state):
        w = torch.clamp(self._weights, 0.0, 1.0)

        def func(a, y):
            return w * y + (1.0 - w) * a

        def scan(func, x):
            res = []
            res.append(x[0].unsqueeze(0))
            a_ = x[0].clone()

            for i in range(1, len(x)):
                res.append(func(a_, x[i]).unsqueeze(0))
                a_ = func(a_, x[i])

            return torch.cat(res)

        res = scan(func, x.permute(2, 0, 1))
        return res.permute(1, 0, 2)
        # for i in x.permute(2,0,1):
        #     print(i)

    # def call(self, inputs: Tensor, initial_state: Tensor):
    #     """Inputs is of shape [batch, seq_length, num_filters]."""
    #     w = torch.clamp(self._weights, 0.0, 1.0)
    #     result = tf.scan(lambda a, x: w * x + (1.0 - w) * a,
    #                      tf.transpose(inputs, (1, 0, 2)),
    #                      initializer=initial_state)
    #     return tf.transpose(result, (1, 0, 2))


class PCENLayer(nn.Module):
    """Per-Channel Energy Normalization.

    This applies a fixed or learnable normalization by an exponential moving
    average smoother, and a compression.
    See https://arxiv.org/abs/1607.05666 for more details.
    """

    def __init__(self,
                 alpha: float = 0.96,
                 smooth_coef: float = 0.04,
                 delta: float = 2.0,
                 root: float = 2.0,
                 floor: float = 1e-6,
                 trainable: bool = False,
                 learn_smooth_coef: bool = False,
                 per_channel_smooth_coef: bool = False,
                 name='PCEN'
                 ):
        """PCEN constructor.

        Args:
          alpha: float, exponent of EMA smoother
          smooth_coef: float, smoothing coefficient of EMA
          delta: float, bias added before compression
          root: float, one over exponent applied for compression (r in the paper)
          floor: float, offset added to EMA smoother
          trainable: bool, False means fixed_pcen, True is trainable_pcen
          learn_smooth_coef: bool, True means we also learn the smoothing
            coefficient
          per_channel_smooth_coef: bool, True means each channel has its own smooth
            coefficient
          name: str, name of the layer
        """
        super(PCENLayer, self).__init__()
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable
        self._learn_smooth_coef = learn_smooth_coef
        self._per_channel_smooth_coef = per_channel_smooth_coef

    def build(self, num_channels):
        alpha_tensor = torch.zeros((num_channels)).type(torch.float32)
        alpha_tensor[:] = self._alpha_init
        self.alpha = nn.Parameter(alpha_tensor, requires_grad=self._trainable)

        delta_tensor = torch.zeros((num_channels)).type(torch.float32)
        delta_tensor[:] = self._delta_init
        self.delta = nn.Parameter(delta_tensor, requires_grad=self._trainable)

        root_tensor = torch.zeros((num_channels)).type(torch.float32)
        root_tensor[:] = self._root_init
        self.root = nn.Parameter(root_tensor, requires_grad=self._trainable)

        if self._learn_smooth_coef:
            self.ema = ExponentialMovingAverage(
                coeff_init=self._smooth_coef,
                per_channel=self._per_channel_smooth_coef,
                trainable=True)
            self.ema.build(num_channels)
        else:
            # TODO: implement simple RNN here
            pass

    def forward(self, x):
        alpha = torch.minimum(self.alpha, torch.ones_like(self.alpha))
        root = torch.maximum(self.root, torch.ones_like(self.root))

        ema_smoother = self.ema(x, x[:, :, 0])
        one_over_root = 1. / root
        output = (x.permute(0, 2, 1) / (self._floor + ema_smoother) ** alpha + self.delta)\
            ** one_over_root - self.delta ** one_over_root
        return output.permute(0, 2, 1)

  # def call(self, inputs):
  #   alpha = tf.math.minimum(self.alpha, 1.0)
  #   root = tf.math.maximum(self.root, 1.0)
  #   ema_smoother = self.ema(inputs, initial_state=tf.gather(inputs, 0, axis=1))
  #   one_over_root = 1. / root
  #   output = ((inputs / (self._floor + ema_smoother)**alpha + self.delta)
  #             **one_over_root - self.delta**one_over_root)
  #   return output


# TODO: implement weight freezing if learn_filters is False

class SquaredModulus(nn.Module):
    """Squared modulus layer.

    Returns a keras layer that implements a squared modulus operator.
    To implement the squared modulus of C complex-valued channels, the expected
    input dimension is N*1*W*(2*C) where channels role alternates between
    real and imaginary part.
    The way the squared modulus is computed is real ** 2 + imag ** 2 as follows:
    - squared operator on real and imag
    - average pooling to compute (real ** 2 + imag ** 2) / 2
    - multiply by 2

    Attributes:
        pool: average-pooling function over the channel dimensions
    """

    def __init__(self):
        super(SquaredModulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        output = 2 * self._pool(x**2)
        return torch.transpose(output, 2, 1)


class Leaf(nn.Module):
    """Pytorch layer that implements time-domain filterbanks.

    Creates a LEAF frontend, a learnable front-end that takes an audio
    waveform as input and outputs a learnable spectral representation. This layer
    can be initialized to replicate the computation of standard mel-filterbanks.
    A detailed technical description is presented in Section 3 of
    https://arxiv.org/abs/2101.08596.

    """

    def __init__(
        self,
        learn_pooling: bool = True,
        learn_filters: bool = True,
        conv1d_cls=GaborConv1D,
        activation=SquaredModulus(),
        pooling_cls=GaussianLowpass,
        n_filters: int = 40,
        sample_rate: int = 16000,
        window_len: float = 25.,
        window_stride: float = 10.,
        # compression_fn=None,
        compression_fn=PCENLayer(
            alpha=0.96,
            smooth_coef=0.04,
            delta=2.0,
            floor=1e-12,
            trainable=True,
            learn_smooth_coef=True,
            per_channel_smooth_coef=True),
        preemp: bool = False,
        preemp_init=PreempInit,
        complex_conv_init=GaborInit,
        pooling_init=ConstantInit,
        # regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
        mean_var_norm: bool = False,
        spec_augment: bool = False,
        name='leaf'
    ):

        super(Leaf, self).__init__()

        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)

        self._preemp = preemp

        if preemp:
            self._preemp_conv = torch.nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                stride=1,
                padding=(2 // 2),
                bias=False,
            )
            preemp_init(self._preemp_conv.weight)

        self._complex_conv = conv1d_cls(
            filters=2 * n_filters,
            kernel_size=window_size,
            strides=1,
            padding=(window_size // 2),  # TODO: validate that this is correct
            use_bias=False,
            input_shape=(None, None, 1),
            kernel_initializer=complex_conv_init,
            # kernel_regularizer=regularizer_fn if learn_filters else None,
            kernel_regularizer=None,
            name='tfbanks_complex_conv',
            trainable=learn_filters)

        self._activation = activation
        self._pooling = pooling_cls(
            kernel_size=window_size,
            strides=window_stride,
            filter_size=n_filters,
            padding=(window_size // 2),
            use_bias=False,
            kernel_initializer=pooling_init,
            # kernel_regularizer=regularizer_fn if learn_pooling else None,
            trainable=learn_pooling)

        self._compress_fn = compression_fn if compression_fn else nn.Identity()
        # Pass number of filters to PCEN layer for on-the-fly building.
        # We do this to avoid adding num_channels as an arg into the class itself to avoid double setting the same arg
        # when instantiating the Leaf class.
        if isinstance(self._compress_fn, PCENLayer):
            self._compress_fn.build(num_channels=n_filters)

    def forward(self, x):
        """Computes the Leaf representation of a batch of waveforms.

        Args:
        x: input audio of shape (batch_size, num_samples) or (batch_size, num_samples, 1).

        Returns:
        Leaf features of shape (batch_size, time_frames, freq_bins).
        """
        outputs = x.unsqueeze(1) if len(x.shape) <= 2 else x  # TODO: validate this
        if self._preemp:
            outputs = self._preemp_conv(x)
            # Pytorch padding trick needed because 'same' doesn't exist and kernel is even.
            # Remove the first value in the feature dim of the tensor to match tf's padding.
            outputs = outputs[:, :, 1:]

        outputs = self._complex_conv(outputs)
        outputs = self._activation(outputs)
        outputs = self._pooling(outputs)
        # As far as I know, torch cannot perform element-wise maximum between a tensor and scalar, here is a workaround.
        output_copy = torch.zeros_like(outputs)
        output_copy[:, :, :] = 1e-5
        outputs = torch.maximum(outputs, output_copy)
        outputs = self._compress_fn(outputs)

        return outputs
