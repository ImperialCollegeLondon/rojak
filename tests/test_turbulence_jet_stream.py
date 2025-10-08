import numpy as np

from rojak.turbulence.jet_stream import get_peak_mask


class TestLocalPeakMask:
    def test_trivial_case(self) -> None:
        # Modified from: https://github.com/scikit-image/scikit-image/blob/959c3b8500cb212dabf3e0ed594a8169d44a113a/tests/skimage/feature/test_peak.py#L16
        trivial = np.zeros((25, 25))
        peak_mask = get_peak_mask(trivial, 0)
        assert not np.any(peak_mask)

    def test_constant_value(self) -> None:
        # Modified from: https://github.com/scikit-image/scikit-image/blob/959c3b8500cb212dabf3e0ed594a8169d44a113a/tests/skimage/feature/test_peak.py#L52
        image = np.full((20, 20), 128, dtype=np.uint8)
        peak_mask = get_peak_mask(image, 0)
        assert not np.any(peak_mask)

    def test_flat_peak(self) -> None:
        # Modified from: https://github.com/scikit-image/scikit-image/blob/959c3b8500cb212dabf3e0ed594a8169d44a113a/tests/skimage/feature/test_peak.py#L57
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1:3, 1:3] = 10
        peak_mask = get_peak_mask(image, 0)
        expected = np.zeros((5, 5), dtype=np.bool_)
        expected[1:3, 1:3] = True
        np.testing.assert_array_equal(peak_mask, expected)

    def test_threshold(self) -> None:
        # Modified from: https://github.com/scikit-image/scikit-image/blob/959c3b8500cb212dabf3e0ed594a8169d44a113a/tests/skimage/feature/test_peak.py#L44
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1, 1] = 10
        image[3, 3] = 20
        peak_mask = get_peak_mask(image, 10)
        expected = np.zeros((5, 5), dtype=np.bool_)
        expected[3, 3] = True
        np.testing.assert_array_equal(peak_mask, expected)

    def test_one_point(self) -> None:
        # Modified from: https://github.com/scikit-image/scikit-image/blob/959c3b8500cb212dabf3e0ed594a8169d44a113a/tests/skimage/feature/test_peak.py#L258
        image = np.zeros((10, 20))
        image[5, 5] = 1
        peak_mask = get_peak_mask(image, 0)
        expected = np.zeros((10, 20), dtype=np.bool_)
        expected[5, 5] = True
        np.testing.assert_array_equal(peak_mask, expected)
