import math

from sampler import top_k_top_p_filtering


def test_top_p_gives_each_token_its_own_probability():
    probs = [0.3, 0.2, 0.5]
    out = top_k_top_p_filtering(probs, top_p=0.8)
    assert out[1] == 0.0
    assert math.isclose(out[2], 0.5 / 0.8)
    assert math.isclose(out[0], 0.3 / 0.8)


def test_top_k_maps_probabilities_to_matching_tokens():
    probs = [0.05, 0.4, 0.1, 0.45]
    out = top_k_top_p_filtering(probs, top_k=2)
    assert out[0] == 0.0 and out[2] == 0.0
    assert math.isclose(out[3], 0.45 / 0.85)
    assert math.isclose(out[1], 0.4 / 0.85)


def test_no_filtering_returns_copy():
    probs = [0.25, 0.75]
    out = top_k_top_p_filtering(probs)
    assert out == probs and out is not probs
