import torch

from ema import WeightAverage


def test_average_moves_toward_the_weights_by_one_minus_decay():
    model = torch.nn.Linear(2, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    average = WeightAverage(model, decay=0.75)
    with torch.no_grad():
        model.weight.fill_(4.0)
    average.update(model)
    assert torch.allclose(average.state_dict(model)["weight"], torch.full((1, 2), 1.0))


def test_state_dict_keeps_buffers_and_leaves_the_model_alone():
    model = torch.nn.BatchNorm1d(2)
    average = WeightAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    average.update(model)
    state = average.state_dict(model)
    assert set(state) == set(model.state_dict())
    assert torch.allclose(state["weight"], torch.full((2,), 2.0))
    assert torch.allclose(model.weight, torch.full((2,), 3.0))


def test_layer_count_reads_the_deepest_encoder_layer():
    from train import layer_count

    state = {
        "embeddings.word_embeddings.weight": 0,
        "encoder.layer.0.output.dense.weight": 0,
        "encoder.layer.11.attention.self.query.weight": 0,
    }
    assert layer_count(state) == 12
