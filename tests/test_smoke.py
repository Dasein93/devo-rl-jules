import torch
from train.ppo import ActorCritic

def test_model_creation():
    """
    Tests that the ActorCritic model can be created.
    """
    num_inputs = 10
    num_actions = 5
    model = ActorCritic(num_inputs, num_actions)
    assert isinstance(model, ActorCritic)

def test_model_forward_pass():
    """
    Tests that a forward pass can be executed without errors.
    """
    num_inputs = 10
    num_actions = 5
    model = ActorCritic(num_inputs, num_actions)

    # Create a dummy input tensor
    batch_size = 4
    dummy_input = torch.randn(batch_size, num_inputs)

    # Test get_value
    value = model.get_value(dummy_input)
    assert value.shape == (batch_size, 1)

    # Test get_action_and_value
    action, log_prob, entropy, value = model.get_action_and_value(dummy_input)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
