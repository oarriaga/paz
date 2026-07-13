from paz.models.feature.lightglue.model import LighterGlueModel

NUM_LAYERS = 6


def port_weights(torch_path):
    import torch

    state = matcher_state(torch.load(torch_path, map_location="cpu"))
    state = {key: value.numpy() for key, value in state.items()}
    model = LighterGlueModel(weights=None)
    set_weights(model, state)
    return model


def matcher_state(state):
    matcher = {}
    for key, value in state.items():
        if key.startswith("matcher."):
            matcher[key[len("matcher."):]] = value
    return matcher


def set_weights(model, state):
    dense(model, state, "input_projection", "input_proj")
    kernel(model, state, "encoding_projection", "posenc.Wr")
    for index in range(NUM_LAYERS):
        set_self_attention(model, state, index)
        set_cross_attention(model, state, index)
    dense(model, state, "assignment_projection", "log_assignment.5.final_proj")
    dense(model, state, "assignment_matchability",
          "log_assignment.5.matchability")


def set_self_attention(model, state, index):
    name, source = f"self_attention_{index}", f"transformers.{index}.self_attn"
    dense(model, state, f"{name}_qkv", f"{source}.Wqkv")
    dense(model, state, f"{name}_projection", f"{source}.out_proj")
    set_feed_forward(model, state, name, f"{source}.ffn")


def set_cross_attention(model, state, index):
    name = f"cross_attention_{index}"
    source = f"transformers.{index}.cross_attn"
    dense(model, state, f"{name}_query", f"{source}.to_qk")
    dense(model, state, f"{name}_value", f"{source}.to_v")
    dense(model, state, f"{name}_projection", f"{source}.to_out")
    set_feed_forward(model, state, name, f"{source}.ffn")


def set_feed_forward(model, state, name, source):
    dense(model, state, f"{name}_expand", f"{source}.0")
    layer_norm(model, state, f"{name}_norm", f"{source}.1")
    dense(model, state, f"{name}_project", f"{source}.3")


def dense(model, state, name, source):
    weights = [state[f"{source}.weight"].T, state[f"{source}.bias"]]
    model.get_layer(name).set_weights(weights)


def kernel(model, state, name, source):
    model.get_layer(name).set_weights([state[f"{source}.weight"].T])


def layer_norm(model, state, name, source):
    weights = [state[f"{source}.weight"], state[f"{source}.bias"]]
    model.get_layer(name).set_weights(weights)


if __name__ == "__main__":
    import sys

    port_weights(sys.argv[1]).save_weights(sys.argv[2])
    print("saved", sys.argv[2])
