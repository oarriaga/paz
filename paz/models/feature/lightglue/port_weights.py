from paz.models.feature.lightglue.model import LighterGlueModel


def port_weights(torch_path):
    import torch

    state = matcher_state(torch.load(torch_path, map_location="cpu"))
    state = {key: value.numpy() for key, value in state.items()}
    model = LighterGlueModel(weights=None)
    set_weights(model, state)
    return model


def matcher_state(state):
    return {key[len("matcher."):]: value for key, value in state.items()
            if key.startswith("matcher.")}


def set_weights(model, state):
    dense(model.input_proj, state, "input_proj")
    model.encoding.projection.set_weights([state["posenc.Wr.weight"].T])
    for index, block in enumerate(model.self_blocks):
        prefix = f"transformers.{index}.self_attn"
        dense(block.qkv, state, f"{prefix}.Wqkv")
        dense(block.out_proj, state, f"{prefix}.out_proj")
        feed_forward(block.ffn, state, f"{prefix}.ffn")
    for index, block in enumerate(model.cross_blocks):
        prefix = f"transformers.{index}.cross_attn"
        dense(block.to_qk, state, f"{prefix}.to_qk")
        dense(block.to_v, state, f"{prefix}.to_v")
        dense(block.out_proj, state, f"{prefix}.to_out")
        feed_forward(block.ffn, state, f"{prefix}.ffn")
    dense(model.assignment.final_proj, state, "log_assignment.5.final_proj")
    dense(model.assignment.matchability, state, "log_assignment.5.matchability")


def feed_forward(ffn, state, prefix):
    dense(ffn.expand, state, f"{prefix}.0")
    ffn.norm.set_weights([state[f"{prefix}.1.weight"],
                          state[f"{prefix}.1.bias"]])
    dense(ffn.project, state, f"{prefix}.3")


def dense(layer, state, prefix):
    layer.set_weights([state[f"{prefix}.weight"].T, state[f"{prefix}.bias"]])


if __name__ == "__main__":
    import sys

    model = port_weights(sys.argv[1])
    model.save_weights(sys.argv[2])
    print("saved", sys.argv[2])
