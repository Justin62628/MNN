import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto

# 加载原始模型
model = onnx.load(r"D:/60-fps-Project/Projects/RIFE GUI/models/scdet/sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx")
graph = model.graph

# 1. 找到原始输入和unsqueeze节点
original_input = None
unsqueeze_node = None
next_node_input = None  # 记录unsqueeze节点的下游节点输入

for input in graph.input:
    if input.name == "input":  # 假设原始输入名称为"input"
        original_input = input
        break

for node in graph.node:
    if node.op_type == "Unsqueeze" and node.input[0] == original_input.name:
        unsqueeze_node = node
        # 找到使用unsqueeze输出的节点
        for downstream_node in graph.node:
            if unsqueeze_node.output[0] in downstream_node.input:
                next_node_input = downstream_node.input
        break

# 2. 删除原始输入和unsqueeze节点
if original_input:
    graph.input.remove(original_input)
if unsqueeze_node:
    graph.node.remove(unsqueeze_node)

# 3. 添加两个新的输入
input1 = helper.make_tensor_value_info(
    "input0", 
    TensorProto.FLOAT, 
    [1, 3, 224, 224]
)
input2 = helper.make_tensor_value_info(
    "input1", 
    TensorProto.FLOAT, 
    [1, 3, 224, 224]
)
graph.input.extend([input1, input2])

# 4. 添加concat节点
concat_output_name = "concat_output"
concat_node = helper.make_node(
    "Concat",
    inputs=["input0", "input1"],
    outputs=[concat_output_name],
    axis=1  # 在batch维度上concat
)
graph.node.append(concat_node)

# 5. 更新下游节点的输入
if next_node_input:
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == unsqueeze_node.output[0]:
                node.input[i] = concat_output_name

# 检查并保存模型
# onnx.checker.check_model(model)
onnx.save(model, "scdet/scdet.onnx")