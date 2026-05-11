from ultralytics import YOLO
from argparse import ArgumentParser

import os



def export(input_file: str, output_file: str, half: bool = False, nms: bool = False) -> None:
    # 加载你的 PyTorch 模型，关闭 verbose 日志
    model = YOLO(input_file, verbose=False)
    model.export(
        format="onnx",
        half=half,
        imgsz=640,  # should be square for non-PyTorch val
        dynamic=False,
        nms=nms,
        batch=1,
        device="0",
        verbose=False,
    )

    os.system("mv {} {}".format(input_file.split(".")[0] + ".onnx", output_file))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", help="path to input .pt file")
    parser.add_argument("output_file", help="path to output .onnx file")
    parser.add_argument("--half", action="store_true", help="export half precision")
    parser.add_argument("--nms", action="store_true", help="export nms precision")
    args = parser.parse_args()

    export(args.input_file, args.output_file, args.half, args.nms)

    """
        导出onnx不应使用内置nms，将会造成大量性能损失以及无法调整的阈值
        图像大小应当取max(w, h)
        ultralytics阶段的half理论上不会导致模型精度下降和推理速度下降，但是可以降低模型大小，
        其原理是将权重用fp16类型表示。如果后续trtexec转化时不选择fp16量化，那么推理过程仍然将
        以fp32数据类型计算，而权重参数将会被上采样至fp32，推理速度和精度可能受影响
        trtexec的fp16选项会对模型进行量化，可能导致模型精度下降，但是会大幅度提升推理速度。其原理
        是分析模型结构，将部分精度不敏感层的参数量化为fp16而保留敏感层参数为fp32，然后计算时
        传入数据类型为fp16，而计算本身以混合精度执行。
        
        ultralytics阶段的half导出只是数据类型的改变
        而涉及到性能优化则主要是tensorrt的量化处理
        trtexec \ 
            --verbose \
            --onnx=<your_model.onnx> \ 
            --saveEngine=<your_model.engine>
            <option:> 
            --optShapes=images:1x3x640x640 # 对特定输入形状进行优化。如果输入模型为静态模型，不需要设置该选项
            --fp16 # 开启fp16半精度量化，注意计算过程将以混合精度模式进行
            
        使用Netron工具或者ONNX-runtime查看输入输出名称与形状
    """