
import os
import paddle2onnx
from paddleocr import TableCellsDetection
import numpy as np


def test_predict():
    model = TableCellsDetection(model_name="RT-DETR-L_wireless_table_cell_det")
    output = model.predict("demo_table_wo_line.png", threshold=0.3, batch_size=1)
    for res in output:
        res.print(json_format=False)
        res.save_to_img("./output/")
        res.save_to_json("./output/res.json")

def convert_to_onnx():
    """
    Convert PaddlePaddle model to ONNX format using paddle2onnx
    paddle2onnx --model_dir /home/tatoao/.paddlex/official_models/RT-DETR-L_wireless_table_cell_det \
                --model_filename inference.json \
                --params_filename inference.pdiparams \
                --save_file table_cell_det.onnx \
                --opset_version 11 \
                --enable_onnx_checker True
    """
    pass
    
# python scripts/paddle_to_onnx.py
if __name__ == "__main__":
    # test_predict()
    convert_to_onnx()