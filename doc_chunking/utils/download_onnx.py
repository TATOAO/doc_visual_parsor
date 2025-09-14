import os
from modelscope import snapshot_download

def download_onnx(cache_dir: str = './model_parameters/layout_detection', model_name: str = 'docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    model_dir = snapshot_download('tatoao/DocLayout-YOLO-DocStructBench-ONNX',
                                  cache_dir=cache_dir)

    model_path = os.path.join(model_dir, model_name)


    dest_path = os.path.join(cache_dir, model_name)
    import shutil
    shutil.move(model_path, dest_path)
    
    return dest_path

# python -m doc_chunking.utils.download_onnx
if __name__ == "__main__":
    download_onnx()
    # import shutil
    # shutil.move('model_parameters/layout_detection/tatoao/DocLayout-YOLO-DocStructBench-ONNX/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx', 'model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx')