import os
from modelscope import snapshot_download

def download_onnx(cache_dir: str = './model_parameters/layout_detection', 
        model_name: str = 'docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx',
        model_repo: str = 'tatoao/DocLayout-YOLO-DocStructBench-ONNX'
        ):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


    # check if the model_name is already in the cache_dir
    if os.path.exists(os.path.join(cache_dir, model_name)):
        return os.path.join(cache_dir, model_name)

    model_dir = snapshot_download(model_repo,
                                  cache_dir=cache_dir)

    model_path = os.path.join(model_dir, model_name)


    dest_path = os.path.join(cache_dir, model_name)
    import shutil
    shutil.move(model_path, dest_path)

    # add a .gitignore file to the cache_dir
    with open(os.path.join(cache_dir, '.gitignore'), 'w') as f:
        f.write('*')
    
    return dest_path

# python -m doc_chunking.utils.download_onnx
if __name__ == "__main__":
    download_onnx()
    # import shutil
    # shutil.move('model_parameters/layout_detection/tatoao/DocLayout-YOLO-DocStructBench-ONNX/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx', 'model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx')