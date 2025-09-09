from doclayout_yolo import YOLOv10
from PIL import Image

model = YOLOv10("model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.pt")

image = Image.open("image.png")
output = model.predict(image)
print(output)
