from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "wc.jpg"), output_image_path=os.path.join(execution_path , "wc.jpg"))

for eachObject in detections:
  print(eachObject["wc"] , " : " , eachObject["percentage_probability"] )
  from PIL import Image
image = Image.open('wc.jpg')
image.show()