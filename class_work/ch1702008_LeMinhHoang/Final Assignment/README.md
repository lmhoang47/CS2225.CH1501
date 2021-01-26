## Table of contents

1. [Introduction](#Introduction)
2. [Install the TensorFlow Object detection API](#Install-the-TensorFlow-Object-detection-API)
3. [Select the model for training](#Select-the-model-for-training)
4. [Image, Label, Annotation, Training vs Testing data](#Image-Label-Annotation-Training-vs-Testing-data)
5. [Process images & labels](#process-images--labels)
6. [Download Tensorflow model which contains Object Detection API](#Download-Tensorflow-model-which-contains-Object-Detection-API)
7. [Generate Tensorflow record](#Generate-Tensorflow-record)
8. [Download the Base Model which was selected above and is used for training](#Download-the-Base-Model-which-was-selected-above-and-isused-for-training)
9. [Configure the Training Pipeline](#Configure-the-Training-Pipeline)
10. [Create Tensorboard link for monitoring the training process](#Create-Tensorboard-link-for-monitoring-the-training-process)
11. [Begin to train model](#Begin-to-train-model)
12. [Export the trained model for reuse - inference](#Export-the-trained-model-for-reuse-inference)
13. [Test video & picture](#test-video--picture)

## Introduction
1. Tên đồ án: Nhận diện hành động dắt xe máy và xe đạp trên video
- Input: Đoạn video, hoặc ảnh
- Output: Các khung hình có hành động dắt xe máy hoặc xe đạp sẽ có rounding box xung quanh đối tượng, cùng với độ chính xác nhận diện từ model
- Hướng tiếp cận: phương pháp giải bài toán chỉ giới hạn ở việc nhận diện hành động dắt xe, trong đó khi ảnh/video được upload lên thì các khung hình sẽ được trích xuất liên tục. Đồng thời thuật toán trong mô hình huấn luyện sẽ quét qua tất cả các hình này và nhận diện xem có hành động dắt xe hay không
- Giới hạn: \
Số lượng đối tượng trong 1 khung ảnh < 10, ít chồng lấn\
Tập dữ liệu huấn luyện của xe đạp phong phú, nhưng cho xe máy rất hạn chế ở góc của đối tượng và loại xe máy
2. Loại bài toán: Object Detection
![alt text](https://drive.google.com/uc?export=view&id=1CdK9yDrjLjsEum6G-5dtziCjemwogG1Q)
- API: \
Protocol buffers để cấu hình tham số của training model\
Tensor Flow, version 1.15.0 để nhận dạng\
COCO: load, parse, visualize khung hình nhận dạng và ghi chú
- Pre-trained model #1: faster_rcnn_inception_v2_coco_2018_01_28\
Pipeline: faster_rcnn_inception_v2\
Number of training steps: 200,000\
Augmentation: random_horizontal_flip
- Pre-trained model #2: ssd_mobilenet_v2_coco_2018_03_29\
Pipeline: ssd_mobilenet_v2_coco\
Number of training steps: 30,000\
Augmentation: sigmoid
3. So sánh kết quả đạt được từ 2 phương pháp SSD và RCNN
![alt text](https://drive.google.com/uc?export=view&id=1-C1JljQwr05WS5Le97i0KV1mhy-uWLnx)

- ssd_mobilenet_v2: \
  Điểm mạnh: thời gian huấn luyện ngắn\
  Điểm yếu: \
    Độ chính xác thấp\
    Nếu 2 lớp đối tượng có feature tương đối giống nhau, thì lớp nào có độ chính xác cao hơn sẽ chiếm ưu thế khi nội suy ra kết quả cuối cùng. Trong trường hợp này, lớp bike_walker thậm chí được suy đoán trong video có xe máy
- faster_rcnn_inception_v2: \
  Điểm mạnh: độ chính xác cải thiện hơn hẳn SSD trong mô hình mà các lớp đối tượng có feature tương đối giống nhau\
  Điểm yếu: nếu các đối tượng có rounding box chồng lấn, phương pháp RCNN không thể xác định chính xác vị trí của đối tượng mà tạo ra nhiều hơn 1 rounding box cho 1 đối tượng được nhận diện
4. Project files
- Python files: *.ipynb
- Folder "walking": output video, training TF data, pipeline config & inference model for RCNN\
Full project folder: https://drive.google.com/drive/folders/1_pcDIoZRL4q_xzHxBToHUM0xBfYIgac2?usp=sharing
- Folder "Walking_SDD" output video, training TF data, pipeline config & inference model for SDD\
Full project folder: https://drive.google.com/drive/folders/1lCsH0mzs-iuqyfrHRyA3ArTiE6PsNbQ_?usp=sharing

## Install the TensorFlow Object detection API
Selected tenorflow version is 1.15.0, object detection API is removed from tf v 2.0+
<pre><code>
...
!pip install gast==0.2.2
!pip install -U --pre tensorflow=="1.15.0"
!pip install tf_slim
...
</code></pre>

Install protobuf
- Protocol Buffers (Protobuf) is a method of serializing structured data. 
- It is useful in developing programs to communicate with each other over a network or for storing data.
<pre><code>
...
!apt-get update && apt-get install alien
...
</code></pre>

- The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. 
- Before the framework can be used, the Protobuf libraries must be compiled. 
- This should be done by running the following command from the models/research/ directory
<pre><code>
...
!apt-get install -qq protobuf-compiler python-pil python-lxml python-tk
!pip install -qq Cython contextlib2 pillow lxml matplotlib
</code></pre>

- COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. 
- COCO API package provides Python APIs that assists in loading, parsing, and visualizing the annotations in COCO, and will be present in your system as pycocotools
<pre><code>
!pip install -qq pycocotools
...
</code></pre>

## Select the model for training
- ssd_mobilenet_v2: the quickest and least accuracy
- faster_rcnn_inception_v2: the best accuracy but worst in bounding rectangles
<pre><code>
...
MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
    }
}
...
</code></pre>

## Image, Label, Annotation, Training vs Testing data
- Download images at: https://drive.google.com/drive/folders/1-3tfEAu9G-tlUgYGF7Mmkez4Yog584Ks?usp=sharing
- Create label by LabelImg. Or download labels at: https://drive.google.com/drive/folders/1-3msVEl8r37AggUwokwcYlEoEn4e1919?usp=sharing and https://drive.google.com/drive/folders/1-2v210myEO1Uf2UX6ijyzx3XGAA5ZxJ1?usp=sharing
- Mix all labels in a random order (not really random, by their hash value instead)
- Moves 20% labels to the testing directory test_labels
<pre><code>
...
!ls data/train_labels/* | sort -R | head -310 | xargs -I{} mv {} data/test_labels
...
</code></pre>

## Process images & labels
- Converting the annotations from xml files to two csv files for each `train_labels/` and `train_labels/`.
- Creating a pbtxt file that specifies the number of class (one class in this case)
- Checking if the annotations for each object are placed within the range of the image width and height.

<pre><code>
...
# for both the train_labels and test_labels csv files, it runs the xml_to_csv() above.
for label_path in ['train_labels', 'test_labels']:
  image_path = os.path.join(os.getcwd(), label_path)
  xml_df, classes = xml_to_csv(label_path)
  xml_df.to_csv(f'{label_path}.csv', index=None)
  print(f'Successfully converted {label_path} xml to csv.')

# Create pbtxt file which contains graph information of all labels in each of image_path
label_map_path = os.path.join("label_map.pbtxt")

pbtxt_content = ""

#creats a pbtxt file the has the class names.
for i, class_name in enumerate(classes):
    # display_name is optional.
    pbtxt_content = (
        pbtxt_content
        + "item {{\n    id: {0}\n    name: '{1}'\n    display_name: '{1}'\n }}\n\n".format(i + 1, class_name)
    )
pbtxt_content = pbtxt_content.strip()
with open(label_map_path, "w") as f:
    f.write(pbtxt_content)
...    
</code></pre>


## Download Tensorflow model which contains Object Detection API
- Clone [Tensorflow models](https://github.com/tensorflow/models.git) from the offical git repo
- Compile the protos and adding folders to the os environment
-* This protobuf file contains everything needed to reconstruct a tensorflow graph. You can load in the graph_protobuf. pbtxt to retrieve the program. Changing the internals of this file is analogous to programming a new graph program
- Test the model builder
<pre><code>
...
# Download Tenorflow
!git clone --q https://github.com/tensorflow/models.git

# Compile protocal buffers
!protoc object_detection/protos/*.proto --python_out=.

# Export the PYTHONPATH environment variable with the reasearch and slim folders' paths
os.environ['PYTHONPATH'] += ':/content/gdrive/My Drive/Classroom/VRA/Walking_SDD/models/research/:/content/gdrive/My Drive/Classroom/VRA/Walking_SDD/models/research/slim/'
...
</code></pre>

## Generate Tensorflow record
- Generate 2 training & testing CSVs to TFRecords files
- Tensorflow accepts the data as tfrecords which is a binary file that run fast with low memory usage. Instead of loading the full data into memory, Tenorflow breaks the data into batches using these TFRecords automatically
<pre><code>
...
for csv in ['train_labels', 'test_labels']:
  writer = tf.io.TFRecordWriter(DATA_BASE_PATH + csv + '.record')
  path = os.path.join(image_dir)
  examples = pd.read_csv(DATA_BASE_PATH + csv + '.csv')
  grouped = split(examples, 'filename')
  for group in grouped:
      tf_example = create_tf_example(group, path)
      writer.write(tf_example.SerializeToString())
...
</code></pre> 

## Download the Base Model which was selected above and is used for training
<pre><code>
...
if not (os.path.exists(MODEL_FILE)):
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
...
</code></pre>

## Configure the Training Pipeline
- Adding the path for the TFRecords files and pbtxt,batch_size,num_steps,num_classes to the configuration file.
- Adding some Image augmentation.
- Creating a directory to save the model at each checkpoint while training. 
<pre><code>
...
%%writefile {model_pipline}
model {
  ssd {
    num_classes: 2 # number of classes to be detected
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
...
</code></pre>

## Create Tensorboard link for monitoring the training process
- Create a link to visualize multiple graph while start training
- Max 20 connection per minute is allowed when using ngrok
<pre><code>
...
# Tensorboard link
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
...    
</code></pre>


## Begin to train model
<pre><code>
...
!python3 model_main.py \
    --pipeline_config_path={"/content/gdrive/My\ Drive/Classroom/VRA/Walking_SDD/models/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config"} \
    --model_dir={"/content/gdrive/My\ Drive/Classroom/VRA/Walking_SDD/models/research/training"} \
    --alsologtostderr \
...
</code></pre>


## Export the trained model for reuse - inference
<pre><code>
...
!python "/content/export_inference_graph.py" \
    --input_type=image_tensor \
    --pipeline_config_path={"/content/ssd_mobilenet_v2_coco.config"} \
    --output_directory={"/content/fine_tuned_model"} \
    --trained_checkpoint_prefix={last_model_path}
...
</code></pre>    

## Test video & picture
<pre><code>
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
 ...
 
#play video preparation
cap = cv2.VideoCapture("/content/gdrive/My Drive/Classroom/VRA/Walking_SDD/data/video/Input_Motor_8_Complicated.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
#w = int(cap.get(3))
#h = int(cap.get(4))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_writer = cv2.VideoWriter("/content/gdrive/My Drive/Classroom/VRA/Walking_SDD/data/video/SDD_Input_Motor_8_Complicated.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
i=0
...
</code></pre>

![alt text](https://drive.google.com/uc?export=view&id=1WAhNBypCcDEpZ7R1n8AM1VyPJpxRClnd)


