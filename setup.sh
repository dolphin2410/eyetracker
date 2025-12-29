python -m venv eyetracker
source ./eyetracker/bin/activate

mv patches/face_parsing.patch face_parsing/patch.patch
cd face_parsing
git apply patch.patch
cd ../
pip install ./face_parsing
pip install -r requirements.txt
curl -L https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.pt -o face_parsing/weights/resnet34.pt
curl -L https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12n-face.pt -o yolov12n-face.pt