import torch
import torchvision
import cv2
import utils
import time
from PIL import Image
from torchvision.transforms import transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

cap = cv2.VideoCapture(0)

frame_count = 0
total_fps = 0


while True:
    ret, frame = cap.read()

    pil_image = Image.fromarray(frame).convert('RGB')

    orig_frame = frame
    # transform the image
    image = transform(pil_image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    # get the start time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
    # get the end time
    end_time = time.time()
    output_image = utils.draw_keypoints(outputs, orig_frame)
    # get the fps
    fps = 1 / (end_time - start_time)
    # add fps to total fps
    total_fps += fps
    # increment frame count
    frame_count += 1
    # wait_time = max(1, int(fps/4))
    # cv2.imshow('Pose detection frame', output_image)

    cv2.imshow('Live Video', output_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()