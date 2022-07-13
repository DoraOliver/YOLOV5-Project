from hashlib import sha1
import os
import torch 
import cv2
import numpy as np
from numpy import transpose
import sys
import argparse

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

@torch.no_grad()
def run():
  
  imgsz=(640, 640)
  img_path = "./data/image_single/street.jpg"
  source=ROOT / 'data/images'
  nosave = False
  project=ROOT / 'runs/detect'
  name='exp'
  exist_ok=False
  save_txt=False

  source = str(source)
  save_img = not nosave and not source.endswith('.txt')  # save inference images
  is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
  is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
  # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
  if is_url and is_file:
      source = check_file(source)  # download

  # Directories
  save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
  (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = DetectMultiBackend(weights='yolov5s.pt', device=device, data='data/coco128.yaml')
  stride, names, pt = model.stride, model.names, model.pt
  imgsz = check_img_size(imgsz, s=stride)

  # dataset = LoadImages('data/image_single', img_size=imgsz, stride=stride, auto=pt)
  # bs = 1
  # # vid_path, vid_writer = [None] * bs, [None] * bs

  # # model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
  # # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
  # seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
  # for path, img, im0s, vid_cap, s in dataset:
  #   img = torch.from_numpy(img).to(device)
  #   img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
  #   img /= 255  # 0 - 255 to 0.0 - 1.0
  #   #如果图片形状为3，新增一个batch维度
  #   if len(img.shape) == 3:
  #       img = img[None]



  #   pred = model(img)
  #   pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.6, multi_label=True)

  #   for i, det in enumerate(pred):  # per image
  #     seen += 1
  #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
  #     p = Path(p)
  #     save_path = str(save_dir / p.name)
  #     # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 
  #     # annotator = Annotator(im0, line_width=3, example=str(names))
  #     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

  #     for *xyxy, conf, cls in reversed(det):
  #       c = int(cls)  # integer class
  #       label = f'{names[c]} {conf:.2f}'
  #       #annotator.box_label(xyxy, label, color=colors(c, True))
  #       box_label(im0, xyxy, label, color=colors(c, True))
  #     im0 = result(im0)
    
  #   cv2.imwrite(save_path, im0)
    
  #   cv2.imshow('detected image', im0)
  #   cv2.waitKey(0)
  #   cv2.destroyAllWindows()

  img_o = cv2.imread(img_path)

  img = letterbox(img_o, new_shape=imgsz, auto=True, color=(0,0,0))[0]
  img = img[:, :, ::-1].transpose(2, 0, 1)
  img = np.ascontiguousarray(img)

  img = torch.from_numpy(img).to(device).float()
  img /= 255.0
  img = img.unsqueeze(0)

  pred = model(img)

  pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.6, multi_label=True)[0]

  pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()

  for *xyxy, conf, cls in reversed(pred):
    c = int(cls)  # integer class
    label =  f'{names[c]} {conf:.2f}'
    img_o = box_label(img_o, xyxy, label, color=colors(c, True))

  img_o = result(img_o)
  cv2.imshow('detected image', img_o)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  cv2.imwrite('./data/image_single/detected_image.jpg', img_o)


def box_label(img, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  lw = 3 or max(round(sum(img.shape) / 2 * 0.003), 2)
  cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)
  return img

def result(img):
  # Return annotated image as array
  return np.asarray(img)

if __name__ == "__main__":
    run()