"""
Original Author: Zylo117 (Yet-Another-EfficientDet-Pytorch)
Updated 2024:
- uses argparse to get arguments from the commandline
- consolidates video, single image, and multi-image inference into a single python file.
"""

"""
Simple Inference Script of EfficientDet-Pytorch
"""

import os
from typing import Generator
import argparse
from tqdm import tqdm

import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, preprocess_video, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

COLOR_LIST = standard_to_bgr(STANDARD_COLORS)
IMAGE_FILE_TYPES : list [ str ] = [ '.jpg', '.png' ]
VIDEO_FILE_TYPES : list [ str ] = [ '.mp4', '.avi' ]


def main(args : argparse.Namespace) -> None:
    # make output_dir if it doesn't already exist
    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)
        
    # load in list of object classes
    obj_list : list [ str ] = load_object_classes(file_path=args.objects_path)
    
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    # enable cudnn properties
    cudnn.fastest = True
    cudnn.benchmark = True

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[args.model_type] if args.forced_input_size is None else args.forced_input_size
    
    # load model backbone
    model = EfficientDetBackbone(compound_coef=args.model_type, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    # set model to use gpu if instructed
    if args.use_gpu:
        model = model.cuda()
        
    # set model to use half precision if instructed
    if args.use_float16:
        model = model.half()
    
    # archive inference data
    data : list [ ModelInput ] = get_model_inputs(args=args, input_size=input_size)
    for i,model_input in enumerate(data):
        print(f"Beginning ModelInput {i}...")
        # get generator for this model input
        data_gen = model_input.gen()
        for x, ori_imgs, framed_metas, file_paths in tqdm(data_gen):
            # pass input into model
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                args.confidence_threshold, args.iou_threshold)

            out = invert_affine(framed_metas, out)
            # write images / display images
            display(out, ori_imgs, file_paths=file_paths, obj_list=obj_list, imshow=args.show_images, imwrite=args.write_data)
        print(f"Finished ModelInput {i}!")


# display detections on the given input images
def display(preds, imgs, file_paths:list[str], obj_list:list[str], imshow:bool=True, imwrite:bool=False) -> None:
    for i in range(len(imgs)):
        # if no predictions are found, skip
        if len(preds[i]['rois']) == 0:
            print(f'File: {file_paths[i]}: No detections found.')
            continue

        imgs[i] = imgs[i].copy()

        person_detected : bool = False
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int64)
            obj = obj_list[preds[i]['class_ids'][j]]
            if obj != "person":
                continue
            person_detected = True
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=COLOR_LIST[get_index_label(obj, obj_list)])

        if person_detected == False:
            print(f'File: {file_paths[i]}: No human detections found.')
            continue

        if imshow == True:
            cv2.imshow(file_paths[i], imgs[i])
            cv2.waitKey(0)
        
        if imwrite == True:
            print(f"Writing Image: {file_paths[i]}")
            cv2.imwrite(file_paths[i], imgs[i])


# loads in a list of object class names from a line separated text file
def load_object_classes ( file_path : str ) -> list [ str ]:
    obj_list : list [ str ] = []
    with open(file=file_path, mode="r") as file:
        for line in file:
            obj_list.append(line.strip())
    return obj_list


# Base class defining input data generation
class ModelInput(object):
    def __init__ ( self, args : argparse.Namespace, input_size : int ) -> None:
        self.args : argparse.Namespace = args
        self.batch_size : int = args.batch_size
        self.use_gpu : bool = args.use_gpu
        self.use_float16 : bool = args.use_float16
        self.input_size : int = input_size
    
    # defined in base classes
    def gen(self) -> Generator [torch.Tensor, None, None]:
        pass
    

# class defining input data generation for video files
class VideoInput(ModelInput):
    def __init__ ( self, args : argparse.Namespace, input_size : int, video_path : str ) -> None:
        super().__init__(args=args, input_size=input_size)
        self.video_path : str = video_path
    
    # generate tensor batches from the video to be processed
    def gen(self) -> Generator [ tuple [ torch.Tensor, object, object ], None, None]:
        # Video capture
        cap = cv2.VideoCapture(self.video_path)

        frame_num : int = 0
        keep_going : bool = True
        while keep_going:
            # get up to batch_size frames
            frames : list [ cv2.typing.MatLike ] = self._get_batch_imgs(cap=cap)
            
            if len(frames) == 0:
                keep_going = False
                continue

            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess_video(*frames, max_size=self.input_size)

            if self.use_gpu:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)
        
            file_paths : list [ str ] = [ 
                os.path.join(args.output_dir, f"{self.video_path}_{i}.jpg") for i in range(frame_num, len(frames)) ]
            frame_num += len(frames)
            
            # yield the values (Generator)
            yield (x, ori_imgs, framed_metas, file_paths)
        # once exhausted, this generator will return None

    def _get_batch_imgs(self, cap : cv2.VideoCapture) -> list [ cv2.typing.MatLike ]:
        # get the image or as many exists until the video stream runs out
        img_count : int = 0
        frames : list [ cv2.typing.MatLike ] = []
        keep_going : bool = True
        while (img_count < self.batch_size) and (keep_going == True):
            # grab a frame from the video
            ret, frame = cap.read()
            
            # if no more frames, finish loop
            if ret == False:
                keep_going = False
                continue
        
            # if there are frames, add them to the list of frames
            frames.append(frame)
            
            # increment img_count
            img_count += 1
        return frames


# class defining input data generation for image files
class ImageInput(ModelInput):
    def __init__ ( self, args : argparse.Namespace, input_size : int, file_paths : list [ str ] ) -> None:
        super().__init__(args=args, input_size=input_size)
        self.file_paths : list [ str ] = file_paths
    
    # generate tensor batches from the image files to be processed
    def gen(self) -> Generator [torch.Tensor, None, None]:
        for i in range(0, len(self.file_paths) // self.batch_size):
            # get up to batch_size frames
            start : int = i*self.batch_size
            stop : int = min((i+1)*self.batch_size, len(self.file_paths))
            file_path_subset : list [ str ] = self.file_paths[start:stop]

            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess(*file_path_subset, max_size=self.input_size)

            if self.use_gpu:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)
        
            file_paths : list [ str ] = [ 
                os.path.join(args.output_dir, f"{os.path.basename(file_path)}_pred.jpg") for file_path in file_path_subset ]
            
            # yield the values (Generator)
            yield (x, ori_imgs, framed_metas, file_paths)
        # once exhausted, this generator will return None


# get the data we want to run the model on.
def get_model_inputs(args : argparse.Namespace, input_size : int) -> list [ ModelInput ]:
    # get all files/folders in the data folder.
    names : list [ str ] = os.listdir(args.data_path)
    paths : list [ str ] = []
    for name in names:
        paths.append(os.path.join(args.data_path, name))

    # divide only grab files, and then separate by video or image format
    image_paths : list [ str ] = []
    video_paths : list [ str ] = []
    for path in paths:
        if os.path.isfile(path) == True:
            _, ext = os.path.splitext(path)
            if ext in IMAGE_FILE_TYPES:
                image_paths.append(path)
            elif ext in VIDEO_FILE_TYPES:
                video_paths.append(path)
            else:
                print(f"File Path: '{path}' skipped - not a supported image or video file type.")
        else:
            print(f"File Path: '{path}' skipped - not a file.")
    
    # create model inputs for both images and videos
    data : list [ ModelInput ] = []

    # create a new VideoInput for each video
    for video_path in video_paths:
        data.append(VideoInput(args=args, input_size=input_size, video_path=video_path))
    
    # create a single ImageInput for all of the images found
    data.append(ImageInput(args=args, input_size=input_size, file_paths=image_paths))        

    return data


if __name__ == "__main__":
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(
        'MODIFIED FROM: Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-m', '--model_type', type=int, default=3, 
                        help='The size of efficientdet model (or submodel architecture) you are using (D0-D7). ex: D3 -> -m 3')
    parser.add_argument('-w', '--weights_path', type=str, default="./weights/efficientdet-d3.pth",
                        help='The path to the pytorch model weights for the efficientdet model.')
    parser.add_argument('-d', '--data_path', type=str, default="./data/",
                        help='The path to the data you want to run inference on. All files in this directory will be ran into the model.')
    parser.add_argument('-i', '--forced_input_size', type= tuple [ int | None ], default=None,
                        help='The input size the images should be forced to. If set to None, will select image input size based on model size (D0-D7).')
    parser.add_argument('-conf_t', '--confidence_threshold', type=float, default=0.2,
                        help='The confidence threshold for which to keep a detection from the model (the confidence the model has that the object is actually what it claims).')
    parser.add_argument('-iou_t', '--iou_threshold', type=float, default=0.2,
                        help='The iou threshold.')
    parser.add_argument('-g', '--use_gpu', type=bool, default=True,
                        help='Whether to load model into GPU or not - Defaults to True.')
    parser.add_argument('-f', '--use_float16', type=bool, default=False,
                        help='Whether to reduce precision to 16 bit precision - Defaults to False (32 bit precision by default).')
    parser.add_argument('-o', '--objects_path', type=str, default='./objects.txt',
                        help='The path to a line-separated file containing the list of objects the model is trained to detect.')
    parser.add_argument('-si', '--show_images', type=bool, default=False,
                        help='Whether to display images when performing inference.')
    parser.add_argument('-wd', '--write_data', type=bool, default=False,
                        help='Whether to write images to a file when performing inference.')
    parser.add_argument('-od', '--output_dir', type=str, default="./inference/",
                        help='The path to the folder to write images or videos, if write_data is true.')
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
                        help='The number of images to process in parallel. The maximum amount you can run depends on the GPU memory your graphics card has.')
    args : argparse.Namespace = parser.parse_args()
    main(args=args)
