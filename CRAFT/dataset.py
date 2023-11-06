import torch
import torch.utils.data as data
import scipy.io as scio
from gaussian import GaussianTransformer
from watershed import watershed1
import re
import itertools
from file_utils import *
from mep import mep
import random
from PIL import Image
import torchvision.transforms as transforms
import craft_utils
import Polygon as plg
import time
import copy
from defaultlist import defaultlist

import configuration as cf
from  ocr_fpc import DataExtraction as DataEx
import albumentations as A

def ratio_area(h, w, box):
    area = h * w
    ratio = 0
    for i in range(len(box)):
        poly = plg.Polygon(box[i])
        box_area = poly.area()
        tem = box_area / area
        if tem > ratio:
            ratio = tem
    return ratio, area


def re_scale(img, word_bboxes, words, min_size):
    height, width = img.shape[:2]
    scale_x = min_size/width
    scale_y = min_size/height
    resized_img = cv2.resize(img,  dsize=(min_size, min_size))
    numpy_boxes = []
    #print("scalex and scaley :", scale_x, scale_y)
    for bboxes in word_bboxes:
        brod_bboxes = bboxes.reshape(1,8)
        for  box in brod_bboxes:
            #print(box)
            #print("before", box)
            box0 = (box[0]*scale_x)
            x1 = check_zero(box0)
            y1 = check_zero(box[1]*scale_y)
            box2 = (box[2]*scale_x) 
            x2 = check_in_image( box2 , min_size)
            y2 = check_zero(box[3]*scale_y)
            box4 = (box[4]*scale_x)
            x3 = check_in_image( box4 , min_size)
            y3 = check_in_image(box[5]*scale_y, min_size)
            box6 = (box[6]*scale_x)
            x4 = check_zero(box6)
            y4 = check_in_image(box[7]*scale_y, min_size)
            
            resized_box= (x1,y1,x2,y2,x3,y3,x4,y4)
            #print("after", resized_box)
            numpy_box = np.array(resized_box).reshape(4, 2)
        numpy_boxes.append((numpy_box))
    word_bboxes = numpy_boxes
    return resized_img, word_bboxes

def check_zero(point):
    if point<0:
        return 0
    return point 
def check_in_image(point, limit):
    if point > limit:
        point = limit
    return point

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


class craft_base_dataset(data.Dataset):
    def __init__(self, process_type, target_size=960, viz=False, debug=False):
        self.target_size = target_size
        self.viz = viz
        self.debug = debug
        self.process_type = process_type
        self.gaussianTransformer = GaussianTransformer(imgSize = self.target_size, region_threshold=0.35, affinity_threshold=0.15)

    def load_image_gt_and_confidencemask(self, index):
        return None, None, None, None, None

    def crop_image_by_bbox(self, image, box):
        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M

    def order_points(self, pts):

        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):

        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped, M

    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        try:
            return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len
        except:
            return 0
    def inference_pursedo_bboxes(self, net, image, word_bbox, word, viz=True):
        word_image, MM = self.four_point_transform(image, word_bbox)
        #if self.process_type == "train":
        DataEx().upload_image(word_image, word)                              #saving the cropped image used only for first epoch
        
        real_word_without_space = word.replace('\s', '')
        real_char_nums = len(real_word_without_space)
        input = word_image.copy()
        scale = 64.0 / input.shape[0]
        input = cv2.resize(input, None, fx=scale, fy=scale)
        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                                                   variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        img_torch = img_torch.type(torch.FloatTensor).to(device)

        scores, _ = net(img_torch)
        region_scores = scores[0, :, :, 0].cpu().data.numpy()
        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
        bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
        bboxes = []
        #print(confidence)

        if True : #modify here 
            width = input.shape[1]
            height = input.shape[0]*0.9

            width_per_char = width / len(word)
            special = 0
            for i, char in enumerate(word):
                if (char == '.' or char == ','):
                    special +=1
            for i, char in enumerate(word):
                if char == ' ':
                    continue 
                if special == 1:
                    if (i < len(word)-1) and char != '.': 
                        h= height
                        width_extra = (width_per_char / len(word))
                        width_char = width_per_char + width_extra
                        left = (i + 0.10) * width_char
                        right = (i + 0.90) * width_char
                        bbox = np.array([[left, 0], [right, 0], [right, h],
                                        [left, h]])
                        bboxes.append(bbox)
                        continue
                    if (char == '.' or char == ',')  and (i==len(word)-1):
                        h1= height/2
                        h= height
                        width_extra = (width_per_char / len(word))
                        width_char = width_per_char + width_extra
                        left = width - (width_extra*1.2)
                        right = width
                        bbox = np.array([[left, h1], [right, h1], [right, h],
                                        [left, h]])
                        bboxes.append(bbox)
                        continue
                    if char == '.' or char == '_'  and (i != len(word)-1):
                        h1= height/2
                        h= height
                        width_extra = (width_per_char / len(word))
                        width_char = width_per_char + width_extra
                        left = (i + 0.10) * width_char
                        right = (i + 0.90) * width_char
                        bbox = np.array([[left, h1], [right, h1], [right, h],
                                        [left, h]])
                        bboxes.append(bbox)
                        continue

                if (char == '.' or char == ',') :
                    h1= height/2
                    h= height
                    left = (i + 0.10) * width_per_char
                    right = (i + 0.90) * width_per_char
                    bbox = np.array([[left, h1], [right, h1], [right, h],
                                    [left, h]])
                    bboxes.append(bbox)
                    continue
                left = (i + 0.10) * width_per_char
                right = (i + 0.90) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)

            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5

        if False:
            _tmp_bboxes = np.int32(bboxes.copy())
            _tmp_bboxes[:, :, 0] = np.clip(_tmp_bboxes[:, :, 0], 0, input.shape[1])
            _tmp_bboxes[:, :, 1] = np.clip(_tmp_bboxes[:, :, 1], 0, input.shape[0])
            for bbox in _tmp_bboxes:
                cv2.polylines(np.uint8(input), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))
            region_scores_color = cv2.applyColorMap(np.uint8(region_scores), cv2.COLORMAP_JET)
            region_scores_color = cv2.resize(region_scores_color, (input.shape[1], input.shape[0]))
            target = self.gaussianTransformer.generate_region(region_scores_color.shape, [_tmp_bboxes])
            target_color = cv2.applyColorMap(target, cv2.COLORMAP_JET)
            viz_image = np.hstack([input[:, :, ::-1], region_scores_color, target_color])
            cv2.imshow("crop_image", viz_image)
            cv2.waitKey()
        bboxes /= scale

        try:
            for j in range(len(bboxes)):
                ones = np.ones((4, 1))
                tmp = np.concatenate([bboxes[j], ones], axis=-1)
                I = np.matrix(MM).I
                ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                bboxes[j] = ori[:, :2]
        except Exception as e: # if its polygon annotation then raises the exception as it will be a singular matrix
            for j in range(len(bboxes)):
                ones = np.ones((4, 1))
                tmp = np.concatenate([bboxes[j], ones], axis=-1)
                I = np.linalg.pinv(MM)
                ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                bboxes[j] = ori[:, :2]
            print(e, word)
        

        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)

        return bboxes, region_scores, confidence

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def get_imagename(self, index):
        return None

    def saveInput(self, imagename, image, region_scores, affinity_scores, confidence_mask):
        boxes, polys = craft_utils.getDetBoxes(region_scores / 255, affinity_scores / 255, 0.7, 0.4, 0.4, False)
        boxes = np.array(boxes, np.int32) * 2
        if len(boxes) > 0:
            np.clip(boxes[:, :, 0], 0, image.shape[1])
            np.clip(boxes[:, :, 1], 0, image.shape[0])
            for box in boxes:
                cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 255, 255))
        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        gt_scores = np.hstack([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color])
        confidence_mask_gray = np.hstack([np.zeros_like(confidence_mask_gray), confidence_mask_gray])
        output = np.concatenate([gt_scores, confidence_mask_gray],
                                axis=0)
        output = np.hstack([image, output])
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + '/output/inputs'), "%s_input.jpg" % imagename)
        print(outpath)
        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)


    def pull_item(self, index, process_type):
        # if self.get_imagename(index) == 'img_59.jpg':
        #     pass
        # else:
        #     return [], [], [], [], np.array([0])
        image, character_bboxes, words, confidence_mask, confidences, word_bboxes = self.load_image_gt_and_confidencemask(index)
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []

        if len(character_bboxes) > 0:

            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,
                                                                                          character_bboxes,
                                                                                          words)
        if self.viz:
            self.saveImage(self.get_imagename(index), image.copy(), character_bboxes, affinity_bboxes, region_scores,
                           affinity_scores,
                           confidence_mask)
        random_transforms = [image, region_scores, affinity_scores, confidence_mask]
        #random_transforms = random_horizontal_flip(random_transforms)

        cvimage, region_scores, affinity_scores, confidence_mask = random_transforms

        region_scores = self.resizeGt(region_scores)
        affinity_scores = self.resizeGt(affinity_scores)
        confidence_mask = self.resizeGt(confidence_mask)

        if False:
            self.saveInput(self.get_imagename(index), cvimage, region_scores, affinity_scores, confidence_mask)
        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        #image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores / 255).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        if self.process_type == 'train':
            return image, region_scores_torch, affinity_scores_torch, confidence_mask_torch, confidences
        if self.process_type == 'validation':
            return image, word_bboxes




class FPC_dataset(craft_base_dataset):
    
    def __init__(self, net, dataframe, process_type, target_size=960, viz=False, debug=False):
        super(FPC_dataset, self).__init__(process_type, target_size, viz, debug)
        self.net = net
        self.net.eval()
        df = dataframe
        self.process_type = process_type
        self.annotation_list = df['annotation_data'].values # co-ordinates

        self.cell_ids_list = df[df.columns[0]].values.tolist() # img_folder , imagenames, images_path
        self.words_list = df['labels'].values.tolist()
        
    def __getitem__(self, index):
        return self.pull_item(index, self.process_type) 

    def __len__(self):
        return len(self.cell_ids_list)

    def get_imagename(self, index):
        return self.cell_ids_list[index]

    def load_image_gt_and_confidencemask(self, index):
        img_path_id = self.cell_ids_list[index]
        word_bboxes, words = self.load_gt(index) # modified
        try:
            resp = DataEx().get_image(img_path_id)
        except:
            print("error in fetching image....please check the server connection")
            resp = DataEx().get_image(img_path_id)    
        image=cv2.imdecode(np.frombuffer(resp, np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #was used
        image, word_bboxes = re_scale(image, word_bboxes, words, self.target_size) #modified
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)
        character_bboxes = []
        new_words = []
        confidences = []
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)): 
                pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net, image,
                                                                                               word_bboxes[i],
                                                                                               words[i],
                                                                                               viz=self.viz)

                confidences.append(confidence)
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                new_words.append(words[i])
                character_bboxes.append(pursedo_bboxes)
        return image, character_bboxes, new_words, confidence_mask, confidences, word_bboxes

    def load_gt(self, index):
        
        bboxes = []
        words = self.words_list[index]
        for l in self.annotation_list[index]:
            #print("individual item of annotation list:", l)
            box = np.array(l).reshape(4, 2)            
            bboxes.append(box)
        
        return bboxes, words
