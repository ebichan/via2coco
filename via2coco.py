from asyncio.log import logger
import json
import os
import cv2
import numpy as np
import argparse
from loguru import logger
from shapely.geometry import Polygon

class Via2CocoConverter:
    '''
    Convert annotation file created by via annotator into coco format.
    Currently, only for bounding box annotation. 
    '''
    def __init__(
        self,
        json_path: str,
        img_root: str,
        out_fname: str='coco_annotation.json'
        ):
        '''
        Args:
            json_path (str): path of via anntation json file.
            img_root (srt): path of image files.
            out_fname (str): file name of output coco json file.
        '''
        self.json_path = json_path
        self.img_root = img_root
        self.out_fname = out_fname
        self.logger = logger
        self.coco_base = {
            "info" : {},
            "licenses": [],
            "images" : [],
            "annotations" : [],
            "categories" : [],
        }
        # query for annotation categories
        self.categories_dic = {} 
        self._parse_json()
        # self.make_coco_annotations()
        # self.save_as_json()
        
    def _create_dict(self):
        coco_images = {
            "id" : 0,
            "license" : 0,
            "coco_url" : "",
            "flickr_url" : {},
            "width" : 0,
            "height" : 0,
            "file_name" : ""
        }
        coco_annotations = {
            "id" : 0,
            "image_id" : 0,
            "category_id" : 1,
            "iscrowd" : 0,
            "segmentation" : [],
            "image_id" : 0,
            "area" : 0,
            "bbox" : [],
        }
        return coco_images, coco_annotations

        
    def _parse_json(self):
        with open(self.json_path, 'r') as f:
            self.json_dict = json.load(f)

    def save_as_json(self):
        with open(self.out_fname, 'w') as f:
            json.dump(self.coco_base, f ,indent=2)

    def make_coco_annotations(self):
        category_count = 1
        annotation_count = 0
        for idx_image, (k, v) in enumerate(self.json_dict.items()):
            try:
                img_path = os.path.join(self.img_root, v['filename'])
            except KeyError:
                break
            im_height, im_width = cv2.imread(img_path).shape[:2]

            coco_images, coco_annotations = self._create_dict()
            coco_images["id"] = idx_image
            coco_images["file_name"] = v["filename"]
            coco_images['width'] = im_width
            coco_images['height'] = im_height

            self.coco_base['images'].append(coco_images.copy())
            # print(self.coco_base['images'])
            annos = v["regions"]
            for idx_anno, anno in enumerate(annos):
                category_name, *_ = anno['region_attributes'].values()
                shape_attr = anno['shape_attributes']
                
                anno_type, *anno_coordinates = shape_attr.values()
                # for object detection
                # bbox = [x, y, width, height], x,y: upper-left coordinates of the bbox
                if anno_type == 'rect': 
                    coco_annotations['bbox'] = anno_coordinates
                    coco_annotations['area'] = shape_attr['width'] * shape_attr['height']
                # for instance segmentation 
                elif anno_type == 'polygon': 
                    px = shape_attr['all_points_x']
                    py = shape_attr['all_points_y']
                    width = np.max(px) - np.min(px)
                    height = np.max(py) - np.min(py)
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]
                    bbox = [np.min(px), np.max(py), np.max(px) - np.min(py), np.max(py) - np.min(py)]
                    coco_annotations['bbox'] = [round(float(x)) for x in bbox]
                    coco_annotations['segmentation'].append(poly)
                    polygon = Polygon(list(zip(px, py)))
                    coco_annotations['area'] = polygon.area
                else:
                    self.logger.info('annotation type : {} is not available.'.format(anno_type))
                    continue

                
                # register category
                if category_name  not in self.categories_dic.keys():
                    category_dic = {
                            'supercategory' : '',
                            'id' : category_count,
                            'name' : category_name
                    }
                    self.categories_dic.setdefault(category_name, category_dic)
                    self.coco_base['categories'].append(category_dic)
                    category_count += 1
                    
                annotation_count += 1

                coco_annotations['id'] = annotation_count
                coco_annotations['image_id'] = idx_image
                coco_annotations['category_id'] = self.categories_dic[category_name]['id']
                self.coco_base['annotations'].append(coco_annotations.copy())
                # print(self.coco_base['annotations'])

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True,
                        help='path of json file created by via annotator.')
    parser.add_argument('--img_root', type=str, required=True,
                        help='path of image files.')
    parser.add_argument('--out_fname', type=str, default='coco_annotations.json',
                        help='file name of output coco json file.')

    args = parser.parse_args()
    converter = Via2CocoConverter(
        json_path=args.json_path,
        img_root=args.img_root,
        out_fname=args.out_fname
    )
    converter.make_coco_annotations()
    converter.save_as_json()