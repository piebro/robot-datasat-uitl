import os
import shutil
import json
import glob
import sys

import cv2
import numpy as np
import imantics
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


def save_combined_annotation_json(json_paths, output_json_path):
  new_annotation_data = {
      "images":[],
      "categories":[],
      "annotations":[]
  }

  for json_path in json_paths:
    with open(json_path) as json_file:
      data = json.load(json_file)
      new_annotation_data["images"].extend(data["images"])
      new_annotation_data["categories"] = data["categories"]
      new_annotation_data["annotations"].extend(data["annotations"])

  with open(output_json_path, 'w') as outfile:
    json.dump(new_annotation_data, outfile)


def get_mask_handle_occlusion(annotation, height, width):
  count = len(annotation)
  mask = np.zeros([height, width, count], dtype=np.uint8)
  
  
  seg_list_robot_and_ball = []
  seg_list_goal = []
  seg_list_rest = []
  # ids: line:1, ball:2, robot:3, centercircle:4, goal:5, penaltycross:6
  # first draw order: goal, line, centercircle, penaltycross, [robot,ball]
  for shape in annotation:
    category_id = shape["category_id"]
    segmentations  = shape["segmentation"]

    if category_id == 2 or category_id == 3:
        seg_list_robot_and_ball.append((category_id, segmentations))
    elif category_id == 6:
        seg_list_goal.append((category_id, segmentations))
    else:
        seg_list_rest.append((category_id, segmentations))
  

  seg_list = []
  seg_list.extend(seg_list_goal)
  seg_list.extend(sorted(seg_list_rest, key=lambda x: x[0]))
  seg_list.extend(seg_list_robot_and_ball)
  
  category_id_list = []
  for i, (category_id, segmentations) in enumerate(seg_list):
    category_id_list.append(category_id)

    pts = [
      np
      .array(anno)
      .reshape(-1, 2)
      .round()
      .astype(int)
      for anno in segmentations
      ]
        
    img = mask[:, :, i:i+1].copy()
    cv2.fillPoly(img, pts, 1)
    mask[:, :, i:i+1] = img
    
  # Handle occlusions
  if(mask.shape[2] > 0): # if at least one mask is there
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(count-2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

  return mask.astype(np.bool), np.array(category_id_list).astype(np.int32)


def save_non_overlapping_annotations_json(json_path, save_json_path):
  coco = COCO(json_path)
  with open(json_path) as json_file:
    coco_json = json.load(json_file)

  coco_json_new = {}
  coco_json_new["categories"] = coco_json["categories"]
  coco_json_new["images"] = coco_json["images"]
  
  new_annotations = []
  for img_count, img_id in enumerate(coco.imgs.keys()):

    annotations = coco.loadAnns(coco.getAnnIds(img_id))
    mask = get_mask_handle_occlusion(annotations, width=640, height=480)
    
    for i, class_id in enumerate(mask[1]):
      im_mask = imantics.Mask(mask[0][:,:,i])
      polygons = im_mask.polygons()
      polygons_new = []
      for poly in polygons:
        polygons_new.append(poly.tolist())

      new_annotation = {
          "id":len(new_annotations),
          "image_id": img_id,
          "category_id": int(class_id),
          "segmentation": polygons_new,
          "bbox": list(im_mask.bbox().bbox()),
          'iscrowd': False,
          'isbbox': False
      }
      new_annotations.append(new_annotation)
      
      sys.stdout.write('\rremoved overlapping for '+str(img_count+1)+' / '+str(len(coco.imgs.keys()))+' images')
      sys.stdout.flush()
    
  coco_json_new["annotations"] = new_annotations
  with open(save_json_path, 'w') as outfile:
    json.dump(coco_json_new, outfile)


def save_train_val_annotations_json(json_path, train_json_path, val_json_path, val_size, shuffel_seed=42):
  coco = COCO(json_path)
  img_ids = sorted(coco.imgs.keys())

  np.random.seed(shuffel_seed)
  np.random.shuffle(img_ids)

  seperate_index = round(len(img_ids)*val_size)
  img_ids_val = img_ids[:seperate_index]
  img_ids_train = img_ids[seperate_index:]

  with open(json_path) as json_file:
    categories_json = json.load(json_file)["categories"]

  for img_ids, save_path in [[img_ids_val, val_json_path], [img_ids_train, train_json_path]]:
    save_json = {
        "images":[],
        "categories":categories_json,
        "annotations":[]
    }

    for img_id in img_ids:
      save_json["images"].append(coco.loadImgs(img_id)[0])
      save_json["annotations"].extend(coco.loadAnns(coco.getAnnIds(img_id)))

    with open(save_path, 'w') as outfile:
      json.dump(save_json, outfile)


def get_dataset(dataset_folder, kaggle_api_token_path):
  """download dataset with fitting Kaggle Key"""
  if os.path.isdir(dataset_folder) and len(os.listdir(dataset_folder)) != 0:
    print("The dataset folder is not empty")
    return

  with open(kaggle_api_token_path) as json_file:
    kaggle_json = json.loads(json_file.read())
    os.environ['KAGGLE_USERNAME'] = kaggle_json["username"]
    os.environ['KAGGLE_KEY'] = kaggle_json["key"]

  import kaggle

  kaggle.api.authenticate()
  os.environ['KAGGLE_USERNAME'] = ""
  os.environ['KAGGLE_KEY'] = ""
  kaggle.api.dataset_download_files('pietbroemmel/naodevils-segmentation-upper-camera', path=dataset_folder, unzip=True)


  image_sub_folders = ["upper_0_500", "upper_500_1000", "upper_1000_1500", "upper_1500_2000"]
  img_root_path = os.path.join(dataset_folder, "images")

  all_images_path = os.path.join(dataset_folder, "all_images")
  os.mkdir(all_images_path)

  for imgage_sub_folder in image_sub_folders:
    sub_folder_path = os.path.join(img_root_path, imgage_sub_folder)
    for file_path in glob.glob(os.path.join(sub_folder_path, '*.*')):
      shutil.move(file_path, all_images_path)

  json_filenames = ["upper_0_500.json", "upper_500_1000.json", "upper_1000_1500.json"]
  annotations_path = os.path.join(dataset_folder, "annotations")
  json_paths = []
  for json_filename in json_filenames:
    json_paths.append(os.path.join(annotations_path, json_filename))

  all_annotations_path = os.path.join(dataset_folder, "all_annotations.json")
  save_combined_annotation_json(json_paths, all_annotations_path)
  save_non_overlapping_annotations_json(all_annotations_path, all_annotations_path)

  save_train_val_annotations_json(
      all_annotations_path,
      os.path.join(dataset_folder, "annotations_train.json"),
      os.path.join(dataset_folder, "annotations_val.json"),
      val_size=0.2)


def save_certain_categories(json_path, save_json_path, categorie_mapping):
  coco = COCO(json_path)

  categories_new = []
  id_to_new_id = {}
  for cat_id in coco.getCatIds():
    cat_data = coco.loadCats(cat_id)[0]
    new_cat_id = categorie_mapping[cat_data["name"]]
    id_to_new_id[cat_id] = new_cat_id
    
    if new_cat_id is not None:
      cat_data["id"] = new_cat_id
      categories_new.append(cat_data)

  imgs_new = []
  anns_new = []

  for img_id in coco.imgs.keys():
    for ann_old in coco.loadAnns(coco.getAnnIds(img_id)):
      new_id = id_to_new_id[ann_old["category_id"]]
      if new_id is not None:
        ann_old["category_id"] = new_id
        anns_new.append(ann_old)


    imgs_new.append(coco.loadImgs(img_id)[0])
  
  new_annotation_data = {
      "images": imgs_new,
      "annotations": anns_new,
      "categories": categories_new
  }
  
  with open(save_json_path, 'w') as outfile:
    json.dump(new_annotation_data, outfile)


def get_figure_of_images(img_array, width, height, size_mult):
    assert(len(img_array) == width*height)

    fig = plt.figure(figsize=(width*size_mult, height*size_mult))
    for row in range(height):
        for column in range(width):
            i = row*width + column
            ax = fig.add_subplot(height, width, i+1)
            ax.set_axis_off()
            ax.title.set_text(img_array[i][0])
            ax.imshow(cv2.cvtColor(img_array[i][1], cv2.COLOR_BGR2RGB))
    
    fig.tight_layout()
    return fig


def get_mask_segmentation(annotation, width, height):
  '''returns mask with all annotations drawn on the mask'''
  mask = np.zeros((height, width), dtype=np.uint8)
  for ann in annotation:
    segmentations = ann['segmentation']
    category_id = ann['category_id']
    pts = [
          np
          .array(anno)
          .reshape(-1, 2)
          .round()
          .astype(int)
          for anno in segmentations
          ]
        
    _mask = mask.copy()
    cv2.fillPoly(_mask, pts, 10)
    mask[_mask == 10] = category_id

  return mask


def get_colored_segmentation_mask(mask, class_id_to_color):
    seg_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in np.unique(mask):
      seg_img[mask == class_id] = class_id_to_color[class_id]
    return seg_img


def show_img_from_coco_json(img_dir, coco_json, class_id_to_color, img_count, fig_size):
  coco = COCO(coco_json)
  img_ids = np.random.choice(list(coco.imgs.keys()), img_count)
  img_array = []
  for img_id in img_ids:
    img_data = coco.loadImgs(int(img_id))[0]
    img = cv2.imread(os.path.join(img_dir, img_data["file_name"]))
    mask = get_mask_segmentation(coco.loadAnns(coco.getAnnIds(img_id)), 640, 480)
    cmask = get_colored_segmentation_mask(mask, class_id_to_color)
    img_array.append([img_data["file_name"], img])
    img_array.append(["mask", cmask])
  fig = get_figure_of_images(img_array, 2, img_count, fig_size)
  fig.show()


def annotations_from_mask_semantic_segmentation(mask):
  annotations = []
  for class_id in np.unique(mask):
    if class_id == 0:
      continue
    class_mask = (mask==class_id)
    im_mask = imantics.Mask(class_mask)
    polygons = im_mask.polygons()
    polygons_new = []
    for poly in polygons:
      polygons_new.append(poly.tolist())
    
    annotation = {
      "category_id": int(class_id),
      "segmentation": polygons_new,
      "bbox": list(im_mask.bbox().bbox()),
    }
    annotations.append(annotation)

  return annotations


def annotations_from_mask_instance_segmentation(class_ids, masks):
  annotations = []
  for i, class_id in enumerate(class_ids): 
    im_mask = imantics.Mask(masks[i])
    polygons = im_mask.polygons()
    polygons_new = []
    for poly in polygons:
      polygons_new.append(poly.tolist())

    annotation = {
      "category_id": int(class_id)+1,
      "segmentation": polygons_new,
      "bbox": list(im_mask.bbox().bbox()),
    }

    annotations.append(annotation)
  
  return annotations


def get_not_annoteted_images(all_manual_annotated_json, img_dir):
  all_images = os.listdir(img_dir)
  with open(all_manual_annotated_json) as json_file:
    manual_json = json.load(json_file)
  
  manual_images = []
  for img_dict in manual_json["images"]:
    manual_images.append(img_dict["file_name"])

  img_to_annotate = [x for x in all_images if x not in manual_images]
  return img_to_annotate


def filename_annotations_list_to_coco_json(filename_annotations_list, categories):
  images = []
  annotations = []
  categories_json = []

  for i, cat_name in enumerate(categories):
    categories_json.append({"id":i+1, "name":cat_name})

  for img_id, filename_annotations in enumerate(filename_annotations_list):
    images.append({
        "id": img_id,
        "path": filename_annotations["filename"],
        "file_name": filename_annotations["filename"],
    })

    for annotation in filename_annotations["annotations"]:
      annotation["id"] = len(annotations)+1
      annotation["image_id"] = img_id
      annotations.append(annotation)

  return {
      "images":images,
      "annotations":annotations,
      "categories":categories_json
  }


def save_predicted_coco_json(img_dir, all_manual_annotated_json, automatic_annotated_json, get_annotation_func, categories):
  img_to_annotate = get_not_annoteted_images(all_manual_annotated_json, img_dir)

  filename_annotations = []
  for i, img_filename in enumerate(img_to_annotate):
    filename_annotations.append({
        "filename": img_filename,
        "annotations": get_annotation_func(os.path.join(img_dir, img_filename))
    })
    sys.stdout.write('\rgenerated annotations to '+str(i+1)+' / '+str(len(img_to_annotate))+' images')
    sys.stdout.flush()
  print()
  
  coco_json = filename_annotations_list_to_coco_json(filename_annotations, categories)

  with open(automatic_annotated_json, 'w') as outfile:
    json.dump(coco_json, outfile)







