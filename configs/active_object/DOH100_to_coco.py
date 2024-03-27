import json

mode = 'val'

p = f'/hd1/shared/100DOH/file/{mode}.json'

images = []
categories = []
annotations = []
cates = ["boardgame", "diy","drink","food","furniture","gardening","housework", "packing", 'puzzle', 'repair', 'study', 'vlog']
for i, cat in enumerate(cates):
    c_data = {
            "id": i,
            "name": cat,
            "supercategory": "none"
        }
    categories.append(c_data)
with open(p) as f:
    data = json.load(f)
    
for image_path, values in data.items():
    
    file_name = image_path
    width, height = values[0]['width'], values[0]['height']
    im_id = len(images)
    im_data = {
        'id': im_id,
        'width': width,
        'height': height,
        "file_name": file_name,
    }
    images.append(im_data)
    c = -1
    for i, cate in enumerate(cates):
        if cate in file_name:
            c=i
    if c ==-1:
        raise
    for v in values:
        obj_bbox = v['obj_bbox']
        if obj_bbox is None:
            continue
        # print(obj_bbox)
        # raise
        x1,x2,y1,y2 = obj_bbox['x1'],obj_bbox['x2'],obj_bbox['y1'],obj_bbox['y2']
        w,h = x2-x1,y2-y1
        x1 =x1*width
        w = w*width
        y1 = y1*height
        h = h*height
        bbox = [x1,y1,w,h]
        an_id = len(annotations)
        an_data = {
            "id": an_id,
            "image_id": im_id,
            "category_id": c,
            "segmentation": [],
            "area": w*h,
            "bbox": bbox,
            "iscrowd": 0,
            "ignore": 0
        }
        annotations.append(an_data)
        
data = {
    'categories': categories,
    'annotations': annotations,
    'images': images, 
}

with open(f'data/100DOH/annotations/{mode}.json', 'w') as f:
    json.dump(data,f)
        
        
