
def get_image_id(image_batch_files):

    img_ids = []
    for image_id in image_batch_files:
        id = image_id.rsplit('.jpg')[0]
        id = id.split('_')[2]
        id = id.lstrip('0')
        img_ids.append(id)

    return img_ids
        





