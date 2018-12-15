
def get_image_id(filename):

    id = filename.rsplit('.jpg')[0]
    id = id.split('_')[2]
    id = id.lstrip('0')
    return id
        





