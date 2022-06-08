import os
import random
import pickle
import cv2
import shutil


def convert_bbox(bbox, h, w):
    """
    Input:
        - bbox: bounding box in form [xTopLeft, yTopLeft, xBottomRight, yBottomRight]
        - h: height of original image
        - w: width of original image
    Return:
        - bbox of form [xCenter, yCenter, width, height] all btw 0 and 1
    """
    xCenter = (bbox[0] + (bbox[2] - bbox[0]) / 2) / w
    yCenter = (bbox[1] + (bbox[3] - bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [xCenter, yCenter, width, height]


def create_yolo_data(annotations, images, type, keypointNames):

    with open(f"coco_kpts/{type}2017.txt", 'w') as imageList:

        for annotation in annotations:

            # Initializing this as valid
            validAnnotation = True

            # Computing the file name
            name = annotation['name'][0:-7]

            # Getting the image so we can read it in
            for image in images:
                if image['name'][0:-4] == annotation['name'][0:-7]:
                    break

            # If an annotation is missing an image we skip it (i.e. don''t include the image)
            if image['name'][0:-4] != name:
                continue

            # Getting dimensions of image
            img = cv2.imread(image['path'])
            h, w, c = img.shape

            # Reading in annotation
            with open(annotation['path'], 'rb') as f:
                data = pickle.load(f)

            # Creating new annotation file
            with open(f"coco_kpts/labels/{type}2017/{name}.txt", 'w') as newAnnotationFile:

                # Iterating through the annotation file
                for a in data.keys():

                    # getting the subannotations (i.e. multiple objects)
                    subAnnotation = data[a]

                    # Computing the new bbox coordinates
                    try:
                        bbox = subAnnotation['bbox']
                        newBbox = [0] + convert_bbox(bbox=bbox, h=h, w=w)

                        # Creating the new annotation
                        newSubAnnotation = ' '.join(str(i) for i in newBbox)
                    except:
                        validAnnotation = False
                        newSubAnnotation = ''


                    # Iterating through the keypoints
                    for keypoint in keypointNames:
                        if keypoint == '_background_':
                            continue

                        # Normalizing the keypoints
                        x, y = subAnnotation[keypoint]

                        if (x == -1) | (y == -1):
                            newX = 0
                            newY = 0
                            obscured = 0
                        else:
                            newX = x / w
                            newY = y / h
                            obscured = 2

                        # Adding the keypoints onto the annotation string
                        newSubAnnotation += ' ' + ' '.join(str(i) for i in [newX, newY, obscured])

                    # Adding new line to end in case of multiple objects
                    newSubAnnotation += '\n'

                    # Writing to disk
                    if validAnnotation:
                        newAnnotationFile.write(newSubAnnotation)

                # Writing image to image list
                imageList.write(f"./images/{type}/{image['name']}\n")

                # Creating copy of image
                shutil.copyfile(src=image['path'], dst=f"coco_kpts/images/{image['name']}")




def main():
    if not os.path.isdir('coco_kpts'):
        os.mkdir('coco_kpts')
        os.mkdir('coco_kpts/images')
        os.mkdir('coco_kpts/labels')
        os.mkdir('coco_kpts/labels/val2017')
        os.mkdir('coco_kpts/labels/train2017')

    random.seed(1)

    annotationsPath = 'Annotations'
    imagePath = '/home/aidan/Downloads/Animals_with_Attributes2/JPEGImages'

    with open(f"{annotationsPath}/class_names.txt") as f:
        keyPointNames = f.readlines()
        keyPointNames = [i.strip() for i in keyPointNames]

    # Iterating through each type of animal
    validationAnnotations = []
    trainAnnotations = []
    images = []
    for animalType in os.listdir(annotationsPath):

        annotations = []
        if animalType in ['class_names.txt', 'Animal_Class.txt', 'skeleton.pkl']:
            continue

        for annotation in os.listdir(f"{annotationsPath}/{animalType}"):
            annotations.append({'path': f"{annotationsPath}/{animalType}/{annotation}", "name": annotation})

        try:
            for image in os.listdir(f"{imagePath}/{animalType}"):
                images.append({'path': f"{imagePath}/{animalType}/{image}", 'name': image})
        except FileNotFoundError:
            continue

        # Computing the number of samples we want for the validation set
        numValidationSamplesPerAnimal = int(len(annotations) * 0.1)

        # Randomly selecting the validation samples
        validationSampleIndexesPerAnimal = random.sample(range(0, len(annotations)), numValidationSamplesPerAnimal)

        # Getting the training sample indexes
        trainSampleIndexesPerAnimal = [i for i in range(0, len(annotations)) if i not in validationSampleIndexesPerAnimal]

        validationAnnotationsPerAnimal = [annotations[i] for i in validationSampleIndexesPerAnimal]
        trainAnnotationsPerAnimal = [annotations[i] for i in trainSampleIndexesPerAnimal]

        validationAnnotations += validationAnnotationsPerAnimal
        trainAnnotations += trainAnnotationsPerAnimal

    create_yolo_data(annotations=trainAnnotations, images=images, type='train', keypointNames=keyPointNames)
    create_yolo_data(annotations=validationAnnotations, images=images, type='val', keypointNames=keyPointNames)


if __name__ == '__main__':
    main()
