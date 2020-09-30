from tensorflow.keras.preprocessing import image

import numpy as np

def test_model(path,classes,model,img_width,img_height):
    img = image.load_img(path,target_size=(img_width,img_height,3))
    img = image.img_to_array(img)
    img = img/255.0

    # plt.imshow(img)
    # Reshaping image
    img = img.reshape(1, img_width, img_height, 3)

    y_prob = model.predict(img)
    top_3_predictions = np.argsort(y_prob[0])[:-4:-1].tolist()

    top_3_classes = []
    
    # Get classes
    for i in range(0,3):
        top_3_classes.append(classes[top_3_predictions[i]])

    return top_3_classes

