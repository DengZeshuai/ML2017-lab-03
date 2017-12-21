from PIL import Image
import os
from feature import NPDFeature
import numpy  as np
import pickle
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

def preprocess(face_path, nonface_path):
    image_list = []
    for image in os.listdir(face_path):
        if image.endswith('.jpg'):
            image_gray_scaled = Image.open(os.path.join(face_path, image)).convert('L').resize((24,24),Image.ANTIALIAS)
            image_list.append(np.array(image_gray_scaled))
    
    for image in os.listdir(nonface_path):
        if image.endswith('.jpg'):
            image_gray_scaled = Image.open(os.path.join(nonface_path, image)).convert('L').resize((24,24),Image.ANTIALIAS)
            image_list.append(np.array(image_gray_scaled))

    return image_list, labels

def  extract_features(image_list):
    for index, image in enumerate(image_list):
        image_list[index] = NPDFeature(image).extract()
    return image_list


def  main(cache_file, max_depth=2, n_weakers_limit=10, epoch=10):
    # dump data to cache
    if not open(cache_file, "rb"):
        #preprocess the image to gray image and resize to 24*24
        image_list = preprocess('datasets/original/face', 'datasets/original/nonface')
        print("dump data to cache!")
        data = extract_features(image_list)
        AdaBoostClassifier.save(data ,'data_cache')
        
    # load data from cache
    samples = AdaBoostClassifier.load(cache_file)
    
    labels = np.ones((len(samples),1))
    for i in range(int(len(samples)/2),len(samples)):
        labels[i] = -1

    # preprocess data to ndarray
    X, y = np.array(samples), np.array(labels).reshape(len(samples),1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.4,random_state=33)
    
    model = AdaBoostClassifier(DecisionTreeClassifier, n_weakers_limit)
    model.fit(X_train,y_train, max_depth)
    
    predict_score_of_Adaboost = model.predict_scores(X_train, y_train)
    print("Train accuracy of Adaboost", predict_score_of_Adaboost)
    predict_score_of_Adaboost = model.predict_scores(X_test, y_test)
    print("Test accuracy of Adaboost", predict_score_of_Adaboost)
    
    print("Finish!")


if __name__ == "__main__":
    # write your code here
    hyperparameter ={
        'max_depth': 1,
        "n_weakers_limit":10,
        "cache_file":"data_cache"
    }
    main(**hyperparameter)






