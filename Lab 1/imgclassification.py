#!/usr/bin/env python

##############
#### Your name: Aryan Shah
##############

import numpy as np
import re, math
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
import ransac_score

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        features = []
        for d in data:
            grayscale = color.rgb2gray(d)
            filtered = filters.gaussian(grayscale, sigma = 0.5)
            equalize = exposure.equalize_hist(filtered)
            fd = feature.hog(equalize)
            features.append(fd)
        feature_data = np.array(features, dtype = object)
        ########################
        # Please do not modify the return type below
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier
        ########################
        self.classifier = svm.SVC()
        self.classifier.fit(train_data, train_labels)
        ########################
        pass

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        predicted_labels = self.classifier.predict(data)
        ########################

        # Please do not modify the return type below
        return predicted_labels
 
    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        ########################
        slope = []
        intercept = []
        for d in data:
            grayscale = color.rgb2gray(d)
            filtered = filters.gaussian(grayscale, sigma = 5)
            canny = feature.canny(filtered)
            edges = np.where(canny)
            x = edges[1]
            y = edges[0]
            lines = []
            slopes = []
            intercepts = []
            sums = []
            for i in range(50):
                j = np.random.choice(len(x) - 1)
                k = np.random.choice(len(x) - 1)
                while(j == k):
                    k = np.random.randint(len(x) - 1)
                x2 = [x[j], x[k]]
                y2 = [y[j], y[k]]
                fit_line = np.polyfit(x2, y2, 1)
                lines.append(fit_line)
                slopes.append(fit_line[0])
                intercepts.append(fit_line[1])
            for j in range(len(slopes)):
                sum = 0
                for k in range(len(x)):
                    distance = (abs(slopes[j]*x[k] - (1*y[k]) + intercepts[j]) / np.sqrt(np.square(slopes[j]) + 1))
                    sum += distance
                sums.append(sum)
            minIndex = sums.index(min(sums))
            slope.append(slopes[minIndex])
            intercept.append(intercepts[minIndex])
        ########################

        # Please do not modify the return type below
        return slope, intercept

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))
    
    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()