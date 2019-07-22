from keras.callbacks import Callback
import cv2
import numpy as np
import itertools, os, time
from Model import get_Model
from parameter import letters
import json
from keras import backend as K
K.set_learning_phase(0)

class TrainCheck(Callback):
    def __init__(self):
        self.epoch = 0
        self.output_path = 0
        self.model_name = 0

    def decode_label(self,out):
    
        out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
        out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
        outstr = ''
        for i in out_best:
            if i < len(letters):
                outstr += letters[i]
        return outstr

    def TrainCheck(self):
        test_dir = './data_japan/test_ep/'
        test_imgs = os.listdir('./data_japan/test_ep/')
        total = 0
        acc = 0
        letter_total = 0
        letter_acc = 0
        start = time.time()
        with open('./data_japan/labels/tes_ep.json') as f:
            data = json.load(f) 
        for test_img in test_imgs:
            img = cv2.imread(test_dir + test_img, cv2.IMREAD_GRAYSCALE)

            img_pred = img.astype(np.float32)
            img_pred = cv2.resize(img_pred, (600, 64))
            img_pred = (img_pred / 255.0) * 2.0 - 1.0
            img_pred = img_pred.T
            img_pred = np.expand_dims(img_pred, axis=-1)
            img_pred = np.expand_dims(img_pred, axis=0)

            net_out_value = self.model.predict(img_pred)

            pred_texts = self.decode_label(net_out_value)

            for i in range(len(data.keys())):
                if test_img == data[str(i)][0]['path'][17:]:
                    true_texts = data[str(i)][0]['class']
                    break 
            for i in range(min(len(pred_texts), len(true_texts))):
                if pred_texts[i] == test_img[i]:
                    letter_acc += 1
            letter_total += max(len(pred_texts), len(true_texts))

            if pred_texts == test_img[0:-4]:
                acc += 1
            total += 1
            print('Predicted: %s  /  True: %s' % (pred_texts, true_texts ))
            
            # cv2.rectangle(img, (0,0), (150, 30), (0,0,0), -1)
            # cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

            #cv2.imshow("q", img)
            #if cv2.waitKey(0) == 27:
            #   break
            #cv2.destroyAllWindows()

        end = time.time()
        total_time = (end - start)
        print("Time : ",total_time / total)
        print("ACC : ", acc / total)
        print("letter ACC : ", letter_acc / letter_total)
