import numpy as np
import cv2
import matplotlib.pyplot as plt


img_path = 'images/tree.jpg'
# load the input image and grab the image dimensions
image = cv2.imread(img_path)
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
resized = cv2.resize(image,(224,224))
blob = cv2.dnn.blobFromImage(resized,1, (224, 224), (104, 117, 123))

# load the class labels from disk
rows = open("Models/synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe("Models/bvlc_googlenet.prototxt",
	"Models/bvlc_googlenet.caffemodel")
net.setInput(blob)
preds = net.forward()

# sort the probabilities (in descending) order, grab the index of the
# top predicted label, and draw it on the input image
number_of_Labels = 5
idx = np.empty([number_of_Labels])
text = np.empty([number_of_Labels])
print(idx)
for i in range(0,number_of_Labels-1):
    idx = np.argsort(preds[0])[::-1][i]
    print(idx)
    text = "Label: {}, {:.2f}%".format(classes[int(idx)],
                                    preds[0][int(idx)] * 100)
    plt.text(0,-30*i, text,fontsize='large',fontweight='bold',color = 'Green' if i == 0 else 'Red')

plt.imshow(image2)
plt.show()
