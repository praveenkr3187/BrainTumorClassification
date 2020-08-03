from flask import *  
import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
import skimage.feature
from scipy.stats import kurtosis, skew
import scipy as sc
import warnings
import cv2
import time
import random
warnings.filterwarnings('ignore')

from prediction import preprocess, predict 

app = Flask(__name__)  
 
app.config['UPLOAD_FOLDER'] = './static'

@app.route('/')  
def upload():  
    return render_template("index.html")  


def returnfeature(img_2d_scaled):
  g = skimage.feature.greycomatrix(img_2d_scaled, [1], [0], levels=256, symmetric=True, normed=True)
  contrast=skimage.feature.greycoprops(g, 'contrast')[0][0]
  energy=skimage.feature.greycoprops(g, 'energy')[0][0]
  homogeneity=skimage.feature.greycoprops(g, 'homogeneity')[0][0]
  correlation=skimage.feature.greycoprops(g, 'correlation')[0][0]
  dissimilarity=skimage.feature.greycoprops(g, 'dissimilarity')[0][0]
  ASM=skimage.feature.greycoprops(g, 'ASM')[0][0]

  feat_lbp=local_binary_pattern(img_2d_scaled,8,1,'uniform')
  feat_lbp=np.uint8((feat_lbp/feat_lbp.max())*255)

  lbp_hist,_=np.histogram(feat_lbp,8)
  lbp_hist=np.array(lbp_hist,dtype=float)
  lbp_prob=np.divide(lbp_hist,np.sum(lbp_hist))
  lbp_energy=np.sum(lbp_prob**2)
  lbp_entropy=-np.sum(np.multiply(lbp_prob,np.log2(lbp_prob)))

  gaborFilt_real,gaborFilt_imag=gabor(img_2d_scaled,frequency=0.6)
  gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2

  gabor_hist,_=np.histogram(gaborFilt,8)
  gabor_hist=np.array(gabor_hist,dtype=float)
  gabor_prob=np.divide(gabor_hist,np.sum(gabor_hist))
  gabor_energy=np.sum(gabor_prob**2)
  gabor_entropy=-np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))

  Kurtosis=kurtosis(img_2d_scaled,axis=None)

  Skew=skew(img_2d_scaled, axis=None)

  Average_smoothness = sc.average(sc.absolute(nd.filters.laplace(img_2d_scaled.astype(float) / 255.0)))

  Std=np.std(img_2d_scaled)

  Mean=np.mean(img_2d_scaled)

  feature_inner=[contrast,energy,homogeneity,correlation,dissimilarity,ASM,lbp_energy,lbp_entropy,gabor_energy,gabor_entropy,Kurtosis,Skew,Average_smoothness,Std,Mean]
  featurename_list=['Contrast','Energy','homogeneity','correlation','dissimilarity','ASM','lbp_energy','lbp_entropy','gabor_energy','gabor_entropy','Kurtosis','Skew','Average_smoothness','Std','Mean']
  d={}
  for i in range(len(feature_inner)):
    name=featurename_list[i]
    d[name]=feature_inner[i]
  return(d)


 
@app.route('/handle_form', methods=['POST'])
def handle_form():
    print("Posted file: {}".format(request.files['file']))
    file = request.files['file']
    file.save(file.filename)
    path = file.filename
    img, mat = preprocess(path)
    _ = plt.imshow(mat,cmap='gray');
    _ = plt.savefig('./static/foo.jpg');
    _ = plt.close();
    ret, thresh1 = cv2.threshold(mat, 90, 255, cv2.THRESH_BINARY )  
    _ = plt.imshow(thresh1,cmap='gray');   
    _ = plt.savefig('./static/goo.jpg');
    _ = plt.close();                                       
    pred = predict(img)
    features = (returnfeature(mat))
    Contrast = features['Contrast']
    Energy = features['Energy']
    Correlation = features['correlation']
    ASM = features['ASM']
    Kurtosis = features['Kurtosis']
    Skew = features['Skew']
    Average_smoothness = features['Average_smoothness']
    Std = features['Std']
    Mean = features['Mean']
    srcfoo = f'foo.jpg?{random.random()}' 
    srcgoo = f'goo.jpg?{random.random()}' 
    if pred == 0:
        pred = 'meningioma'
    elif pred == 1:
        pred = 'glioma'
    elif pred == 2:
        pred = 'pituitary tumor'
    else:
        pred = -1
    return render_template('result.html',srcfoo=srcfoo,srcgoo=srcgoo,pred = pred,Contrast = Contrast,Energy = Energy,Correlation = Correlation,ASM = ASM,Kurtosis = Kurtosis,Skew = Skew,Average_smoothness = Average_smoothness,Std = Std,Mean = Mean)

@app.route('/test')
def test():
    return render_template('result.html')

@app.route('/definition')
def definition():
    return render_template('definitions.html')

@app.route('/contact')
def contact():
    return render_template('contacts.html')

if __name__ == '__main__':  
    app.run(debug = True)  