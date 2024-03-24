import cv2
import numpy as np

# Fundamental Matrix Estimation
#https://scikit-image.org/docs/stable/auto_examples/transform/plot_fundamental_matrix.html
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform,EssentialMatrixTransform

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Extractor(object):
    def __init__(self, K):
        self.GX = 16//2
        self.GY = 12//2
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.orb = cv2.ORB_create() 
        self.last = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) # brute force matcher - used for matching the features of the first image with another image.

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]
    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        print(ret)
        return int(round(ret[0])), int(round(ret[1])) 
    
    def extract(self, img):

        # cv2.detect had some problems here - features were only visible within small grid like spaces
        """
        sy = img.shape[0]//self.GY
        sx = img.shape[1]//self.GX
        akp = []
        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                # keypoints and descriptors
                img_chunk = img[ry:ry+sy , rx: rx+sx]
                # print(img_chunk.shape)
                # kp, des = self.orb.detectAndCompute(img_chunk,None)
                kp = self.orb.detect(img_chunk, None)
                for p in kp:
                    print(p.pt)
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    # u, v = map(lambda x: int(round(x)), p.pt)
                    # cv2.circle(img,(u,v), color=(0,255,0), radius = 3)
                    akp.append(p)
        return akp
        """
        #detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        #extraction - keyopints, descriptors
        kps = [cv2.KeyPoint(x=f[0][0] ,y= f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        #matching - not perfect here, some stupid noise, but most is gone, mostly the courtesy of ransac+fundamental matrix transform
        ret =[]
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last["des"], k=2) # basically matching prev & current keypoints
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last["kps"][m.trainIdx].pt
                    ret.append((kp1, kp2))
            print((f"{len(matches)} matches"))

        # filter - find good points among lots of noisy points too - using RANSAC
        if len(ret) > 0 :
            ret = np.array(ret)
            print(ret.shape)
            print(ret[:,0].shape)

            # normalize coordinates : subtract to move to 0
            ret[:,0,:] = self.normalize(ret[:,0,:])
            ret[:,1,:] = self.normalize(ret[:,1,:])

            # Ransac algo - used to fit model efficiently in presense of excessive noise and wrong data points
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform, 
                                    min_samples=8,
                                    residual_threshold=1, 
                                    max_trials=100)

            # notice difference without the inliers - try commenting it
            ret = ret[inliers]
        self.last = {"kps":kps, "des":des}

        # print(des)

        # We do this for DMatch - for maching keypoint descriptors
        return ret