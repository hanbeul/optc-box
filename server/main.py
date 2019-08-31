import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

MIN_MATCH_COUNT = 100
MIN_MATCH = 10

def readImages():
  #img1 = cv2.imread('images/lucy.png')
  #img1 = cv2.imread('images/big_mom.png')
  #img1 = cv2.imread('images/warco.png')
  #img1 = cv2.imread('images/coli_lucy.png')
  img2 = cv2.imread('images/box_half.jpeg')

  img1 = cv2.resize(img1, (64, 64))
  img2_ratio = img2.shape[0] / img2.shape[1]
  img2 = cv2.resize(img2, (64*6, int(64*6*img2_ratio)))

  return img1, img2

def main():
  img1, img2 = readImages()
  ## Create ORB object and BF object(using HAMMING)
  orb = cv2.ORB_create()
  brisk = cv2.BRISK_create()

  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  ## Find the keypoints and descriptors with ORB
  kpts1, descs1 = brisk.detectAndCompute(gray1,None)
  kpts2, descs2 = brisk.detectAndCompute(gray2,None)

  ## match descriptors and sort them in the order of their distance
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  start = time.time()
  matches = bf.match(descs1, descs2)
  end = time.time()
  print(round(end - start, 4), 'sec')

  good = []
  for m in matches:
    #print(m.distance)
    if m.distance < MIN_MATCH_COUNT:
      good.append(m)
  
  print(len(good), 'matching features...')
  if len(good) > MIN_MATCH:
    print('Object found!')
  else:
    print('Object not found!')


  ## extract the matched keypoints
  #src_pts  = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
  #dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

  ## find homography matrix and do perspective transform
  #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
  #h,w = img1.shape[:2]
  #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
  #dst = cv2.perspectiveTransform(pts,M)

  ## draw found regions
  #img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
  #cv2.imshow("found", img2)

  ## draw match lines
  res = cv2.drawMatches(img1, kpts1, img2, kpts2, good, None,flags=2)
  kpi = cv2.drawKeypoints(img1, kpts1, None)
  cv2.imshow("orb_match", res)

  cv2.waitKey();cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
