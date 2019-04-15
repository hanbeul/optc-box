const cv = require('opencv4nodejs');

let imageName = process.argv[2];
let img1 = cv.imread('./images/' + imageName + '.png');
let img2 = cv.imread('./images/box.jpeg');
img1 = img1.cvtColor(cv.COLOR_BGR2GRAY);
img2 = img2.cvtColor(cv.COLOR_BGR2GRAY);

const KP_THRESH = .20
const OBJ_THRESH = 15

// let SIFT = new cv.SIFTDetector();
let SURF = new cv.SURFDetector();
// let BRISK = new cv.BRISKDetector();

// let kp1 = SIFT.detect(img1);
// let kp2 = SIFT.detect(img2);
// let desc1 = SIFT.compute(img1, kp1);
// let desc2 = SIFT.compute(img2, kp2);

let kp1 = SURF.detect(img1);
let kp2 = SURF.detect(img2);
let desc1 = SURF.compute(img1, kp1);
let desc2 = SURF.compute(img2, kp2);

// let kp1 = BRISK.detect(img1);
// let kp2 = BRISK.detect(img2);
// let desc1 = BRISK.compute(img1, kp1);
// let desc2 = BRISK.compute(img2, kp2);

// bf = new cv.BFMatcher(cv.NORM_HAMMING, crossCheck=true);

let start = process.hrtime();

let matches = cv.matchFlannBased(desc1, desc2);

// let matches = bf.match(desc1, desc2);
// matches = bf.match(desc1, desc2);

console.log(matches);
console.log('');

let goodMatches = [];
matches.forEach(match => {
  if (match.distance < KP_THRESH)
    goodMatches.push(match);
});

if (goodMatches.length > OBJ_THRESH) 
  console.log('Object Found!');
else
  console.log('Object not found.');
console.log('%d matches found', goodMatches.length);

let end = process.hrtime(start);
console.log('%d.%ds', end[0], end[1]);

// let result = cv.drawMatches(img1, img2, kp1, kp2, goodMatches);
//
// cv.imshow('Matching keypoints', result);
// cv.waitKey();
