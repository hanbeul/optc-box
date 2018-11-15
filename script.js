let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');
let templateElement = document.getElementById('templateInput');
let templateImg = document.getElementById('templateSrc');

inputElement.addEventListener('change', (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

templateElement.addEventListener('change', (e) => {
  templateImg.src = URL.createObjectURL(e.target.files[0]);
}, false);

imgElement.onload = function() {
  let mat = cv.imread(imgElement);
  cv.imshow('canvasOutput', mat);
  mat.delete();
  searchForBorders();
  detectSquares();
};

templateImg.onload = function() {
  if (imgElement.src != '') {
    searchForTemplate();
  };
};

function searchForTemplate() {
  console.log("Searching for Template!");
  let src = cv.imread(imgElement);
  let template = cv.imread(templateImg);
  let output = new cv.Mat();
  let mask = new cv.Mat();
  cv.matchTemplate(src, template, output, cv.TM_CCOEFF, mask);
  let result = cv.minMaxLoc(output, mask);
  let maxPoint = result.maxLoc;
  let color = new cv.Scalar(255, 0, 0, 255);
  let point = new cv.Point(maxPoint.x + template.cols, maxPoint.y + template.rows);
  cv.rectangle(src, maxPoint, point, color, 2, cv.LINE_8, 0);
  cv.imshow('canvasOutput', src);
  src.delete(); 
  template.delete();
  output.delete();
  mask.delete();

};

function searchForBorders() {
  console.log("Searching for borders!");
  let src = cv.imread(imgElement);
  let processedImg = new cv.Mat();

  let low = new cv.Mat(src.rows, src.cols, src.type(), [242, 215, 112, 250]);
  let high = new cv.Mat(src.rows, src.cols, src.type(), [243, 216, 113, 255]);

  cv.inRange(src, low, high, processedImg);

  cv.imshow('borderDetect', processedImg);
  src.delete();
}

function detectSquares() {
  console.log('Detecting squares!');
  let src = cv.imread(imgElement);
  let gray = new cv.Mat();
  let processedImg = new cv.Mat();

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
  cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

  cv.imshow('squareDetect', gray);
  src.delete();
}

function onOpenCvReady() {
  document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
};
