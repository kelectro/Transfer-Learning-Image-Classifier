let net;
//crreate webcam element
const webcamElement = document.getElementById('webcam');
//create the classifier
const classifier = knnClassifier.create();

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');
//make prediction through the webcam
  await setupWebcam();

//Now we will read an image via the webcam and we will associate it with
//a specific class
const addExample=classId => {
  //get intermediate activation of mobile net 'conv_preds' and pass it to knn
  const activation =net.infer(webcamElement,'conv_preds');
//pass activation to knnClassifier
classifier.addExample(activation,classId);
};

//add an example with the button click
document.getElementById('class-a').addEventListener('click',() => addExample(0));
document.getElementById('class-b').addEventListener('click',() => addExample(1));
document.getElementById('class-c').addEventListener('click',() => addExample(2));


//infinite loop to receive and analyze every frame
while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}

      `;
    }
    // document.write("<hr>");
    // document.write("welcome");
    // document.write("<hr>");
    //



    //   document.getElementById("demo").innerHTML = "Good day!";


    if (result.confidences[result.classIndex]>0.1) {
    document.write("<hr>");
    document.write("welcome");
    document.write("<hr>");
  }
    await tf.nextFrame();
  }
}
app();
