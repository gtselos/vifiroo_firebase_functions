//import * as functions from 'firebase-functions';
import * as express from 'express';
import * as path from 'path';
import * as multer from 'multer';
import * as bodyPix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs-node';
import * as Jimp from 'jimp';


// // Start writing Firebase Functions
// // https://firebase.google.com/docs/functions/typescript
//
// export const helloWorld = functions.https.onRequest((request, response) => {
//  response.send("Hello from Firebase!");
// });

const port = 3000
const app = express()
let net: bodyPix.BodyPix
   
const UPLOAD_PATH = 'uploads';
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
      cb(null, `${UPLOAD_PATH}/`)
    },
    filename: function (req, file, cb) {
      cb(null, Date.now() + path.extname(file.originalname)) //Appending extension
    }
  })
  
  var upload = multer({ storage: storage });


// define a route handler for the default home page
app.get( "/", ( req: any, res: any ) => {
    // render the index template
    res.sendFile(path.join(__dirname+'/html/index.html'));
} );

app.post('/api/upload', upload.single('uploadedPic'), async (req, res) => {
    const file = req.file;
    if (!file) {
      const error = new Error('Please upload a file')
      res.status(400);
      res.send('error: ' + error);
      return;
    }
    let img = await readFile(req.file.path);
    let tensorImage = uint8ArrayToTensorImage(img);
    const segmentation = await net.segmentPersonParts(tensorImage);
    console.log(segmentation);
    res.status(200);
    res.send('File uploaded!');
  })

// start the express server
app.listen( port, async () => {
    // tslint:disable-next-line:no-console
    net = await bodyPix.load({
        architecture: 'ResNet50',
        outputStride: 16,
        quantBytes: 2
      });
    console.log( `server started at http://localhost:${ port }` );
} );

async function readFile(tempFilePath: string) : Promise<Uint8Array> {
    let image = await Jimp.read(tempFilePath);
    let imageToUint8Array  = await image.getBufferAsync(getImageType(tempFilePath));
    return imageToUint8Array;
}

function getImageType(imagePath: string) : string {
    let extension = imagePath.split('.').pop();
    switch (extension) {
        case 'jpeg':
            return Jimp.MIME_JPEG;
        case 'jpg':
            return Jimp.MIME_JPEG;
        case 'png':
            return Jimp.MIME_PNG;
        default:
            throw new Error("Not Supported image type");
    }
}

function uint8ArrayToTensorImage(image: Uint8Array) : tf.Tensor3D {
    return tf.node.decodeImage(image) as tf.Tensor3D;
}