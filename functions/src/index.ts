//import * as functions from 'firebase-functions';
import * as express from 'express';
import * as fs from 'fs';
import * as path from 'path';
import * as multer from 'multer';
import * as Jimp from 'jimp';
import * as Canvas from 'canvas';
import * as bodyPix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs-node';
import { SemanticPartSegmentation } from '@tensorflow-models/body-pix/dist/types';


// // Start writing Firebase Functions
// // https://firebase.google.com/docs/functions/typescript
//
// export const helloWorld = functions.https.onRequest((request, response) => {
//  response.send("Hello from Firebase!");
// });
var RAINBOW_PART_COLORS = [
    [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
    [238, 67, 149], [255, 78, 125], [255, 94, 99], [255, 115, 75],
    [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
    [175, 240, 91], [135, 245, 87], [96, 247, 96], [64, 243, 115],
    [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
    [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
];

const port = 3000;
const app = express();
const DELETE_ORIGINAL = true;
let net: bodyPix.BodyPix;
   
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
    var jimpImage = await readFile(req.file.path);
    let img = await jimpToUint8Array(jimpImage, req.file.path);
    let tensorImage = uint8ArrayToTensorImage(img);
    const segmentation = await net.segmentPersonParts(tensorImage,{
        flipHorizontal: false,
        internalResolution: 'high',
        segmentationThreshold: 0.7,
        maxDetections: 1,
      });
    let maskFilePath = await createMaskImage(segmentation, jimpImage.bitmap.width, jimpImage.bitmap.height);
    addMaskToOriginal(jimpImage, await Jimp.read(maskFilePath));
    if (DELETE_ORIGINAL) {
        fs.unlinkSync(req.file.path);
    }
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

async function readFile(tempFilePath: string) : Promise<Jimp> {
    let image = await Jimp.read(tempFilePath);
    return image;
}

async function jimpToUint8Array(image: Jimp, tempFilePath: string) : Promise<Uint8Array> {
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

async function createMaskImage(segmentation: SemanticPartSegmentation, width: number, height: number) : Promise<string> {
    let outputFilePath = 'output/test.jpeg';
    const coloredPartImageData = toColoredPartMask(segmentation, RAINBOW_PART_COLORS);
    if (coloredPartImageData == null) {
        throw Error("Mask failed");
    }
    const canvas = Canvas.createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(coloredPartImageData, 0, 0);
    const out = fs.createWriteStream(outputFilePath);
    const stream = canvas.createJPEGStream()
    stream.pipe(out);
    let end = new Promise<string>(function(resolve, reject) {
        out.on('finish', () => {
            console.log('The JPEG mask file was created.');
            resolve(outputFilePath);
        });
    });
    return end;
}

function toColoredPartMask(partSegmentation: SemanticPartSegmentation, partColors: number[][]) {
    if (Array.isArray(partSegmentation) && partSegmentation.length === 0) {
        return null;
    }
    var multiPersonPartSegmentation;
    if (!Array.isArray(partSegmentation)) {
        multiPersonPartSegmentation = [partSegmentation];
    }
    else {
        multiPersonPartSegmentation = partSegmentation;
    }
    var _a = multiPersonPartSegmentation[0], width = _a.width, height = _a.height;
    var bytes = new Uint8ClampedArray(width * height * 4);
    for (var i = 0; i < height * width; ++i) {
        var j = i * 4;
        bytes[j + 0] = 255;
        bytes[j + 1] = 255;
        bytes[j + 2] = 255;
        bytes[j + 3] = 255;
        for (var k = 0; k < multiPersonPartSegmentation.length; k++) {
            var partId = multiPersonPartSegmentation[k].data[i];
            if (partId !== -1) {
                var color = partColors[partId];
                if (!color) {
                    throw new Error("No color could be found for part id " + partId);
                }
                bytes[j + 0] = color[0];
                bytes[j + 1] = color[1];
                bytes[j + 2] = color[2];
                bytes[j + 3] = 255;
            }
        }
    }
    return Canvas.createImageData(bytes, width, height);
}

function addMaskToOriginal(original: Jimp, mask: Jimp) : void {
    original.composite(mask, 0, 0,
       {
         mode: Jimp.BLEND_SCREEN,
         opacitySource: 0.5,
         opacityDest: 1
       },
       (err, img) => {
           if (err) {
               throw err;
           }
           img.write('output/test.jpeg');
           console.log('The JPEG output file was created.');
       }
   );
}