//import * as functions from 'firebase-functions';
import * as express from 'express';
import * as path from 'path';
import * as multer from 'multer';

// // Start writing Firebase Functions
// // https://firebase.google.com/docs/functions/typescript
//
// export const helloWorld = functions.https.onRequest((request, response) => {
//  response.send("Hello from Firebase!");
// });

const port = 3000;
const app = express();
   
const UPLOAD_PATH = 'uploads';
const upload = multer({ dest: `${UPLOAD_PATH}/` });


// define a route handler for the default home page
app.get( "/", ( req: any, res: any ) => {
    // render the index template
    res.sendFile(path.join(__dirname+'/html/index.html'));
} );

app.post('/api/upload', upload.single('uploadedPic'), (req, res) => {
    const file = req.file;
    if (!file) {
      const error = new Error('Please upload a file')
      res.status(400);
      res.send('error: ' + error);
      return;
    }
    res.status(200);
    res.send('File uploaded!');
  })

// start the express server
app.listen( port, () => {
    // tslint:disable-next-line:no-console
    console.log( `server started at http://localhost:${ port }` );
} );