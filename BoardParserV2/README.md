# Checkpoint Report: Preliminaries

## Data collection

For this checkpoint I decided to try and train a computer vision model that can recognize and classify chess pieces for in person, over the board games. I tried to fine a pre-trained YOLOv5 model. I used a dataset I found online which was specifically designed for chess piece detection. The dataset is publicly available [here](https://public.roboflow.com/object-detection/chess-full/23). The dataset consists of images of chess pieces that have been annotated with bounding boxes for each piece. Each piece is also given a class notifying the computer what the piece is (e.g. rook, king, queen, etc...)

I decided to use the YOLOv5 (You Only Look Once) architecture for this since it is well suited for real-time detection tasks. YOLOv5 is a state of the art model which is both extremely efficient and fast. For the project, I used transfer learning by finetuning the pretrained weights for YOLOv5x which allowed me to build off of the already state of the art model and use it for this project. Unfortunately due to the size of the weights I cannot add them to this repo, (The finalized weights have been added to the repo via git lfs). To train the model for yourself you can  cd into the repository directory and run the following command: `py BoardParserV2/yolov5/train.py --weights yolov5x.pt --data chess/data.yaml --device-cpu --batch-size 8`. My finalized model weights can be found under `BoardParserV2/yolov5/runs/train/exp6/weights/best.pt`

## Training Process & Prelimnary Results 
### Training Process
The model was fine tuned for 15 epochs via transfer learning. Throughout the training process I ensured that the order the data was fed to the model was randomized so it would not just end up memorizing images, and was able to generalize. While the model showed steady improvement in its ability to classify chess pieces, there is still room for improvement as there are still quite a bit of misclassification when testing on a live environment. 

### Preliminary Results
The following video shows the preliminary results of the model. ![](preliminary_results.gif)

As we can see from the video, the model is still not very confident in its predictions. The model is confusing pieces for others and is predicting more than one class for some pieces. For example the white king is being predicted as both a white king and a white queen. It makes the most misclassifications with the king and queen, I presume this could be due to my chess set as its a lower end set with minimal distinction between the two pieces.

## Future Improvements

- **More Training**: Given that the model was only trained for 15 epochs and the dataset if fairly small, it is likely that training it with a more robust dataset and for a longer period of time could help reduce the number of misclassification.

- **Data Augmentation**: I plan on introducing more data augmentation techniques since the dataset has the images of the pieces in mostly the same perspective. Adding zoom and random rotations could potentially help the model generalize better
- **Further Functionality**: The final thing I plan on doing is to expand this program to add some functionality to this model so it does the following:

    - **Digital Board**: I want this model to be able to classify pieces for digital boards as well so people can paste screenshots of online games for further analysis

    - **FEN String Conversion**: The final thing I would like to add to this model is the ability to recognize the position of the pieces in relation to the board (e.g. `Classifying rook on a8` rather than `rook detected`) and return a FEN string representation of the board. The fen string can then in the future be fed into a reinforcement learning model that will try and analyse the position.
