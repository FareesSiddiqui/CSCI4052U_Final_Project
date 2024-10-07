# Checkpoint Report: Preliminaries

## Data collection

For this checkpoint I decided to try and train a computer vision model that can recognize and classify chess pieces for in person, over the board games. I tried to fine a pre-trained YOLOv5 model. I used a dataset I found online which was specifically designed for chess piece detection. The dataset is publicly available [here](https://public.roboflow.com/object-detection/chess-full/23). The dataset consists of images of chess pieces that have been annotated with bounding boxes for each piece. Each piece is also given a class notifying the computer what the piece is (e.g. rook, king, queen, etc...)