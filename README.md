# Face Identification

## Description

Fine tune the pre-trained vggface2 FaceNet model, Classifiy faces to indentify faces in the images, with fine tuned model.

## Getting Started

### Dependencies

```
pip install facenet-pytorch tensorboard matplotlib opencv-python
```

### Executing program

* Iterate raw_images to get the cropped faces.

```
python crop.py
```

* Fine tune the ResNet model

```
python tune.py
```

* Classify faces
```
python classifiy.py
```

* Identify faces in the image
```
python identify.py
```

## Authors

Contributors names and contact info

[Yancy Qin](https://www.linkedin.com/in/yancyqin/)

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

* [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
