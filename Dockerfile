FROM tensorflow/tensorflow:latest-gpu  
USER root
RUN apt-get update -y && apt-get upgrade -y 
# && apt install libgl1-mesa-dev -y

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install opencv-python sklearn pillow tensorflow_hub
 
# CMD [ "python3", "opt/main.py" ]