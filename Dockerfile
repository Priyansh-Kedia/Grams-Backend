FROM python:3.7-buster

RUN apt-get update && apt-get install nginx vim -y --no-install-recommends

RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV DockerHOME=/usr/src/app

RUN mkdir -p $DockerHOME  
WORKDIR $DockerHOME  


COPY . $DockerHOME 

# set environment variables  
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1  
# install dependencies  
RUN pip install --upgrade pip  

RUN pip install -r requirements.txt  

EXPOSE 8000  

CMD [ "python","manage.py","qcluster", "0.0.0.0:6379"]

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
