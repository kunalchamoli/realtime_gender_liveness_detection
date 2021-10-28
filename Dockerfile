FROM python:3.6

#update
RUN apt-get update

COPY ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip3 install -r requirements.txt

COPY . /real_time_detection

WORKDIR /real_time_detection

CMD ["python3" , "main.py"]
