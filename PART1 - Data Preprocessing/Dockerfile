# DEFINE THE CONTAINER INIT
FROM python:latest
LABEL maintainer="jrauljperez02.dev@gmail.com"

# DEFINE THE MAIN PATH
WORKDIR /usr/app/src

# INSTALL ALL THE DEPENDENCIES INSIDE THIS PROJECT
COPY requirements.txt ./
RUN pip install -r requirements.txt

# COPY THE Data.csv and move it to the root directory
COPY Data.csv ./

# COPY THE ROOT OF THE MAIN APP
COPY app.py ./

# RUN THE PROJECT
CMD [ "python", "./app.py" ]