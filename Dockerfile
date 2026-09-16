# syntax=docker/dockerfile:1
FROM python:3.13

RUN python3 -m ensurepip
RUN pip3 install --no-cache ortools
RUN pip3 install --no-cache grpcio grpcio-tools
RUN pip3 install --no-cache fastapi uvicorn
RUN pip3 install --no-cache attrs
RUN pip3 install --no-cache black
RUN pip3 install --no-cache certifi
RUN pip3 install --no-cache charset-normalizer
RUN pip3 install --no-cache cycler
RUN pip3 install --no-cache dateparser
RUN pip3 install --no-cache idna
RUN pip3 install --no-cache kiwisolver
RUN pip3 install --no-cache kubernetes
RUN pip3 install --no-cache matplotlib
RUN pip3 install --no-cache mysql-connector
RUN pip3 install --no-cache mysql-connector-python
RUN pip3 install --no-cache networkx
#RUN pip3 install --no-cache numpy==1.26.2
RUN pip3 install --no-cache pandas
#RUN pip3 install --no-cache Pillow
#RUN pip3 install --no-cache elasticsearch==8.17.2
RUN pip3 install --no-cache pylint
RUN pip3 install --no-cache pyparsing
RUN pip3 install --no-cache python-dateutil
RUN pip3 install --no-cache pytz
RUN pip3 install --no-cache regex
RUN pip3 install --no-cache requests
RUN pip3 install --no-cache shortuuid
RUN pip3 install --no-cache six
RUN pip3 install --no-cache tabulate
RUN pip3 install --no-cache tzlocal
RUN pip3 install --no-cache urllib3
RUN pip3 install --no-cache google-api-python-client

ENV PYTHONPATH="/usr/local/ramite2_optimizations/:/usr/local/ramite2_optimizations/ramite:."