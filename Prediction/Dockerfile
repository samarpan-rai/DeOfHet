FROM python:3.7-alpine

ENV SPACY_VERSION "2.3.0"

ENV LANG nl_core_news_lg
ENV PORT 3000

ENV SPACY_LOG_LEVEL error

RUN apk update && apk add --no-cache python3 tini bash libgomp && \
    apk add --no-cache --virtual .build-deps \
        build-base \
        python3-dev \
        g++



# Set up spacy

RUN python3 -m pip install spacy==${SPACY_VERSION}

RUN python3 -m spacy download ${LANG} && \
    pip show spacy > /etc/spacy_info



COPY requirements.txt /

RUN pip install -r /requirements.txt

RUN apk del .build-deps \
        build-base \
        subversion \
        python3-dev \
        g++ && \

    rm -r /usr/lib/python*/ensurepip && \
    rm -r /root/.cache


COPY src/ /app/src

COPY model/ /app/model

WORKDIR /app



# Call hugs
CMD [ "hug", "-f","src/predict.py"]
