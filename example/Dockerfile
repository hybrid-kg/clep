FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    r-base \
    r-cran-randomforest \
    python3.7 \
    python3-pip \
    python3-setuptools \
    python3-dev

RUN apt-get update \
	&& apt-get install -t unstable -y --no-install-recommends \
	    apt-utils \
	    openssl \
	    libcurl4-openssl-dev \
	    libxml2-dev

RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile

# Install development tools
RUN R -e 'install.packages("devtools")'

RUN run.sh
