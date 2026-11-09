FROM rocker/r-ver:4.3.2

WORKDIR /app

# install packages first so docker can cache this step
COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic_model.R .

# the script looks for train.csv / test.csv here (we mount src/data into this folder when running)
ENV DATA_DIR=/app/data

CMD ["Rscript", "titanic_model.R"]
