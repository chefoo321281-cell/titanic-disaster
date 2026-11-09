# packages the model script needs
install.packages(c("readr", "dplyr"))

# fail the docker build loudly if something didn't install
stopifnot(all(c("readr", "dplyr") %in% rownames(installed.packages())))
