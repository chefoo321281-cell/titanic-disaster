# titanic-disaster

Predicting who survived the Titanic with logistic regression. I did it twice: once in **Python** and once in **R**, each one running inside its own **Docker** container.

The data is NOT in this repo (the assignment says not to include it), so you have to download it first. It takes about a minute, steps below.

## What you need

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed **and open/running** (check that the whale icon is there)
- A free [Kaggle](https://www.kaggle.com) account (only for downloading the data)

You do NOT need Python or R installed, Docker handles everything.

## What's in the repo

```
titanic-disaster/
├── .gitignore
├── README.md
└── src/
    ├── data/              <- you create this, it's empty at first (and gitignored)
    ├── python_model/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r_model/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

## Step 1: Clone the repo

```bash
git clone <https://github.com/chefoo321281-cell/titanic-disaster>
cd titanic-disaster
```

## Step 2: Download the data

1. Go to https://www.kaggle.com/competitions/titanic/data and log in.
2. Click **Download All** (you might need to click "Join Competition" / accept the rules first).
3. Unzip it. You only need two files: `train.csv` and `test.csv`.
4. Make the data folder and put both files in it:

```bash
mkdir -p src/data
```

Now `src/data/` should contain exactly:

```
src/data/train.csv
src/data/test.csv
```

(The other files in the Kaggle zip, like `gender_submission.csv`, aren't needed.)

## Step 3: Run the Python version

Run these from the **main folder** of the repo (the one with this README):

```bash
docker build -t titanic-python -f src/python_model/Dockerfile src/python_model
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

The first build takes a minute or two (it's installing packages). After that it's fast.

## Step 4: Run the R version

Same deal, again from the main folder:

```bash
docker build -t titanic-r -f src/r_model/Dockerfile src/r_model
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

The R build is slower the first time (maybe 3-5 minutes), that's normal, it's installing R packages.

**Windows users:** in PowerShell, swap `$(pwd)` for `${PWD}`, like this:

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

## What you should see

Each script prints its progress in 8 steps: loading data, a quick look at it, splitting, cleaning/new features, training, training accuracy, predictions on `test.csv`, and test-side accuracy. At the end it says `All done!`.

It also saves a predictions file into `src/data/` (`predictions_python.csv` or `predictions_r.csv`). Those are in Kaggle submission format and they're gitignored with the rest of the data folder.

## About the model

- Plain logistic regression (scikit-learn in Python, `glm` in R).
- Features: passenger class, sex, age, fare, port of embarkation, plus three I made up: family size, traveling alone, and is-a-kid (under 12).
- Missing ages/fares get filled with the median, missing ports with the most common one.
- **Heads up about "test accuracy":** Kaggle's `test.csv` doesn't come with the answers (no `Survived` column), so nobody can measure accuracy on it directly. What I do instead: I hold out 20% of `train.csv`, don't let the model see it, and report accuracy on that. I still run predictions on the real `test.csv` and save them. If you want a true score for those, upload the predictions file to Kaggle.
- Expect roughly 80% accuracy on the hold-out. It'll be a little different between Python and R because they split the data differently.

## Troubleshooting

- **"Cannot connect to the Docker daemon"** -> Docker Desktop isn't running. Open it and wait for it to finish starting.
- **Script says it can't find train.csv / test.csv** -> the files aren't in `src/data/`, or the names are different. Check the folder, and make sure you ran the commands from the main repo folder.
- **Weird "no such file" error on `docker build`** -> same thing, you're probably not in the main repo folder. Run `pwd` (or `cd` on Windows) to check.
