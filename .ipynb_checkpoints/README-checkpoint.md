# Content-Based Netflix Recommendation System 

## Dataset
* **Source:** The dataset used for this project is from Kaggle: Netflix Shows Dataset.
* **Description:** The dataset contains information about movies and TV shows available on Netflix, including columns such as title, description, listed_in, type, release_year, and more.
* **Steps to load:**
    * Download the dataset from the provided [Kaggle link](https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download).
    * Place the netflix_titles.csv file in the same directory as your Jupyter Notebook.
    * Load the dataset using pandas:
          df = pd.read_csv("netflix_titles.csv")

## Setup
1. **Python Version:** This project is compatible with Python 3.8 or higher
2. **Virtual Environment:**
    * Create a virtual environment by running:
          python3 -m venv venv
    * Activate the virtual environment:
        * On macOS:
              source venv/bin/activate
3. **Install Dependencies:**
    * Install the required libraries by running:
      pip install -r requirements.txt
    * Example requirements.txt:
          pandas
          scikit-learn
          numpy
          jupyter

## Running the Code in Jupyter Notebook
1. Start Jupyter Notebook:
    * Once your virtual environment is activated and dependencies are installed, you can start the Jupyter Notebook server by running:
          jupyter notebook
    * This will open the Jupyter interface in your browser.
2. Open the Notebook:
    * In the Jupyter interface, navigate to the notebook where the code is written and open it.
3. Run the Cells:
    * Once the notebook is open, you can run each code cell sequentially. The code will execute in the order the cells are run, and the output will appear directly below each cell.

## Results
**Example query:**
    "I love thrilling action movies set in space, with a comedic twist."

**Sample output:**
```
Recommendations:
title: Incoming
listed_in: action & adventure, sci-fi & fantasy
description: When an imprisoned terrorist cell hijacks a high-security prison in outer space, a CIA agent becomes the one chance of stopping them....
----------------------------------------
title: Lockout
listed_in: action & adventure, international movies, sci-fi & fantasy
description: A government agent wrongly accused of a crime gets a shot at freedom if he can engineer a high-risk rescue mission to outer space....
----------------------------------------
title: Defiance
listed_in: action & adventure, dramas
description: In this action-packed drama based on an extraordinary true story, four brothers protect more than 1,000 Jewish refugees during World War II....
----------------------------------------
title: Star Trek: Deep Space Nine
listed_in: tv action & adventure, tv sci-fi & fantasy
description: In this "Star Trek" spin-off, Commander Sisko leads the multi-species crew of Deep Space Nine, a Federation space station with a complex mission....
----------------------------------------
title: Small Soldiers
listed_in: action & adventure, comedies, sci-fi & fantasy
description: When the Commando Elite, a group of toy action figures, are released before they've been tested, they attack the children playing with them....
----------------------------------------
```
