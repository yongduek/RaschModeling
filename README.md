# RaschModeling
Rasch Modeling Experiment

## Project Structure

### Python Scripts

*   **`check_pvals.py`**: Calculates and reports item statistics (specifically p-values/difficulty) for high-ability subgroups, focusing on specific target items to check for potential bias or drift.
*   **`draw_pathway.py`**: Generates a **Bond & Fox Pathway Plot**, which plots Fit Statistics (Infit t) against Measures. This is a diagnostic tool for identifying misfitting items or persons.
*   **`draw_writemap.py`**: Generates a **Wright Map (Variable Map)**. This visualizes the distributions of Person Ability (Theta) and Item Difficulty (Delta) on the same logit scale to check for targeting.
*   **`plot_icc.py`**: Contains functions to plot **Item Characteristic Curves (ICC)**, comparing the theoretical model curve against empirical data binning.
*   **`rasch_bayesian.py`**: Performs **Bayesian Estimation** of the Rasch model using **Stan** (`cmdstanpy`). It estimates Theta and Delta, calculates Infit/Outfit statistics, and generates PPC (Posterior Predictive Check) results.
*   **`rasch_cmle.py`**: Implements **Conditional Maximum Likelihood Estimation (CMLE)** for item parameters (mathematically removing person parameters during estimation) and subsequent MLE for person measures.
*   **`rasch_jmle.py`**: Implements **Joint Maximum Likelihood Estimation (JMLE)**, an iterative algorithm (Newton-Raphson) to estimate person and item parameters simultaneously.
*   **`rasch_ssa.py`**: Performs **Split-Sample Analysis** (using CMLE) to verify parameter invariance. It splits the sample in half and checks the correlation and stability of item estimates between the two groups.
*   **`rasch_ssa_bayes.py`**: Performs **Split-Sample Analysis using Bayesian Estimation**. It runs the Stan model on two halves of the data to check for parameter invariance using Bayesian estimates.
*   **`readBLOT.py`**: A utility module responsible for reading, parsing, and cleaning the raw dataset (`blot.txt`) into a usable pandas DataFrame.

### Jupyter Notebooks

*   **`eda.ipynb`**: **Exploratory Data Analysis**. A notebook for initial data inspection, descriptive statistics, and understanding the raw response patterns before modeling.
*   **`rasch_bayesian_visuals.ipynb`**: Interactive notebook for running the Bayesian analysis pipeline and generating visualizations (Wright Maps, Pathway Plots, etc.) specifically for the Bayesian results.
*   **`rasch_cmle_visuals.ipynb`**: Interactive notebook for running the CMLE analysis pipeline and generating visualizations for the CMLE/MLE results.
*   **`rasch_ssa_visuals.ipynb`**: Interactive notebook specialized for visualizing the results of the Split-Sample Analysis (e.g., scatter plots of estimates from Sample 1 vs. Sample 2).
