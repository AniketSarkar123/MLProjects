# Dog vs Cat Classifier (Linear Regression)

This project demonstrates a simple **linear regression-based classifier** that separates dogs and cats based on:

- **Whisker Length** (X-axis)
- **Ear Flappiness** (Y-axis)

## How It Works
1. We create a dataset where:
   - Cats generally have **longer whiskers** but **less flappy ears**.
   - Dogs generally have **shorter whiskers** but **more flappy ears**.
2. A separating line (positive slope) is chosen to split the clusters.
3. A new test point is classified based on which side of the line it lies on.

## Visualization
The plot shows:
- Blue points → Cats
- Red points → Dogs
- Black line → Separation boundary

## Running the Notebook in Google Colab
1. Open the `.ipynb` file in Colab.
2. Run all cells.
3. Adjust the slope (`m`) and intercept (`b`) to improve separation.

---