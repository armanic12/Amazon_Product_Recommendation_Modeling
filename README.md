# Amazon Product Recommendation Modeling

A model-based collaborative filtering project that recommends products based on a user’s past purchases and ratings from similar users. The notebook builds an item similarity model using dimensionality reduction and correlation on the Amazon Beauty ratings dataset.

## Files in this repository
- `Amazon_Product_Recommendation.ipynb`  
  End-to-end workflow. Loads ratings, builds an item-item recommendation model with TruncatedSVD, computes correlations, and returns similar products for a selected item.
- `README.md`  
  Project summary and usage notes.
- Data file: `ratings_Beauty.csv`
  Sample-sized version of the file in the repository root. Download full-seized file from the source listed below.

## Dataset
- Name: Amazon Beauty product ratings  
- Format: CSV
- Source: [Kaggle - Amazon Ratings Dataset](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)

## Project overview
- **Goal**  
  Recommend products to a shopper based on item similarity learned from historical ratings.

- **Approach**  
  1. Load and clean the ratings data.  
  2. Build an item-user matrix.  
  3. Apply TruncatedSVD to reduce dimensionality and capture latent factors.  
  4. Compute an item-item correlation matrix in the reduced space.  
  5. Select a product and rank similar products by correlation.  
  6. Return the top results after removing the selected product.

## Methods and workflow
- **Data loading and cleaning**  
  Read `ratings_Beauty.csv`, drop missing values, inspect shape and head.

- **Matrix construction**  
  Create an item-user utility matrix suitable for model-based collaborative filtering.

- **Dimensionality reduction**  
  Use `sklearn.decomposition.TruncatedSVD` to project the utility matrix into a lower-rank space. This step preserves key structure while reducing noise.

- **Similarity computation**  
  Compute an item-item correlation matrix from the reduced representation with `numpy.corrcoef`.

- **Recommendation logic**  
  Pick a product index, map to the correlation vector, filter out the selected product, apply a correlation threshold, and show the top recommendations.

## Tools and skills used
- Python  
- NumPy, pandas  
- scikit-learn (TruncatedSVD)  
- Matrix factorization  
- Item-item collaborative filtering  
- Correlation analysis  
- Data cleaning and preprocessing  
- matplotlib for quick inspection plots  

## How to run
1. Download the dataset.  
2. Save as `ratings_Beauty.csv` in the repository root.  
3. Open `Amazon_Product_Recommendation.ipynb`.  
4. Run all cells.
