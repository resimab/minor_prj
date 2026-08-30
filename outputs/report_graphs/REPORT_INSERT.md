# Report-ready additions

## Cluster-number selection

K-Means clustering was applied to the entropy-weighted feasibility score. The
elbow curve shows a pronounced reduction in within-cluster variation up to
approximately three clusters, followed by progressively smaller practical
gains. At `k = 3`, inertia was 145,368.1; increasing to four clusters
reduced it to 85,708.3, but would divide the decision output into an
additional tier that was not required by the Low–Moderate–High interpretation.
The silhouette score for three clusters was 0.544, indicating reasonable
separation. Although the numerical maximum occurred at two clusters
(0.577), that solution cannot represent the important intermediate
"Moderate" feasibility category. Therefore, `k = 3` was selected as a balance
between compactness, separation, and decision usefulness.

**Figure caption:** *Elbow and silhouette analysis for K-Means solutions from
one to ten clusters. The selected three-cluster solution provides reasonable
statistical separation and directly maps to Low, Moderate, and High restaurant
location feasibility tiers.*

## Three-cluster profile

The final clusters contained 1,102 Low,
1,692 Moderate, and
1,378 High observations. Their mean feasibility
scores were 19.60,
35.76, and
53.38, respectively. The learned score intervals
were approximately 4.10–27.65
for Low, 27.67–44.53
for Moderate, and 44.57–81.60
for High.

**Figure caption:** *Distribution and within-tier spread of entropy-weighted
feasibility scores for the three K-Means clusters.*

## Feature list and importance

The final model used four numerical criteria—Demand, Accessibility,
Competition, and Population—and one categorical feature, search area. The
feasibility score was constructed with entropy-derived weights of 41.82% for
Demand, 33.37% for Population, 16.01% for Competition, and 8.79% for
Accessibility. Permutation analysis of the deployed classifier similarly
identified Demand and Population as the most influential predictive inputs.
The search-area feature had near-zero global permutation importance, suggesting
that most predictive information was carried by the four substantive criteria.

**Figure caption:** *Entropy weights used in score construction and permutation
importance of the five inputs to the final feasibility classifier. Error bars
represent one standard deviation over repeated permutations.*

## Learning curve

The final tuned Logistic Regression pipeline was evaluated using a stratified
five-fold learning curve with macro F1 as the scoring measure. At the largest
training size, mean training macro F1 was
0.998, while mean validation macro F1
was 0.995. The small gap of
0.003
indicates limited overfitting, and the validation curve's stabilization shows
that the model generalizes consistently as more training data are added.

**Figure caption:** *Five-fold stratified learning curve for the final tuned
Logistic Regression model. Shaded bands show ±1 standard deviation across
folds.*

## Comparative analysis

Four classifiers were compared using an untouched test set, stratified
five-fold cross-validation, and area-held-out validation. XGBoost produced the
highest initial test macro F1 (0.9803), but Logistic Regression had nearly
identical cross-validation performance (0.9792) and the strongest area-held-out
macro F1 (0.9734). The area-held-out result was important because it tests
transfer to an unseen study area. Logistic Regression was therefore selected
for tuning; the saved final model achieved a test accuracy of 0.9940 and test
macro F1 of 0.9942.

**Figure caption:** *Macro F1 comparison of candidate models under test-set,
five-fold cross-validation, and area-held-out evaluation. Logistic Regression
showed the strongest spatial generalization and was selected for deployment.*

## Methodological note

The feasibility labels were produced from an entropy-weighted score and then
learned by a supervised classifier. Thus, the reported accuracy measures how
consistently the model reproduces the constructed feasibility tiers; it should
not be interpreted as external proof of restaurant business success. A future
validation study should compare predictions with independent outcomes such as
revenue, survival, rent-adjusted profitability, or observed foot traffic.
