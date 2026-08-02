import json
from prediction_service import MODEL_FEATURES, predict_feasibility

sample = {f: 0 for f in MODEL_FEATURES}
sample['search_area'] = 'Baneshwor'
if 'avg_restaurant_rating_500m' in sample:
    sample['avg_restaurant_rating_500m'] = 0.0
if 'avg_review_ratings_500m' in sample:
    sample['avg_review_ratings_500m'] = 0.0
if 'nearest_restaurant_m' in sample:
    sample['nearest_restaurant_m'] = 500.0

print('MODEL_FEATURES =', MODEL_FEATURES)
try:
    result = predict_feasibility(sample)
    print(json.dumps(result, indent=2))
except Exception as e:
    print('ERROR:', repr(e))
