# User-Based Collaborative Filtering Recommender (Portfolio Summary)

> Note: Source code intentionally withheld due to company ownership. This README documents the design, scope, and impact for evaluation purposes.

## Project Snapshot
- Type: Memory‑based recommender system
- Paradigm: User–User Collaborative Filtering (explicit ratings)
- Dataset Schema: MovieLens 100K–style (users, items, 1–5 ratings, titles)
- Goal: Predict unseen item ratings and surface top‑N (prototype focused on top‑1) recommendations
- Tech: Python, pandas, NumPy, scikit-learn (LabelEncoder)

## Why It Matters
Delivers an interpretable baseline recommender:
- Fast to prototype
- Transparent similarity logic (useful for stakeholder trust)
- Serves as a yardstick before moving to latent factor or deep models

## Core Architecture
1. Data Preparation
   - Load ratings + metadata
   - Deduplicate user–item pairs
   - Label-encode users and items for compact integer IDs
2. Per-User Baseline
   - Use user median rating (robust vs. mean under skew/outliers)
3. Normalized Residual Store
   - For each user: item → (rating − user_median)
4. Inverted Index
   - Item → list of users who rated it (speeds neighbor lookup)
5. Similarity
   - On-demand cosine similarity over overlapping items (no full similarity matrix)
6. KNN Retrieval
   - Filter to users who rated target item, rank by similarity, keep top K
7. Prediction
   - user_median(u) + Σ(sim_uv * (r_v,i − user_median(v))) / Σ|sim_uv|
8. Recommendation
   - Score all unseen items for a user; pick argmax (extendable to top‑N)
9. Evaluation
   - Full pass over known ratings computing RMSE + mean absolute deviation (prototype level)

## Key Functions (Conceptual)
- similarity(u, v): Cosine over shared residuals
- get_KNN(k, u, item): Neighbor set restricted to raters of item
- predict_rating(u, item, k): Baseline + weighted deviation
- recommend_best_movie(u, k): Highest predicted rating among unseen items
- evaluate_recommender(k): Bulk RMSE / average absolute error

## Design Choices & Rationale
| Aspect | Choice | Reason |
|--------|--------|--------|
| Baseline | Median | Reduces influence of user outliers |
| Weight Denominator | Σ&vert;sim&vert; | Avoids cancellation of opposite-signed residuals |
| Similarity Computation | On-demand | Memory efficient for sparse user–item space |
| Data Structures | Dicts / inverted index | Quick overlap & neighbor filtering |
| Evaluation | In-sample RMSE (prototype) | Fast feedback loop; future: temporal or CV splits |

## Example Workflow (Abstracted Pseudocode)
```
load_ratings()
encode_ids()
compute_user_medians()
build_residual_maps()
build_item_to_users()

for each (u, i):
    neighbors = topK(similarity(u, v) for v in users_who_rated(i))
    predict via baseline + weighted deviations
```

## Performance Notes
- Complexity per prediction: O(R_i + K * avg_overlap) where R_i = users who rated item
- Amenable to:
  - Caching (similarity(u,v))
  - Vectorization / sparse ops
  - Parallel scoring for recommendation phase

## Potential Enhancements
- Significance weighting (down-weight small overlaps)
- Shrinkage-adjusted user baselines
- Item-based CF or hybrid blending
- Latent factor model (e.g., ALS / SVD) for comparison
- Cold-start strategies (content features, popularity priors)
- Top‑N ranking metrics (Precision@K, MAP, NDCG) on held-out split
- Similarity caching with LRU + invalidation hooks

## Limitations
- No temporal dynamics
- No handling of implicit feedback
- Cold-start users/items default to baseline
- Current evaluation lacks proper train/test separation (prototype constraint)

## Impact / Outcome
- Established a transparent benchmark
- Informed decisions on whether to invest in matrix factorization next
- Provided explainability: “Recommended because similar users rated it X above their norm”

## Ethical / Privacy Note
Implementation details and original proprietary adaptations are excluded. No personal user data beyond anonymized IDs was utilized.

## At a Glance (CV Blurb)
Implemented a user-based collaborative filtering recommender (median-centered cosine KNN) with on-demand similarity, achieving low baseline RMSE on a MovieLens-style dataset and establishing an interpretable benchmark for future hybrid and factorization models.
