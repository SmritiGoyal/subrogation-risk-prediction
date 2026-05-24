# Data

This project uses an auto insurance claims dataset from a competition sponsored by a major US property & casualty insurance carrier. The dataset is referred to throughout the codebase as **TriGuard**, the masked carrier identifier used in the original challenge.

- **Source:** Academic competition dataset, restricted distribution
- **Domain:** Auto insurance subrogation (recovery from at-fault third parties)
- **Size:** ~18,000 training claims + ~12,000 test claims
- **Target:** `subrogation` — binary indicator of whether a claim presented a subrogation opportunity
- **Positive class rate:** ~22.9% (imbalanced)

## Files expected in this directory

| Filename | Rows | Columns | Description |
|---|---:|---:|---|
| `Training_TriGuard.csv` | 18,001 | 29 | Training data with `subrogation` target |
| `Testing_TriGuard.csv` | 12,000 | 28 | Test data, no target column |

These files are **not committed** to this repository because the underlying competition data is not redistributable. To run the pipeline you need to obtain the competition data through the original channel and place it at the paths above.

## Schema

The 29 columns in the training file (28 features + 1 target) cover four broad categories:

### Identifier (1 column)

| Column | Type | Description |
|---|---|---|
| `claim_number` | int | Row identifier — not a model feature |

### Driver & policyholder attributes (8 columns)

| Column | Type | Description |
|---|---|---|
| `year_of_born` | int | Driver's birth year (used to derive `driver_age`) |
| `gender` | str | Driver's gender |
| `living_status` | str | Housing status (e.g., owner, renter) |
| `annual_income` | float | Reported annual income, log-transformed during processing |
| `high_education_ind` | int | Binary indicator of higher-education attainment |
| `address_change_ind` | int | Binary indicator of recent address change |
| `email_or_tel_available` | int | Binary indicator of contact info on file |
| `age_of_DL` | int | Age at which the driver obtained their license |

### Claim attributes (10 columns)

| Column | Type | Description |
|---|---|---|
| `claim_date` | date | Date the claim was filed (used to extract `claim_year`, `claim_month`) |
| `zip_code` | str | Claimant zip code — dropped during processing |
| `claim_day_of_week` | str | Day of week the claim was filed |
| `accident_site` | str | Location category where the accident occurred |
| `accident_type` | str | Categorical classification of the accident |
| `witness_present_ind` | int | Binary witness present indicator |
| `policy_report_filed_ind` | int | Binary police report filed indicator |
| `past_num_of_claims` | int | Prior claim count for this policyholder |
| `liab_prct` | float | Estimated liability percentage (0-100) |
| `claim_est_payout` | float | Estimated claim payout amount, log-transformed during processing |

### Vehicle & operational attributes (9 columns)

| Column | Type | Description |
|---|---|---|
| `vehicle_made_year` | int | Manufacturing year (used to derive `vehicle_age`) |
| `vehicle_category` | str | Vehicle classification |
| `vehicle_color` | str | Vehicle color |
| `vehicle_price` | float | Vehicle market price, log-transformed during processing |
| `vehicle_weight` | float | Vehicle curb weight |
| `vehicle_mileage` | float | Recorded mileage, log-transformed during processing |
| `safety_rating` | float | Standardized vehicle safety rating |
| `in_network_bodyshop` | int | Binary in-network repair indicator |
| `channel` | str | Customer acquisition channel |

### Target (1 column, training only)

| Column | Type | Description |
|---|---|---|
| `subrogation` | int | 1 if claim had subrogation opportunity, else 0 |

## Data quality

The competition data is exceptionally clean: **~0.01% missingness** across all columns. The pipeline uses `dropna()` rather than column-specific imputation, dropping ~2 rows total. This is appropriate at this missingness rate; heavier imputation would be unjustified.

There are no duplicate `claim_number` values in either split.

## Reproducing the pipeline

Once the data is in place, from the repository root:

```bash
pip install -r requirements.txt
cp config.example.py config.py
python src/run_pipeline.py
```

Expected runtime: ~30-45 minutes on a modern laptop (Intel i9, 32 GB RAM) once Windows Defender is configured to exclude the project folder. Without that exclusion the pipeline can stall for hours during stacked ensemble training as antivirus scans the venv files.

## Why the dataset is masked

The original competition was sponsored by a specific US auto insurance carrier. To preserve the technical content of this portfolio project while respecting the competition's confidentiality conventions, the carrier name has been replaced with "TriGuard" throughout the codebase — including filenames, comments, and documentation. The model methodology and results are unchanged.