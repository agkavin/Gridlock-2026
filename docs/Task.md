# Gridlock

# Traffic demand prediction

Cities worldwide are increasingly turning to AI-powered solutions to
tackle the issue of traffic congestion. This problem disrupts the smooth
flow of transportation and poses a significant barrier to economic
growth. To address this challenge effectively, the first step is to
understand travel demand and patterns within urban areas
comprehensively. By harnessing the power of AI, cities, and regions aim
to gather critical insights into transportation dynamics. This will enable
them to implement data-driven strategies and solutions to alleviate
traffic congestion and promote more efficient mobility. Ultimately, this
endeavor will foster economic development and prosperity.

## Task

Design a system that helps us provide valuable insights into passenger
travel patterns, booking behavior, and trip cancellations, which can be
used for various analyses and predict demand in the travel industry.

## Dataset description

The dataset folder contains the following files:
train.csv: 77299 x  11
test.csv: 41778 x 10
sample_submission.csv: 5 x 2

## Variable description

The columns provided in the dataset are as follows:

| Column name | Description |
| --- | --- |
| Index | Represents the unique identification of datapoint |
| geohash | Represents geographic information regarding a place |
| day | Represents the day when the information is recorded |
| timestamp | Represents the timestamp of the record inserted in the system |
| RoadType | Represents the type of road in the nearby location |
| NumberofLanes | Represents the number of roads/lanes present in the location |
| LargeVehicles | Represents whether large vehicles are permitted on the specific roads/lanes |
| Landmarks | Represents whether there are any landmarks near the location |
| Temperature | Represents the temperature of the place |
| Weather | Represents the weather of the place |
| demand | Represents the demand of the traffic at the timestamp |

## Evaluation Metric

- **Score Formula:**
    
    \text{score}=\max\left(0,100\times R^2\right)
    
- Where:
    - `R²` is calculated using `metrics.r2_score(actual, predicted)`.
    - The final score is clipped at a minimum value of `0`.

## Result Submission Guidelines

- The index column is **`Index`**.
- The target column is **`demand`**.
- The submission file must be in **`.csv`** format only.
- The submission file size must be **41778 × 2**.
- Ensure that the submission file contains:
    - Correct index values as provided in the test file.
    - Correct column names as specified in `sample_submission.csv`.
