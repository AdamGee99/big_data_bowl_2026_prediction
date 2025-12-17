# NFL Big Data Bowl 2026 - Prediction

## Overview
- Goal: predict player movement after the ball is thrown using NFL player tracking data. Full info on [kaggle](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview).
- Six [CatBoost](https://catboost.ai/) models that predict player's future position using kinematics and derived features.
- Built on over 5 million frames across 14,108 NFL plays. 
- CV movement error ~ 0.77 yards (under 2.5 times the length of a football).
- Repo contains full pipeline that cleans player tracking data, performs EDA, derives important features, and trains/generates predictions in 5-fold CV in parallel.

## Data
- NFL player tracking [data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data) with over 5 million frames across the 2023-2024 season.
- On each play, the players to predict were the targeted receiver and any defensive player within 5 yards of the target receiver at throw, or is able to reach the ball landing location.
- Data cleaned to remove impossible speeds and accelerations or plays that were left recording too long. 

## Modelling Approach
- Predict future frame's direction, speed, acceleration.
- Contrasts most other approaches which predicted the future x,y positions. I figured updating the position based on kinematics is universal across the entire field whereas predicting x,y might depend on the current field position. 
- Used [CatBoost](https://catboost.ai/) since it provides quick accurate predictions with little tuning (biggest advantage since data is so large).
- Speed model predicted future log(speed) and back-transformed (since speeds are strictly positive). 
- Models trained on all the frames where the ball was in the air, or frames before the ball is thrown but the play is 62.5% overall complete. Initial 62.5% of frames in the play were not used to train the models. This cutoff was tuned through CV.


## Feature Engineering
Features list:
- **kinematics**: current and lagged difference in direction, speed, log(speed), acceleration. 
- **time features**: time until play complete, time elapsed, time elapsed post-throw, frame_id, frame_id of throw.
- **player-player features**: closest opponent and closest teammate distance, difference in direction to each other, difference in direction to ball, speed, acceleration.
- **ball landing point features**: distance to ball landing point, difference in direction to ball landing point, ball landing point distance to out of bounds.
- **other**: distance to out of bounds, difference in direction to nearest out of bounds, player height, player weight, absolute yardline number.

Each model used a different combination of these features (eg, speed model used current log(speed) instead of current speed).

<p align="center">
  <img src="visuals/feature_importance/dir_o_feat_imp_plot.png" width="49%" />
  <img src="visuals/feature_importance/speed_o_feat_imp_plot.png" width="49%" />
</p>

## Results
- Final results TBA. They are scored on the remaining games in the 2025-2026 regular season.
- Training data CV rmse around 0.77.
- These are accurate predictions. Average error of player movement when ball is in the air is roughly equivalent to 2.5 times the length of a football.

<p align="center">
  <img src="visuals/24/24_v2.gif" width="49%" />
  <img src="visuals/267/267_v2.gif" width="49%" />
</p>

<p align="center">
  <em>Predicted vs true position when ball is in the air for two sample plays.</em>
</p>

## Key Lessons
- Defense harder to predict than offense. Offense almost always goes straight to the ball but defense may be guarding other players.
- Predominantly important features were kinematics, especially in speed, acceleration models. Then ball features, then various time features. Surprisingly, player-player features did not seem to help much.
- Parallelizing CV over fold and plays was crucial for efficient training and tuning.
- Creating basic helper functions like calculating direction differences made life so much easier.
- Probably spent too much time on feature engineering, should have prioritized model architecture a bit more.

## Future Improvements
- Implement a deep learning approach with CNNs.
- Ensemble with other models (XGBoost, Light GBM, CNNs, ...).
- Data augmentation, like rotating plays around an axis, can help increase training set size.
  

