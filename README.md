# NFL Big Data Bowl 2026 - Prediction

## Overview

Goal is to predict player movement after the ball is thrown using NFL player tracking data. Full information can be found on [kaggle](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview).

## Data

My models were only trained on the data provided in the kaggle competition. Includes over 5 million frames across 14,108 plays in the 2023-2024 NFL season. 

On each play, the players to predict were the targetted receiver and any defensive player within a 5 yard radius of the target receiver at throw, or is able to reach the ball landing location at 12yds/second.

## Approach

Six total models that predict the proceeding frame's direction difference, speed, and acceleration for offensive and defensive players separately. Contrasts most other approaches which predicted the future x,y positions. I figured updating the position based on direction, speed, and acceleration is universal across the entire field whereas predicting x,y might depend on the current field position. 

I used [CatBoost](https://catboost.ai/) since it provides quick accurate predictions with little tuning. The little tuning was the biggest advantage since the data is so large.

Speed model predicted the proceeding frame's log(speed) and back-transformed to the original scale, since speeds are strictly positive. 

Prior to training, data was cleaned to remove impossible speeds and accelerations or plays that were left recording for too long. 

Models trained on all the frames where the ball was in the air, or frames before the ball is thrown but the play is 62.5% overall complete. The initial 62.5% of frames in the play were not used to train the models. This cutoff was tuned through CV.


## Feature Engineering
Features list:
- **kinematics**: current direction, speed, log(speed), acceleration. Difference in direction, acceleration, log(speed) from previous frame. 
- **time features**: time until play complete, time elapsed, time elapsed post-throw, frame_id of throw.
- **player-player features**: closest opponent and closest teammate distance, direction difference, direction, speed, acceleration.
- **player info**: player height, weight.
- **ball landing point features**: distance to ball landing point, difference in direction to ball landing point.
- **out of bounds features**: distance to out of bounds, difference in direction to nearest out of bounds.

Each model used a different combination of these features (eg, speed model used current log(speed) instead of current speed).

The predominantly important features were kinematics, followed by the ball landing point features, then various time features. Surprisingly, the player-player features did not seem to help much. 

**show feature importance figures here**

## Results
- Final results TBA. They are scored on the remaining games in the 2025-2026 regular season.
- Training data CV rmse around 0.77.
- These are accurate predictions. My average error of player movement when ball is in the air is roughly equivalent to 2.5 times the length of a football.

<p align="center">
  <img src="visuals/267/267.gif" width="600"/>
  <br>
  <em>Predicted vs Actual Movement — sample play 1</em>
</p>

**Key Findings:**
  -Defense harder to predict than offense. Offense generally go where ball is landing but defense may go or may be already gaurding another player.
  

