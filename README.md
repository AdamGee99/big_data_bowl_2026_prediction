# NFL Big Data Bowl 2026 - Prediction

## Overview

The goal of this competition is to use NFL player tracking data to predict where players will move once the quarterback throws the ball. Full information can be found on [kaggle](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview).

## Data

My models were only trained on the data provided in the kaggle competition. This included over 5 million frames across 14,108 plays in the 2023-2024 NFL season. 

On each play, the players required for prediction included the target receiver and any defensive player within a 5 yard radius of the target receiver at throw, or is able to reach the ball landing location at 12yds/second.

## Approach

Six total models that predict the direction difference, speed, and acceleration of the proceeding frame for offensive and defensive players respectively. This contrasts most other approaches which predicted the future x,y positions. I figured updating the position based on direction, speed, and acceleration is a more universal approach across the entire field whereas predicting x,y might depend on the current field position.

I used [CatBoost](https://catboost.ai/) since it provides quick accurate predictions with little tuning. 

Speed model predicted the proceeding frame's log(speed) and back-transformed to the original scale, since speeds are strictly positive. 

Prior to training, data was cleaned to remove clearly impossible speeds and accelerations or plays that were left recording for too long. 

Models were trained on all the frames where the ball was in the air, or frames before the ball is thrown but the play is 62.5% overall complete. The initial 62.5% of frames in the play were not used to train the models. This cutoff was tuned through CV.


## Feature Engineering
Below is the entire feature list used to predict the direction, speed, and acceleration in the proceeding frame:
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

- Final results are to be announced. They are scored on the remaining games in the 2025-2026 regular season. My CV score was around 0.77 which seems to be quite far off the top scores of around 0.45. 
- Still these are good accurate predictions. On average my predictions when ball is in the air were 0.77 yards off the truth, which is equivalent to roughly 2.5 times the length of a football. 
- **ADD VISUALS**.. defense harder to predict..
