############################################### Libraries ############################################### 

library(tidyverse)
library(here)

############################################### TO DO ############################################### 

#' Steps
#' 1. Generate end model
#'      -by end model, I mean final in the line of models to generate the position predictions
#'      -this model simply predicts the player's position on the next frame using direction, speed, acceleration of current frame
#'      -this is an exact formula, no estimation...(simple kinematics formula)
#'      -just use the true observed direction, speed, acceleration values for now, don't worry about estimating them
#'      -see how well this matches up with true observed values
#'      -figure out what features are needed here (is direction, speed, acceleration all or is more neeeded? orientation?)
#' 
#' 
#' 2. start developing models for the each features needed above (direction, speed, acceleration)



############################################### Step 1 ############################################### 

#import data
train = read.csv(file = here("data", "train_clean.csv"))

source(here("helper.R"))

#' end model is a simple kinematics formula:
#' pos_next = pos_curr + vel_curr*0.1 + 0.5*acc_curr*(0.1^2) - 0.1 seconds between frames
#' this is a straight line in the current direction (dir_curr)


#' right now use true s, a, dir to predict next frame
#' if you know the player's true speed, acc, direction at frame i, how well can we predict next frame assuming they are constant?
#' 
#' 

#just use previous frame only for calculations for now
lead_frames = 0
lag_frames = 1
window_size = lag_frames + lead_frames

test = train %>%
  filter(player_to_predict) %>%
  group_by(game_id, nfl_id, play_id) %>%
  filter(cur_group_id() == 1) %>% #use this to filter for only 1 player in 1 play in 1 game
  select(game_id, play_id, throw, s, a, dir, o, frame_id, player_name, x, y) %>%
  ungroup() %>%
  #first calculate s, a, dir from true values of x,y after throw
  mutate(x_diff = lead(x, n = lead_frames) - lag(x, n = lag_frames), #the difference in x direction from previous frame in yards
         y_diff = lead(y, n = lead_frames) - lag(y, n = lag_frames), #the difference in y direction from previous frame in yards
         dist_diff = sqrt(x_diff^2 + y_diff^2), #distance travelled from previous frame in yards
         est_speed = dist_diff/((window_size)/10), #yards/second (1 frame is 0.1 seconds)
         est_acc_vector = (lead(est_speed, n = lead_frames) - lag(est_speed, n = lag_frames))/((window_size)/10), #this has directions (negative accelerations)
         est_acc_scalar = abs(est_acc_vector),
         est_dir = get_dir(x_diff = x_diff, y_diff = y_diff)) %>% 
  #now calculate position in next frame using true s, a, dir
  mutate(pred_dist_diff = est_speed*0.1 + est_acc_vector*0.5*0.1^2, #using speed + acc
         pred_dist_diff_2 = est_speed*0.1, #using speed only
         pred_x_sa = x + cos(est_dir*pi/180)*pred_dist_diff,
         pred_y_sa = y + sin(est_dir*pi/180)*pred_dist_diff,
         pred_x_s = x + cos(est_dir*pi/180)*pred_dist_diff_2,
         pred_y_s = y + sin(est_dir*pi/180)*pred_dist_diff_2,
  ) %>%
  #shift the predictions to the leading row (were predicting the next frame)
  mutate(across(starts_with("pred_"), lag))

#' the predicted position from the next frame is the exact same movement from the previous frame
#' that is, the player is going the same speed (travels the same distance) in the same direction

#direction of previous frame vs direction of next frame
ggplot(data = test, aes(x = est_dir, y = lead(est_dir))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  scale_x_continuous(limits = c(0, 360), breaks = c(0, 90, 180, 270, 360)) +
  scale_y_continuous(limits = c(0, 360), breaks = c(0, 90, 180, 270, 360)) +
  labs(x = "Direction in Current Frame", y = "Direction in Next Frame") +
  theme_bw()

#clearly this assumption doesn't make sense, this would mean player is travelling in straight line the whole time
#same with speed and acceleration, they aren't the same throughout the play

ggplot(data = test, aes(x = est_speed, y = lead(est_speed))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "Speed in Current Frame", y = "Speed in Next Frame") +
  theme_bw()

ggplot(data = test, aes(x = est_acc_vector, y = lead(est_acc_vector))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "Acceleration in Current Frame", y = "Acceleration in Next Frame") +
  theme_bw()


#' lets see how well it can predict using the true x,y coordinates though

#predicted position vs actual position
ggplot(data = test, mapping = aes(x = x, y = y)) + #use lead since the predictions are for the next (leading) frame
  geom_point(colour = "black") +
  geom_point(mapping = aes(x = pred_x_sa, y = pred_y_sa), colour = "blue") +
  geom_point(mapping = aes(x = pred_x_s, y = pred_y_s), colour = "red") + 
  scale_x_continuous(n.breaks = 20) +
  scale_y_continuous(n.breaks = 20) +
  theme_bw()

#some minor differences between using s only vs s + a


#get predictions post-throw
test_preds = test %>% filter(throw == "post") %>% select(frame_id, x, y, pred_x_sa, pred_y_sa, pred_x_s, pred_y_s)

#speed + acceleration
get_rmse(true_x = test_preds$x, true_y = test_preds$y, 
         pred_x = test_preds$pred_x_sa, pred_y = test_preds$pred_y_sa)
#speed only
get_rmse(true_x = test_preds$x, true_y = test_preds$y, 
         pred_x = test_preds$pred_x_s, pred_y = test_preds$pred_y_s)

#speed only is better, why?
#0.53 rmse, why isn't this better?






# # do it in a for loop to make sure everything's working properly
# # just use previous frame only
# 
# results = data.frame()
# for (i in 2:length(test)) {
#   if (i == 2) { #start by using first observation (true vals)
#     curr_x = test$x[2]
#     curr_y = test$y[2]
#     prev_x = test$x[1]
#     prev_y = test$y[1]
#     
#     #difference in x,y from previous frame
#     x_diff = curr_x - prev_x
#     y_diff = curr_y - prev_y
#     dist_diff = sqrt(x_diff^2 + y_diff^2)
#     prev_s = test$s[1]
#     
#     #metric calculations
#     curr_s = dist_diff/0.1
#     curr_acc_vector = curr_s - prev_s
#     curr_dir = (90 - (atan2(y = y_diff, x = x_diff)*180/pi)) %% 360
#     
#     #predict next frame position
#     pred_dist_diff = curr_s*0.1 + curr_acc_vector*0.5*0.1^2
#     pred_x = curr_x + cos(curr_dir*pi/180)*pred_dist_diff
#     pred_y = curr_y + sin(curr_dir*pi/180)*pred_dist_diff
#     
#     #store predicted values
#     results = results %>% rbind(c(
#       test$x[i], test$y[i], curr_x, curr_y, pred_x, pred_y, curr_s, curr_acc_vector, curr_dir
#     ))
#     
#     #update values
#     curr_x = pred_x
#     curr_y = pred_y
#     prev_x = curr_x
#     prev_y = curr_y
#     prev_s = curr_s
#   }
#   
#   #difference in x,y from previous frame
#   x_diff = curr_x - prev_x
#   y_diff = curr_y - prev_y
#   dist_diff = sqrt(x_diff^2 + y_diff^2)
#   
#   #metric calculations
#   curr_s = dist_diff/0.1
#   curr_acc_vector = curr_s - prev_s
#   curr_dir = (90 - (atan2(y = y_diff, x = x_diff)*180/pi)) %% 360
#   
#   #predict next frame position
#   pred_dist_diff = curr_s*0.1 + curr_acc_vector*0.5*0.1^2
#   pred_x = curr_x + cos(curr_dir*pi/180)*pred_dist_diff
#   pred_y = curr_y + sin(curr_dir*pi/180)*pred_dist_diff
#   
#   #store predicted values
#   results = results %>% rbind(c(
#     test$x[i], test$y[i], curr_x, curr_y, pred_x, pred_y, curr_s, curr_acc_vector, curr_dir
#   ))
#   
#   #update values
#   curr_x = pred_x
#   curr_y = pred_y
#   prev_x = curr_x
#   prev_y = curr_y
#   prev_s = curr_s
# }
# colnames(results) = c("true_x", "true_y", "curr_x", "curr_y", "pred_x", "pred_y", "curr_s", "curr_acc", "curr_dir")
# results





############################################### Misc Ideas I'll get to eventually ############################################### 

#' 1. check if theres anything in test thats not in train... (any player for eg)
#'      there will be

#' 2. add speed, direction, acceleration from predictions of model at every frame 
#'      recorded features s and a are magnitudes, not vectors
#'      experiment with including vectors (negative acceleration) in model, I think this makes a lot of sense - see which recorded or estimated gives better CV

#' 3. figure out how to incorporate orientation, cannot estimate from x,y - they use sensors in player's shoulder pads
#'      maybe estimate it? but how helpful will this even be?

#' 4. make a model to predict direction - then use that
#'      train model to predict direction, 
#'      if a player is turning for eg, the direction isn't just a straight line between current and previous frame, its going to keep curving
#'      this depends on speed, acc, and where the ball is landing
#'      two solutions to this, the gibbs way as before, or fit a model to predict direction

#' 5. certain players are dominant to one side, eg always like to cut right, use this info?



